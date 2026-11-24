print('### USING HDRNET BT709 PAPER LEGACY (pre affine-fix) ARCH FROM:', __file__)

# =====================================================================
# 基于 hdrnet_bt709_paper_arch.py 创建的 Legacy 自包含版本。
#
# 与原文件的区别：
#   1. GuideNetBT709Legacy / BilateralSliceApplyBT709Legacy 不再从
#      hdrnet_bt709_arch.py 导入，而是全部内联到本文件中
#   2. 仿射变换保留修复前的旧实现 (x.reshape(B, 3, 1, H, W))，
#      即行和缩放语义 x[i] * Σ_j A[i,j]，而非真正的 A @ x
#
# 用途：加载/复现 2026-10-21 仿射广播修复之前训练的旧权重
#      （如 train_hdrnet_bt709_sdr2hdr_*_v2 等实验的 net_g_*.pth）。
#      旧权重是在旧语义下学出来的，必须搭配本文件实现才能复现。
#
# 注意：新训练请使用 hdrnet_bt709_paper_arch.py（已修复，真正的 A @ x + b）。
# =====================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY


# =====================================================================
# 工具函数
# =====================================================================
def conv_lrelu(in_ch, out_ch, k, s, p, bias=True):
    """Conv + LeakyReLU 组合块"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=bias),
        nn.LeakyReLU(0.2, inplace=True),
    )


# =====================================================================
# 系数网络 LowResNet（严格对齐论文 Table 1）
# =====================================================================
class LowResNetBT709PaperLegacy(nn.Module):
    """
    对齐论文 "Deep Bilateral Learning for Real-Time Image Enhancement"
    (SIGGRAPH 2017) Table 1 的通道配置：

      S1~S4 (splat conv):  通道 = cm * 2^i * gd      （论文: 8→16→32→64）
      G1~G2 (global conv): 通道 = gc = cm * 8 * gd    （论文: 64）
      G3~G5 (global fc):   16*gc → 4*gc → 2*gc → gc   （论文: 1024→256→128→64，先升后降）
      L1~L2 (local conv):  通道 = gc                  （论文: 64）
      F     (fusion):      relu(local + global)，宽度 = gc
      A     (pointwise linear): 1x1 线性卷积 gc → D*12  （论文: 64→96）

    输出 grid: (B, D, Hg, Wg, Cc)
    """

    def __init__(self, in_ch=3, base_channels=16, num_downsample=3, num_residual=2,
                 grid_H=16, grid_W=16, grid_D=8, coeffs_channels=12, fixed_ds_size=256,
                 global_channels=None):
        super().__init__()

        self.grid_H = grid_H
        self.grid_W = grid_W
        self.grid_D = grid_D
        self.coeffs_channels = coeffs_channels
        self.fixed_ds_size = fixed_ds_size

        cm = base_channels
        gd = grid_D
        gc = global_channels if global_channels is not None else cm * 8 * gd
        n_ds_layers = int(np.log2(fixed_ds_size / grid_H))

        # ---- S1~S4 splat: 下采样到 spatial_bin ----
        splat_layers = []
        ch = in_ch
        for i in range(n_ds_layers):
            out_ch = cm * (2 ** i) * gd
            splat_layers.append(conv_lrelu(ch, out_ch, k=3, s=2, p=1))
            ch = out_ch
        self.splat = nn.Sequential(*splat_layers)
        splat_out_ch = ch

        # ---- G1~G2 global conv: 继续下采样到 4x4 ----
        n_global_layers = max(1, int(np.log2(grid_H / 4)))
        global_layers = []
        for i in range(n_global_layers):
            global_layers.append(conv_lrelu(ch, gc, k=3, s=2, p=1))
            ch = gc
        self.global_trunk = nn.Sequential(*global_layers) if global_layers else nn.Identity()

        # ---- G3~G5 global fc: 先升维到 4*gc 再降维（论文 1024→256→128→64）----
        # 注意：论文中 4x4 特征图直接展平（不做全局池化），向量长度 = gc*4*4 = 16*gc
        self.global_fc1 = nn.Linear(gc * 4 * 4, gc * 4)
        self.global_fc2 = nn.Linear(gc * 4, gc * 2)
        self.global_fc3 = nn.Linear(gc * 2, gc)

        # ---- L1~L2 local conv: 保持 spatial_bin 分辨率，宽度 = gc ----
        if splat_out_ch != gc:
            # 显式指定 global_channels 导致宽度不一致时，用 1x1 投影对齐
            self.local_proj = nn.Conv2d(splat_out_ch, gc, kernel_size=1)
        else:
            self.local_proj = nn.Identity()
        self.local_conv1 = conv_lrelu(gc, gc, k=3, s=1, p=1)
        self.local_conv2 = conv_lrelu(gc, gc, k=3, s=1, p=1)

        # ---- A: pointwise linear，1x1 线性卷积（无激活）输出系数 ----
        self.coeff_conv = nn.Conv2d(gc, grid_D * coeffs_channels, kernel_size=1)

        # ---- 恒等初始化 ----
        # weight=0、bias 编码恒等系数（A=I, b=0），整张网格（含训练数据未覆盖的
        # 死区，如 guide 高区顶层）默认输出 = 输入，而非随机值。
        # 系数通道布局: k∈[0,9) 为 3x3 矩阵行优先展开（A[i][j] = k=i*3+j），
        # k∈[9,12) 为 bias。恒等即 A 对角线为 1、其余为 0。
        # 注: 旧式广播 (x_vec singleton 在 dim2) 下有效变换为逐通道缩放
        # y[i] = x[i]·Σ_j A[i,j] + b[i]，A=I 同样是恒等，故初始化方式不变。
        # 对旧 checkpoint 无影响（加载时被覆盖）；仅从头训练时生效。
        nn.init.zeros_(self.coeff_conv.weight)
        nn.init.zeros_(self.coeff_conv.bias)
        with torch.no_grad():
            self.coeff_conv.bias.view(grid_D, coeffs_channels)[:, :9] = \
                torch.eye(3).flatten()  # A=I

    def forward(self, x):
        """x: (B, 3, fixed_ds_size, fixed_ds_size) -> grid: (B, D, Hg, Wg, Cc)"""
        B = x.size(0)

        # splat 下采样
        splat_feat = self.splat(x)

        # global 分支: 下采样到 4x4 + 展平 + FC（输出宽度 = gc）
        g = self.global_trunk(splat_feat)
        g = g.view(B, -1)  # (B, gc*4*4)
        g = F.leaky_relu(self.global_fc1(g), 0.2)
        g = F.leaky_relu(self.global_fc2(g), 0.2)
        g = self.global_fc3(g).view(B, -1, 1, 1)  # (B, gc, 1, 1)

        # local 分支: 投影 + 空间卷积（输出宽度 = gc）
        local_feat = self.local_proj(splat_feat)
        local_feat = self.local_conv1(local_feat)
        local_feat = self.local_conv2(local_feat)

        # F fusion: local + global（广播）-> relu
        fusion = F.relu(local_feat + g)

        # A pointwise linear -> 系数 -> reshape (B, D, Hg, Wg, Cc)
        grid = self.coeff_conv(fusion)
        grid = grid.view(B, self.grid_D, self.coeffs_channels, self.grid_H, self.grid_W)
        grid = grid.permute(0, 1, 3, 4, 2).contiguous()
        return grid


# =====================================================================
# BT.709 亮度 GuideNet（固定公式，无需学习）—— 自包含副本
# =====================================================================
class GuideNetBT709Legacy(nn.Module):
    """
    使用 BT.709 标准亮度公式作为 guide：
    Y = 0.2126 * R + 0.7152 * G + 0.0722 * B

    无需学习参数，深度完全由输入 RGB 决定
    """

    def __init__(self, in_ch=3):
        super().__init__()
        # BT.709 系数（固定，不参与训练）
        self.register_buffer('coeffs', torch.tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1))

    def forward(self, x):
        """x: (B, 3, H, W) in [0, 1] -> guide: (B, 1, H, W) in [0, 1]"""
        # 计算 BT.709 亮度
        guide = (x * self.coeffs).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        # 已经是 [0, 1] 范围，无需 sigmoid
        return guide


# =====================================================================
# 三线性 Slicing + 仿射变换（修复前实现，自包含，不从别处导入）
# =====================================================================
class BilateralSliceApplyBT709Legacy(nn.Module):
    """
    在 bilateral grid 上做三线性插值 + 仿射变换

    ⚠️ 仿射变换为修复前的旧实现：
       x_vec = x.reshape(B, 3, 1, H, W)，singleton 在 dim2，
       实际计算是行和缩放 x[i] * Σ_j A[i,j]，而非 A @ x。
       仅用于旧权重推理复现，不要用于新训练。
    """

    def __init__(self, coeffs_channels=12, grid_D=8, grid_H=16, grid_W=16):
        super().__init__()
        assert coeffs_channels == 12, "需要 12 个系数 (3x3 矩阵 + 3 bias)"
        self.Cc = coeffs_channels
        self.D = grid_D
        self.Hg = grid_H
        self.Wg = grid_W

    def forward(self, grid, guide, x):
        B, D, Hg, Wg, Cc = grid.shape
        _, _, H, W = x.shape
        device = x.device

        # 三线性插值坐标
        z = guide * (D - 1)
        ys_lin = torch.linspace(0, Hg - 1, H, device=device)
        xs_lin = torch.linspace(0, Wg - 1, W, device=device)
        ys_2d, xs_2d = torch.meshgrid(ys_lin, xs_lin, indexing='ij')
        ys = ys_2d.view(1, 1, H, W).expand(B, 1, H, W)
        xs = xs_2d.view(1, 1, H, W).expand(B, 1, H, W)

        # 下取整 / 上取整索引
        x0, y0, z0 = torch.floor(xs), torch.floor(ys), torch.floor(z)
        x1 = torch.clamp(x0 + 1, max=Wg - 1)
        y1 = torch.clamp(y0 + 1, max=Hg - 1)
        z1 = torch.clamp(z0 + 1, max=D - 1)
        dx, dy, dz = xs - x0, ys - y0, z - z0
        x0, y0, z0 = x0.long(), y0.long(), z0.long()
        x1, y1, z1 = x1.long(), y1.long(), z1.long()

        # grid 展平 + gather 8 个顶点
        grid_flat = grid.view(B, D * Hg * Wg, Cc)

        def idx(z_, y_, x_):
            return (z_ * Hg + y_) * Wg + x_

        def gather(grid_flat, idx_):
            idx_flat = idx_.view(B, -1)
            idx_expand = idx_flat.unsqueeze(-1).expand(-1, -1, Cc)
            return torch.gather(grid_flat, 1, idx_expand).permute(0, 2, 1).contiguous().view(B, Cc, H, W)

        g000 = gather(grid_flat, idx(z0, y0, x0))
        g001 = gather(grid_flat, idx(z0, y0, x1))
        g010 = gather(grid_flat, idx(z0, y1, x0))
        g011 = gather(grid_flat, idx(z0, y1, x1))
        g100 = gather(grid_flat, idx(z1, y0, x0))
        g101 = gather(grid_flat, idx(z1, y0, x1))
        g110 = gather(grid_flat, idx(z1, y1, x0))
        g111 = gather(grid_flat, idx(z1, y1, x1))

        # 三线性插值
        w000 = (1 - dx) * (1 - dy) * (1 - dz)
        w001 = dx * (1 - dy) * (1 - dz)
        w010 = (1 - dx) * dy * (1 - dz)
        w011 = dx * dy * (1 - dz)
        w100 = (1 - dx) * (1 - dy) * dz
        w101 = dx * (1 - dy) * dz
        w110 = (1 - dx) * dy * dz
        w111 = dx * dy * dz
        coeff = (g000*w000 + g001*w001 + g010*w010 + g011*w011 +
                 g100*w100 + g101*w101 + g110*w110 + g111*w111)

        # 仿射变换（修复前的旧实现，勿改）
        # singleton 在 dim2: x 通道与 A 的行索引 i 配对 -> 行和缩放 x[i] * Σ_j A[i,j]
        A = coeff[:, :9, :, :].reshape(B, 3, 3, H, W)
        b = coeff[:, 9:, :, :].reshape(B, 3, 1, H, W)
        x_vec = x.reshape(B, 3, 1, H, W)
        return (A * x_vec).sum(dim=2) + b.squeeze(2)


# =====================================================================
# 完整 HDRNet BT.709 论文对齐版（Legacy，修复前仿射实现）
# =====================================================================
@ARCH_REGISTRY.register()
class HDRnetBT709PaperLegacy(nn.Module):
    """
    HDRNet BT.709 论文对齐版（Legacy）：
      LowResNet  - 通道配置严格对齐论文 Table 1
      GuideNet   - 固定 BT.709 亮度公式（自包含，无需学习）
      SliceApply - 三线性插值 + 仿射变换（修复前旧实现，自包含）

    ⚠️ 仅用于加载 2026-10-21 仿射修复之前训练的旧权重。
       网络结构与 HDRnetBT709Paper 完全相同，权重可直接互载
      （state_dict 的参数名一致），但前向仿射语义为旧版。
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, grid_H=16, grid_W=16, grid_D=8,
                 base_channels=16, num_downsample=3, num_residual=2, fixed_ds_size=256,
                 global_channels=None):
        super().__init__()
        assert num_in_ch == 3 and num_out_ch == 3

        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.grid_H = grid_H
        self.grid_W = grid_W
        self.grid_D = grid_D
        self.coeffs_channels = 12
        self.fixed_ds_size = fixed_ds_size

        self.lowres_net = LowResNetBT709PaperLegacy(
            in_ch=num_in_ch, base_channels=base_channels,
            num_downsample=num_downsample, num_residual=num_residual,
            grid_H=grid_H, grid_W=grid_W, grid_D=grid_D,
            coeffs_channels=self.coeffs_channels, fixed_ds_size=fixed_ds_size,
            global_channels=global_channels,
        )
        self.guide_net = GuideNetBT709Legacy(in_ch=num_in_ch)  # 固定 BT.709
        self.slice_apply = BilateralSliceApplyBT709Legacy(
            coeffs_channels=self.coeffs_channels, grid_D=grid_D,
            grid_H=grid_H, grid_W=grid_W,
        )

    def forward(self, x, strength: float = 1.0):
        """x: (B, 3, H, W) in [0, 1]"""
        # Grid 网络: 固定尺寸下采样输入
        x_low = F.interpolate(x, size=(self.fixed_ds_size, self.fixed_ds_size),
                              mode='bilinear', align_corners=False)
        grid = self.lowres_net(x_low)

        # Guide 网络: BT.709 亮度（固定公式）
        guide = self.guide_net(x)

        # 三线性插值 + 仿射变换（旧实现）
        out = self.slice_apply(grid, guide, x)

        if strength < 1.0:
            out = strength * out + (1.0 - strength) * x

        if self.training:
            return out
        return torch.clamp(out, 0.0, 1.0)
