print('### USING HDRNET BT709 ARCH FROM:', __file__)

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
# 系数网络 LowResNet（双分支 + 融合，对齐 hdrnet_arch.py）
# =====================================================================
class LowResNetBT709(nn.Module):
    """
    双分支系数网络（和 hdrnet_arch.py 一样）：
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

        # ---- splat: 下采样到 spatial_bin ----
        splat_layers = []
        ch = in_ch
        for i in range(n_ds_layers):
            out_ch = cm * (2 ** i) * gd
            splat_layers.append(conv_lrelu(ch, out_ch, k=3, s=2, p=1))
            ch = out_ch
        self.splat = nn.Sequential(*splat_layers)
        splat_out_ch = ch

        # ---- global: 继续下采样到 4x4 -> FC -> 全局特征 ----
        n_global_layers = max(1, int(np.log2(grid_H / 4)))
        global_layers = []
        for i in range(n_global_layers):
            global_layers.append(conv_lrelu(ch, gc, k=3, s=2, p=1))
            ch = gc
        self.global_trunk = nn.Sequential(*global_layers) if global_layers else nn.Identity()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc1 = nn.Linear(gc, gc // 2)
        self.global_fc2 = nn.Linear(gc // 2, gc // 4)
        self.global_fc3 = nn.Linear(gc // 4, cm * 1 * gd)

        # ---- local: 保持 spatial_bin 分辨率 ----
        self.local_bottleneck = nn.Conv2d(splat_out_ch, gc, kernel_size=1)
        self.local_conv1 = conv_lrelu(gc, gc, k=3, s=1, p=1)
        self.local_conv2 = nn.Conv2d(gc, cm * 1 * gd, kernel_size=3, stride=1, padding=1, bias=False)

        # ---- prediction: 1x1 conv -> D*Cc 通道系数 ----
        self.pred_conv = nn.Conv2d(cm * 1 * gd, grid_D * coeffs_channels, kernel_size=1)

        # ---- 恒等初始化 ----
        # weight=0、bias 编码恒等系数（A=I, b=0），整张网格（含训练数据未覆盖的
        # 死区，如 guide 高区顶层）默认输出 = 输入，而非随机值。
        # 系数通道布局: k∈[0,9) 为 3x3 矩阵行优先展开（A[i][j] = k=i*3+j），
        # k∈[9,12) 为 bias。恒等即 A 对角线为 1、其余为 0。
        # 对旧 checkpoint 无影响（加载时被覆盖）；仅从头训练时生效。
        nn.init.zeros_(self.pred_conv.weight)
        nn.init.zeros_(self.pred_conv.bias)
        with torch.no_grad():
            self.pred_conv.bias.view(grid_D, coeffs_channels)[:, :9] = \
                torch.eye(3).flatten()  # A=I

    def forward(self, x):
        """x: (B, 3, fixed_ds_size, fixed_ds_size) -> grid: (B, D, Hg, Wg, Cc)"""
        B = x.size(0)

        # splat 下采样
        splat_feat = self.splat(x)

        # global 分支: 下采样 + 全局池化 + FC
        g = self.global_trunk(splat_feat)
        g = self.global_pool(g).view(B, -1)
        g = F.leaky_relu(self.global_fc1(g), 0.2)
        g = F.leaky_relu(self.global_fc2(g), 0.2)
        g = self.global_fc3(g).view(B, 1, 1, -1).permute(0, 3, 1, 2)

        # local 分支: 降维 + 空间卷积
        local_feat = self.local_bottleneck(splat_feat)
        local_feat = self.local_conv1(local_feat)
        local_feat = self.local_conv2(local_feat)

        # fusion: local + global -> relu -> 1x1 conv
        fusion = F.relu(local_feat + g)
        grid = self.pred_conv(fusion)

        # reshape -> (B, D, Hg, Wg, Cc)
        grid = grid.view(B, self.grid_D, self.coeffs_channels, self.grid_H, self.grid_W)
        grid = grid.permute(0, 1, 3, 4, 2).contiguous()
        return grid


# =====================================================================
# BT.709 亮度 GuideNet（固定公式，无需学习）
# =====================================================================
class GuideNetBT709(nn.Module):
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
# 三线性 Slicing + 仿射变换（和 hdrnet_arch.py 一样）
# =====================================================================
class BilateralSliceApplyBT709(nn.Module):
    """
    在 bilateral grid 上做三线性插值 + 仿射变换
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

        # 仿射变换: y = A @ x + b
        A = coeff[:, :9, :, :].reshape(B, 3, 3, H, W)
        b = coeff[:, 9:, :, :].reshape(B, 3, 1, H, W)
        # singleton 必须在 dim1 (输入通道轴): 广播沿输出维 i 复制 x,
        # dim2 收缩时 x 通道 j 与 A 列索引 j 配对 -> Σ_j A[i,j]*x[j]
        x_vec = x.reshape(B, 1, 3, H, W)
        return (A * x_vec).sum(dim=2) + b.squeeze(2)


# =====================================================================
# 完整 HDRNet BT.709 版本（可部署为纯 3D LUT）
# =====================================================================
@ARCH_REGISTRY.register()
class HDRnetBT709(nn.Module):
    """
    HDRNet BT.709 版本：
      LowResNet  - 和 hdrnet_arch.py 一样（双分支）
      GuideNet   - 固定 BT.709 亮度公式（无需学习）
      SliceApply - 三线性插值 + 仿射变换

    注意：由于 Bilateral Grid 具有空间变化特性，模型无法简化为纯 3D LUT
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

        self.lowres_net = LowResNetBT709(
            in_ch=num_in_ch, base_channels=base_channels,
            num_downsample=num_downsample, num_residual=num_residual,
            grid_H=grid_H, grid_W=grid_W, grid_D=grid_D,
            coeffs_channels=self.coeffs_channels, fixed_ds_size=fixed_ds_size,
            global_channels=global_channels,
        )
        self.guide_net = GuideNetBT709(in_ch=num_in_ch)  # 固定 BT.709
        self.slice_apply = BilateralSliceApplyBT709(
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

        # 三线性插值 + 仿射变换
        out = self.slice_apply(grid, guide, x)

        if strength < 1.0:
            out = strength * out + (1.0 - strength) * x

        if self.training:
            return out
        return torch.clamp(out, 0.0, 1.0)
