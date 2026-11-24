print('### USING HDRNET BT709 PAPER LUMA-ONLY ARCH FROM:', __file__)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.archs.hdrnet_bt709_arch import GuideNetBT709
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
# 系数网络 LowResNet（亮度版，只输出 2 个系数）
# =====================================================================
class LowResNetBT709PaperLumaOnly(nn.Module):
    """
    基于 LowResNetBT709Paper，只输出 2 个系数（scale, offset）。

    Grid 输出: (B, D, Hg, Wg, 2)
    每个格点的 2 个系数含义:
      coeff[..., 0] = scale   (亮度缩放因子)
      coeff[..., 1] = offset  (亮度偏移)

    Y' = scale × Y + offset
    最终 RGB 通过等比缩放得到: R' = R × (Y'/Y), G' = G × (Y'/Y), B' = B × (Y'/Y)
    """

    def __init__(self, in_ch=3, base_channels=16, grid_H=16, grid_W=16, grid_D=8,
                 fixed_ds_size=256, global_channels=None):
        super().__init__()

        self.grid_H = grid_H
        self.grid_W = grid_W
        self.grid_D = grid_D
        self.coeffs_channels = 2  # 固定为 2: scale + offset
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

        # ---- G3~G5 global fc: 先升维到 4*gc 再降维 ----
        self.global_fc1 = nn.Linear(gc * 4 * 4, gc * 4)
        self.global_fc2 = nn.Linear(gc * 4, gc * 2)
        self.global_fc3 = nn.Linear(gc * 2, gc)

        # ---- L1~L2 local conv: 保持 spatial_bin 分辨率 ----
        if splat_out_ch != gc:
            self.local_proj = nn.Conv2d(splat_out_ch, gc, kernel_size=1)
        else:
            self.local_proj = nn.Identity()
        self.local_conv1 = conv_lrelu(gc, gc, k=3, s=1, p=1)
        self.local_conv2 = conv_lrelu(gc, gc, k=3, s=1, p=1)

        # ---- A: pointwise linear -> 2 系数 (scale + offset) ----
        self.coeff_conv = nn.Conv2d(gc, grid_D * 2, kernel_size=1)

        # ---- 恒等初始化 ----
        # weight=0、bias 编码恒等系数（scale=1, offset=0），整张网格（含训练数据
        # 未覆盖的死区，如 guide 高区顶层）默认输出 = 输入，而非随机值。
        # 训练时梯度会正常更新 weight/bias，已训区域照常学习。
        nn.init.zeros_(self.coeff_conv.weight)
        nn.init.zeros_(self.coeff_conv.bias)
        with torch.no_grad():
            self.coeff_conv.bias.view(grid_D, 2)[:, 0] = 1.0  # scale=1

    def forward(self, x):
        """x: (B, 3, fixed_ds_size, fixed_ds_size) -> grid: (B, D, Hg, Wg, 2)"""
        B = x.size(0)

        # splat 下采样
        splat_feat = self.splat(x)

        # global 分支: 下采样到 4x4 + 展平 + FC
        g = self.global_trunk(splat_feat)
        g = g.view(B, -1)  # (B, gc*4*4)
        g = F.leaky_relu(self.global_fc1(g), 0.2)
        g = F.leaky_relu(self.global_fc2(g), 0.2)
        g = self.global_fc3(g).view(B, -1, 1, 1)  # (B, gc, 1, 1)

        # local 分支: 投影 + 空间卷积
        local_feat = self.local_proj(splat_feat)
        local_feat = self.local_conv1(local_feat)
        local_feat = self.local_conv2(local_feat)

        # F fusion: local + global（广播）-> relu
        fusion = F.relu(local_feat + g)

        # A pointwise linear -> 系数 -> reshape (B, D, Hg, Wg, 2)
        grid = self.coeff_conv(fusion)
        grid = grid.view(B, self.grid_D, 2, self.grid_H, self.grid_W)
        grid = grid.permute(0, 1, 3, 4, 2).contiguous()
        return grid


# =====================================================================
# 亮度三线性插值 + 等比缩放
# =====================================================================
class BilateralSliceApplyLumaOnly(nn.Module):
    """
    在 bilateral grid 上做三线性插值，得到每像素的 (scale, offset)，
    计算 Y' = scale × Y + offset，然后 RGB 等比缩放。

    等比缩放保证色度不变: R' = R × (Y'/Y), G' = G × (Y'/Y), B' = B × (Y'/Y)
    """

    def __init__(self, grid_D=8, grid_H=16, grid_W=16):
        super().__init__()
        self.D = grid_D
        self.Hg = grid_H
        self.Wg = grid_W

    def forward(self, grid, guide, x):
        """
        grid:  (B, D, Hg, Wg, 2)  — 每个格点 (scale, offset)
        guide: (B, 1, H, W)       — BT.709 亮度 Y ∈ [0, 1]
        x:     (B, 3, H, W)       — 原始 RGB 输入

        return: (B, 3, H, W)      — 等比缩放后的 RGB
        """
        B, D, Hg, Wg, Cc = grid.shape
        _, _, H, W = x.shape
        device = x.device

        # ---- 三线性插值坐标 ----
        z = torch.clamp(guide, 0.0, 1.0) * (D - 1)  # (B, 1, H, W)
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

        # 三线性插值权重
        w000 = (1 - dx) * (1 - dy) * (1 - dz)
        w001 = dx * (1 - dy) * (1 - dz)
        w010 = (1 - dx) * dy * (1 - dz)
        w011 = dx * dy * (1 - dz)
        w100 = (1 - dx) * (1 - dy) * dz
        w101 = dx * (1 - dy) * dz
        w110 = (1 - dx) * dy * dz
        w111 = dx * dy * dz

        # 插值得到每像素的 (scale, offset)
        coeff = (g000*w000 + g001*w001 + g010*w010 + g011*w011 +
                 g100*w100 + g101*w101 + g110*w110 + g111*w111)
        # coeff: (B, 2, H, W)

        scale = coeff[:, 0:1, :, :]    # (B, 1, H, W)
        offset = coeff[:, 1:2, :, :]   # (B, 1, H, W)

        # ---- 计算 Y' = scale × Y + offset ----
        Y = guide  # (B, 1, H, W)
        Y_prime = scale * Y + offset   # (B, 1, H, W)

        # ---- 等比缩放: ratio = Y' / Y ----
        # 防止除零: Y 很小时 ratio = 1 (不缩放)
        eps = 1e-6
        ratio = Y_prime / (Y + eps)

        # Y 极小时（接近黑色），不做缩放
        safe_mask = (Y > eps).float()
        ratio = ratio * safe_mask + (1.0 - safe_mask)

        # ---- RGB 等比缩放 ----
        return x * ratio  # (B, 3, H, W)


# =====================================================================
# 完整 HDRNet BT.709 亮度版（只改亮度，色度不变）
# =====================================================================
@ARCH_REGISTRY.register()
class HDRnetBT709PaperLumaOnly(nn.Module):
    """
    HDRNet BT.709 亮度版：
      LowResNet  - 输出 2 系数 grid (scale, offset)
      GuideNet   - 固定 BT.709 亮度公式（无需学习）
      SliceApply - 三线性插值 + Y' = scale×Y + offset + RGB 等比缩放

    特性:
      - 只改变亮度，色度严格不变（RGB 等比缩放）
      - 参数量比 12 系数版本大幅减少（coeff_conv 输出从 D×12 降为 D×2）
      - 输出保证 RGB 比例不变，即 R:G:B 恒定

    配置示例 (yml):
      network_g:
        type: HDRnetBT709PaperLumaOnly
        num_in_ch: 3
        num_out_ch: 3
        grid_H: 32
        grid_W: 32
        grid_D: 16
        base_channels: 1
        fixed_ds_size: 512
        global_channels: 128
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, grid_H=16, grid_W=16, grid_D=8,
                 base_channels=16, fixed_ds_size=256, global_channels=None):
        super().__init__()
        assert num_in_ch == 3 and num_out_ch == 3

        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.grid_H = grid_H
        self.grid_W = grid_W
        self.grid_D = grid_D
        self.fixed_ds_size = fixed_ds_size

        self.lowres_net = LowResNetBT709PaperLumaOnly(
            in_ch=num_in_ch, base_channels=base_channels,
            grid_H=grid_H, grid_W=grid_W, grid_D=grid_D,
            fixed_ds_size=fixed_ds_size,
            global_channels=global_channels,
        )
        self.guide_net = GuideNetBT709(in_ch=num_in_ch)  # 固定 BT.709
        self.slice_apply = BilateralSliceApplyLumaOnly(
            grid_D=grid_D, grid_H=grid_H, grid_W=grid_W,
        )

    def forward(self, x, strength: float = 1.0):
        """x: (B, 3, H, W) in [0, 1]"""
        # Grid 网络: 固定尺寸下采样输入
        x_low = F.interpolate(x, size=(self.fixed_ds_size, self.fixed_ds_size),
                              mode='bilinear', align_corners=False)
        grid = self.lowres_net(x_low)

        # Guide 网络: BT.709 亮度（固定公式）
        guide = self.guide_net(x)

        # 三线性插值 + 等比缩放
        out = self.slice_apply(grid, guide, x)

        if strength < 1.0:
            out = strength * out + (1.0 - strength) * x

        if self.training:
            return out
        return torch.clamp(out, 0.0, 1.0)
