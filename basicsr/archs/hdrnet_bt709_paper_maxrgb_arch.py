print('### USING HDRNET BT709 PAPER MAXRGB ARCH FROM:', __file__)

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
# maxRGB GuideNet（固定公式，无需学习）
# =====================================================================
class GuideNetMaxRGB(nn.Module):
    """
    使用 max(R, G, B) 作为 guide 信号。

    与 BT.709 Y 相比，maxRGB 直接反映最接近削顶的通道，
    对高光保护更自然。
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """x: (B, 3, H, W) in [0, 1] -> guide: (B, 1, H, W) in [0, 1]"""
        guide, _ = x.max(dim=1, keepdim=True)  # (B, 1, H, W)
        return guide


# =====================================================================
# 系数网络 LowResNet（maxRGB 版，只输出 2 个系数）
# =====================================================================
class LowResNetBT709PaperMaxRGB(nn.Module):
    """
    基于 LowResNetBT709Paper，输出 2 个系数 (scale, offset)。

    Grid 输出: (B, D, Hg, Wg, 2)
      coeff[..., 0] = scale
      coeff[..., 1] = offset

    maxRGB' = scale × maxRGB + offset
    RGB_out = RGB_in × (maxRGB' / maxRGB)
    """

    def __init__(self, in_ch=3, base_channels=16, grid_H=16, grid_W=16, grid_D=8,
                 fixed_ds_size=256, global_channels=None):
        super().__init__()

        self.grid_H = grid_H
        self.grid_W = grid_W
        self.grid_D = grid_D
        self.coeffs_channels = 2
        self.fixed_ds_size = fixed_ds_size

        cm = base_channels
        gd = grid_D
        gc = global_channels if global_channels is not None else cm * 8 * gd
        n_ds_layers = int(np.log2(fixed_ds_size / grid_H))

        # ---- Splat: 下采样到 spatial_bin ----
        splat_layers = []
        ch = in_ch
        for i in range(n_ds_layers):
            out_ch = cm * (2 ** i) * gd
            splat_layers.append(conv_lrelu(ch, out_ch, k=3, s=2, p=1))
            ch = out_ch
        self.splat = nn.Sequential(*splat_layers)
        splat_out_ch = ch

        # ---- Global conv: 继续下采样到 4x4 ----
        n_global_layers = max(1, int(np.log2(grid_H / 4)))
        global_layers = []
        for i in range(n_global_layers):
            global_layers.append(conv_lrelu(ch, gc, k=3, s=2, p=1))
            ch = gc
        self.global_trunk = nn.Sequential(*global_layers) if global_layers else nn.Identity()

        # ---- Global FC: 先升维再降维 ----
        self.global_fc1 = nn.Linear(gc * 4 * 4, gc * 4)
        self.global_fc2 = nn.Linear(gc * 4, gc * 2)
        self.global_fc3 = nn.Linear(gc * 2, gc)

        # ---- Local conv: 保持 spatial_bin 分辨率 ----
        if splat_out_ch != gc:
            self.local_proj = nn.Conv2d(splat_out_ch, gc, kernel_size=1)
        else:
            self.local_proj = nn.Identity()
        self.local_conv1 = conv_lrelu(gc, gc, k=3, s=1, p=1)
        self.local_conv2 = conv_lrelu(gc, gc, k=3, s=1, p=1)

        # ---- 系数预测: 1x1 线性卷积 -> 2 系数 ----
        self.coeff_conv = nn.Conv2d(gc, grid_D * 2, kernel_size=1)

    def forward(self, x):
        """x: (B, 3, fixed_ds_size, fixed_ds_size) -> grid: (B, D, Hg, Wg, 2)"""
        B = x.size(0)

        splat_feat = self.splat(x)

        # global 分支
        g = self.global_trunk(splat_feat)
        g = g.view(B, -1)
        g = F.leaky_relu(self.global_fc1(g), 0.2)
        g = F.leaky_relu(self.global_fc2(g), 0.2)
        g = self.global_fc3(g).view(B, -1, 1, 1)

        # local 分支
        local_feat = self.local_proj(splat_feat)
        local_feat = self.local_conv1(local_feat)
        local_feat = self.local_conv2(local_feat)

        # fusion
        fusion = F.relu(local_feat + g)

        # 系数 -> reshape (B, D, Hg, Wg, 2)
        grid = self.coeff_conv(fusion)
        grid = grid.view(B, self.grid_D, 2, self.grid_H, self.grid_W)
        grid = grid.permute(0, 1, 3, 4, 2).contiguous()
        return grid


# =====================================================================
# maxRGB 三线性插值 + 等比缩放
# =====================================================================
class BilateralSliceApplyMaxRGB(nn.Module):
    """
    在 bilateral grid 上做三线性插值，得到每像素的 (scale, offset)，
    计算 maxRGB' = scale × maxRGB + offset，然后 RGB 等比缩放。

    等比缩放保证色度不变: R' = R × ratio, G' = G × ratio, B' = B × ratio
    其中 ratio = maxRGB' / maxRGB
    """

    def __init__(self, grid_D=8, grid_H=16, grid_W=16):
        super().__init__()
        self.D = grid_D
        self.Hg = grid_H
        self.Wg = grid_W

    def forward(self, grid, guide, x):
        """
        grid:  (B, D, Hg, Wg, 2)  — 每个格点 (scale, offset)
        guide: (B, 1, H, W)       — maxRGB ∈ [0, 1]
        x:     (B, 3, H, W)       — 原始 RGB 输入

        return: (B, 3, H, W)      — 等比缩放后的 RGB
        """
        B, D, Hg, Wg, Cc = grid.shape
        _, _, H, W = x.shape
        device = x.device

        # ---- 三线性插值坐标 ----
        z = torch.clamp(guide, 0.0, 1.0) * (D - 1)
        ys_lin = torch.linspace(0, Hg - 1, H, device=device)
        xs_lin = torch.linspace(0, Wg - 1, W, device=device)
        ys_2d, xs_2d = torch.meshgrid(ys_lin, xs_lin, indexing='ij')
        ys = ys_2d.view(1, 1, H, W).expand(B, 1, H, W)
        xs = xs_2d.view(1, 1, H, W).expand(B, 1, H, W)

        x0, y0, z0 = torch.floor(xs), torch.floor(ys), torch.floor(z)
        x1 = torch.clamp(x0 + 1, max=Wg - 1)
        y1 = torch.clamp(y0 + 1, max=Hg - 1)
        z1 = torch.clamp(z0 + 1, max=D - 1)
        dx, dy, dz = xs - x0, ys - y0, z - z0
        x0, y0, z0 = x0.long(), y0.long(), z0.long()
        x1, y1, z1 = x1.long(), y1.long(), z1.long()

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

        scale = coeff[:, 0:1, :, :]
        offset = coeff[:, 1:2, :, :]

        # ---- maxRGB' = scale × maxRGB + offset ----
        maxRGB = guide  # (B, 1, H, W)
        maxRGB_prime = scale * maxRGB + offset

        # ---- 等比缩放: ratio = maxRGB' / maxRGB ----
        eps = 1e-6
        ratio = maxRGB_prime / (maxRGB + eps)

        # maxRGB 极小时不缩放
        safe_mask = (maxRGB > eps).float()
        ratio = ratio * safe_mask + (1.0 - safe_mask)

        return x * ratio


# =====================================================================
# 完整 HDRNet BT.709 maxRGB 版
# =====================================================================
@ARCH_REGISTRY.register()
class HDRnetBT709PaperMaxRGB(nn.Module):
    """
    HDRNet BT.709 maxRGB 版：
      GuideNet   - max(R, G, B) 作为 guide（固定公式）
      LowResNet  - 输出 2 系数 grid (scale, offset)
      SliceApply - 三线性插值 + maxRGB' = scale×maxRGB + offset + RGB 等比缩放

    与 HDRnetBT709PaperLumaOnly 的区别：
      - Guide 用 maxRGB 而非 BT.709 Y
      - maxRGB 直接反映最接近削顶的通道，高光保护更自然
      - ratio = maxRGB'/maxRGB 保证输出 max 通道 ≤ maxRGB'

    配置示例 (yml):
      network_g:
        type: HDRnetBT709PaperMaxRGB
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

        self.lowres_net = LowResNetBT709PaperMaxRGB(
            in_ch=num_in_ch, base_channels=base_channels,
            grid_H=grid_H, grid_W=grid_W, grid_D=grid_D,
            fixed_ds_size=fixed_ds_size,
            global_channels=global_channels,
        )
        self.guide_net = GuideNetMaxRGB()
        self.slice_apply = BilateralSliceApplyMaxRGB(
            grid_D=grid_D, grid_H=grid_H, grid_W=grid_W,
        )

    def forward(self, x, strength: float = 1.0):
        """x: (B, 3, H, W) in [0, 1]"""
        x_low = F.interpolate(x, size=(self.fixed_ds_size, self.fixed_ds_size),
                              mode='bilinear', align_corners=False)
        grid = self.lowres_net(x_low)

        guide = self.guide_net(x)

        out = self.slice_apply(grid, guide, x)

        if strength < 1.0:
            out = strength * out + (1.0 - strength) * x

        if self.training:
            return out
        return torch.clamp(out, 0.0, 1.0)
