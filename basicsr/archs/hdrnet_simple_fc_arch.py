print('### USING HDRNET SIMPLE FC ARCH FROM:', __file__)

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY


# ---------------------------
# 工具函数
# ---------------------------
def conv_lrelu(in_ch, out_ch, k, s, p, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=bias),
        nn.LeakyReLU(0.2, inplace=True),
    )


def srgb_to_linear(x):
    x = x.clamp(1e-6, 1.0)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = x.clamp(1e-6, 1.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1.0 / 2.4)) - 0.055)


# ---------------------------
# LowResNet (global pool + FC)
# ---------------------------
class LowResNetFC(nn.Module):
    """
    简化版低分辨率网络：
    - trunk 下采样 + global pool + FC 生成 grid
    - 输出 grid: (B, D, Hg, Wg, 1) 每格 1 个 scale
    """

    def __init__(self, in_ch=3, base_channels=16, num_downsample=3, num_residual=2,
                 grid_H=32, grid_W=32, grid_D=16, fixed_ds_size=512):
        super().__init__()

        self.grid_H = grid_H
        self.grid_W = grid_W
        self.grid_D = grid_D
        self.fixed_ds_size = fixed_ds_size

        # trunk: 下采样到 grid_H × grid_W
        trunk_layers = []
        ch = in_ch
        for i in range(num_downsample):
            out_ch = base_channels * (2 ** i)
            trunk_layers.append(conv_lrelu(ch, out_ch, k=3, s=2, p=1))
            ch = out_ch
        # residual layers
        for _ in range(num_residual):
            trunk_layers.append(conv_lrelu(ch, ch, k=3, s=1, p=1))
        self.trunk = nn.Sequential(*trunk_layers)

        # global pool + FC
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(ch, grid_D * grid_H * grid_W * 1)

    def forward(self, x):
        """
        x: (B, 3, H, W) in [0,1]
        return: grid (B, D, Hg, Wg, 1)
        """
        B = x.size(0)
        feat = self.trunk(x)
        pooled = self.pool(feat).view(B, -1)
        grid = self.fc(pooled).view(B, self.grid_D, self.grid_H, self.grid_W, 1)
        return grid


# ---------------------------
# GuideNet (3 层 conv)
# ---------------------------
class GuideNetFC(nn.Module):
    """
    引导网络：
    - 3 层 conv + sigmoid
    - 全分辨率输入
    """

    def __init__(self, in_ch=3, guide_channels=16):
        super().__init__()
        self.conv1 = conv_lrelu(in_ch, guide_channels, k=1, s=1, p=0)
        self.conv2 = conv_lrelu(guide_channels, guide_channels, k=3, s=1, p=1)
        self.conv3 = conv_lrelu(guide_channels, guide_channels, k=3, s=1, p=1)
        self.conv_out = nn.Conv2d(guide_channels, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        g = self.conv1(x)
        g = self.conv2(g)
        g = self.conv3(g)
        return torch.sigmoid(self.conv_out(g))


# ---------------------------
# BilateralSliceScale (三线性插值 + scale)
# ---------------------------
class BilateralSliceScaleFC(nn.Module):
    """
    简化版 bilateral slice apply：
    - 只输出 scale 系数
    - 在线性空间做同比例缩放
    """

    def __init__(self, grid_D=16, grid_H=32, grid_W=32):
        super().__init__()
        self.D = grid_D
        self.Hg = grid_H
        self.Wg = grid_W

    def forward(self, grid, guide, x):
        """
        grid:  (B, D, Hg, Wg, 1)  每格 1 个 scale
        guide: (B, 1, H, W)       in [0,1]
        x:     (B, 3, H, W)       sRGB [0,1]
        return: (B, 3, H, W)      sRGB [0,1]
        """
        B, D, Hg, Wg, _ = grid.shape
        _, _, H, W = x.shape
        device = x.device

        # 转成线性空间
        x_linear = srgb_to_linear(x)

        # 深度坐标 z ∈ [0, D-1]
        z = guide * (D - 1)

        # 空间坐标
        ys_lin = torch.linspace(0, Hg - 1, H, device=device)
        xs_lin = torch.linspace(0, Wg - 1, W, device=device)
        ys_2d, xs_2d = torch.meshgrid(ys_lin, xs_lin, indexing='ij')
        ys = ys_2d.view(1, 1, H, W).expand(B, 1, H, W)
        xs = xs_2d.view(1, 1, H, W).expand(B, 1, H, W)

        # 下取整 / 上取整
        x0, y0, z0 = torch.floor(xs), torch.floor(ys), torch.floor(z)
        x1 = torch.clamp(x0 + 1, max=Wg - 1)
        y1 = torch.clamp(y0 + 1, max=Hg - 1)
        z1 = torch.clamp(z0 + 1, max=D - 1)
        dx, dy, dz = xs - x0, ys - y0, z - z0
        x0, y0, z0 = x0.long(), y0.long(), z0.long()
        x1, y1, z1 = x1.long(), y1.long(), z1.long()

        # grid 展平 + gather
        grid_flat = grid.view(B, D * Hg * Wg, 1)

        def idx(z_, y_, x_):
            return (z_ * Hg + y_) * Wg + x_

        def gather(grid_flat, idx_):
            idx_flat = idx_.view(B, -1)
            idx_expand = idx_flat.unsqueeze(-1).expand(-1, -1, 1)
            g = torch.gather(grid_flat, 1, idx_expand)
            return g.permute(0, 2, 1).contiguous().view(B, 1, H, W)

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

        scale = (g000*w000 + g001*w001 + g010*w010 + g011*w011 +
                 g100*w100 + g101*w101 + g110*w110 + g111*w111)

        # 约束 scale
        scale = scale.clamp(0.01, 10.0)

        # 在线性空间做同比例缩放
        y_linear = x_linear * scale

        # 转回 sRGB
        return linear_to_srgb(y_linear)


# ---------------------------
# 完整 HDRNet Simple FC
# ---------------------------
@ARCH_REGISTRY.register()
class HDRnetSimpleFC(nn.Module):
    """
    简化版 HDRNet (FC 版本)：
    - LowResNet: global pool + FC 生成 grid
    - GuideNet: 3 层 conv
    - SliceApply: 三线性插值 + 线性空间缩放
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, grid_H=32, grid_W=32, grid_D=16,
                 base_channels=16, num_downsample=3, num_residual=2, fixed_ds_size=512):
        super().__init__()
        assert num_in_ch == 3 and num_out_ch == 3

        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.grid_H = grid_H
        self.grid_W = grid_W
        self.grid_D = grid_D
        self.fixed_ds_size = fixed_ds_size

        self.lowres_net = LowResNetFC(
            in_ch=num_in_ch, base_channels=base_channels,
            num_downsample=num_downsample, num_residual=num_residual,
            grid_H=grid_H, grid_W=grid_W, grid_D=grid_D,
            fixed_ds_size=fixed_ds_size,
        )
        self.guide_net = GuideNetFC(in_ch=num_in_ch)
        self.slice_apply = BilateralSliceScaleFC(
            grid_D=grid_D, grid_H=grid_H, grid_W=grid_W,
        )

    def forward(self, x, strength: float = 1.0):
        """x: (B, 3, H, W) in [0, 1]"""
        # Grid 网络: 固定尺寸下采样输入
        x_low = F.interpolate(x, size=(self.fixed_ds_size, self.fixed_ds_size),
                              mode='bilinear', align_corners=False)
        grid = self.lowres_net(x_low)    # (B, D, Hg, Wg, 1)

        # Guide 网络: 全分辨率输入
        guide = self.guide_net(x)        # (B, 1, H, W)

        # 三线性插值 + 线性空间缩放
        out = self.slice_apply(grid, guide, x)

        if strength < 1.0:
            out = strength * out + (1.0 - strength) * x

        if self.training:
            return out
        return torch.clamp(out, 0.0, 1.0)
