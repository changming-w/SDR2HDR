print('### USING HDRNET SIMPLE 1D ARCH FROM:', __file__)

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
# LowResNet (global pool + FC → 1D curve)
# ---------------------------
class LowResNet1D(nn.Module):
    """
    简化版低分辨率网络：
    - 去掉空间维度，只输出亮度维度的 scale 曲线
    - 输出 grid: (B, D) 每格 1 个 scale，共 D 格
    """

    def __init__(self, in_ch=3, base_channels=16, num_downsample=3, num_residual=2,
                 grid_D=16, fixed_ds_size=512):
        super().__init__()

        self.grid_D = grid_D
        self.fixed_ds_size = fixed_ds_size

        # trunk: 下采样
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

        # global pool + FC → 1D curve
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(ch, grid_D)

    def forward(self, x):
        """
        x: (B, 3, H, W) in [0,1]
        return: grid (B, D) - 1D scale curve
        """
        B = x.size(0)
        feat = self.trunk(x)
        pooled = self.pool(feat).view(B, -1)
        grid = self.fc(pooled)  # (B, D)
        return grid


# ---------------------------
# GuideNet (3 层 conv)
# ---------------------------
class GuideNet1D(nn.Module):
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
# 1D Slicing + scale (only depth interpolation)
# ---------------------------
class BilateralSlice1D(nn.Module):
    """
    1D bilateral slice：
    - 只沿深度维度插值
    - 在线性空间做同比例缩放
    """

    def __init__(self, grid_D=16):
        super().__init__()
        self.D = grid_D

    def forward(self, grid, guide, x):
        """
        grid:  (B, D)       1D scale curve
        guide: (B, 1, H, W) in [0,1] - depth coordinate
        x:     (B, 3, H, W) sRGB [0,1]
        return: (B, 3, H, W) sRGB [0,1]
        """
        B, D = grid.shape
        _, _, H, W = x.shape
        device = x.device

        # 转成线性空间
        x_linear = srgb_to_linear(x)

        # 深度坐标 z ∈ [0, D-1]
        z = guide * (D - 1)  # (B, 1, H, W)

        # 下取整 / 上取整
        z0 = torch.floor(z).clamp(0, D - 1).long()
        z1 = torch.clamp(z0 + 1, max=D - 1)
        dz = z - z0.float()

        # 从 grid 中 gather scale
        # grid: (B, D) -> (B, 1, D) for gather
        grid_expanded = grid.unsqueeze(1)  # (B, 1, D)

        # gather z0 and z1
        scale0 = torch.gather(grid_expanded, 2, z0.view(B, 1, H, W).view(B, 1, -1))
        scale0 = scale0.view(B, 1, H, W)

        scale1 = torch.gather(grid_expanded, 2, z1.view(B, 1, H, W).view(B, 1, -1))
        scale1 = scale1.view(B, 1, H, W)

        # 线性插值
        scale = scale0 * (1 - dz) + scale1 * dz

        # 约束 scale
        scale = scale.clamp(0.01, 10.0)

        # 在线性空间做同比例缩放
        y_linear = x_linear * scale

        # 转回 sRGB
        return linear_to_srgb(y_linear)


# ---------------------------
# 完整 HDRNet Simple 1D
# ---------------------------
@ARCH_REGISTRY.register()
class HDRnetSimple1D(nn.Module):
    """
    极简版 HDRNet：
    - LowResNet: global pool + FC → 1D scale curve (D,)
    - GuideNet: 3 层 conv → guide map
    - SliceApply: 1D interpolation along depth only
    - 部署时就是 1D LUT: 亮度 → scale
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, grid_D=16,
                 base_channels=16, num_downsample=3, num_residual=2, fixed_ds_size=512):
        super().__init__()
        assert num_in_ch == 3 and num_out_ch == 3

        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.grid_D = grid_D
        self.fixed_ds_size = fixed_ds_size

        self.lowres_net = LowResNet1D(
            in_ch=num_in_ch, base_channels=base_channels,
            num_downsample=num_downsample, num_residual=num_residual,
            grid_D=grid_D, fixed_ds_size=fixed_ds_size,
        )
        self.guide_net = GuideNet1D(in_ch=num_in_ch)
        self.slice_apply = BilateralSlice1D(grid_D=grid_D)

    def forward(self, x, strength: float = 1.0):
        """x: (B, 3, H, W) in [0, 1]"""
        # Grid 网络: 固定尺寸下采样输入 → 1D curve
        x_low = F.interpolate(x, size=(self.fixed_ds_size, self.fixed_ds_size),
                              mode='bilinear', align_corners=False)
        grid = self.lowres_net(x_low)    # (B, D)

        # Guide 网络: 全分辨率输入
        guide = self.guide_net(x)        # (B, 1, H, W)

        # 1D interpolation + scale
        out = self.slice_apply(grid, guide, x)

        if strength < 1.0:
            out = strength * out + (1.0 - strength) * x

        if self.training:
            return out
        return torch.clamp(out, 0.0, 1.0)
