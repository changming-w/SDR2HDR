"""HDRNet 纯 1D LUT 架构。

LowResNet 输出 (B, N, 12) 仿射系数，N 为亮度 bin 数量。
每个 bin 对应一组 3x3 仿射矩阵 + bias（共 12 个参数）。
部署时：BT.709 亮度 → 查 bin → 仿射变换 → 可展开为 3D LUT。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY


def conv_lrelu(in_ch, out_ch, k, s, p, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=bias),
        nn.LeakyReLU(0.2, inplace=True),
    )


class LowResNet1DLUT(nn.Module):
    """系数网络，输出 N 个亮度 bin 的仿射系数 (B, N, 12)。

    Args:
        in_ch: 输入通道数。
        base_channels: 基础通道倍数。
        num_bins: 亮度 bin 数量 (N)。
        coeffs_channels: 每个 bin 的系数数（12 = 3x3 矩阵 + 3 bias）。
        fixed_ds_size: 固定下采样输入尺寸。
        num_downsample: 下采样层数。
    """

    def __init__(self, in_ch=3, base_channels=16, num_bins=4096,
                 coeffs_channels=12, fixed_ds_size=512, num_downsample=4):
        super().__init__()

        self.num_bins = num_bins
        self.coeffs_channels = coeffs_channels
        self.fixed_ds_size = fixed_ds_size

        cm = base_channels
        n_ds_layers = num_downsample

        # 下采样主干网络
        splat_layers = []
        ch = in_ch
        for i in range(n_ds_layers):
            out_ch = cm * (2 ** min(i, 3))
            splat_layers.append(conv_lrelu(ch, out_ch, k=3, s=2, p=1))
            ch = out_ch
        self.splat = nn.Sequential(*splat_layers)

        # 全局池化 + FC 头 → (B, N*12)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(ch, cm * 4)
        self.fc2 = nn.Linear(cm * 4, cm * 8)
        self.fc3 = nn.Linear(cm * 8, num_bins * coeffs_channels)

    def forward(self, x):
        """前向传播。

        Args:
            x: (B, 3, fixed_ds_size, fixed_ds_size) 下采样后的输入。

        Returns:
            grid: (B, N, 12) 仿射系数。
        """
        B = x.size(0)
        feat = self.splat(x)
        g = self.pool(feat).view(B, -1)
        g = F.leaky_relu(self.fc1(g), 0.2)
        g = F.leaky_relu(self.fc2(g), 0.2)
        g = self.fc3(g)
        g = g.view(B, self.num_bins, self.coeffs_channels)
        return g


class SliceApply1DLUT(nn.Module):
    """按 BT.709 亮度查 1D LUT，支持可微线性插值。

    每个像素：BT.709 亮度 → 找 bin → 取 12 系数 → 仿射变换。
    """

    def __init__(self, num_bins=4096, coeffs_channels=12):
        super().__init__()
        self.N = num_bins
        self.Cc = coeffs_channels

    def forward(self, grid, x):
        """通过亮度索引的仿射系数应用 1D LUT。

        Args:
            grid: (B, N, 12) 每帧的仿射系数。
            x: (B, 3, H, W) 输入 RGB，范围 [0, 1]。

        Returns:
            out: (B, 3, H, W) 变换后的 RGB，范围 [0, 1]。
        """
        B, _, H, W = x.shape

        # BT.709 亮度 → 连续 bin 索引
        luminance = 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]
        z = luminance * (self.N - 1)  # (B, 1, H, W)

        # 线性插值的相邻 bin
        z0 = torch.floor(z).clamp(0, self.N - 1).long()
        z1 = (z0 + 1).clamp(0, self.N - 1)
        dz = (z - z0.float()).squeeze(1).unsqueeze(-1)  # (B, H, W, 1)

        # 从 grid 中收集系数
        coeff0 = torch.gather(grid, 1, z0.view(B, -1).unsqueeze(-1).expand(-1, -1, self.Cc))
        coeff1 = torch.gather(grid, 1, z1.view(B, -1).unsqueeze(-1).expand(-1, -1, self.Cc))
        coeff0 = coeff0.view(B, H, W, self.Cc)
        coeff1 = coeff1.view(B, H, W, self.Cc)

        # 线性插值
        coeff = coeff0 * (1 - dz) + coeff1 * dz  # (B, H, W, 12)

        # 仿射变换: y = A @ x + b
        A = coeff[..., :9].reshape(B, H, W, 3, 3)
        b = coeff[..., 9:].reshape(B, H, W, 3, 1)
        x_vec = x.permute(0, 2, 3, 1).unsqueeze(-1)  # (B, H, W, 3, 1)
        out = (A @ x_vec).squeeze(-1) + b.squeeze(-1)  # (B, H, W, 3)
        return out.permute(0, 3, 1, 2)  # (B, 3, H, W)


@ARCH_REGISTRY.register()
class HDRnet1DLUT(nn.Module):
    """纯 1D LUT 版 HDRNet（无 GuideNet）。

    LowResNet 输出 N=4096 个 BT.709 亮度 bin 的仿射系数 (N, 12)。
    每个 bin 存储一组 3x3 仿射矩阵 + 3 bias。每帧内容自适应。
    部署时可展开为 3D LUT（遍历 RGB 立方体）。

    Args:
        num_in_ch: 输入通道数。
        num_out_ch: 输出通道数。
        num_bins: 亮度 bin 数量。
        base_channels: LowResNet 基础通道倍数。
        num_downsample: LowResNet 下采样层数。
        fixed_ds_size: LowResNet 固定输入尺寸。
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_bins=4096,
                 base_channels=16, num_downsample=4, fixed_ds_size=512):
        super().__init__()
        assert num_in_ch == 3 and num_out_ch == 3

        self.num_bins = num_bins
        self.fixed_ds_size = fixed_ds_size

        self.lowres_net = LowResNet1DLUT(
            in_ch=num_in_ch, base_channels=base_channels,
            num_bins=num_bins, coeffs_channels=12,
            fixed_ds_size=fixed_ds_size, num_downsample=num_downsample,
        )
        self.slice_apply = SliceApply1DLUT(
            num_bins=num_bins, coeffs_channels=12,
        )

    def forward(self, x, strength: float = 1.0):
        """前向传播。

        Args:
            x: (B, 3, H, W) 输入 RGB，范围 [0, 1]。
            strength: 混合强度（1.0 = 全效果，0.0 = 直通）。

        Returns:
            out: (B, 3, H, W) 输出 RGB，范围 [0, 1]。
        """
        x_low = F.interpolate(x, size=(self.fixed_ds_size, self.fixed_ds_size),
                              mode='bilinear', align_corners=False)
        grid = self.lowres_net(x_low)  # (B, N, 12)
        out = self.slice_apply(grid, x)

        if strength < 1.0:
            out = strength * out + (1.0 - strength) * x

        if self.training:
            return out
        return torch.clamp(out, 0.0, 1.0)
