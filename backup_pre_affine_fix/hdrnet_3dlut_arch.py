print('### USING HDRNET 3D LUT ARCH FROM:', __file__)

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
# 系数网络 LowResNet（双分支 + 融合，输出 1D 曲线）
# =====================================================================
class LowResNet3D(nn.Module):
    """
    双分支系数网络（压缩版）：
      - 去掉空间维度，只输出亮度维度的 12 系数曲线
      - 输出 grid: (B, D, 12) - 每格 12 个系数（3x3 矩阵 + 3 bias）

    部署时就是 3D LUT: (D, 12) = (16, 12)
    """

    def __init__(self, in_ch=3, base_channels=16, num_downsample=3, num_residual=2,
                 grid_D=16, coeffs_channels=12, fixed_ds_size=512):
        super().__init__()

        self.grid_D = grid_D
        self.coeffs_channels = coeffs_channels
        self.fixed_ds_size = fixed_ds_size

        cm = base_channels
        gd = grid_D

        # ---- splat: 下采样到 4x4 ----
        splat_layers = []
        ch = in_ch
        n_ds_layers = int(np.log2(fixed_ds_size / 4))
        for i in range(n_ds_layers):
            out_ch = cm * (2 ** i) * gd
            splat_layers.append(conv_lrelu(ch, out_ch, k=3, s=2, p=1))
            ch = out_ch
        self.splat = nn.Sequential(*splat_layers)

        # ---- global: 4x4 -> pool -> FC ----
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc1 = nn.Linear(ch, cm * 4 * gd)
        self.global_fc2 = nn.Linear(cm * 4 * gd, cm * 2 * gd)
        self.global_fc3 = nn.Linear(cm * 2 * gd, cm * 1 * gd)

        # ---- prediction: FC -> D * 12 系数 ----
        self.pred_fc = nn.Linear(cm * 1 * gd, grid_D * coeffs_channels)

    def forward(self, x):
        """x: (B, 3, fixed_ds_size, fixed_ds_size) -> grid: (B, D, 12)"""
        B = x.size(0)

        # splat 下采样
        feat = self.splat(x)

        # global: pool + FC
        g = self.global_pool(feat).view(B, -1)
        g = F.leaky_relu(self.global_fc1(g), 0.2)
        g = F.leaky_relu(self.global_fc2(g), 0.2)
        g = self.global_fc3(g)  # (B, cm*1*gd)

        # prediction: FC -> D * 12
        grid = self.pred_fc(g)  # (B, D * 12)
        grid = grid.view(B, self.grid_D, self.coeffs_channels)  # (B, D, 12)
        return grid


# =====================================================================
# 引导网络 GuideNet（对齐 hdrnet_arch.py）
# =====================================================================
class GuideNet3D(nn.Module):
    """
    全分辨率逐像素引导网络：
    2 层 1x1 conv -> sigmoid -> guide map in [0, 1]
    """

    def __init__(self, in_ch=3, guide_channels=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, guide_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(guide_channels, 1, kernel_size=1)

    def forward(self, x):
        """x: (B, 3, H, W) -> guide: (B, 1, H, W) in [0, 1]"""
        g = F.leaky_relu(self.conv1(x), 0.2)
        return torch.sigmoid(self.conv2(g))


# =====================================================================
# 1D Slicing + 仿射变换（只沿深度插值）
# =====================================================================
class BilateralSliceApply3D(nn.Module):
    """
    1D bilateral slice apply：
      grid:  (B, D, 12)  - 12 = 3x3 矩阵 + 3 bias
      guide: (B, 1, H, W) in [0, 1] - 深度轴采样坐标
      x:     (B, 3, H, W) - 全分辨率 RGB
    输出: y = A(z) * x + b(z), shape (B, 3, H, W)

    只沿深度轴插值，空间位置共享同一组系数曲线
    """

    def __init__(self, coeffs_channels=12, grid_D=16):
        super().__init__()
        assert coeffs_channels == 12, "需要 12 个系数 (3x3 矩阵 + 3 bias)"
        self.Cc = coeffs_channels
        self.D = grid_D

    def forward(self, grid, guide, x):
        """
        grid:  (B, D, 12)
        guide: (B, 1, H, W) in [0, 1]
        x:     (B, 3, H, W)
        """
        B, D, Cc = grid.shape
        _, _, H, W = x.shape
        device = x.device

        # 深度坐标 z ∈ [0, D-1]
        z = guide * (D - 1)  # (B, 1, H, W)

        # 下取整 / 上取整
        z0 = torch.floor(z).clamp(0, D - 1).long()  # (B, 1, H, W)
        z1 = torch.clamp(z0 + 1, max=D - 1)
        dz = z - z0.float()  # (B, 1, H, W)

        # grid 展平用于 gather
        grid_flat = grid.view(B, D, Cc)  # (B, D, 12)

        # gather z0 和 z1 的系数
        # z0: (B, 1, H, W) -> (B, 1, H*W) -> (B, 1, H*W, 12)
        z0_flat = z0.view(B, 1, H * W).unsqueeze(-1).expand(-1, -1, -1, Cc)
        coeff0 = torch.gather(grid_flat.unsqueeze(2).expand(-1, -1, H * W, -1), 1, z0_flat)
        coeff0 = coeff0.view(B, Cc, H, W)

        z1_flat = z1.view(B, 1, H * W).unsqueeze(-1).expand(-1, -1, -1, Cc)
        coeff1 = torch.gather(grid_flat.unsqueeze(2).expand(-1, -1, H * W, -1), 1, z1_flat)
        coeff1 = coeff1.view(B, Cc, H, W)

        # 线性插值
        coeff = coeff0 * (1 - dz) + coeff1 * dz  # (B, 12, H, W)

        # 仿射变换: y = A @ x + b
        A = coeff[:, :9, :, :].reshape(B, 3, 3, H, W)
        b = coeff[:, 9:, :, :].reshape(B, 3, 1, H, W)
        x_vec = x.reshape(B, 3, 1, H, W)
        return (A * x_vec).sum(dim=2) + b.squeeze(2)


# =====================================================================
# 完整 HDRNet 3D LUT 版本
# =====================================================================
@ARCH_REGISTRY.register()
class HDRnet3DLUT(nn.Module):
    """
    HDRNet 3D LUT 版本：
      LowResNet  - 下采样输入 -> 双分支 -> 1D 系数曲线 (D, 12)
      GuideNet   - 全分辨率输入 -> per-pixel guide map
      SliceApply - 只沿深度轴插值 + 仿射变换

    部署时 grid 就是 3D LUT: (D, 12) = (16, 12)
    每个像素只需要根据 guide 值在深度轴上做线性插值
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, grid_D=16,
                 base_channels=16, num_downsample=3, num_residual=2, fixed_ds_size=512):
        super().__init__()
        assert num_in_ch == 3 and num_out_ch == 3, "HDRNet 仅支持 3 通道输入输出"

        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.grid_D = grid_D
        self.coeffs_channels = 12
        self.fixed_ds_size = fixed_ds_size

        self.lowres_net = LowResNet3D(
            in_ch=num_in_ch, base_channels=base_channels,
            num_downsample=num_downsample, num_residual=num_residual,
            grid_D=grid_D, coeffs_channels=self.coeffs_channels, fixed_ds_size=fixed_ds_size,
        )
        self.guide_net = GuideNet3D(in_ch=num_in_ch)
        self.slice_apply = BilateralSliceApply3D(
            coeffs_channels=self.coeffs_channels, grid_D=grid_D,
        )

    def forward(self, x, strength: float = 1.0):
        """x: (B, 3, H, W) in [0, 1]"""
        # Grid 网络: 固定尺寸下采样输入 -> 1D 系数曲线
        x_low = F.interpolate(x, size=(self.fixed_ds_size, self.fixed_ds_size),
                              mode='bilinear', align_corners=False)
        grid = self.lowres_net(x_low)  # (B, D, 12)

        # Guide 网络: 全分辨率输入
        guide = self.guide_net(x)  # (B, 1, H, W)

        # 1D 插值 + 仿射变换
        out = self.slice_apply(grid, guide, x)  # (B, 3, H, W)

        # 强度混合
        if strength < 1.0:
            out = strength * out + (1.0 - strength) * x

        if self.training:
            return out
        return torch.clamp(out, 0.0, 1.0)


# =====================================================================
# 分支网络（用于 ONNX 分段部署）
# =====================================================================
@ARCH_REGISTRY.register()
class HDRnet3DLUTGuideBranch(nn.Module):
    """Guide 分支: 全分辨率 -> guide map"""
    def __init__(self, num_in_ch=3):
        super().__init__()
        self.num_in_ch = num_in_ch
        self.guide_net = GuideNet3D(in_ch=num_in_ch)

    def forward(self, x):
        return self.guide_net(x)


@ARCH_REGISTRY.register()
class HDRnet3DLUTGridBranch(nn.Module):
    """Grid 分支: 低分辨率输入 -> 1D 系数曲线 + 1D slicing + 仿射变换"""
    def __init__(self, num_in_ch=3, num_out_ch=3, grid_D=16,
                 base_channels=16, num_downsample=3, num_residual=2, fixed_ds_size=512):
        super().__init__()
        assert num_in_ch == 3 and num_out_ch == 3

        self.lowres_net = LowResNet3D(
            in_ch=num_in_ch, base_channels=base_channels,
            num_downsample=num_downsample, num_residual=num_residual,
            grid_D=grid_D, coeffs_channels=12, fixed_ds_size=fixed_ds_size,
        )
        self.slice_apply = BilateralSliceApply3D(
            coeffs_channels=12, grid_D=grid_D,
        )

    def forward(self, x, guide):
        x_low = F.interpolate(x, size=(self.lowres_net.fixed_ds_size,) * 2,
                              mode='bilinear', align_corners=False)
        grid = self.lowres_net(x_low)
        out = self.slice_apply(grid, guide, x)
        return torch.clamp(out, 0.0, 1.0)
