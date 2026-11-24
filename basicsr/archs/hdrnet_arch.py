print('### USING HDRNET ARCH FROM:', __file__)

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
# 系数网络 LowResNet（双分支 + 融合，对齐 google/hdrnet _coefficients）
# =====================================================================
class LowResNet(nn.Module):
    """
    双分支系数网络：
      splat   - 下采样到 spatial_bin 分辨率
      global  - 继续下采样到 4x4，经 FC 提取全局语义
      local   - 保持 spatial_bin 分辨率，提取空间细节
      fusion  - relu(local + global) -> 1x1 conv -> grid 系数

    输出 grid: (B, D, Hg, Wg, Cc)
    """

    def __init__(self, in_ch=3, base_channels=16, num_downsample=3, num_residual=2,
                 grid_H=16, grid_W=16, grid_D=8, coeffs_channels=12, fixed_ds_size=256):
        super().__init__()

        self.grid_H = grid_H
        self.grid_W = grid_W
        self.grid_D = grid_D
        self.coeffs_channels = coeffs_channels
        self.fixed_ds_size = fixed_ds_size

        cm = base_channels  # channel_multiplier
        gd = grid_D         # luma_bins
        n_ds_layers = int(np.log2(fixed_ds_size / grid_H))  # 动态下采样层数

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
            global_layers.append(conv_lrelu(ch, cm * 8 * gd, k=3, s=2, p=1))
            ch = cm * 8 * gd
        self.global_trunk = nn.Sequential(*global_layers) if global_layers else nn.Identity()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc1 = nn.Linear(cm * 8 * gd, cm * 4 * gd)
        self.global_fc2 = nn.Linear(cm * 4 * gd, cm * 2 * gd)
        self.global_fc3 = nn.Linear(cm * 2 * gd, cm * 1 * gd)

        # ---- local: 保持 spatial_bin 分辨率 ----
        self.local_bottleneck = nn.Conv2d(splat_out_ch, cm * 8 * gd, kernel_size=1)
        self.local_conv1 = conv_lrelu(cm * 8 * gd, cm * 8 * gd, k=3, s=1, p=1)
        self.local_conv2 = nn.Conv2d(cm * 8 * gd, cm * 1 * gd, kernel_size=3, stride=1, padding=1, bias=False)

        # ---- prediction: 1x1 conv -> D*Cc 通道系数 ----
        self.pred_conv = nn.Conv2d(cm * 1 * gd, grid_D * coeffs_channels, kernel_size=1)

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
        g = self.global_fc3(g).view(B, 1, 1, -1).permute(0, 3, 1, 2)  # (B, cm*gd, 1, 1)

        # local 分支: 降维 + 空间卷积
        local_feat = self.local_bottleneck(splat_feat)
        local_feat = self.local_conv1(local_feat)
        local_feat = self.local_conv2(local_feat)  # (B, cm*gd, Hg, Wg)

        # fusion: local + global -> relu -> 1x1 conv
        fusion = F.relu(local_feat + g)
        grid = self.pred_conv(fusion)  # (B, D*Cc, Hg, Wg)

        # reshape -> (B, D, Hg, Wg, Cc)
        grid = grid.view(B, self.grid_D, self.coeffs_channels, self.grid_H, self.grid_W)
        grid = grid.permute(0, 1, 3, 4, 2).contiguous()
        return grid


# =====================================================================
# 引导网络 GuideNet（对齐 HDRNetPointwiseNNGuide）
# =====================================================================
class GuideNet(nn.Module):
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
# 三线性 Slicing + 仿射变换
# =====================================================================
class BilateralSliceApply(nn.Module):
    """
    在 bilateral grid 上做三线性插值 + 仿射变换：
      grid:  (B, D, Hg, Wg, 12)  - 12 = 3x3 矩阵 + 3 bias
      guide: (B, 1, H, W) in [0, 1] - 深度轴采样坐标
      x:     (B, 3, H, W) - 全分辨率 RGB
    输出: y = A(x) * x + b(x), shape (B, 3, H, W)
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
        z = guide * (D - 1)  # 深度轴 (B, 1, H, W)
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
# 完整 HDRNet（对齐 google/hdrnet）
# =====================================================================
@ARCH_REGISTRY.register()
class HDRnet(nn.Module):
    """
    完整 HDRNet:
      LowResNet  - 下采样输入 -> 双分支 -> bilateral grid 系数
      GuideNet   - 全分辨率输入 -> per-pixel guide map
      SliceApply - 在全分辨率上做三线性插值 + 仿射变换

    推理支持任意分辨率输入（Guide 和输出保持原始尺寸）。
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, grid_H=16, grid_W=16, grid_D=8,
                 base_channels=16, num_downsample=3, num_residual=2, fixed_ds_size=256):
        super().__init__()
        assert num_in_ch == 3 and num_out_ch == 3, "HDRNet 仅支持 3 通道输入输出"

        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.grid_H = grid_H
        self.grid_W = grid_W
        self.grid_D = grid_D
        self.coeffs_channels = 12
        self.fixed_ds_size = fixed_ds_size

        self.lowres_net = LowResNet(
            in_ch=num_in_ch, base_channels=base_channels,
            num_downsample=num_downsample, num_residual=num_residual,
            grid_H=grid_H, grid_W=grid_W, grid_D=grid_D,
            coeffs_channels=self.coeffs_channels, fixed_ds_size=fixed_ds_size,
        )
        self.guide_net = GuideNet(in_ch=num_in_ch)
        self.slice_apply = BilateralSliceApply(
            coeffs_channels=self.coeffs_channels, grid_D=grid_D,
            grid_H=grid_H, grid_W=grid_W,
        )

    def forward(self, x, strength: float = 1.0):
        """x: (B, 3, H, W) in [0, 1]"""
        # Grid 网络: 固定 512x512 下采样输入
        x_low = F.interpolate(x, size=(self.fixed_ds_size, self.fixed_ds_size),
                              mode='bilinear', align_corners=False)
        grid = self.lowres_net(x_low)  # (B, D, Hg, Wg, Cc)

        # Guide 网络: 全分辨率输入
        guide = self.guide_net(x)  # (B, 1, H, W)

        # 三线性插值 + 仿射变换
        out = self.slice_apply(grid, guide, x)  # (B, 3, H, W)

        # 强度混合（用于推理时控制增强程度）
        if strength < 1.0:
            out = strength * out + (1.0 - strength) * x

        if self.training:
            return out
        return torch.clamp(out, 0.0, 1.0)


# =====================================================================
# 分支网络（用于 ONNX 分段部署）
# =====================================================================
@ARCH_REGISTRY.register()
class HDRnetGuideBranch(nn.Module):
    """Guide 分支: 全分辨率 -> guide map"""
    def __init__(self, num_in_ch=3):
        super().__init__()
        self.num_in_ch = num_in_ch
        self.guide_net = GuideNet(in_ch=num_in_ch)

    def forward(self, x):
        return self.guide_net(x)


@ARCH_REGISTRY.register()
class HDRnetGridBranch(nn.Module):
    """Grid 分支: 低分辨率输入 -> grid + slicing + 仿射变换"""
    def __init__(self, num_in_ch=3, num_out_ch=3, grid_H=16, grid_W=16, grid_D=8,
                 base_channels=16, num_downsample=3, num_residual=2, fixed_ds_size=256):
        super().__init__()
        assert num_in_ch == 3 and num_out_ch == 3

        self.lowres_net = LowResNet(
            in_ch=num_in_ch, base_channels=base_channels,
            num_downsample=num_downsample, num_residual=num_residual,
            grid_H=grid_H, grid_W=grid_W, grid_D=grid_D,
            coeffs_channels=12, fixed_ds_size=fixed_ds_size,
        )
        self.slice_apply = BilateralSliceApply(
            coeffs_channels=12, grid_D=grid_D, grid_H=grid_H, grid_W=grid_W,
        )

    def forward(self, x, guide):
        x_low = F.interpolate(x, size=(self.lowres_net.fixed_ds_size,) * 2,
                              mode='bilinear', align_corners=False)
        grid = self.lowres_net(x_low)
        out = self.slice_apply(grid, guide, x)
        return torch.clamp(out, 0.0, 1.0)
