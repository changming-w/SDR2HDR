print('### USING HDRNET SPATIAL LUT ARCH FROM:', __file__)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY


# =====================================================================
# 工具函数
# =====================================================================
def conv_lrelu(in_ch, out_ch, k, s, p, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=bias),
        nn.LeakyReLU(0.2, inplace=True),
    )


# =====================================================================
# LowResNet：输出 (D, N, 12)，N = Hg × Wg
# =====================================================================
class LowResNetSpatial(nn.Module):
    """
    双分支系数网络，输出展平空间维度：
    grid: (B, D, N, Cc)，N = Hg × Wg
    """

    def __init__(self, in_ch=3, base_channels=16, num_downsample=3, num_residual=2,
                 grid_H=64, grid_W=64, grid_D=16, coeffs_channels=12, fixed_ds_size=512):
        super().__init__()

        self.grid_H = grid_H
        self.grid_W = grid_W
        self.grid_D = grid_D
        self.coeffs_channels = coeffs_channels
        self.fixed_ds_size = fixed_ds_size

        cm = base_channels
        gd = grid_D
        n_ds_layers = int(np.log2(fixed_ds_size / grid_H))

        # splat: 下采样到 spatial_bin
        splat_layers = []
        ch = in_ch
        for i in range(n_ds_layers):
            out_ch = cm * (2 ** i) * gd
            splat_layers.append(conv_lrelu(ch, out_ch, k=3, s=2, p=1))
            ch = out_ch
        self.splat = nn.Sequential(*splat_layers)
        splat_out_ch = ch

        # global: 继续下采样到 4x4 -> FC -> 全局特征
        n_global_layers = max(1, int(np.log2(grid_H / 4)))
        global_layers = []
        for i in range(n_global_layers):
            global_layers.append(conv_lrelu(ch, cm * 8 * gd, k=3, s=2, p=1))
            ch = cm * 8 * gd
        self.global_trunk = nn.Sequential(*global_layers) if n_global_layers > 0 else nn.Identity()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc1 = nn.Linear(cm * 8 * gd, cm * 4 * gd)
        self.global_fc2 = nn.Linear(cm * 4 * gd, cm * 2 * gd)
        self.global_fc3 = nn.Linear(cm * 2 * gd, cm * 1 * gd)

        # local: 保持 spatial_bin 分辨率
        self.local_bottleneck = nn.Conv2d(splat_out_ch, cm * 8 * gd, kernel_size=1)
        self.local_conv1 = conv_lrelu(cm * 8 * gd, cm * 8 * gd, k=3, s=1, p=1)
        self.local_conv2 = nn.Conv2d(cm * 8 * gd, cm * 1 * gd, kernel_size=3, stride=1, padding=1, bias=False)

        # prediction: 1x1 conv -> D*Cc 通道系数
        self.pred_conv = nn.Conv2d(cm * 1 * gd, grid_D * coeffs_channels, kernel_size=1)

    def forward(self, x):
        """x: (B, 3, fixed_ds_size, fixed_ds_size) -> grid: (B, D, N, Cc)"""
        B = x.size(0)

        splat_feat = self.splat(x)

        # global 分支
        g = self.global_trunk(splat_feat)
        g = self.global_pool(g).view(B, -1)
        g = F.leaky_relu(self.global_fc1(g), 0.2)
        g = F.leaky_relu(self.global_fc2(g), 0.2)
        g = self.global_fc3(g).view(B, 1, 1, -1).permute(0, 3, 1, 2)

        # local 分支
        local_feat = self.local_bottleneck(splat_feat)
        local_feat = self.local_conv1(local_feat)
        local_feat = self.local_conv2(local_feat)

        # fusion
        fusion = F.relu(local_feat + g)
        grid = self.pred_conv(fusion)

        # reshape -> (B, D, Hg, Wg, Cc) -> (B, D, N, Cc)
        grid = grid.view(B, self.grid_D, self.coeffs_channels, self.grid_H, self.grid_W)
        grid = grid.permute(0, 1, 3, 4, 2).contiguous()
        grid = grid.view(B, self.grid_D, self.grid_H * self.grid_W, self.coeffs_channels)
        return grid


# =====================================================================
# BT.709 亮度 GuideNet（固定公式）
# =====================================================================
class GuideNetBT709(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        self.register_buffer('coeffs', torch.tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1))

    def forward(self, x):
        guide = (x * self.coeffs).sum(dim=1, keepdim=True)
        return guide


# =====================================================================
# 空间索引 Slicing：每个像素精确映射到空间位置，仅深度插值
# =====================================================================
class SpatialSliceApply(nn.Module):
    """
    部署友好的 slicing：
    - 空间维度：每个像素精确映射到最近的空间索引（无插值）
    - 深度维度：BT.709 亮度 → 深度插值
    
    grid: (B, D, N, 12)
    输出: (B, 3, H, W)
    """

    def __init__(self, coeffs_channels=12, grid_D=16, grid_H=64, grid_W=64):
        super().__init__()
        assert coeffs_channels == 12
        self.Cc = coeffs_channels
        self.D = grid_D
        self.Hg = grid_H
        self.Wg = grid_W
        self.N = grid_H * grid_W

    def forward(self, grid, guide, x):
        """
        grid: (B, D, N, 12)
        guide: (B, 1, H, W) - BT.709 亮度 [0, 1]
        x: (B, 3, H, W) - 输入 RGB
        """
        B, _, H, W = x.shape
        device = x.device

        # 每个像素的空间索引（精确映射，无插值）
        # 像素 (i, j) → 空间索引 = i * Wg + j（最近邻）
        pixel_h = torch.linspace(0, self.Hg - 1, H, device=device)
        pixel_w = torch.linspace(0, self.Wg - 1, W, device=device)
        ph, pw = torch.meshgrid(pixel_h, pixel_w, indexing='ij')
        spatial_idx = (ph.long() * self.Wg + pw.long()).view(1, 1, H, W).expand(B, 1, H, W)
        # clamp
        spatial_idx = spatial_idx.clamp(0, self.N - 1)

        # 深度插值
        z = guide * (self.D - 1)  # (B, 1, H, W)
        z0 = torch.floor(z).clamp(0, self.D - 1).long()
        z1 = (z0 + 1).clamp(0, self.D - 1)
        dz = (z - z0.float()).unsqueeze(-1)  # (B, 1, H, W, 1)

        # 取系数：gather
        # grid: (B, D, N, 12) -> (B, D*N, 12)
        grid_flat = grid.view(B, self.D * self.N, self.Cc)

        def gather_coeff(z_idx, s_idx):
            """在 (D, N) 中按 (z, spatial) 索引取系数"""
            idx = (z_idx * self.N + s_idx).view(B, -1)  # (B, H*W)
            idx = idx.unsqueeze(-1).expand(-1, -1, self.Cc)  # (B, H*W, 12)
            coeff = torch.gather(grid_flat, 1, idx)  # (B, H*W, 12)
            return coeff.view(B, H, W, self.Cc)

        coeff0 = gather_coeff(z0, spatial_idx.squeeze(1))  # (B, H, W, 12)
        coeff1 = gather_coeff(z1, spatial_idx.squeeze(1))

        # 深度插值
        coeff = coeff0 * (1 - dz) + coeff1 * dz  # (B, H, W, 12)

        # 仿射变换: y = A @ x + b
        A = coeff[..., :9].reshape(B, H, W, 3, 3)
        b = coeff[..., 9:].reshape(B, H, W, 3, 1)
        x_vec = x.permute(0, 2, 3, 1).unsqueeze(-1)  # (B, H, W, 3, 1)
        out = (A @ x_vec).squeeze(-1) + b.squeeze(-1)  # (B, H, W, 3)
        return out.permute(0, 3, 1, 2)  # (B, 3, H, W)


# =====================================================================
# 完整模型
# =====================================================================
@ARCH_REGISTRY.register()
class HDRnetSpatialLUT(nn.Module):
    """
    空间 1D LUT 版本：
    - LowResNet 输出 (D, N, 12)，N = Hg × Wg
    - GuideNet 用 BT.709 固定亮度
    - SliceApply: 每像素精确空间索引 + 深度插值
    
    部署时：
    - 每帧 LowResNet → (D, N, 12)
    - 每个像素: 空间索引 + BT.709亮度 → 查 12 系数 → 仿射
    - 等效: 每个空间位置一个 1D LUT (亮度 → RGB)
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, grid_H=64, grid_W=64, grid_D=16,
                 base_channels=16, num_downsample=3, num_residual=2, fixed_ds_size=512):
        super().__init__()
        assert num_in_ch == 3 and num_out_ch == 3

        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.grid_H = grid_H
        self.grid_W = grid_W
        self.grid_D = grid_D
        self.coeffs_channels = 12
        self.fixed_ds_size = fixed_ds_size

        self.lowres_net = LowResNetSpatial(
            in_ch=num_in_ch, base_channels=base_channels,
            num_downsample=num_downsample, num_residual=num_residual,
            grid_H=grid_H, grid_W=grid_W, grid_D=grid_D,
            coeffs_channels=self.coeffs_channels, fixed_ds_size=fixed_ds_size,
        )
        self.guide_net = GuideNetBT709(in_ch=num_in_ch)
        self.slice_apply = SpatialSliceApply(
            coeffs_channels=self.coeffs_channels, grid_D=grid_D,
            grid_H=grid_H, grid_W=grid_W,
        )

    def forward(self, x, strength: float = 1.0):
        """x: (B, 3, H, W) in [0, 1]"""
        x_low = F.interpolate(x, size=(self.fixed_ds_size, self.fixed_ds_size),
                              mode='bilinear', align_corners=False)
        grid = self.lowres_net(x_low)  # (B, D, N, 12)
        guide = self.guide_net(x)       # (B, 1, H, W)
        out = self.slice_apply(grid, guide, x)

        if strength < 1.0:
            out = strength * out + (1.0 - strength) * x

        if self.training:
            return out
        return torch.clamp(out, 0.0, 1.0)
