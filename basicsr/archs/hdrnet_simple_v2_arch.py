print('### USING HDRNET SIMPLE V2 ARCH FROM:', __file__)

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
# sRGB <-> Linear 转换
# =====================================================================
def srgb_to_linear(x):
    x = x.clamp(1e-6, 1.0)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = x.clamp(1e-6, 1.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1.0 / 2.4)) - 0.055)


# =====================================================================
# 系数网络 LowResNet（双分支 + 融合，对齐 hdrnet_arch.py，输出 1 个 scale）
# =====================================================================
class LowResNetV2(nn.Module):
    """
    双分支系数网络（和 hdrnet_arch.py 的 LowResNet 结构一样）：
      splat   - 下采样到 grid_H × grid_W 分辨率
      global  - 继续下采样到 4×4，经 FC 提取全局语义
      local   - 保持 grid_H × grid_W 分辨率，提取空间细节
      fusion  - relu(local + global) -> 1x1 conv -> grid 系数

    输出 grid: (B, D, Hg, Wg, 1)  每格 1 个 scale
    """

    def __init__(self, in_ch=3, base_channels=16, num_downsample=3, num_residual=2,
                 grid_H=32, grid_W=32, grid_D=16, fixed_ds_size=512):
        super().__init__()

        self.grid_H = grid_H
        self.grid_W = grid_W
        self.grid_D = grid_D
        self.fixed_ds_size = fixed_ds_size

        cm = base_channels
        gd = grid_D
        n_ds_layers = int(np.log2(fixed_ds_size / grid_H))

        # ---- splat: 下采样到 grid_H × grid_W ----
        splat_layers = []
        ch = in_ch
        for i in range(n_ds_layers):
            out_ch = cm * (2 ** i)
            splat_layers.append(conv_lrelu(ch, out_ch, k=3, s=2, p=1))
            ch = out_ch
        self.splat = nn.Sequential(*splat_layers)
        splat_out_ch = ch

        # ---- global: 继续下采样到 4×4 -> FC -> 全局特征 ----
        n_global_layers = max(1, int(np.log2(grid_H / 4)))
        global_layers = []
        gch = ch
        for i in range(n_global_layers):
            global_layers.append(conv_lrelu(gch, cm * 8, k=3, s=2, p=1))
            gch = cm * 8
        self.global_trunk = nn.Sequential(*global_layers) if global_layers else nn.Identity()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc1 = nn.Linear(cm * 8, cm * 4)
        self.global_fc2 = nn.Linear(cm * 4, cm * 2)
        self.global_fc3 = nn.Linear(cm * 2, cm * 1)

        # ---- local: 保持 grid_H × grid_W 分辨率 ----
        self.local_bottleneck = nn.Conv2d(splat_out_ch, cm * 8, kernel_size=1)
        self.local_conv1 = conv_lrelu(cm * 8, cm * 8, k=3, s=1, p=1)
        self.local_conv2 = nn.Conv2d(cm * 8, cm * 1, kernel_size=3, stride=1, padding=1, bias=False)

        # ---- prediction: 1x1 conv -> D 通道系数（每格 1 个 scale）----
        self.pred_conv = nn.Conv2d(cm * 1, grid_D, kernel_size=1)

    def forward(self, x):
        """x: (B, 3, fixed_ds_size, fixed_ds_size) -> grid: (B, D, Hg, Wg, 1)"""
        B = x.size(0)

        # splat 下采样
        splat_feat = self.splat(x)

        # global 分支: 下采样 + 全局池化 + FC
        g = self.global_trunk(splat_feat)
        g = self.global_pool(g).view(B, -1)
        g = F.leaky_relu(self.global_fc1(g), 0.2)
        g = F.leaky_relu(self.global_fc2(g), 0.2)
        g = self.global_fc3(g).view(B, 1, 1, -1).permute(0, 3, 1, 2)  # (B, cm*1, 1, 1)

        # local 分支: 降维 + 空间卷积
        local_feat = self.local_bottleneck(splat_feat)
        local_feat = self.local_conv1(local_feat)
        local_feat = self.local_conv2(local_feat)  # (B, cm*1, Hg, Wg)

        # fusion: local + global -> relu -> 1x1 conv
        fusion = F.relu(local_feat + g)
        grid = self.pred_conv(fusion)  # (B, D, Hg, Wg)

        # reshape -> (B, D, Hg, Wg, 1)
        grid = grid.unsqueeze(-1)
        return grid


# =====================================================================
# 引导网络 GuideNet（对齐 hdrnet_arch.py）
# =====================================================================
class GuideNetV2(nn.Module):
    """
    逐像素引导网络（和 hdrnet_arch.py 一样）：
    1x1 conv -> leaky_relu -> 1x1 conv -> sigmoid
    """

    def __init__(self, in_ch=3, guide_channels=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, guide_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(guide_channels, 1, kernel_size=1)

    def forward(self, x):
        g = F.leaky_relu(self.conv1(x), 0.2)
        return torch.sigmoid(self.conv2(g))


# =====================================================================
# 三线性 Slicing + 单 scale 缩放（线性空间）
# =====================================================================
class BilateralSliceScaleV2(nn.Module):
    """
    和 hdrnet_arch.py 的 BilateralSliceApply 一样的三线性插值，
    只是把 12 系数的仿射变换改成 1 个 scale 的线性空间缩放。
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
        """
        B, D, Hg, Wg, _ = grid.shape
        _, _, H, W = x.shape
        device = x.device

        # 转成线性空间
        x_linear = srgb_to_linear(x)

        # 深度坐标 z \in [0, D-1]
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
        y = linear_to_srgb(y_linear)
        return y


# =====================================================================
# 完整 HDRNet Simple V2
# =====================================================================
@ARCH_REGISTRY.register()
class HDRnetSimpleV2(nn.Module):
    """
    基于 hdrnet_arch.py 的双分支 grid 网络，只改输出为 1 个 scale：
      LowResNet  - 双分支（global + local + fusion）-> bilateral grid
      GuideNet   - 全分辨率输入 -> per-pixel guide map
      SliceApply - 三线性插值 + 线性空间同比例缩放
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

        self.lowres_net = LowResNetV2(
            in_ch=num_in_ch, base_channels=base_channels,
            num_downsample=num_downsample, num_residual=num_residual,
            grid_H=grid_H, grid_W=grid_W, grid_D=grid_D,
            fixed_ds_size=fixed_ds_size,
        )
        self.guide_net = GuideNetV2(in_ch=num_in_ch)
        self.slice_apply = BilateralSliceScaleV2(
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
