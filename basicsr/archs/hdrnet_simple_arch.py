print('### USING HDRNET SIMPLE ARCH FROM:', __file__)

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY


# ---------------------------
# 1. 工具函数：LeakyReLU Conv
# ---------------------------
def conv_lrelu(in_ch, out_ch, k, s, p, bias=True):
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=bias),
        nn.LeakyReLU(0.2, inplace=True),
    ]
    return nn.Sequential(*layers)


# ---------------------------
# 2. sRGB <-> Linear 转换
# ---------------------------
def srgb_to_linear(x):
    """sRGB [0,1] -> 线性空间 [0,1]"""
    x = x.clamp(1e-6, 1.0)  # 防止 0 导致梯度问题
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    """线性空间 [0,1] -> sRGB [0,1]"""
    x = x.clamp(1e-6, 1.0)  # 防止 0 导致梯度问题
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1.0 / 2.4)) - 0.055)


# ---------------------------
# 3. 低分辨率网络（输出 scale grid）
# ---------------------------
class LowResNetSimple(nn.Module):
    """
    简化版低分辨率网络：
    - 输出 grid: (B, 1, H_g, W_g, D_g) 每格 1 个 scale 系数
    """

    def __init__(
        self,
        in_ch=3,
        base_channels=16,
        num_downsample=3,
        num_residual=2,
        grid_H=32,
        grid_W=32,
        grid_D=16,
    ):
        super().__init__()

        self.grid_H = grid_H
        self.grid_W = grid_W
        self.grid_D = grid_D

        # 下采样 trunk
        layers = []
        ch = in_ch
        for i in range(num_downsample):
            out_ch = base_channels * (2 ** i)
            layers.append(conv_lrelu(ch, out_ch, k=3, s=2, p=1))
            ch = out_ch

        # residual conv blocks
        for _ in range(num_residual):
            layers.append(conv_lrelu(ch, ch, k=3, s=1, p=1))

        self.trunk = nn.Sequential(*layers)

        # 用卷积代替 FC，保留空间结构
        self.grid_conv = nn.Conv2d(ch, self.grid_D, kernel_size=3, padding=1)
        # depth 方向平滑卷积
        self.depth_smooth = nn.Conv1d(1, 1, kernel_size=3, padding=1)

    def forward(self, x):
        """
        x: (B,3,H,W) in [0,1]
        return:
            grid: (B, 1, Hg, Wg, D) 每格 1 个 scale
        """
        B = x.size(0)
        feat = self.trunk(x)            # (B, ch, H', W')

        # 用卷积生成 grid，保留空间结构
        grid = self.grid_conv(feat)     # (B, D, H', W')

        # 上采样到目标 grid 大小
        grid = F.interpolate(grid, size=(self.grid_H, self.grid_W),
                             mode='bilinear', align_corners=False)  # (B, D, Hg, Wg)

        # depth 方向平滑
        B, D, Hg, Wg = grid.shape
        grid = grid.view(B, D, Hg * Wg).permute(0, 2, 1).contiguous()  # (B, Hg*Wg, D)
        grid = grid.view(B * Hg * Wg, D).unsqueeze(1)  # (B*Hg*Wg, 1, D)
        grid = self.depth_smooth(grid)  # (B*Hg*Wg, 1, D)
        grid = grid.squeeze(1).view(B, Hg * Wg, D).permute(0, 2, 1).contiguous()  # (B, D, Hg*Wg)
        grid = grid.view(B, D, Hg, Wg)

        # 调整维度顺序
        grid = grid.permute(0, 2, 3, 1).contiguous()  # (B, Hg, Wg, D)
        grid = grid.view(B, 1, self.grid_H, self.grid_W, self.grid_D)
        return grid


# ---------------------------
# 4. 引导网络（小图输入）
# ---------------------------
class GuideNetSimple(nn.Module):
    """
    小型 CNN -> 每像素 1 维 guide（sigmoid 到 [0,1]）
    输入：下采样小图 (384x216)
    输出：上采样回全分辨率
    """

    def __init__(self, in_ch=3, guide_H=216, guide_W=384):
        super().__init__()
        self.guide_H = guide_H
        self.guide_W = guide_W
        self.conv1 = conv_lrelu(in_ch, 16, k=1, s=1, p=0)
        self.conv2 = conv_lrelu(16, 16, k=3, s=1, p=1)
        self.conv3 = conv_lrelu(16, 16, k=3, s=1, p=1)
        self.conv_out = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        x: (B,3,H,W) 全分辨率
        return:
            guide: (B,1,H,W) in [0,1] 全分辨率
        """
        B, _, H, W = x.shape
        # 下采样到小图
        x_small = F.interpolate(x, size=(self.guide_H, self.guide_W),
                                mode='bilinear', align_corners=False)
        x_small = self.conv1(x_small)
        x_small = self.conv2(x_small)
        x_small = self.conv3(x_small)
        g = self.conv_out(x_small)
        g = torch.sigmoid(g)
        # 上采样回全分辨率
        g = F.interpolate(g, size=(H, W), mode='bilinear', align_corners=False)
        return g


# ---------------------------
# 5. 简化版三线性 Slicing + 同比例缩放
# ---------------------------
class BilateralSliceScale(nn.Module):
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
        grid:  (B, 1, Hg, Wg, D)  每格 1 个 scale
        guide: (B, 1, H, W)       in [0,1]
        x:     (B, 3, H, W)       sRGB [0,1]
        return: (B, 3, H, W)      sRGB [0,1]
        """
        B, _, Hg, Wg, D = grid.shape
        _, _, H, W = x.shape
        device = x.device

        # 转成线性空间
        x_linear = srgb_to_linear(x)  # (B, 3, H, W)

        # 深度坐标 z \in [0, D-1]
        z = guide * (D - 1)  # (B,1,H,W)

        # 空间坐标
        ys_lin = torch.linspace(0, Hg - 1, H, device=device)
        xs_lin = torch.linspace(0, Wg - 1, W, device=device)
        ys_2d, xs_2d = torch.meshgrid(ys_lin, xs_lin, indexing='ij')

        ys = ys_2d.view(1, 1, H, W).expand(B, 1, H, W)
        xs = xs_2d.view(1, 1, H, W).expand(B, 1, H, W)

        # 下取整与上取整
        x0 = torch.floor(xs).clamp(min=0, max=Wg - 1)
        y0 = torch.floor(ys).clamp(min=0, max=Hg - 1)
        z0 = torch.floor(z).clamp(min=0, max=D - 1)

        x1 = (x0 + 1).clamp(max=Wg - 1)
        y1 = (y0 + 1).clamp(max=Hg - 1)
        z1 = (z0 + 1).clamp(max=D - 1)

        # 保存原始的浮点坐标用于插值权重
        dx = xs - torch.floor(xs)
        dy = ys - torch.floor(ys)
        dz = z - torch.floor(z)
        # clamp 权重到 [0, 1]
        dx = dx.clamp(0, 1)
        dy = dy.clamp(0, 1)
        dz = dz.clamp(0, 1)

        x0 = x0.long()
        y0 = y0.long()
        z0 = z0.long()
        x1 = x1.long()
        y1 = y1.long()
        z1 = z1.long()

        # grid: (B, 1, Hg, Wg, D) -> 重排为 (B, D, Hg, Wg) 方便索引
        # 原始: grid[b, 0, y, x, d]
        # 重排后: grid_perm[b, d, y, x]
        grid_perm = grid.squeeze(1).permute(0, 3, 1, 2)  # (B, D, Hg, Wg)
        grid_perm = grid_perm.contiguous().view(B, D * Hg * Wg)  # (B, D*Hg*Wg)

        def idx(y_, x_, z_):
            # 索引计算：grid_perm 形状是 (D, Hg, Wg)
            # flat_idx = z * (Hg * Wg) + y * Wg + x
            return z_ * (Hg * Wg) + y_ * Wg + x_

        i000 = idx(y0, x0, z0)
        i001 = idx(y0, x0, z1)
        i010 = idx(y0, x1, z0)
        i011 = idx(y0, x1, z1)
        i100 = idx(y1, x0, z0)
        i101 = idx(y1, x0, z1)
        i110 = idx(y1, x1, z0)
        i111 = idx(y1, x1, z1)

        def gather(grid_flat, idx_):
            """
            grid_flat: (B, N)
            idx_:      (B,1,H,W) long
            返回:      (B, 1, H, W)
            """
            B_local, N = grid_flat.shape
            idx_flat = idx_.view(B, -1)  # (B, H*W)
            # 确保索引在范围内
            idx_flat = idx_flat.clamp(0, N - 1)
            g = torch.gather(grid_flat, 1, idx_flat)  # (B, H*W)
            g = g.view(B, 1, H, W)
            return g

        g000 = gather(grid_perm, i000)
        g001 = gather(grid_perm, i001)
        g010 = gather(grid_perm, i010)
        g011 = gather(grid_perm, i011)
        g100 = gather(grid_perm, i100)
        g101 = gather(grid_perm, i101)
        g110 = gather(grid_perm, i110)
        g111 = gather(grid_perm, i111)

        wx0 = 1.0 - dx
        wy0 = 1.0 - dy
        wz0 = 1.0 - dz
        wx1 = dx
        wy1 = dy
        wz1 = dz

        w000 = wx0 * wy0 * wz0
        w001 = wx1 * wy0 * wz0
        w010 = wx0 * wy1 * wz0
        w011 = wx1 * wy1 * wz0
        w100 = wx0 * wy0 * wz1
        w101 = wx1 * wy0 * wz1
        w110 = wx0 * wy1 * wz1
        w111 = wx1 * wy1 * wz1

        # 三线性插值得到 scale
        scale = (
            g000 * w000 +
            g001 * w001 +
            g010 * w010 +
            g011 * w011 +
            g100 * w100 +
            g101 * w101 +
            g110 * w110 +
            g111 * w111
        )  # (B, 1, H, W)

        # 约束 scale 在合理范围内，防止 NaN
        scale = scale.clamp(0.01, 10.0)

        # 在线性空间做同比例缩放
        y_linear = x_linear * scale  # (B, 3, H, W)

        # 转回 sRGB
        y = linear_to_srgb(y_linear)
        return y


# ---------------------------
# 6. 完整简化版 HDRNet
# ---------------------------
@ARCH_REGISTRY.register()
class HDRnetSimple(nn.Module):
    """
    简化版 HDRNet：
    - Grid 只输出 scale 系数
    - 在线性空间做同比例缩放
    - 部署时可转成 4D LUT (32, 32, 16, 256)
    """

    def __init__(
        self,
        num_in_ch=3,
        num_out_ch=3,
        grid_H=32,
        grid_W=32,
        grid_D=16,
        base_channels=16,
        num_downsample=3,
        num_residual=2,
        grid_input_H=216,
        grid_input_W=384,
        guide_input_H=216,
        guide_input_W=384,
    ):
        super().__init__()
        assert num_in_ch == 3 and num_out_ch == 3, \
            "HDRNet 默认 3 通道输入输出"

        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.grid_H = grid_H
        self.grid_W = grid_W
        self.grid_D = grid_D
        self.grid_input_H = grid_input_H
        self.grid_input_W = grid_input_W

        self.lowres_net = LowResNetSimple(
            in_ch=num_in_ch,
            base_channels=base_channels,
            num_downsample=num_downsample,
            num_residual=num_residual,
            grid_H=grid_H,
            grid_W=grid_W,
            grid_D=grid_D,
        )
        self.guide_net = GuideNetSimple(in_ch=num_in_ch, guide_H=guide_input_H, guide_W=guide_input_W)
        self.slice_apply = BilateralSliceScale(
            grid_D=grid_D,
            grid_H=grid_H,
            grid_W=grid_W,
        )

    def forward(self, x, strength: float = 1.0):
        """
        x: (B,3,H,W) in [0,1] sRGB
        """
        # Grid 网络: 下采样到小图
        x_low = F.interpolate(x, size=(self.grid_input_H, self.grid_input_W),
                              mode='bilinear', align_corners=False)
        grid = self.lowres_net(x_low)    # (B, 1, Hg, Wg, D)

        # Guide 网络: 全分辨率输入
        guide = self.guide_net(x)        # (B,1,H,W)

        out = self.slice_apply(grid, guide, x)  # (B,3,H,W)

        if strength < 1.0:
            out = strength * out + (1.0 - strength) * x

        if self.training:
            return out
        else:
            return torch.clamp(out, 0.0, 1.0)
