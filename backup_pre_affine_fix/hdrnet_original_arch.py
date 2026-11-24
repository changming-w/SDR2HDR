print('### USING HDRNET ARCH FROM:', __file__)

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
# 2. 低分辨率网络（输出官方 layout: (B, D, Hg, Wg, Cc)）
# ---------------------------
class LowResNet(nn.Module):
    """
    对齐 google/hdrnet.models.hdrnet.HdrNet._low_res_net 的思路：

      - 下采样 trunk: 3x3 conv + stride=2
      - 多层 3x3 conv
      - global pooling + fc 预测 bilateral grid

    输出 grid layout: (B, D, H_g, W_g, Cc)
    """

    def __init__(
        self,
        in_ch=3,
        base_channels=16,
        num_downsample=3,
        num_residual=2,
        grid_H=16,
        grid_W=16,
        grid_D=8,
        coeffs_channels=12,
    ):
        super().__init__()

        self.grid_H = grid_H
        self.grid_W = grid_W
        self.grid_D = grid_D
        self.coeffs_channels = coeffs_channels  # 3x3 + 3 = 12

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

        # global pooling 到 1x1，然后 fc 输出 D*Hg*Wg*Cc
        self.pool = nn.AdaptiveAvgPool2d(1)

        # ch -> (D * Hg * Wg * Cc)
        self.fc = nn.Linear(
            ch,
            self.grid_D * self.grid_H * self.grid_W * self.coeffs_channels,
        )

    def forward(self, x):
        """
        x: (B,3,H,W) in [0,1]
        return:
            grid: (B, D, H_g, W_g, Cc)
        """
        B = x.size(0)
        feat = self.trunk(x)            # (B, ch, H', W')
        pooled = self.pool(feat)        # (B, ch, 1, 1)
        pooled = pooled.view(B, -1)     # (B, ch)

        grid_flat = self.fc(pooled)     # (B, D*Hg*Wg*Cc)
        grid = grid_flat.view(
            B, self.grid_D, self.grid_H, self.grid_W, self.coeffs_channels
        )  # (B, D, Hg, Wg, Cc)
        return grid


# ---------------------------
# 3. 引导网络
# ---------------------------
class GuideNet(nn.Module):
    """
    小型全分辨率 CNN -> 每像素 1 维 guide（sigmoid 到 [0,1]）
    """

    def __init__(self, in_ch=3):
        super().__init__()
        self.conv1 = conv_lrelu(in_ch, 16, k=1, s=1, p=0)
        self.conv2 = conv_lrelu(16, 16, k=3, s=1, p=1)
        self.conv3 = conv_lrelu(16, 16, k=3, s=1, p=1)
        self.conv_out = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        x: (B,3,H,W)
        return:
            guide: (B,1,H,W) in [0,1]
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        g = self.conv_out(x)
        g = torch.sigmoid(g)
        return g


# ---------------------------
# 4. 三线性 Slicing + 3x3 仿射  (官方 layout 版)
# ---------------------------
class BilateralSliceApply(nn.Module):
    """
    PyTorch 版 bilateral slice apply，兼容官方 layout:

    输入：
      - grid:   (B, D, H_g, W_g, Cc)   官方：D, Hg, Wg, 通道
      - guide:  (B, 1, H, W)           in [0,1]
      - x:      (B, 3, H, W)           full-res RGB

    输出：
      - y: (B,3,H,W) = A(x)*x + b(x)
    """

    def __init__(self, coeffs_channels=12, grid_D=8, grid_H=16, grid_W=16):
        super().__init__()
        assert coeffs_channels == 12, "HDRNet 使用 12 个系数（3x3+3）"
        self.Cc = coeffs_channels
        self.D = grid_D
        self.Hg = grid_H
        self.Wg = grid_W

    def forward(self, grid, guide, x):
        """
        grid:  (B, D, Hg, Wg, Cc)
        guide: (B, 1, H, W)
        x:     (B, 3, H, W)
        """
        B, D, Hg, Wg, Cc = grid.shape
        _, _, H, W = x.shape
        device = x.device

        # 深度坐标 z \in [0, D-1]
        z = guide * (D - 1)  # (B,1,H,W)

        # 使用 meshgrid 构造 2D 坐标，兼容 H≠W
        ys_lin = torch.linspace(0, Hg - 1, H, device=device)  # (H,)
        xs_lin = torch.linspace(0, Wg - 1, W, device=device)  # (W,)
        ys_2d, xs_2d = torch.meshgrid(ys_lin, xs_lin, indexing='ij')  # (H,W),(H,W)

        ys = ys_2d.view(1, 1, H, W).expand(B, 1, H, W)  # (B,1,H,W)
        xs = xs_2d.view(1, 1, H, W).expand(B, 1, H, W)  # (B,1,H,W)

        # 下取整与上取整
        x0 = torch.floor(xs)
        y0 = torch.floor(ys)
        z0 = torch.floor(z)

        x1 = torch.clamp(x0 + 1, max=Wg - 1)
        y1 = torch.clamp(y0 + 1, max=Hg - 1)
        z1 = torch.clamp(z0 + 1, max=D - 1)

        dx = xs - x0
        dy = ys - y0
        dz = z - z0

        x0 = x0.long()
        y0 = y0.long()
        z0 = z0.long()
        x1 = x1.long()
        y1 = y1.long()
        z1 = z1.long()

        # 官方 layout (D, Hg, Wg, Cc) 展平到 N=D*Hg*Wg，通道 Cc 保留在最后
        # grid: (B, D, Hg, Wg, Cc) -> (B, N, Cc)
        grid_flat = grid.view(B, D * Hg * Wg, Cc)  # (B, N, Cc)

        def idx(z_, y_, x_):
            # flatten index 与 grid_flat 的展平顺序对应
            return (z_ * Hg + y_) * Wg + x_  # (B,1,H,W), 值域 [0, N-1]

        i000 = idx(z0, y0, x0)
        i001 = idx(z0, y0, x1)
        i010 = idx(z0, y1, x0)
        i011 = idx(z0, y1, x1)
        i100 = idx(z1, y0, x0)
        i101 = idx(z1, y0, x1)
        i110 = idx(z1, y1, x0)
        i111 = idx(z1, y1, x1)

        def gather(grid_flat, idx_):
            """
            grid_flat: (B, N, Cc)
            idx_:      (B,1,H,W) long
            返回:      (B, Cc, H, W)
            """
            B_local, N, Cc_local = grid_flat.shape
            assert B_local == B and Cc_local == Cc

            idx_flat = idx_.view(B, -1)  # (B, H*W)
            # 扩展成 (B, H*W, Cc)，按 dim=1 gather，然后再转成 (B,Cc,H,W)
            idx_expand = idx_flat.unsqueeze(-1).expand(-1, -1, Cc)  # (B, H*W, Cc)

            # gather 在 dim=1 上，使用整型索引
            g = torch.gather(grid_flat, 1, idx_expand)  # (B, H*W, Cc)
            g = g.permute(0, 2, 1).contiguous().view(B, Cc, H, W)  # (B, Cc, H, W)
            return g

        g000 = gather(grid_flat, i000)
        g001 = gather(grid_flat, i001)
        g010 = gather(grid_flat, i010)
        g011 = gather(grid_flat, i011)
        g100 = gather(grid_flat, i100)
        g101 = gather(grid_flat, i101)
        g110 = gather(grid_flat, i110)
        g111 = gather(grid_flat, i111)

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

        coeff = (
            g000 * w000 +
            g001 * w001 +
            g010 * w010 +
            g011 * w011 +
            g100 * w100 +
            g101 * w101 +
            g110 * w110 +
            g111 * w111
        )  # (B, Cc, H, W)

        # ---- 防御性检查 ----
        Bc, Cc2, Hc, Wc = coeff.shape
        assert Bc == B and Hc == H and Wc == W, \
            f'coeff shape mismatch: {coeff.shape}, x: {(B, 3, H, W)}'
        assert Cc2 == 12, f'coeff channels should be 12, but got {Cc2}'

        Cin, Cout = 3, 3

        # 9 个系数做 3x3 线性变换，3 个做 bias
        A = coeff[:, :Cin * Cout, :, :].reshape(B, Cout, Cin, H, W)  # (B,3,3,H,W)
        b = coeff[:, Cin * Cout:, :, :].reshape(B, Cout, 1, H, W)    # (B,3,1,H,W)
        x_vec = x.reshape(B, Cin, 1, H, W)                           # (B,3,1,H,W)

        prod = A * x_vec                    # (B,3,3,H,W)
        # 修复广播：sum 后 (B,3,H,W)，b.squeeze(2) 也是 (B,3,H,W)
        y = prod.sum(dim=2) + b.squeeze(2)  # (B,3,H,W)
        return y


# ---------------------------
# 5. 完整 HDRNet（官方 layout 版）
# ---------------------------
@ARCH_REGISTRY.register()
class HDRnetOrig(nn.Module):
    """
    完整 HDRNet 结构（bilateral grid 使用官方 layout: (B, D, Hg, Wg, Cc)）：

      - LowResNet: 预测双边网格系数（bilateral grid）
      - GuideNet: 预测 per-pixel guide
      - BilateralSliceApply: 三线性插值 + 仿射变换
    """

    def __init__(
        self,
        num_in_ch=3,
        num_out_ch=3,
        grid_H=16,
        grid_W=16,
        grid_D=8,
        base_channels=16,
        num_downsample=3,
        num_residual=2,
        fixed_ds_size=256,
    ):
        super().__init__()
        assert num_in_ch == 3 and num_out_ch == 3, \
            "HDRNet 默认 3 通道输入输出，如需修改需要调整系数维度。"

        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.grid_H = grid_H
        self.grid_W = grid_W
        self.grid_D = grid_D
        self.coeffs_channels = 12
        self.fixed_ds_size = fixed_ds_size

        self.lowres_net = LowResNet(
            in_ch=num_in_ch,
            base_channels=base_channels,
            num_downsample=num_downsample,
            num_residual=num_residual,
            grid_H=grid_H,
            grid_W=grid_W,
            grid_D=grid_D,
            coeffs_channels=self.coeffs_channels,
        )
        self.guide_net = GuideNet(in_ch=num_in_ch)
        self.slice_apply = BilateralSliceApply(
            coeffs_channels=self.coeffs_channels,
            grid_D=grid_D,
            grid_H=grid_H,
            grid_W=grid_W,
        )

    def forward(self, x, strength: float = 1.0):
        """
        x: (B,3,H,W) in [0,1]
        """
        # Grid 网络: 下采样到固定分辨率
        x_low = F.interpolate(x, size=(self.fixed_ds_size, self.fixed_ds_size),
                              mode='bilinear', align_corners=False)
        grid = self.lowres_net(x_low)    # (B, D, Hg, Wg, Cc)

        # Guide 网络: 全分辨率输入
        guide = self.guide_net(x)        # (B,1,H,W)

        out = self.slice_apply(grid, guide, x)  # (B,3,H,W)

        if strength < 1.0:
            out = strength * out + (1.0 - strength) * x

        if self.training:
            return out
        else:
            return torch.clamp(out, 0.0, 1.0)


# ---------------------------
# 6. 分支网络（用于 ONNX / 分段部署）
# ---------------------------
@ARCH_REGISTRY.register()
class HDRnetOrigGuideBranch(nn.Module):
    """
    仅输出 guide 的分支。
    用于：先在一个模型里 x->guide，再在另一个模型里 x+guide->y。
    """
    def __init__(self, num_in_ch=3):
        super().__init__()
        self.num_in_ch = num_in_ch
        self.guide_net = GuideNet(in_ch=num_in_ch)

    def forward(self, x):
        return self.guide_net(x)  # (B,1,H,W)


@ARCH_REGISTRY.register()
class HDRnetOrigGridBranch(nn.Module):
    """
    低分辨率网格 + slicing + 应用 的分支。
    输入：x, guide
    输出：y
    """
    def __init__(
        self,
        num_in_ch=3,
        num_out_ch=3,
        grid_H=16,
        grid_W=16,
        grid_D=8,
        base_channels=16,
        num_downsample=3,
        num_residual=2,
    ):
        super().__init__()
        assert num_in_ch == 3 and num_out_ch == 3

        self.lowres_net = LowResNet(
            in_ch=num_in_ch,
            base_channels=base_channels,
            num_downsample=num_downsample,
            num_residual=num_residual,
            grid_H=grid_H,
            grid_W=grid_W,
            grid_D=grid_D,
            coeffs_channels=12,
        )
        self.slice_apply = BilateralSliceApply(
            coeffs_channels=12,
            grid_D=grid_D,
            grid_H=grid_H,
            grid_W=grid_W,
        )

    def forward(self, x, guide):
        grid = self.lowres_net(x)           # (B, D, Hg, Wg, Cc)
        out = self.slice_apply(grid, guide, x)
        return torch.clamp(out, 0.0, 1.0)
