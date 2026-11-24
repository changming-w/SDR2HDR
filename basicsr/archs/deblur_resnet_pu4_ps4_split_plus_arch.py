import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY


class ResidualBlockNoBN(nn.Module):
    """去掉 BN 的残差块，更适合图像复原。
       结构: x -> Conv(3x3) -> LeakyReLU -> Conv(3x3) + skip
    """
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.act1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        return x + res


class SE_Module(nn.Module):
    def __init__(self, num_channels, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_channels, num_channels // reduction, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels // reduction, num_channels, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class KeyFusion(nn.Module):
    """三路特征融合模块，通道/空间自适应加权"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # in_channels = 64, out_channels = 64
        self.conv_low = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv_mid = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv_skip = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

        self.attn = nn.Sequential(
            nn.Conv2d(out_channels * 3, 3, 1, 1, 0),
            nn.Softmax(dim=1)
        )
        self.fuse_conv = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

    def forward(self, low_feat, mid_feat, skip_feat):
        # 三路特征默认同分辨率
        low = self.conv_low(low_feat)
        mid = self.conv_mid(mid_feat)
        skip = self.conv_skip(skip_feat)

        concat_feat = torch.cat([low, mid, skip], dim=1)
        weights = self.attn(concat_feat)
        fused = (
            low * weights[:, 0:1, :, :] +
            mid * weights[:, 1:2, :, :] +
            skip * weights[:, 2:3, :, :]
        )
        return self.fuse_conv(fused)


class WindowAttentionBlock(nn.Module):
    """轻量窗口自注意力块（在 H/4 上使用，避免算力爆炸）

    - 把 feature 按窗口划分 (window_size x window_size)，在每个窗口内做多头自注意力；
    - 窗口内 token 数固定（如 8x8 = 64），复杂度 O(64^2) 常数级；
    - 用一个 1x1 Conv 做 QKV 映射，用另一个 1x1 Conv 做输出映射；
    - 外面包一层残差，整体类似 Transformer block，但简化很多。
    """
    def __init__(self, dim, num_heads=4, window_size=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size

        self.qkv = nn.Conv2d(dim, dim * 3, 1, 1, 0)
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0)
        self.scale = (dim // num_heads) ** -0.5

        self.act = nn.GELU()
        # 可选 channel mixing
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1, 1, 0)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.window_size

        # 如果 H,W 不是 window_size 的整数倍，简单 pad 到最近倍数
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            B, C, H, W = x.shape

        # B,C,H,W -> B,3C,H,W
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        # 分窗口：先 reshape 成 (B, C, num_h*ws, num_w*ws)
        # 再 -> (B*num_windows, C, ws, ws)
        num_h = H // ws
        num_w = W // ws

        def window_partition(t):
            # t: B,C,H,W -> (B*num_h*num_w, C, ws, ws)
            t = t.view(B, C, num_h, ws, num_w, ws)
            t = t.permute(0, 2, 4, 1, 3, 5).contiguous()  # B, num_h, num_w, C, ws, ws
            t = t.view(B * num_h * num_w, C, ws, ws)
            return t

        def window_reverse(t):
            # t: (B*num_h*num_w, C, ws, ws) -> B,C,H,W
            t = t.view(B, num_h, num_w, C, ws, ws)
            t = t.permute(0, 3, 1, 4, 2, 5).contiguous()
            t = t.view(B, C, H, W)
            return t

        q = window_partition(q)  # Bn, C, ws, ws
        k = window_partition(k)
        v = window_partition(v)

        Bn, Cn, _, _ = q.shape
        # -> Bn, num_heads, head_dim, ws*ws
        head_dim = Cn // self.num_heads
        q = q.view(Bn, self.num_heads, head_dim, ws * ws)
        k = k.view(Bn, self.num_heads, head_dim, ws * ws)
        v = v.view(Bn, self.num_heads, head_dim, ws * ws)

        # 注意力: q^T k
        attn = torch.matmul(q.transpose(-2, -1), k) * self.scale  # Bn, num_heads, ws*ws, ws*ws
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v.transpose(-2, -1))  # Bn, num_heads, ws*ws, head_dim
        out = out.transpose(-2, -1).contiguous()
        out = out.view(Bn, Cn, ws, ws)

        # 合并窗口
        out = window_reverse(out)  # B,C,H,W

        # 投影 + FFN + 残差
        out = self.proj(out)
        out = self.act(out)
        out = self.ffn(out) + x  # 这里用 x 为残差基准

        # 去掉 pad
        if pad_h != 0 or pad_w != 0:
            out = out[:, :, :H - pad_h, :W - pad_w]

        return out


@ARCH_REGISTRY.register()
class Deblur_ResNet_ConvDown_PS4_Attn_Split_Plus_Model(nn.Module):
    """
    去模糊版 SRResNet（卷积下采样 + PixelShuffle 上采样 + 轻量窗口自注意力）：

    - 前端：stride=4 Conv2d 下采样 ×4。
    - 中间：
        * H/4 上的低层残差堆叠 (6 个 ResidualBlockNoBN + SE)；
        * H/8 轻量分支（2 个 ResidualBlockNoBN）扩大感受野；
        * H/4 上的中层残差 (3 个 ResidualBlockNoBN) + 1 个 WindowAttentionBlock 加强长程建模 + SE。
    - 后端：KeyFusion 输出 64 通道，Conv(64->3*16) + PixelShuffle(4) 上采样。
    - 输入:  (B, 3, H, W)   (H,W 是 4 的倍数)
      中间:  (B, 64, H/4, W/4)
      输出:  (B, 3, H, W)，clamp 到 [0, 1]。
    """

    def __init__(self, num_channels=64, in_channels=3, deep_blocks=2,
                 attn_heads=4, attn_window_size=8):
        super().__init__()
        self.num_channels = num_channels
        self.in_channels = in_channels

        # 1. 前端卷积下采样 ×4： (B,3,H,W) -> (B,64,H/4,W/4)
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, kernel_size=4, stride=4, padding=0),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # 2. 低层残差堆叠（无 BN）
        self.res_low = nn.Sequential(*[ResidualBlockNoBN(num_channels) for _ in range(6)])
        self.se_low = SE_Module(num_channels)

        # 3. 轻量多尺度增强分支：在 low_feat 上做 H/4 -> H/8 -> H/4 的小循环
        self.down_8 = nn.Conv2d(num_channels, num_channels, 3, 2, 1)  # H/4 -> H/8
        self.res_deep = nn.Sequential(*[ResidualBlockNoBN(num_channels) for _ in range(deep_blocks)])
        self.up_8 = nn.ConvTranspose2d(num_channels, num_channels, 4, 2, 1)  # H/8 -> H/4

        # 4. skip 分支：只做特征变换，不改变分辨率
        self.skip_conv = nn.Conv2d(num_channels, num_channels, 3, 1, 1)
        self.se_skip = SE_Module(num_channels)

        self.up_relu1 = nn.LeakyReLU(0.1, inplace=True)
        self.up_conv = nn.Conv2d(num_channels, num_channels, 3, 1, 1)
        self.up_relu2 = nn.LeakyReLU(0.1, inplace=True)

        # 5. 中层残差堆叠 + 轻量窗口自注意力 + SE
        self.res_mid = nn.Sequential(*[ResidualBlockNoBN(num_channels) for _ in range(3)])
        self.attn_mid = WindowAttentionBlock(num_channels,
                                             num_heads=attn_heads,
                                             window_size=attn_window_size)
        self.se_mid = SE_Module(num_channels)

        # 6. 特征融合：输入/输出都用 64 通道
        self.fusion = KeyFusion(num_channels, num_channels)  # 64 -> 64

        # 7. 最终上采样：
        #    先映射到 3*16 通道，再 PixelShuffle(4) -> (B,3,H,W)
        self.to_shuffle = nn.Conv2d(num_channels, in_channels * 16, 3, 1, 1)
        self.up_final = nn.PixelShuffle(4)

        self.out_act = nn.Identity()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # 输入高宽必须是4的倍数
        assert x.shape[-2] % 4 == 0 and x.shape[-1] % 4 == 0, "输入图片尺寸必须是4的倍数"

        # 下采样到 H/4,W/4（卷积方式）
        x = self.down_conv(x)   # B, 64, H/4, W/4

        # 低层特征
        low_feat = self.res_low(x)
        low_feat = self.se_low(low_feat)

        # 轻量多尺度增强：H/4 -> H/8 -> H/4
        deep = self.down_8(low_feat)   # B,64,H/8,W/8
        deep = self.res_deep(deep)     # 极少 block，计算量小
        deep = self.up_8(deep)         # B,64,H/4,W/4

        # 与 low_feat 融合，增强感受野
        low_feat = low_feat + deep

        # skip 分支（不改变分辨率）
        skip_feat = self.skip_conv(low_feat)
        skip_feat = self.se_skip(skip_feat)

        # 中间进一步特征提取 + 自注意力
        x = self.up_relu1(skip_feat)
        x = self.up_conv(x)
        x = self.up_relu2(x)

        mid_feat = self.res_mid(x)
        mid_feat = self.attn_mid(mid_feat)  # 轻量窗口自注意力
        mid_feat = self.se_mid(mid_feat)

        # 三路特征融合，输出 64 通道
        fused = self.fusion(low_feat, mid_feat, skip_feat)  # B, 64, H/4, W/4

        # 映射到 3*16 通道，再 PixelShuffle(4) 恢复回原分辨率
        feat_for_up = self.to_shuffle(fused)   # B, 3*16, H/4, W/4
        output = self.up_final(feat_for_up)    # B, 3,    H,   W

        output = self.out_act(output)
        output = torch.clamp(output, 0.0, 1.0)
        return output

    def prepare_for_inference(self):
        self.eval()