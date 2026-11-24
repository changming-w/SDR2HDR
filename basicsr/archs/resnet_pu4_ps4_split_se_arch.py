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


@ARCH_REGISTRY.register()
class Deblur_ResNet_ConvDown_PS4_Split_SE_Model(nn.Module):
    """
    去模糊版 SRResNet（卷积下采样 + PixelShuffle 上采样）：
    这一版只去掉了窗口自注意力，保留所有 SE 通道注意力模块。
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

        # 2. 低层残差堆叠（无 BN）+ SE
        self.res_low = nn.Sequential(*[ResidualBlockNoBN(num_channels) for _ in range(6)])
        self.se_low = SE_Module(num_channels)

        # 3. 轻量多尺度增强分支：在 low_feat 上做 H/4 -> H/8 -> H/4 的小循环
        self.down_8 = nn.Conv2d(num_channels, num_channels, 3, 2, 1)  # H/4 -> H/8
        self.res_deep = nn.Sequential(*[ResidualBlockNoBN(num_channels) for _ in range(deep_blocks)])
        self.up_8 = nn.ConvTranspose2d(num_channels, num_channels, 4, 2, 1)  # H/8 -> H/4

        # 4. skip 分支：只做特征变换，不改变分辨率 + SE
        self.skip_conv = nn.Conv2d(num_channels, num_channels, 3, 1, 1)
        self.se_skip = SE_Module(num_channels)

        self.up_relu1 = nn.LeakyReLU(0.1, inplace=True)
        self.up_conv = nn.Conv2d(num_channels, num_channels, 3, 1, 1)
        self.up_relu2 = nn.LeakyReLU(0.1, inplace=True)

        # 5. 中层残差堆叠（仅 ResBlock + SE，去掉自注意力）
        self.res_mid = nn.Sequential(*[ResidualBlockNoBN(num_channels) for _ in range(3)])
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

        # 中间进一步特征提取
        x = self.up_relu1(skip_feat)
        x = self.up_conv(x)
        x = self.up_relu2(x)

        mid_feat = self.res_mid(x)
        # 去掉 WindowAttentionBlock，这里直接接 SE
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