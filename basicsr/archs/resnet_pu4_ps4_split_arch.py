import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY


class ResidualBlockNoBN(nn.Module):
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


class KeyFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_low = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv_mid = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv_skip = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

        self.attn = nn.Sequential(
            nn.Conv2d(out_channels * 3, 3, 1, 1, 0),
            nn.Softmax(dim=1)
        )
        self.fuse_conv = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

    def forward(self, low_feat, mid_feat, skip_feat):
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
class Deblur_ResNet_ConvDown_PS4_Split_Model(nn.Module):
    """
    三个 block 数量可配置版本 + 大跳连：
      - num_low_blocks:  低层 H/4 上的 Residual blocks 数
      - num_deep_blocks: H/8 分支的 Residual blocks 数
      - num_mid_blocks:  中层 H/4 上的 Residual blocks 数

    结构概览：
      输入 x
        ├─ 下采样 ×4 → res_low (num_low_blocks) → low_feat
        │      └─ H/8 分支：down_8 → res_deep → up_8 → 加回 low_feat
        ├─ skip_conv(low_feat) → skip_feat → up_* → res_mid(num_mid_blocks) → mid_feat
        ├─ KeyFusion(low_feat, mid_feat, skip_feat) → fused
        ├─ to_shuffle + PixelShuffle(4) → out_feat (残差)
        └─ output = out_feat + x   （大跳连）
    """

    def __init__(self,
                 num_channels=64,
                 in_channels=3,
                 num_low_blocks=6,
                 num_deep_blocks=2,
                 num_mid_blocks=3):
        super().__init__()
        self.num_channels = num_channels
        self.in_channels = in_channels

        # 1. 下采样到 H/4, W/4
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, kernel_size=4, stride=4, padding=0),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # 2. 低层残差堆叠（H/4）
        self.res_low = nn.Sequential(
            *[ResidualBlockNoBN(num_channels) for _ in range(num_low_blocks)]
        )

        # 3. H/8 多尺度分支
        self.down_8 = nn.Conv2d(num_channels, num_channels, 3, 2, 1)   # H/4 -> H/8
        self.res_deep = nn.Sequential(
            *[ResidualBlockNoBN(num_channels) for _ in range(num_deep_blocks)]
        )
        self.up_8 = nn.ConvTranspose2d(num_channels, num_channels, 4, 2, 1)  # H/8 -> H/4

        # 4. skip 分支（仍在 H/4）
        self.skip_conv = nn.Conv2d(num_channels, num_channels, 3, 1, 1)

        self.up_relu1 = nn.LeakyReLU(0.1, inplace=True)
        self.up_conv = nn.Conv2d(num_channels, num_channels, 3, 1, 1)
        self.up_relu2 = nn.LeakyReLU(0.1, inplace=True)

        # 5. 中层残差堆叠（H/4）
        self.res_mid = nn.Sequential(
            *[ResidualBlockNoBN(num_channels) for _ in range(num_mid_blocks)]
        )

        # 6. 三路特征融合 + 上采样
        self.fusion = KeyFusion(num_channels, num_channels)
        self.to_shuffle = nn.Conv2d(num_channels, in_channels * 16, 3, 1, 1)
        self.up_final = nn.PixelShuffle(4)

        self.out_act = nn.Identity()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            # 注意：如果你后面把激活函数改成别的，kaiming 的 nonlinearity 要对应改
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # 保存原始输入用于大跳连
        x_in = x

        assert x.shape[-2] % 4 == 0 and x.shape[-1] % 4 == 0, "输入图片尺寸必须是4的倍数"

        # 下采样到 H/4, W/4
        x = self.down_conv(x)   # B, 64, H/4, W/4

        # 低层特征
        low_feat = self.res_low(x)

        # H/8 多尺度分支
        deep = self.down_8(low_feat)   # B,64,H/8,W/8
        deep = self.res_deep(deep)
        deep = self.up_8(deep)         # B,64,H/4,W/4

        # 多尺度增强融合回 low_feat
        low_feat = low_feat + deep

        # skip 分支
        skip_feat = self.skip_conv(low_feat)

        # 中间特征提取
        x = self.up_relu1(skip_feat)
        x = self.up_conv(x)
        x = self.up_relu2(x)
        mid_feat = self.res_mid(x)

        # 三路特征融合
        fused = self.fusion(low_feat, mid_feat, skip_feat)  # B, 64, H/4, W/4

        # 上采样，得到残差
        feat_for_up = self.to_shuffle(fused)   # B, 3*16, H/4, W/4
        out_feat = self.up_final(feat_for_up)  # B, 3,    H,   W

        # 大跳连：输出 = 输入 + 残差
        output = out_feat + x_in

        output = self.out_act(output)
        output = torch.clamp(output, 0.0, 1.0)
        return output

    def prepare_for_inference(self):
        self.eval()