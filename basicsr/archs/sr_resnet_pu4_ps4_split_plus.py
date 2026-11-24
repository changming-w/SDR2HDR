import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY

# 修复：简化残差块，移除不稳定的重参数化逻辑（优先保证训练收敛）
class RepResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.act1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x + res  # 残差连接，保证梯度传递

# 关键节点通道注意力（保留原有逻辑，无bug）
class SE_Module(nn.Module):
    def __init__(self, num_channels, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_channels, num_channels//reduction, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels//reduction, num_channels, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

# 特征融合模块（保留原有逻辑，无bug）
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
        # 确保输入尺寸一致
        low = self.conv_low(F.interpolate(low_feat, size=mid_feat.shape[-2:], mode='bilinear', align_corners=False))
        mid = self.conv_mid(mid_feat)
        skip = self.conv_skip(skip_feat)

        concat_feat = torch.cat([low, mid, skip], dim=1)
        weights = self.attn(concat_feat)
        fused = low * weights[:, 0:1, :, :] + mid * weights[:, 1:2, :, :] + skip * weights[:, 2:3, :, :]
        return self.fuse_conv(fused)

# 修复后的核心模型（解决通道不匹配、激活函数错误等问题）
@ARCH_REGISTRY.register()
class SRResNet_PU4_PS4_Split_Plus_Model(nn.Module):
    def __init__(self, num_channels=64, in_channels=3):  # 增加输入通道数参数（默认RGB 3通道）
        super().__init__()
        self.num_channels = num_channels
        self.in_channels = in_channels

        # 修复1：动态计算PixelUnshuffle后的通道数（避免写死48导致的不匹配）
        self.down_pixel = nn.PixelUnshuffle(4)  # 输入尺寸需为4的倍数（如256×256）
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels * 16, num_channels, 3, 1, 1),  # 3×16=48（RGB图），动态适配输入通道
            nn.LeakyReLU(0.1, inplace=True)
        )

        # 修复2：简化残差块序列（6+3=9个残差块，保证容量且不冗余）
        self.res_low = nn.Sequential(*[RepResidualBlock(num_channels) for _ in range(6)])
        self.se_low = SE_Module(num_channels)

        # 上采样分支（保留原有结构，无bug）
        self.up_deconv = nn.ConvTranspose2d(num_channels, num_channels, 4, 2, 1)  # 2倍上采样
        self.up_relu1 = nn.LeakyReLU(0.1, inplace=True)
        self.up_conv = nn.Conv2d(num_channels, num_channels, 3, 1, 1)
        self.up_relu2 = nn.LeakyReLU(0.1, inplace=True)
        self.se_skip = SE_Module(num_channels)

        self.res_mid = nn.Sequential(*[RepResidualBlock(num_channels) for _ in range(3)])
        self.se_mid = SE_Module(num_channels)

        # 特征融合（保留原有逻辑）
        self.fusion = KeyFusion(num_channels, in_channels * 16)  # 输出通道匹配PixelShuffle需求

        # 最终上采样（4倍，与输入PixelUnshuffle对应）
        self.up_final = nn.PixelShuffle(4)

        # 修复3：移除Sigmoid激活，用Identity保证输出动态范围（0-1）
        self.out_act = nn.Identity()

        # 权重初始化（适配LeakyReLU）
        self.init_weights()

    def init_weights(self):
        """权重初始化，保证训练稳定性"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # 输入尺寸检查（可选，避免尺寸错误）
        assert x.shape[-2] % 4 == 0 and x.shape[-1] % 4 == 0, "输入图片尺寸必须是4的倍数"

        # 前向传播（修复后无特征全零问题）
        x = self.down_pixel(x)  # 通道数：3→48，尺寸：256×256→64×64
        x = self.conv_in(x)     # 通道数：48→64
        low_feat = self.res_low(x)
        low_feat = self.se_low(low_feat)

        # 上采样分支
        skip_feat = self.up_deconv(low_feat)  # 尺寸：64×64→128×128
        skip_feat = self.se_skip(skip_feat)
        x = self.up_relu1(skip_feat)
        x = self.up_conv(x)
        x = self.up_relu2(x)
        mid_feat = self.res_mid(x)
        mid_feat = self.se_mid(mid_feat)

        # 特征融合
        fused = self.fusion(low_feat, mid_feat, skip_feat)  # 输出通道：48

        # 最终上采样（4倍）
        output = self.up_final(fused)  # 通道数：48→3，尺寸：128×128→256×256
        output = self.out_act(output)

        # 确保输出在0-1范围（冗余保障）
        output = torch.clamp(output, 0.0, 1.0)
        return output

    def prepare_for_inference(self):
        """推理时无需重参数化（已简化残差块）"""
        self.eval()