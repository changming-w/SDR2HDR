import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)


@LOSS_REGISTRY.register()
class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        if reduction not in ['mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: mean | sum')
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight, reduction=reduction)

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]

        y_diff = super().forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super().forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)

        loss = x_diff + y_diff

        return loss


@LOSS_REGISTRY.register()
class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

class GradientLoss(nn.Module):
    """全图梯度约束 loss（边缘约束），支持 L1 / L2。

    L = |∇pred - ∇target|_p
    其中 ∇ 用简单差分实现：
      gx = x[:,:,:,1:] - x[:,:,:,:-1]
      gy = x[:,:,1:,:] - x[:,:,:-1,:]

    Args:
        loss_weight (float): loss 权重.
        mode (str): 'l1' or 'l2'（对应 L1 / MSE）.
    """

    def __init__(self, loss_weight=1.0, mode='l1'):
        super(GradientLoss, self).__init__()
        self.loss_weight = loss_weight
        assert mode in ['l1', 'l2']
        self.mode = mode

    def _gradient(self, x):
        # x: [B, C, H, W]
        gx = x[:, :, :, 1:] - x[:, :, :, :-1]  # 水平梯度
        gy = x[:, :, 1:, :] - x[:, :, :-1, :]  # 垂直梯度
        return gx, gy

    def forward(self, pred, target, **kwargs):
        """
        pred, target: [B, C, H, W]
        """
        assert pred.shape == target.shape, 'pred and target must have same shape'

        gx_p, gy_p = self._gradient(pred)
        gx_t, gy_t = self._gradient(target)

        if self.mode == 'l1':
            loss_x = (gx_p - gx_t).abs().mean()
            loss_y = (gy_p - gy_t).abs().mean()
        else:  # 'l2'
            loss_x = ((gx_p - gx_t) ** 2).mean()
            loss_y = ((gy_p - gy_t) ** 2).mean()

        loss = (loss_x + loss_y) * 0.5
        return self.loss_weight * loss


@LOSS_REGISTRY.register()
class ChromaLoss(nn.Module):
    """色度损失：在 YCbCr 空间约束 Cb/Cr 通道，保持颜色一致性（BT.709）。

    将 RGB 转换到 BT.709 YCbCr 空间后，只计算 Cb/Cr 通道的 L1 损失。
    Y 通道（亮度）不计算损失，让模型自由学习亮度映射。

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super(ChromaLoss, self).__init__()
        self.loss_weight = loss_weight
        # BT.709 RGB -> YCbCr 转换矩阵（固定，不参与训练）
        self.register_buffer('rgb_to_ycbcr', torch.tensor([
            [0.2126, 0.7152, 0.0722],      # Y
            [-0.114572, -0.385428, 0.5],   # Cb
            [0.5, -0.454153, -0.045847]    # Cr
        ]))

    def _rgb_to_ycbcr(self, img):
        """RGB -> YCbCr 转换。

        Args:
            img: [B, 3, H, W] RGB 图像，值域 [0, 1]
        Returns:
            [B, 3, H, W] YCbCr 图像
        """
        # img: [B, 3, H, W] -> [B, H, W, 3]
        img_perm = img.permute(0, 2, 3, 1)
        # 矩阵乘法: [B, H, W, 3] @ [3, 3] -> [B, H, W, 3]
        ycbcr = torch.matmul(img_perm, self.rgb_to_ycbcr.t())
        # Cb/Cr 偏移 [0, 0.5] -> [B, H, W, 3]
        offset = torch.tensor([0.0, 0.5, 0.5], device=img.device)
        ycbcr = ycbcr + offset
        # -> [B, 3, H, W]
        return ycbcr.permute(0, 3, 1, 2)

    def forward(self, pred, target, **kwargs):
        """
        Args:
            pred (Tensor): [B, 3, H, W] 预测输出
            target (Tensor): [B, 3, H, W] Ground truth
        """
        pred_ycbcr = self._rgb_to_ycbcr(pred)
        target_ycbcr = self._rgb_to_ycbcr(target)

        # 只计算 Cb (index 1) 和 Cr (index 2) 的损失
        cb_loss = (pred_ycbcr[:, 1:2] - target_ycbcr[:, 1:2]).abs().mean()
        cr_loss = (pred_ycbcr[:, 2:3] - target_ycbcr[:, 2:3]).abs().mean()

        loss = (cb_loss + cr_loss) * 0.5
        return self.loss_weight * loss


@LOSS_REGISTRY.register()
class LumaL1Loss(nn.Module):
    """亮度 L1 损失：在 Y 通道上计算 L1 loss。

    支持 pred 和 target 使用不同色域的 Y 系数：
      - pred 为 BT.709 RGB（网络输出）→ 用 BT.709 系数算 Y
      - target 为 BT.2020 RGB（GT）→ 用 BT.2020 系数算 Y
    数学上等价于先把 GT 转到 BT.709 再算 Y，但无需矩阵乘法、无 clip 损失。

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
        reduction (str): 'none' | 'mean' | 'sum'. Default: 'mean'.
        target_colorspace (str): GT 色域空间，'bt709' 或 'bt2020'。Default: 'bt709'.
    """

    # 不同色域的亮度系数
    _LUMA_COEFFS = {
        'bt709':  [0.2126, 0.7152, 0.0722],
        'bt2020': [0.2627, 0.6780, 0.0593],
    }

    def __init__(self, loss_weight=1.0, reduction='mean', target_colorspace='bt709'):
        super(LumaL1Loss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

        assert target_colorspace in self._LUMA_COEFFS, \
            f'Unsupported target_colorspace: {target_colorspace}. Choose from {list(self._LUMA_COEFFS.keys())}'

        # pred 始终用 BT.709（网络输出在 BT.709 空间）
        self.register_buffer('pred_luma_coeffs',
                             torch.tensor(self._LUMA_COEFFS['bt709']).view(1, 3, 1, 1))
        # target 用指定色域的系数
        self.register_buffer('target_luma_coeffs',
                             torch.tensor(self._LUMA_COEFFS[target_colorspace]).view(1, 3, 1, 1))

    def forward(self, pred, target, **kwargs):
        """
        Args:
            pred (Tensor): [B, 3, H, W] 预测输出 (BT.709 RGB)
            target (Tensor): [B, 3, H, W] Ground truth (色域由 target_colorspace 指定)
        """
        pred_y = (pred * self.pred_luma_coeffs).sum(dim=1, keepdim=True)
        target_y = (target * self.target_luma_coeffs).sum(dim=1, keepdim=True)

        if self.reduction == 'mean':
            loss = F.l1_loss(pred_y, target_y, reduction='mean')
        elif self.reduction == 'sum':
            loss = F.l1_loss(pred_y, target_y, reduction='sum')
        else:
            loss = F.l1_loss(pred_y, target_y, reduction='none')

        return self.loss_weight * loss


@LOSS_REGISTRY.register()
class MaxRGBL1Loss(nn.Module):
    """maxRGB L1 损失：在 max(R,G,B) 通道上计算 L1 loss。

    与 LumaL1Loss 的区别：
      - LumaL1Loss 用加权 Y 系数，BT.2020 可直接用 BT.2020 系数（数学等价）
      - MaxRGBL1Loss 用 max(R,G,B)，BT.2020 没有等价关系，必须先把 GT 转为 BT.709 再取 max

    流程：
      pred:   BT.709 RGB → max(R, G, B)
      target: BT.2020 RGB → 转 BT.709 (clip) → max(R, G, B)

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
        reduction (str): 'none' | 'mean' | 'sum'. Default: 'mean'.
        target_colorspace (str): GT 色域空间，'bt709' 或 'bt2020'。Default: 'bt709'.
    """

    # BT.2020 → BT.709 转换矩阵
    _BT2020_TO_BT709 = torch.tensor([
        [ 1.6605, -0.5876, -0.0728],
        [-0.1246,  1.1329, -0.0083],
        [-0.0182, -0.1006,  1.1187],
    ])

    def __init__(self, loss_weight=1.0, reduction='mean', target_colorspace='bt709'):
        super(MaxRGBL1Loss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.target_colorspace = target_colorspace

        if target_colorspace == 'bt2020':
            self.register_buffer('conv_matrix',
                                 self._BT2020_TO_BT709.clone())
        else:
            self.conv_matrix = None

    def _to_bt709(self, target):
        """BT.2020 RGB → BT.709 RGB（clip 到 [0,1]）"""
        bt709 = torch.einsum('ij,bjhw->bihw', self.conv_matrix, target)
        return bt709.clamp(0.0, 1.0)

    def forward(self, pred, target, **kwargs):
        """
        Args:
            pred (Tensor): [B, 3, H, W] 预测输出 (BT.709 RGB)
            target (Tensor): [B, 3, H, W] Ground truth (色域由 target_colorspace 指定)
        """
        pred_max = pred.max(dim=1, keepdim=True)[0]

        if self.target_colorspace == 'bt2020':
            target_bt709 = self._to_bt709(target)
            target_max = target_bt709.max(dim=1, keepdim=True)[0]
        else:
            target_max = target.max(dim=1, keepdim=True)[0]

        if self.reduction == 'mean':
            loss = F.l1_loss(pred_max, target_max, reduction='mean')
        elif self.reduction == 'sum':
            loss = F.l1_loss(pred_max, target_max, reduction='sum')
        else:
            loss = F.l1_loss(pred_max, target_max, reduction='none')

        return self.loss_weight * loss