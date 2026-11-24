import os
import torch
import torch.nn as nn

from basicsr.archs.hdrnet_arch import HDRnet, LowResNet, GuideNet, BilateralSliceApply


# ===========================
# 1. 三个 ONNX wrapper 定义
# ===========================

class GridNetONNX(nn.Module):
    """只负责：低分辨率 RGB -> bilateral grid"""
    def __init__(self, lowres_net: LowResNet):
        super().__init__()
        self.lowres_net = lowres_net

    def forward(self, x_low):
        """
        x_low: (B,3,H_low,W_low)  例如 512x512
        return:
            grid: (B,12,D,Hg,Wg)
        """
        return self.lowres_net(x_low)


class GuideNetONNX(nn.Module):
    """只负责：全分辨率 RGB -> guide map"""
    def __init__(self, guide_net: GuideNet):
        super().__init__()
        self.guide_net = guide_net

    def forward(self, x_full):
        """
        x_full: (B,3,H_full,W_full)
        return:
            guide: (B,1,H_full,W_full)
        """
        return self.guide_net(x_full)


class SliceApplyONNX(nn.Module):
    """
    只负责：三线性插值 + 仿射
    输入:
      grid:   (B,12,D,Hg,Wg)
      guide:  (B,1,H_full,W_full)
      x_full: (B,3,H_full,W_full)
    输出:
      y_full: (B,3,H_full,W_full)
    """
    def __init__(self, slice_apply: BilateralSliceApply):
        super().__init__()
        self.slice_apply = slice_apply

    def forward(self, grid, guide, x_full):
        y = self.slice_apply(grid, guide, x_full)
        # 导出推理模型，一般直接 clamp 到 [0,1]
        return torch.clamp(y, 0.0, 1.0)


# ===========================
# 2. 主导出逻辑
# ===========================

def export_hdrnet_onnx(
    pth_path,
    save_dir='onnx_exports',
    grid_low_res=(512, 512),      # grid 分支用的低分辨率大小
    full_res_example=(2160, 3840),# 示例 full-res，用于导出时的 dummy
    opset_version=11
):
    """
    从训练好的 HDRnet pth 权重导出三个 ONNX:
      1) grid_net.onnx       : x_low -> grid
      2) guide_net.onnx      : x_full -> guide_full
      3) slice_apply.onnx    : (grid, guide_full, x_full) -> y_full
    """

    os.makedirs(save_dir, exist_ok=True)

    # ------- 2.1 加载完整 HDRnet 权重 -------
    device = torch.device('cpu')
    print(f'Loading model from {pth_path}')
    state_dict = torch.load(pth_path, map_location=device)

    # 如果 pth 里是 BasicSR 的字典形式（包含 'params' 或 'params_ema'），做一下兼容
    if isinstance(state_dict, dict) and (
        'params' in state_dict or 'params_ema' in state_dict
    ):
        # 优先用 EMA 权重
        if 'params_ema' in state_dict:
            state_dict = state_dict['params_ema']
            print('Using params_ema from state_dict')
        else:
            state_dict = state_dict['params']
            print('Using params from state_dict')

    # 根据你的训练配置初始化 HDRnet
    # 注意：grid_H, grid_W, grid_D, base_channels, num_downsample, num_residual
    # 要和训练时保持一致
    model = HDRnet(
        num_in_ch=3,
        num_out_ch=3,
        grid_H=16,
        grid_W=16,
        grid_D=8,
        base_channels=16,
        num_downsample=3,
        num_residual=2,
    ).to(device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print('HDRnet weights loaded.')

    # ------- 2.2 构建并导出 GridNet ONNX -------
    grid_net = GridNetONNX(model.lowres_net).to(device).eval()

    H_low, W_low = grid_low_res
    dummy_low = torch.randn(1, 3, H_low, W_low, device=device)

    grid_onnx_path = os.path.join(save_dir, 'hdrnet_grid_net.onnx')
    print(f'Exporting GridNet ONNX to {grid_onnx_path}')
    torch.onnx.export(
        grid_net,
        dummy_low,
        grid_onnx_path,
        input_names=['x_low'],
        output_names=['grid'],
        opset_version=opset_version,
        # 如果你打算固定低分辨率（例如永远 512x512），可以不写 dynamic_axes
        dynamic_axes={
            'x_low': {0: 'batch', 2: 'H_low', 3: 'W_low'},
            # grid 尺寸为 (B,12,D,Hg,Wg)，一般固定，不设动态轴
            'grid': {0: 'batch'},
        },
    )
    print('GridNet ONNX exported.')

    # ------- 2.3 构建并导出 GuideNet ONNX -------
    guide_net = GuideNetONNX(model.guide_net).to(device).eval()

    H_full, W_full = full_res_example
    dummy_full = torch.randn(1, 3, H_full, W_full, device=device)

    guide_onnx_path = os.path.join(save_dir, 'hdrnet_guide_net.onnx')
    print(f'Exporting GuideNet ONNX to {guide_onnx_path}')
    torch.onnx.export(
        guide_net,
        dummy_full,
        guide_onnx_path,
        input_names=['x_full'],
        output_names=['guide_full'],
        opset_version=opset_version,
        dynamic_axes={
            'x_full': {0: 'batch', 2: 'H_full', 3: 'W_full'},
            'guide_full': {0: 'batch', 2: 'H_full', 3: 'W_full'},
        },
    )
    print('GuideNet ONNX exported.')

    # ------- 2.4 构建并导出 SliceApply ONNX -------
    slice_net = SliceApplyONNX(model.slice_apply).to(device).eval()

    # grid shape: (B,12,D,Hg,Wg)
    D = model.grid_D
    Hg = model.grid_H
    Wg = model.grid_W
    dummy_grid = torch.randn(1, 12, D, Hg, Wg, device=device)

    dummy_guide = torch.randn(1, 1, H_full, W_full, device=device)
    dummy_x_full = torch.randn(1, 3, H_full, W_full, device=device)

    slice_onnx_path = os.path.join(save_dir, 'hdrnet_slice_apply.onnx')
    print(f'Exporting SliceApply ONNX to {slice_onnx_path}')
    torch.onnx.export(
        slice_net,
        (dummy_grid, dummy_guide, dummy_x_full),
        slice_onnx_path,
        input_names=['grid', 'guide_full', 'x_full'],
        output_names=['y_full'],
        opset_version=opset_version,
        dynamic_axes={
            'grid': {0: 'batch'},  # D,Hg,Wg 固定
            'guide_full': {0: 'batch', 2: 'H_full', 3: 'W_full'},
            'x_full': {0: 'batch', 2: 'H_full', 3: 'W_full'},
            'y_full': {0: 'batch', 2: 'H_full', 3: 'W_full'},
        },
    )
    print('SliceApply ONNX exported.')

    print('All ONNX models exported successfully.')


# ===========================
# 3. 命令行入口
# ===========================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pth',
        type=str,
        required=True,
        help='Path to HDRnet .pth (e.g., experiments/xxx/net_g_latest.pth)',
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='onnx_exports',
        help='Directory to save ONNX files',
    )
    parser.add_argument(
        '--grid_h',
        type=int,
        default=512,
        help='Low-res height for grid branch dummy input',
    )
    parser.add_argument(
        '--grid_w',
        type=int,
        default=512,
        help='Low-res width for grid branch dummy input',
    )
    parser.add_argument(
        '--full_h',
        type=int,
        default=2160,
        help='Example full-res height for guide/slice dummy input',
    )
    parser.add_argument(
        '--full_w',
        type=int,
        default=3840,
        help='Example full-res width for guide/slice dummy input',
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=11,
        help='ONNX opset version',
    )
    args = parser.parse_args()

    export_hdrnet_onnx(
        pth_path=args.pth,
        save_dir=args.save_dir,
        grid_low_res=(args.grid_h, args.grid_w),
        full_res_example=(args.full_h, args.full_w),
        opset_version=args.opset,
    )