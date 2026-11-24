#!/usr/bin/env python3
"""可视化 HDRNet Bilateral Grid"""

import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from basicsr.archs.hdrnet_bt709_arch import HDRnetBT709
from basicsr.utils import img2tensor


def load_model(model_path, device='cuda'):
    """加载模型"""
    model = HDRnetBT709(
        num_in_ch=3, num_out_ch=3,
        grid_H=32, grid_W=32, grid_D=16,
        base_channels=16, num_downsample=3, num_residual=2,
        fixed_ds_size=512
    )

    checkpoint = torch.load(model_path, map_location=device)
    if 'params_ema' in checkpoint:
        model.load_state_dict(checkpoint['params_ema'])
    elif 'params' in checkpoint:
        model.load_state_dict(checkpoint['params'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    print(f"[info] 模型加载完成: {model_path}")
    return model


def extract_grid(model, img_tensor, device='cuda'):
    """提取 Bilateral Grid"""
    with torch.no_grad():
        # 下采样到 fixed_ds_size
        x_low = F.interpolate(img_tensor, size=(512, 512), mode='bilinear', align_corners=False)
        # 提取 grid
        grid = model.lowres_net(x_low)  # (1, 12*16, 32, 32)
    return grid


def visualize_gain_maps(grid, img_h, img_w, device='cuda'):
    """可视化每个 bin 每个通道的增益比例（灰度图）

    grid shape: (1, 16, 32, 32, 12) 或 (1, 192, 32, 32)

    增益 = 输出/输入，对于单位输入 (1,1,1)：
    - R_out = a0 + a1 + a2 + a3
    - G_out = a4 + a5 + a6 + a7
    - B_out = a8 + a9 + a10 + a11

    布局：16 行 × 3 列（每行一个 bin，每列 R/G/B 通道）
    """
    # 处理不同的 grid 形状
    if grid.dim() == 5:
        # (1, 16, 32, 32, 12)
        grid_reshaped = grid[0]  # (16, 32, 32, 12)
    else:
        # (1, 192, 32, 32) -> (1, 12, 16, 32, 32) -> (16, 32, 32, 12)
        grid_reshaped = grid.reshape(1, 12, 16, 32, 32).permute(0, 2, 3, 4, 1)[0]  # (16, 32, 32, 12)

    cell_size = 512  # 每个增益图的大小
    cols = 3  # R, G, B
    rows = 16  # 16 个 bins

    canvas = np.ones((cell_size * rows, cell_size * cols), dtype=np.float32) * 0.5  # 中灰背景

    for b in range(16):
        # 提取 12 个系数
        coeffs = grid_reshaped[b]  # (32, 32, 12)

        # 计算对于单位输入 (1,1,1) 的输出
        # R_out = a0 + a1 + a2 + a3
        r_out = coeffs[:, :, 0] + coeffs[:, :, 1] + coeffs[:, :, 2] + coeffs[:, :, 3]  # (32, 32)
        # G_out = a4 + a5 + a6 + a7
        g_out = coeffs[:, :, 4] + coeffs[:, :, 5] + coeffs[:, :, 6] + coeffs[:, :, 7]  # (32, 32)
        # B_out = a8 + a9 + a10 + a11
        b_out = coeffs[:, :, 8] + coeffs[:, :, 9] + coeffs[:, :, 10] + coeffs[:, :, 11]  # (32, 32)

        # 增益 = 输出 / 输入，输入为 1，所以增益 = 输出
        gain_r = r_out
        gain_g = g_out
        gain_b = b_out

        # 上采样到 cell_size
        gain_r_up = F.interpolate(
            gain_r.unsqueeze(0).unsqueeze(0),
            size=(cell_size, cell_size),
            mode='bilinear',
            align_corners=False
        ).squeeze().cpu().numpy()

        gain_g_up = F.interpolate(
            gain_g.unsqueeze(0).unsqueeze(0),
            size=(cell_size, cell_size),
            mode='bilinear',
            align_corners=False
        ).squeeze().cpu().numpy()

        gain_b_up = F.interpolate(
            gain_b.unsqueeze(0).unsqueeze(0),
            size=(cell_size, cell_size),
            mode='bilinear',
            align_corners=False
        ).squeeze().cpu().numpy()

        # 放到 canvas 上
        row_start = b * cell_size
        row_end = (b + 1) * cell_size

        # R 通道增益
        canvas[row_start:row_end, 0:cell_size] = gain_r_up
        # G 通道增益
        canvas[row_start:row_end, cell_size:2*cell_size] = gain_g_up
        # B 通道增益
        canvas[row_start:row_end, 2*cell_size:3*cell_size] = gain_b_up

    # 裁剪到合理范围 [0, 2]，中灰 = 1
    canvas = np.clip(canvas, 0, 2)

    # 转成 uint8
    canvas_uint8 = (canvas * 127.5).astype(np.uint8)  # 1.0 -> 127 (中灰)

    # 画网格线
    canvas_bgr = cv2.cvtColor(canvas_uint8, cv2.COLOR_GRAY2BGR)
    for i in range(1, rows):
        cv2.line(canvas_bgr, (0, i * cell_size), (cols * cell_size, i * cell_size), (100, 100, 100), 1)
    for j in range(1, cols):
        cv2.line(canvas_bgr, (j * cell_size, 0), (j * cell_size, rows * cell_size), (100, 100, 100), 1)

    # 标注
    # 列标题
    cv2.putText(canvas_bgr, "R gain", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(canvas_bgr, "G gain", (cell_size + 50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(canvas_bgr, "B gain", (2 * cell_size + 50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 行标签（bin 编号）
    for b in range(16):
        y = b * cell_size + cell_size // 2 + 10
        cv2.putText(canvas_bgr, f"bin{b}", (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return canvas_bgr


def draw_grid_lines(img, grid_h, grid_w, color=(0, 255, 0), thickness=1):
    """在图像上绘制网格线"""
    h, w = img.shape[:2]
    img_out = img.copy()

    # 水平线
    for i in range(1, grid_h):
        y = int(i * h / grid_h)
        cv2.line(img_out, (0, y), (w, y), color, thickness)

    # 垂直线
    for i in range(1, grid_w):
        x = int(i * w / grid_w)
        cv2.line(img_out, (x, 0), (x, h), color, thickness)

    return img_out


def main():
    parser = argparse.ArgumentParser(description="可视化 Bilateral Grid")
    parser.add_argument("--input", required=True, help="输入图片路径")
    parser.add_argument("--model", required=True, help="模型权重路径")
    parser.add_argument("--output", default="./grid_viz", help="输出目录")
    parser.add_argument("--device", default="cuda", help="设备")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 加载图片
    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(args.input)
    img_h, img_w = img.shape[:2]
    print(f"[info] 图片尺寸: {img_w}x{img_h}")

    # 转 tensor
    img_tensor = img2tensor(img, bgr2rgb=True, float32=True)
    img_tensor = img_tensor.unsqueeze(0).to(args.device)  # (1, 3, H, W)
    img_tensor = img_tensor / 255.0

    # 加载模型
    model = load_model(args.model, args.device)

    # 提取 grid
    grid = extract_grid(model, img_tensor, args.device)
    print(f"[info] Grid shape: {grid.shape}")

    # 计算 BT.709 亮度图
    # Y = 0.2126*R + 0.7152*G + 0.0722*B
    img_rgb = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    luma_bt709 = 0.2126 * img_rgb[:, :, 0] + 0.7152 * img_rgb[:, :, 1] + 0.0722 * img_rgb[:, :, 2]
    luma_uint8 = (luma_bt709 * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(args.output, "luma_bt709.png"), luma_uint8)

    # 运行推理
    with torch.no_grad():
        output = model(img_tensor)

    # 保存输入和输出
    input_rgb = (img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    output_rgb = (output.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    input_rgb = cv2.cvtColor(input_rgb, cv2.COLOR_RGB2BGR)
    output_rgb = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)

    cv2.imwrite(os.path.join(args.output, "input.png"), input_rgb)
    cv2.imwrite(os.path.join(args.output, "output.png"), output_rgb)

    # 可视化增益图（灰度图，16行×3列）
    gain_viz = visualize_gain_maps(grid, img_h, img_w, args.device)
    cv2.imwrite(os.path.join(args.output, "gain_maps.png"), gain_viz)

    # 在输出图上叠加网格线
    output_with_grid = draw_grid_lines(output_rgb, 32, 32, color=(0, 255, 0), thickness=1)
    cv2.imwrite(os.path.join(args.output, "output_with_grid.png"), output_with_grid)

    # 在输入图上叠加网格线
    input_with_grid = draw_grid_lines(input_rgb, 32, 32, color=(0, 255, 0), thickness=1)
    cv2.imwrite(os.path.join(args.output, "input_with_grid.png"), input_with_grid)

    # 混合可视化：gain_viz + output
    gain_resized = cv2.resize(gain_viz, (output_rgb.shape[1], output_rgb.shape[0]))
    blended = cv2.addWeighted(output_rgb, 0.7, gain_resized, 0.3, 0)
    cv2.imwrite(os.path.join(args.output, "blended.png"), blended)

    print(f"[done] 可视化结果保存到: {args.output}")
    print(f"  - input.png: 输入图片")
    print(f"  - luma_bt709.png: BT.709 亮度图 (bins 的分档依据)")
    print(f"  - output.png: 模型输出")
    print(f"  - gain_maps.png: 增益图 (16行×3列，灰度=增益比例，亮=增强，暗=抑制)")
    print(f"  - input_with_grid.png: 输入 + 32x32 网格线")
    print(f"  - output_with_grid.png: 输出 + 32x32 网格线")
    print(f"  - blended.png: 输出与增益图混合")


if __name__ == "__main__":
    main()
