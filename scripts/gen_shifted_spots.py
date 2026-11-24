#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在中灰背景上放置多个相同的 3×3 亮点，位置依次偏移 1 像素。

用途: 观察像素级位移对亮点的视觉影响，常用于测试插值/抗锯齿/亚像素定位。
依赖: pip install numpy opencv-python
"""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def parse_args():
    p = argparse.ArgumentParser(
        description="中灰图上放多个相同 3×3 亮点，位置依次偏移 1 像素"
    )
    p.add_argument("--width",  type=int, default=3840,
                   help="图像宽度 (像素, 默认 3840)")
    p.add_argument("--height", type=int, default=2160,
                   help="图像高度 (像素, 默认 2160)")
    p.add_argument("--bg",      type=int,   default=128,
                   help="中灰背景亮度 0~255 (默认 128)")
    p.add_argument("--spot",    type=int,   default=255,
                   help="亮点亮度 0~255 (默认 255)")
    p.add_argument("--spotsize",type=int,   default=3,
                   help="亮点尺寸 (默认 3, 即 3×3)")
    p.add_argument("--nspots",  type=int,   default=16,
                   help="亮点数量 (默认 16)")
    p.add_argument("--cols",    type=int,   default=0,
                   help="每行亮点数 (0=自动, 默认 0)")
    p.add_argument("--bitdepth",type=int,   default=8, choices=[8, 16, 32],
                   help="输出位深 (默认 8)")
    p.add_argument("-o", "--out", type=str, default="shifted_spots.png",
                   help="输出文件路径 (默认 shifted_spots.png)")
    p.add_argument("--preview", action="store_true",
                   help="生成放大预览图 (需要 matplotlib)")
    return p.parse_args()


def main():
    args = parse_args()

    spot = args.spotsize
    n = args.nspots
    img_w = args.width
    img_h = args.height

    # 自动计算列数: 使亮点分布接近图像宽高比
    if args.cols <= 0:
        aspect = img_w / img_h
        cols = max(1, round(math.sqrt(n * aspect)))
    else:
        cols = args.cols
    rows = math.ceil(n / cols)

    # 根据图像尺寸反算间距，使亮点均匀分布
    cell_w = (img_w - spot) / max(cols - 1, 1) if cols > 1 else img_w
    cell_h = (img_h - spot) / max(rows - 1, 1) if rows > 1 else img_h
    gap_x = int(cell_w - spot)
    gap_y = int(cell_h - spot)

    # 创建中灰背景
    if args.bitdepth == 8:
        bg_val = args.bg
        spot_val = args.spot
        img = np.full((img_h, img_w), bg_val, dtype=np.uint8)
    elif args.bitdepth == 16:
        bg_val = int(args.bg / 255.0 * 65535)
        spot_val = int(args.spot / 255.0 * 65535)
        img = np.full((img_h, img_w), bg_val, dtype=np.uint16)
    else:
        bg_val = args.bg / 255.0
        spot_val = args.spot / 255.0
        img = np.full((img_h, img_w), bg_val, dtype=np.float32)

    # 放置亮点，每个亮点偏移 1 像素
    for i in range(n):
        row_idx = i // cols
        col_idx = i % cols
        # 基础位置 (使用 float 计算后取整)
        y0 = int(round(row_idx * cell_h))
        x0 = int(round(col_idx * cell_w))
        # 依次偏移 1 像素 (对角线方向)
        dy = i % spot   # 0,1,2,0,1,2,...
        dx = i % spot
        y_start = y0 + dy
        x_start = x0 + dx
        y_end = y_start + spot
        x_end = x_start + spot
        img[y_start:y_end, x_start:x_end] = spot_val

    # 保存
    outpath = Path(args.out)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(outpath), img)
    print(f"[OK] {outpath}  ({img_w}×{img_h}, {n} 个 {spot}×{spot} 亮点)")

    # 预览: 缩小显示 (大图不适合放大)
    if args.preview and HAS_MPL:
        img_disp = cv2.imread(str(outpath), cv2.IMREAD_UNCHANGED)
        if img_disp.dtype == np.uint16:
            img_disp = (img_disp / 256).astype(np.uint8)
        # 大图缩小到适合屏幕，小图放大
        max_dim = 1920
        if max(img_w, img_h) > max_dim:
            scale_preview = max_dim / max(img_w, img_h)
            disp_w = int(img_w * scale_preview)
            disp_h = int(img_h * scale_preview)
            img_show = cv2.resize(img_disp, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
            scale = scale_preview
        else:
            img_show = img_disp
            scale = 1.0
        fig, ax = plt.subplots(figsize=(14, 14 * img_h / img_w))
        ax.imshow(img_show, cmap="gray", vmin=0, vmax=255)
        ax.set_title(
            f"{n} spots ({spot}x{spot}) | {img_w}x{img_h} | "
            f"bg={args.bg}, spot={args.spot} | each shifted by 1px (diagonal)",
            fontsize=11,
        )
        # 标注每个亮点的位置偏移
        for i in range(n):
            row_idx = i // cols
            col_idx = i % cols
            dy = i % spot
            dx = i % spot
            cy = (row_idx * cell_h + dy + spot / 2) * scale
            cx = (col_idx * cell_w + dx + spot / 2) * scale
            ax.annotate(f"#{i}+{dx},{dy}",
                        xy=(cx, cy), color="red", fontsize=7,
                        ha="center", va="center",
                        fontweight="bold")
        ax.axis("off")
        preview_path = outpath.with_name(outpath.stem + "_preview.png")
        fig.savefig(str(preview_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[预览] {preview_path}")


if __name__ == "__main__":
    main()
