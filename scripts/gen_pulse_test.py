#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成单点脉冲测试图：用于验证 Gaussian 差分核 (DoG) 响应。

用途: 输入到 MTK 亮度差分金字塔，观察各级尺度的空间响应是否符合 DoG 模型:
    h_i(r) = a * [G(r, σ_{i-1}) - G(r, σ_i)]

预期响应特征:
  - 中心主响应 + 周围符号相反/衰减的环形响应
  - 不同级别响应半径按 ~2 倍关系增长
  - L1 范围最大，L6 范围最小

测试图:
  正脉冲: 背景 Y=128, 中心 3×3 像素 Y=144  (Δ=+16)
  负脉冲: 背景 Y=128, 中心 3×3 像素 Y=112  (Δ=-16)

依赖: pip install numpy opencv-python
"""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description="生成单点脉冲测试图 (正/负脉冲)，用于验证 DoG 核响应"
    )
    p.add_argument("--bg",       type=int, default=128,
                   help="背景亮度 Y (0~255, 默认 128)")
    p.add_argument("--delta",    type=int, default=16,
                   help="脉冲幅度 |ΔY| (默认 16, 即正脉冲=bg+16, 负脉冲=bg-16)")
    p.add_argument("--spot",     type=int, default=3, choices=[1, 3, 5, 7],
                   help="中心脉冲尺寸 N×N (默认 3, 推荐 3 避免单像素特殊处理)")
    p.add_argument("--width",    type=int, default=3840,
                   help="图像宽度 (像素, 默认 3840)")
    p.add_argument("--height",   type=int, default=2160,
                   help="图像高度 (像素, 默认 2160)")
    p.add_argument("--center",   type=str, default=None,
                   help="脉冲中心坐标 'x,y' (默认图像中心)")
    p.add_argument("--bitdepth", type=int, default=8, choices=[8, 16],
                   help="输出位深 (默认 8)")
    p.add_argument("-o", "--outdir", type=str, default="pulse_test",
                   help="输出目录 (默认 pulse_test)")
    p.add_argument("--both",     action="store_true", default=True,
                   help="同时生成正脉冲和负脉冲 (默认开启)")
    p.add_argument("--pos-only", action="store_true",
                   help="仅生成正脉冲")
    p.add_argument("--neg-only", action="store_true",
                   help="仅生成负脉冲")
    p.add_argument("--crosshair", action="store_true",
                   help="在脉冲中心画十字准线 (便于定位)")
    p.add_argument("--grid",     type=int, default=0,
                   help="在背景上叠加网格线间距 (0=关闭, 默认 0)")
    return p.parse_args()


def make_pulse_image(width: int, height: int, bg: int, delta: int, spot: int,
                     cx: int, cy: int, bitdepth: int,
                     crosshair: bool = False,
                     grid_spacing: int = 0) -> np.ndarray:
    """
    生成单点脉冲图像。

    Parameters
    ----------
    width, height : int
        图像尺寸。
    bg : int
        背景亮度 (0~255)。
    delta : int
        脉冲幅度。正=亮脉冲，负=暗脉冲。
    spot : int
        中心脉冲区域尺寸 (spot × spot)。
    cx, cy : int
        脉冲中心坐标。
    bitdepth : int
        8 或 16。
    crosshair : bool
        是否在中心画十字准线。
    grid_spacing : int
        网格线间距 (0=无网格)。

    Returns
    -------
    img : ndarray
    """
    if bitdepth == 8:
        dtype = np.uint8
        max_val = 255
        bg_val = np.clip(bg, 0, 255)
        spot_val = np.clip(bg + delta, 0, 255)
    else:
        dtype = np.uint16
        max_val = 65535
        bg_val = int(np.clip(bg, 0, 255) / 255.0 * 65535)
        spot_val = int(np.clip(bg + delta, 0, 255) / 255.0 * 65535)

    img = np.full((height, width), bg_val, dtype=dtype)

    # 绘制中心脉冲
    half = spot // 2
    y0 = max(0, cy - half)
    y1 = min(height, cy + half + 1)
    x0 = max(0, cx - half)
    x1 = min(width, cx + half + 1)
    img[y0:y1, x0:x1] = spot_val

    # 十字准线 (用 bg±8 的微弱标记，不干扰脉冲本身)
    if crosshair:
        if bitdepth == 8:
            mark_val = np.clip(bg - 8, 0, 255)
        else:
            mark_val = int(np.clip(bg - 8, 0, 255) / 255.0 * 65535)
        # 水平线
        img[cy, max(0, cx - 20):max(0, cx - half - 1)] = mark_val
        img[cy, min(width, cx + half + 1):min(width, cx + 21)] = mark_val
        # 垂直线
        img[max(0, cy - 20):max(0, cy - half - 1), cx] = mark_val
        img[min(height, cy + half + 1):min(height, cy + 21), cx] = mark_val

    # 网格线
    if grid_spacing > 0:
        if bitdepth == 8:
            grid_val = np.clip(bg - 4, 0, 255)
        else:
            grid_val = int(np.clip(bg - 4, 0, 255) / 255.0 * 65535)
        for pos in range(grid_spacing, width, grid_spacing):
            img[:, pos] = grid_val
        for pos in range(grid_spacing, height, grid_spacing):
            img[pos, :] = grid_val

    return img


def main():
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 中心坐标
    if args.center:
        cx, cy = map(int, args.center.split(","))
    else:
        cx = args.width // 2
        cy = args.height // 2

    # 确保脉冲不越界
    half = args.spot // 2
    if cx - half < 0 or cx + half >= args.width or cy - half < 0 or cy + half >= args.height:
        raise SystemExit(
            f"脉冲中心 ({cx},{cy}) + 尺寸 {args.spot} 超出图像范围 {args.width}×{args.height}"
        )

    gen_pos = not args.neg_only
    gen_neg = not args.pos_only

    results = []

    if gen_pos:
        img_pos = make_pulse_image(
            width=args.width, height=args.height, bg=args.bg, delta=+args.delta, spot=args.spot,
            cx=cx, cy=cy, bitdepth=args.bitdepth,
            crosshair=args.crosshair, grid_spacing=args.grid,
        )
        fname = f"pulse_pos_bg{args.bg}_d+{args.delta}_{args.spot}x{args.spot}.png"
        fpath = outdir / fname
        cv2.imwrite(str(fpath), img_pos)
        results.append(("正脉冲", fpath, +args.delta))
        print(f"[OK] {fpath}  (ΔY=+{args.delta}, {args.spot}×{args.spot}, center=({cx},{cy}))")

    if gen_neg:
        img_neg = make_pulse_image(
            width=args.width, height=args.height, bg=args.bg, delta=-args.delta, spot=args.spot,
            cx=cx, cy=cy, bitdepth=args.bitdepth,
            crosshair=args.crosshair, grid_spacing=args.grid,
        )
        fname = f"pulse_neg_bg{args.bg}_d-{args.delta}_{args.spot}x{args.spot}.png"
        fpath = outdir / fname
        cv2.imwrite(str(fpath), img_neg)
        results.append(("负脉冲", fpath, -args.delta))
        print(f"[OK] {fpath}  (ΔY=-{args.delta}, {args.spot}×{args.spot}, center=({cx},{cy}))")

    # 生成一张正负脉冲并排对比图
    if gen_pos and gen_neg:
        if args.bitdepth == 8:
            combined = np.hstack([img_pos, img_neg])
        else:
            combined = np.hstack([
                (img_pos / 256).astype(np.uint8),
                (img_neg / 256).astype(np.uint8),
            ])
        # 中间加分隔线
        sep_x = args.width
        combined[:, sep_x - 1:sep_x + 1] = 0
        fname_combined = f"pulse_combined_{args.spot}x{args.spot}.png"
        fpath_combined = outdir / fname_combined
        cv2.imwrite(str(fpath_combined), combined)
        print(f"[OK] {fpath_combined}  (正|负 并排对比)")

    # 打印测试说明
    print(f"\n{'='*60}")
    print("测试方法:")
    print("  1. 将脉冲图输入 MTK pipeline，所有尺度参数设为中性，记录基准输出")
    print("  2. 每次只调整一个尺度参数，记录增强后输出")
    print("  3. 计算 ΔY = Y_增强 - Y_基准")
    print("  4. 观察各级尺度的 ΔY 空间分布")
    print(f"\n预期 DoG 响应特征:")
    print("  - 中心主响应 + 周围符号相反/衰减的环形响应")
    print("  - 不同级别响应半径按 ~2 倍关系增长")
    print("  - L1 范围最大，L6 范围最小")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
