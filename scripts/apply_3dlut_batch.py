#!/usr/bin/env python3
"""批量应用 3D LUT (.npy) 到图片目录。

用法:
    python scripts/apply_3dlut_batch.py \
        --lut skin_3dlut.npy \
        --input_dir /path/to/tcl_skin \
        --output_dir /path/to/output
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np


def load_lut_npy(path):
    """加载 .npy 格式的 3D LUT，shape (N,N,N,3)。"""
    lut = np.load(path).astype(np.float64)
    assert lut.ndim == 4 and lut.shape[-1] == 3, f"Invalid LUT shape: {lut.shape}"
    return lut


def apply_3dlut_trilinear(image_float, lut):
    """
    对 float64 RGB [0,1] 图像应用 3D LUT（三线性插值）。

    Args:
        image_float: (H, W, 3) float64, [0, 1]
        lut: (N, N, N, 3) float64

    Returns:
        (H, W, 3) float64, clipped to [0, 1]
    """
    size = lut.shape[0]
    h, w, _ = image_float.shape

    # 将像素映射到 LUT 坐标空间 [0, size-1]
    coords = np.clip(image_float, 0.0, 1.0) * (size - 1)

    # 整数部分和小数部分
    c0 = np.floor(coords).astype(np.int32)
    c0 = np.minimum(c0, size - 2)
    frac = coords - c0

    r0, g0, b0 = c0[..., 0], c0[..., 1], c0[..., 2]
    fr, fg, fb = frac[..., 0], frac[..., 1], frac[..., 2]

    # 8 个角点的三线性插值
    def lerp(a, b, t):
        return a + t * (b - a)

    # 获取 8 个邻居
    c000 = lut[r0,     g0,     b0    ]
    c100 = lut[r0 + 1, g0,     b0    ]
    c010 = lut[r0,     g0 + 1, b0    ]
    c110 = lut[r0 + 1, g0 + 1, b0    ]
    c001 = lut[r0,     g0,     b0 + 1]
    c101 = lut[r0 + 1, g0,     b0 + 1]
    c011 = lut[r0,     g0 + 1, b0 + 1]
    c111 = lut[r0 + 1, g0 + 1, b0 + 1]

    # R 方向插值
    c00 = lerp(c000, c100, fr[..., None])
    c10 = lerp(c010, c110, fr[..., None])
    c01 = lerp(c001, c101, fr[..., None])
    c11 = lerp(c011, c111, fr[..., None])

    # G 方向插值
    c0 = lerp(c00, c10, fg[..., None])
    c1 = lerp(c01, c11, fg[..., None])

    # B 方向插值
    result = lerp(c0, c1, fb[..., None])

    return np.clip(result, 0.0, 1.0)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def main():
    parser = argparse.ArgumentParser(description="批量应用 3D LUT")
    parser.add_argument("--lut", required=True, help="LUT .npy 文件路径")
    parser.add_argument("--input_dir", required=True, help="输入图片目录")
    parser.add_argument("--output_dir", required=True, help="输出图片目录")
    args = parser.parse_args()

    print(f"加载 LUT: {args.lut}")
    lut = load_lut_npy(args.lut)
    print(f"LUT shape: {lut.shape}")

    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted([
        f for f in os.listdir(args.input_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ])

    print(f"共 {len(files)} 张图片")

    t_start = time.time()
    for i, fname in enumerate(files, 1):
        src_path = os.path.join(args.input_dir, fname)
        dst_path = os.path.join(args.output_dir, fname)

        img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  [{i}/{len(files)}] 跳过（无法读取）: {fname}")
            continue

        # BGR -> RGB, uint8 -> float64
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
        rgb = img[..., :3][..., ::-1].astype(np.float64) / 255.0

        out_rgb = apply_3dlut_trilinear(rgb, lut)

        # float64 -> uint8, RGB -> BGR
        out_bgr = (out_rgb * 255.0).clip(0, 255).astype(np.uint8)[..., ::-1]
        cv2.imwrite(dst_path, out_bgr)

        if i % 20 == 0 or i == len(files):
            elapsed = time.time() - t_start
            speed = i / elapsed
            eta = (len(files) - i) / speed if speed > 0 else 0
            print(f"  [{i}/{len(files)}] {speed:.1f} img/s, ETA {eta:.0f}s")

    total = time.time() - t_start
    print(f"\n完成！{len(files)} 张，耗时 {total:.1f}s")
    print(f"输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
