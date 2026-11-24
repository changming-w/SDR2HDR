#!/usr/bin/env python3
"""BT.709 RGB → BT.2020 RGB 批量转换"""
import os
import sys
import numpy as np
from PIL import Image
from pathlib import Path

# BT.709 → BT.2020 转换矩阵
M = np.array([
    [0.6274, 0.3293, 0.0433],
    [0.0691, 0.9195, 0.0114],
    [0.0164, 0.0880, 0.8956],
], dtype=np.float64)

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def convert_image(src_path, dst_path):
    img = Image.open(src_path).convert("RGB")
    arr = np.asarray(img, dtype=np.float64) / 255.0
    # 直接在 gamma 空间转换（与 GT 生成流程一致）
    arr_2020 = arr @ M.T
    arr_2020 = np.clip(np.rint(arr_2020 * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(arr_2020).save(dst_path)


def main():
    if len(sys.argv) < 2:
        print("用法: python bt709_to_bt2020.py <输入目录> [输出目录]")
        print("  若不指定输出目录，则在输入目录同级创建 <dirname>_bt2020")
        sys.exit(1)

    src_dir = Path(sys.argv[1])
    if not src_dir.is_dir():
        print(f"错误: {src_dir} 不是目录")
        sys.exit(1)

    if len(sys.argv) >= 3:
        dst_dir = Path(sys.argv[2])
    else:
        dst_dir = src_dir.parent / f"{src_dir.name}_bt2020"

    dst_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        p for p in src_dir.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )

    print(f"输入: {src_dir} ({len(files)} 张)")
    print(f"输出: {dst_dir}")

    for i, src_path in enumerate(files, 1):
        dst_path = dst_dir / src_path.name
        convert_image(src_path, dst_path)
        if i % 50 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] {src_path.name}")

    print(f"[done] {len(files)} 张 -> {dst_dir}")


if __name__ == "__main__":
    main()
