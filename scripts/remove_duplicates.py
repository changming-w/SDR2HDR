#!/usr/bin/env python3
"""检测并删除重复/相似图片（配对数据集）。

使用 SSIM 检测相似图片，保留每组相似图中的第一张。
同时删除对应的 GT 图片。
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


def compute_similarity(img1_path, img2_path):
    """计算两张图片的相似度（直方图相关系数）。"""
    try:
        img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
        img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)

        if img1 is None or img2 is None:
            return 0.0

        # Resize to smaller for speed
        img1 = cv2.resize(img1, (256, 256))
        img2 = cv2.resize(img2, (256, 256))

        # Compute histograms for each channel
        h1 = cv2.calcHist([img1], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
        h2 = cv2.calcHist([img2], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])

        # Normalize
        h1 = cv2.normalize(h1, h1).flatten()
        h2 = cv2.normalize(h2, h2).flatten()

        # Correlation
        return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
    except Exception as e:
        return 0.0


def find_duplicates(img_dir, threshold=0.95):
    """找出相似图片组。

    Args:
        img_dir: 图片目录
        threshold: 直方图相关系数阈值，越高越严格（推荐 0.95-0.99）

    Returns:
        duplicates: 要删除的文件列表
    """
    # 收集所有图片
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.webp'}
    img_files = sorted([
        f for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])

    print(f"[info] 找到 {len(img_files)} 张图片")

    # 预计算所有直方图
    print("[info] 计算直方图...")
    histograms = {}
    for fname in tqdm(img_files, desc="Histogram"):
        path = os.path.join(img_dir, fname)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.resize(img, (256, 256))
            h = cv2.calcHist([img], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
            histograms[fname] = cv2.normalize(h, h).flatten()

    print(f"[info] 成功计算 {len(histograms)} 张直方图")

    # 找相似图片
    print(f"[info] 检测相似图片 (相关系数阈值={threshold})...")
    duplicates = set()

    files = list(histograms.keys())
    for i in tqdm(range(len(files)), desc="Comparing"):
        fname_i = files[i]
        if fname_i in duplicates:
            continue

        h_i = histograms[fname_i]

        # 与后面的图片比较
        for j in range(i + 1, len(files)):
            fname_j = files[j]
            if fname_j in duplicates:
                continue

            h_j = histograms[fname_j]
            corr = cv2.compareHist(h_i, h_j, cv2.HISTCMP_CORREL)

            if corr >= threshold:
                duplicates.add(fname_j)
                print(f"  相似: {fname_i} <-> {fname_j} (相关={corr:.3f})")

    print(f"\n[info] 发现 {len(duplicates)} 张重复/相似图片")
    return duplicates


def main():
    parser = argparse.ArgumentParser(description="去除重复/相似图片")
    parser.add_argument("--sdr", required=True, help="SDR 图片目录")
    parser.add_argument("--hdr", required=True, help="HDR 图片目录")
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="直方图相关系数阈值 (默认 0.95，越高越严格)")
    parser.add_argument("--dry-run", action="store_true",
                        help="只检测不删除")
    args = parser.parse_args()

    # 找重复
    duplicates = find_duplicates(args.sdr, args.threshold)

    if not duplicates:
        print("[info] 没有发现重复图片")
        return

    if args.dry_run:
        print(f"\n[dry-run] 将删除以下 {len(duplicates)} 个文件:")
        for f in sorted(duplicates):
            print(f"  {f}")
        print(f"\n[dry-run] 使用 --no-dry-run 执行删除")
        return

    # 删除文件
    print(f"\n[info] 开始删除 {len(duplicates)} 对图片...")
    deleted = 0
    for fname in sorted(duplicates):
        sdr_path = os.path.join(args.sdr, fname)
        hdr_path = os.path.join(args.hdr, fname)

        try:
            if os.path.exists(sdr_path):
                os.remove(sdr_path)
            if os.path.exists(hdr_path):
                os.remove(hdr_path)
            deleted += 1
        except Exception as e:
            print(f"[ERROR] 删除失败 {fname}: {e}")

    print(f"\n[done] 成功删除 {deleted} 对图片")
    print(f"[info] 剩余图片数量: {len(os.listdir(args.sdr))}")


if __name__ == "__main__":
    main()
