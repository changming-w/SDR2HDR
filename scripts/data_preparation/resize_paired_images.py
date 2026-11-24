#!/usr/bin/env python3
"""
将 4K SDR/HDR 配对图像 resize 后保存，加速训练数据加载。

用法:
  python scripts/data_preparation/resize_paired_images.py \
      --hdr_dir /mnt/data/wcm/data/hdr/train/hdrframes \
      --sdr_dir /mnt/data/wcm/data/hdr/train/sdrframes_jianying_daili_hevc_60fps \
      --out_hdr_dir /mnt/data/wcm/data/hdr/train/hdrframes_1080p \
      --out_sdr_dir /mnt/data/wcm/data/hdr/train/sdrframes_1080p \
      --short_side 1080
"""

import os
import cv2
import argparse
import numpy as np

IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.webp')


def scandir_files(folder):
    """简单文件扫描，不依赖 basicsr"""
    return [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]


def resize_and_save(src_path, dst_path, short_side, is_hdr=False):
    """读取图片，resize 使短边 = short_side，保存。"""
    if is_hdr:
        img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f'WARN: failed to read {src_path}')
            return False
        h, w = img.shape[:2]
        scale = short_side / min(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(dst_path, img_resized)
    else:
        img = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f'WARN: failed to read {src_path}')
            return False
        h, w = img.shape[:2]
        scale = short_side / min(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(dst_path, img_resized)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdr_dir', type=str, required=True, help='原始 HDR 图片目录')
    parser.add_argument('--sdr_dir', type=str, required=True, help='原始 SDR 图片目录')
    parser.add_argument('--out_hdr_dir', type=str, required=True, help='输出 HDR 目录')
    parser.add_argument('--out_sdr_dir', type=str, required=True, help='输出 SDR 目录')
    parser.add_argument('--short_side', type=int, default=1080, help='resize 后短边像素数')
    args = parser.parse_args()

    os.makedirs(args.out_hdr_dir, exist_ok=True)
    os.makedirs(args.out_sdr_dir, exist_ok=True)

    # 收集配对文件
    hdr_files = sorted([f for f in scandir_files(args.hdr_dir) if f.lower().endswith(IMG_EXTENSIONS)])
    sdr_files = sorted([f for f in scandir_files(args.sdr_dir) if f.lower().endswith(IMG_EXTENSIONS)])

    hdr_names = {os.path.splitext(f)[0]: f for f in hdr_files}
    sdr_names = {os.path.splitext(f)[0]: f for f in sdr_files}

    common_keys = sorted(set(hdr_names.keys()) & set(sdr_names.keys()))
    print(f'HDR: {len(hdr_files)}, SDR: {len(sdr_files)}, 配对: {len(common_keys)}')
    print(f'目标短边: {args.short_side}px, 输出: {args.out_hdr_dir}, {args.out_sdr_dir}')

    for i, key in enumerate(common_keys):
        if (i + 1) % 50 == 0 or i == 0:
            print(f'[{i+1}/{len(common_keys)}] processing...')
        # HDR (16bit)
        src_hdr = os.path.join(args.hdr_dir, hdr_names[key])
        dst_hdr = os.path.join(args.out_hdr_dir, hdr_names[key])
        resize_and_save(src_hdr, dst_hdr, args.short_side, is_hdr=True)

        # SDR (8bit)
        src_sdr = os.path.join(args.sdr_dir, sdr_names[key])
        dst_sdr = os.path.join(args.out_sdr_dir, sdr_names[key])
        resize_and_save(src_sdr, dst_sdr, args.short_side, is_hdr=False)

    print(f'Done! 处理了 {len(common_keys)} 对图片')

    # 验证
    sample_hdr = cv2.imread(os.path.join(args.out_hdr_dir, hdr_names[common_keys[0]]), cv2.IMREAD_UNCHANGED)
    sample_sdr = cv2.imread(os.path.join(args.out_sdr_dir, sdr_names[common_keys[0]]), cv2.IMREAD_UNCHANGED)
    print(f'HDR sample: shape={sample_hdr.shape}, dtype={sample_hdr.dtype}, range=[{sample_hdr.min()}, {sample_hdr.max()}]')
    print(f'SDR sample: shape={sample_sdr.shape}, dtype={sample_sdr.dtype}, range=[{sample_sdr.min()}, {sample_sdr.max()}]')


if __name__ == '__main__':
    main()
