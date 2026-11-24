#!/usr/bin/env python3
"""把目录内图片统一成目标尺寸（默认 3840x2160），全部输出为无损 PNG。

规则：
  1. 原始宽高都 >= 目标：居中裁切成目标尺寸；
  2. 任一维度 < 目标：直接 resize 到目标尺寸（可能改变宽高比）。

注意：源文件是 JPG 时，其解码结果本身已含压缩损失，这是源的问题；
本脚本输出端一律 PNG 无损写入，不再引入任何额外损失。

用法：
  python crop_resize_to_target.py --input <输入目录> [--out <输出目录>]
                                  [--width 3840] [--height 2160]
"""

import argparse
import os

import cv2

EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def resize_to(img, tw: int, th: int):
    """缩放到目标尺寸：缩小用 INTER_AREA（抗混叠最佳），放大用 Lanczos。"""
    h, w = img.shape[:2]
    interp = cv2.INTER_AREA if (w >= tw and h >= th) else cv2.INTER_LANCZOS4
    return cv2.resize(img, (tw, th), interpolation=interp)


def center_crop_16x9(img):
    """整图居中裁出最大 16:9 矩形（纯数组切片，不改像素值）。"""
    h, w = img.shape[:2]
    if w * 9 > h * 16:  # 太宽：裁宽，高保留
        nw, nh = h * 16 // 9, h
        x0, y0 = (w - nw) // 2, 0
    else:               # 太高（或刚好）：裁高，宽保留
        nw, nh = w, w * 9 // 16
        x0, y0 = 0, (h - nh) // 2
    return img[y0:y0 + nh, x0:x0 + nw]


def process_one(src: str, dst: str, tw: int, th: int, mode: str = 'auto') -> str:
    img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f'读取失败: {src}')
    h, w = img.shape[:2]

    if mode == 'crop16x9':
        # 先整图居中裁成 16:9，再 resize 到目标（宽高比已匹配，不会拉伸变形）
        img = center_crop_16x9(img)
        op = 'crop16x9'
    elif mode == 'auto' and w >= tw and h >= th:
        # 居中裁切：纯数组切片，不改变任何像素值
        y0 = (h - th) // 2
        x0 = (w - tw) // 2
        img = img[y0:y0 + th, x0:x0 + tw]
        op = 'crop'
    else:
        # 任一维度不足：直接拉伸到目标（可能改变宽高比）
        op = 'resize'

    if img.shape[:2] != (th, tw):
        img = resize_to(img, tw, th)

    ok = cv2.imwrite(dst, img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise RuntimeError(f'写入失败: {dst}')
    return f'{op}  {w}x{h} -> {tw}x{th}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--out', default=None, help='默认 <input>_3840x2160')
    ap.add_argument('--width', type=int, default=3840)
    ap.add_argument('--height', type=int, default=2160)
    ap.add_argument('--mode', choices=['auto', 'resize', 'crop16x9'],
                    default='auto',
                    help='auto=大图中居裁切/小图 resize；resize=全部直接 resize；'
                         'crop16x9=先居中裁成 16:9 再 resize 到目标')
    a = ap.parse_args()

    out = a.out or f'{a.input.rstrip("/")}_{a.width}x{a.height}'
    os.makedirs(out, exist_ok=True)

    files = sorted(f for f in os.listdir(a.input)
                   if os.path.splitext(f)[1].lower() in EXTS)
    print(f'输入: {a.input}  共 {len(files)} 张图片')
    print(f'输出: {out}  目标尺寸 {a.width}x{a.height} (PNG 无损)')

    n_crop = n_resize = n_fail = 0
    for i, name in enumerate(files, 1):
        src = os.path.join(a.input, name)
        dst = os.path.join(out, os.path.splitext(name)[0] + '.png')
        try:
            info = process_one(src, dst, a.width, a.height, a.mode)
        except Exception as e:
            n_fail += 1
            print(f'[{i}/{len(files)}] [FAIL] {name}: {e}')
            continue
        if info.startswith('crop'):
            n_crop += 1
        else:
            n_resize += 1
        print(f'[{i}/{len(files)}] {name}  {info}')

    print(f'完成: 居中裁切 {n_crop} 张, resize {n_resize} 张, 失败 {n_fail} 张')


if __name__ == '__main__':
    main()
