#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按亮度分离人脸/主体（亮）与背景（暗），只把背景 RGB 按比例提亮。
支持 16bit PNG，全程保持位深与精度。

用法：
  # 1) 先用预览模式调参（只处理一张，输出 8bit 色调映射预览，方便肉眼看）
  python scripts/brighten_background.py --preview

  # 2) 参数满意后，批量处理整个目录
  python scripts/brighten_background.py --run

调参说明（改下面 CONFIG 或用命令行覆盖）：
  T_HIGH : 亮度 >= T_HIGH 判为人脸/主体，完全不动（保护）
  T_LOW  : 亮度 <= T_LOW  判为背景，施加满额提亮
           两者之间平滑过渡，避免人脸边缘出现硬光晕
  BOOST  : 背景提亮比例，0.20 = +20%（对背景 RGB 乘以 1.20）
  SIGMA  : 掩码羽化强度，越大过渡越柔和
"""

import os
import argparse
import numpy as np
import cv2

# ======================= 可调参数 =======================
CONFIG = dict(
    src_dir='/mnt/data/wcm/data/hdr/train/dark_scene_1200_1080p/hdr',
    dst_dir='/mnt/data/wcm/data/hdr/train/dark_scene_1200_1080p/hdr_brightened_0.2',
    T_LOW=2000.0,      # 背景阈值（<= 满额提亮）
    T_HIGH=5000.0,    # 人脸阈值（>= 完全保护）
    BOOST=0.20,       # 提亮比例 +20%
    SIGMA=3.0,        # 掩码羽化
    RESIZE_TO=None,   # 缩放到此分辨率，格式 (W, H)，如 (3840, 2160)；None=不缩放
    NUM_WORKERS=8,
)
# =======================================================


def compute_factor(img, T_LOW, T_HIGH, BOOST, SIGMA):
    """返回逐像素提亮系数 factor (H, W)，背景=1+BOOST，人脸=1.0，中间平滑。"""
    b = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    r = img[:, :, 2].astype(np.float32)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b          # BT.709 亮度

    alpha = np.clip((T_HIGH - lum) / (T_HIGH - T_LOW), 0, 1)  # 1=暗(背景) 0=亮(人脸)
    if SIGMA > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=SIGMA)  # 羽化，避免硬边光晕
    return 1.0 + BOOST * alpha


def brighten(img, cfg):
    """对单张 16bit 图施加背景提亮，返回同 dtype 结果。"""
    factor = compute_factor(img, cfg['T_LOW'], cfg['T_HIGH'], cfg['BOOST'], cfg['SIGMA'])
    maxv = 65535 if img.dtype == np.uint16 else 255
    out = np.clip(img.astype(np.float32) * factor[..., None], 0, maxv).astype(img.dtype)
    return out


def _tonemap(x):
    """16bit -> 8bit 色调映射预览（gamma + 99.5 分位归一化），仅用于肉眼查看。"""
    xf = x.astype(np.float32) / (65535.0 if x.dtype == np.uint16 else 255.0)
    n = np.percentile(xf, 99.5)
    xf = np.clip(xf / max(n, 1e-6), 0, 1)
    return (np.power(xf, 1 / 2.2) * 255).astype(np.uint8)


def preview(cfg, out_dir='/tmp'):
    """只处理目录里第一张图，输出原图/结果/掩码预览，方便调参。"""
    files = sorted(f for f in os.listdir(cfg['src_dir']) if f.lower().endswith('.png'))
    if not files:
        print('没有找到 PNG 图片')
        return
    fname = files[0]
    img = cv2.imread(os.path.join(cfg['src_dir'], fname), cv2.IMREAD_UNCHANGED)
    out = brighten(img, cfg)
    factor = compute_factor(img, cfg['T_LOW'], cfg['T_HIGH'], cfg['BOOST'], cfg['SIGMA'])

    cv2.imwrite(f'{out_dir}/preview_orig.png', _tonemap(img))
    cv2.imwrite(f'{out_dir}/preview_result.png', _tonemap(out))
    # 掩码：亮=背景(被提亮) 暗=人脸(被保护)
    mask_vis = ((factor - 1.0) / max(cfg['BOOST'], 1e-6) * 255).astype(np.uint8)
    cv2.imwrite(f'{out_dir}/preview_mask.png', mask_vis)

    boosted = factor > (1.0 + cfg['BOOST'] * 0.5)
    face = factor < (1.0 + cfg['BOOST'] * 0.05)
    print(f'预览图片: {fname}')
    print(f"  背景占比 {100*boosted.mean():.1f}%  均值 {img[boosted].mean():.0f} -> {out[boosted].mean():.0f}")
    print(f"  人脸占比 {100*face.mean():.1f}%  均值 {img[face].mean():.0f} -> {out[face].mean():.0f}")
    print(f'  预览已写入 {out_dir}/preview_orig.png / preview_result.png / preview_mask.png')


def run(cfg):
    """批量处理整个目录。"""
    from multiprocessing import Pool
    os.makedirs(cfg['dst_dir'], exist_ok=True)
    files = [f for f in os.listdir(cfg['src_dir']) if f.lower().endswith('.png')]

    def _worker(fname):
        img = cv2.imread(os.path.join(cfg['src_dir'], fname), cv2.IMREAD_UNCHANGED)
        if img is None:
            return f'读取失败 {fname}'
        out = brighten(img, cfg)
        cv2.imwrite(os.path.join(cfg['dst_dir'], fname), out)
        return None

    # multiprocessing 无法直接 pickle 闭包，这里用模块级函数配 starmap
    with Pool(cfg['NUM_WORKERS']) as p:
        res = p.starmap(_run_one, [(f, cfg) for f in files])
    errs = [x for x in res if x]
    print(f"✓ 完成 {len(files)-len(errs)}/{len(files)} 张 -> {cfg['dst_dir']}")
    for e in errs:
        print(' ', e)


def _run_one(fname, cfg):
    img = cv2.imread(os.path.join(cfg['src_dir'], fname), cv2.IMREAD_UNCHANGED)
    if img is None:
        return f'读取失败 {fname}'
    out = brighten(img, cfg)
    cv2.imwrite(os.path.join(cfg['dst_dir'], fname), out)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preview', action='store_true', help='只处理第一张并输出预览，用于调参')
    ap.add_argument('--run', action='store_true', help='批量处理整个目录')
    ap.add_argument('--src', help='输入目录，覆盖 CONFIG')
    ap.add_argument('--dst', help='输出目录，覆盖 CONFIG')
    ap.add_argument('--t-low', type=float, help='背景阈值 T_LOW')
    ap.add_argument('--t-high', type=float, help='人脸阈值 T_HIGH')
    ap.add_argument('--boost', type=float, help='提亮比例，如 0.2')
    ap.add_argument('--sigma', type=float, help='掩码羽化强度')
    args = ap.parse_args()

    cfg = dict(CONFIG)
    if args.src: cfg['src_dir'] = args.src
    if args.dst: cfg['dst_dir'] = args.dst
    if args.t_low is not None: cfg['T_LOW'] = args.t_low
    if args.t_high is not None: cfg['T_HIGH'] = args.t_high
    if args.boost is not None: cfg['BOOST'] = args.boost
    if args.sigma is not None: cfg['SIGMA'] = args.sigma

    print(f"参数: T_LOW={cfg['T_LOW']} T_HIGH={cfg['T_HIGH']} BOOST={cfg['BOOST']} SIGMA={cfg['SIGMA']}")

    if args.preview:
        preview(cfg)
    elif args.run:
        run(cfg)
    else:
        print('请指定 --preview（调参预览）或 --run（批量处理）')


if __name__ == '__main__':
    main()
