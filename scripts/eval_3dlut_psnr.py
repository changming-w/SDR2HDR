#!/usr/bin/env python3
import numpy as np
import cv2
from pathlib import Path


def apply_3dlut(image, lut):
    size = lut.shape[0]
    img = np.clip(image.astype(np.float64), 0.0, 1.0) * (size - 1)
    base = np.floor(img).astype(np.int32)
    base = np.minimum(base, size - 2)
    frac = img - base
    r0, g0, b0 = base[..., 0], base[..., 1], base[..., 2]
    fr, fg, fb = frac[..., 0], frac[..., 1], frac[..., 2]
    out = np.zeros_like(image, dtype=np.float64)
    for dr in (0, 1):
        for dg in (0, 1):
            for db in (0, 1):
                ri = r0 + dr if dr else r0
                gi = g0 + dg if dg else g0
                bi = b0 + db if db else b0
                w = ((fr if dr else 1 - fr) *
                     (fg if dg else 1 - fg) *
                     (fb if db else 1 - fb))
                out += w[..., None] * lut[ri, gi, bi]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def compute_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(1.0 / mse)


import os

lut = np.load('/mnt/data/wcm/20260711_color_enhance_plants2/fitted_lut_17.npy')
sdr_dir = Path('/mnt/data/wcm/20260711_color_enhance_plants2/sdr')
edr_dir = Path('/mnt/data/wcm/20260711_color_enhance_plants2/edr')
out_dir = Path('/mnt/data/wcm/20260711_color_enhance_plants2/sdr_lut_output')
out_dir.mkdir(exist_ok=True)

psnr_list = []
files = sorted(sdr_dir.glob('*.png'))
for i, f in enumerate(files):
    edr_path = edr_dir / f.name
    if not edr_path.exists():
        continue
    sdr = cv2.imread(str(f), cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32) / 255.0
    edr = cv2.imread(str(edr_path), cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32) / 255.0
    if sdr.shape != edr.shape:
        continue
    result = apply_3dlut(sdr, lut)

    # 保存 LUT 推理结果（BGR 格式，uint8 PNG）
    out_bgr = (np.clip(result, 0.0, 1.0) * 255.0).astype(np.uint8)[..., ::-1]
    cv2.imwrite(str(out_dir / f.name), out_bgr)

    psnr = compute_psnr(result, edr)
    psnr_list.append(psnr)
    if (i + 1) % 50 == 0:
        print(f'  已处理 {i+1}/{len(files)} 张，当前平均 PSNR: {np.mean(psnr_list):.4f} dB')

print(f'\n总图片数: {len(psnr_list)}')
print(f'平均 PSNR: {np.mean(psnr_list):.4f} dB')
print(f'中位 PSNR: {np.median(psnr_list):.4f} dB')
print(f'最高 PSNR: {np.max(psnr_list):.4f} dB')
print(f'最低 PSNR: {np.min(psnr_list):.4f} dB')
print(f'输出目录: {out_dir}')
