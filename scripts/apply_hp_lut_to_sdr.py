#!/usr/bin/env python3
# 用高精度拟合的 3D LUT 处理 sdr 目录并保存输出

import numpy as np
import cv2
from pathlib import Path


def apply_3dlut(image, lut):
    """三线性插值应用 3D LUT。"""
    size = lut.shape[0]
    img = np.clip(image.astype(np.float64), 0.0, 1.0) * (size - 1)
    base = np.floor(img).astype(np.int32)
    base = np.minimum(base, size - 2)
    frac = img - base

    out = np.zeros_like(image, dtype=np.float64)
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                wi = frac[:, :, 0] if di else (1.0 - frac[:, :, 0])
                wj = frac[:, :, 1] if dj else (1.0 - frac[:, :, 1])
                wk = frac[:, :, 2] if dk else (1.0 - frac[:, :, 2])
                w = (wi * wj * wk)[:, :, None]
                idx = (base[:, :, 0] + di, base[:, :, 1] + dj, base[:, :, 2] + dk)
                out += w * lut[idx]
    return np.clip(out, 0.0, 1.0)


lut = np.load('/mnt/data/wcm/20260711_color_enhance_plants2/fitted_lut_17_hp.npy')
src_dir = Path('/mnt/data/wcm/20260711_color_enhance_plants2/sdr')
out_dir = Path('/mnt/data/wcm/20260711_color_enhance_plants2/sdr_lut_output_hp')
out_dir.mkdir(parents=True, exist_ok=True)

count = 0
for img_path in sorted(src_dir.iterdir()):
    if img_path.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}:
        continue

    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img_bgr is None:
        continue

    # BGR -> RGB, float64
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float64)
    if np.issubdtype(img_bgr.dtype, np.integer):
        img_rgb /= float(np.iinfo(img_bgr.dtype).max)

    out_rgb = apply_3dlut(img_rgb, lut)
    out_bgr = (np.clip(out_rgb, 0.0, 1.0) * 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out_bgr, cv2.COLOR_RGB2BGR)

    out_path = out_dir / img_path.name
    cv2.imwrite(str(out_path), out_bgr)
    count += 1

print(f"完成，共处理 {count} 张图片")
print(f"输出目录：{out_dir}")
