#!/usr/bin/env python3
"""用单张图片（105.png）拟合 3D LUT，然后应用到所有 sdr 图片上。"""

import numpy as np
import cv2
from pathlib import Path
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import spsolve


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


def fit_lut_single_image(sdr, edr, lut_size=17):
    """用单张图片的所有像素拟合 3D LUT。"""
    x = sdr.reshape(-1, 3).astype(np.float64)
    y = edr.reshape(-1, 3).astype(np.float64)

    # 过滤无效像素
    valid = (np.isfinite(x).all(axis=1)) & (np.isfinite(y).all(axis=1))
    x, y = x[valid], y[valid]
    print(f"有效像素数: {x.shape[0]}")

    # 构建三线性插值矩阵
    pos = np.clip(x, 0, 1) * (lut_size - 1)
    base = np.minimum(np.floor(pos).astype(np.int32), lut_size - 2)
    frac = pos - base

    corners = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],
                        [1,0,0],[1,0,1],[1,1,0],[1,1,1]])
    nodes = base[:, None, :] + corners[None, :, :]
    cols = (nodes[:, :, 0] * lut_size + nodes[:, :, 1]) * lut_size + nodes[:, :, 2]
    w_per = np.where(corners[None, :, :] == 1, frac[:, None, :], 1 - frac[:, None, :])
    weights = np.prod(w_per, axis=2)

    n = x.shape[0]
    rows = np.repeat(np.arange(n), 8)
    A = csr_matrix((weights.reshape(-1), (rows, cols.reshape(-1))),
                   shape=(n, lut_size ** 3))

    # 求解
    ATA = (A.T @ A).tocsc()
    ATy = A.T @ y

    # 平滑正则 + 恒等先验
    ATA = ATA + 0.001 * eye(lut_size ** 3)
    prior = np.linspace(0, 1, lut_size)
    gr, gg, gb = np.meshgrid(prior, prior, prior, indexing='ij')
    identity_lut = np.stack([gr, gg, gb], axis=-1).reshape(-1, 3)
    ATA = ATA + 0.0001 * eye(lut_size ** 3)
    ATy = ATy + 0.0001 * identity_lut

    lut_flat = np.zeros((lut_size ** 3, 3))
    for c in range(3):
        lut_flat[:, c] = spsolve(ATA, ATy[:, c])
    lut_flat = np.clip(lut_flat, 0, 1)

    return lut_flat.reshape(lut_size, lut_size, lut_size, 3)


# === 主流程 ===
base_dir = Path('/mnt/data/wcm/20260711_color_enhance_plants2')
sdr_dir = base_dir / 'sdr'
edr_dir = base_dir / 'edr'
out_dir = base_dir / 'sdr_lut_output_105'
out_dir.mkdir(exist_ok=True)

# 1. 读取 105.png 拟合 LUT
print("读取 105.png 拟合 LUT...")
sdr_img = cv2.imread(str(sdr_dir / '105.png'), cv2.IMREAD_COLOR)[..., ::-1].astype(np.float64) / 255.0
edr_img = cv2.imread(str(edr_dir / '105.png'), cv2.IMREAD_COLOR)[..., ::-1].astype(np.float64) / 255.0

lut = fit_lut_single_image(sdr_img, edr_img, lut_size=17)

# 评估 105.png 本身的拟合精度
result_105 = apply_3dlut(sdr_img, lut)
psnr_105 = compute_psnr(result_105, edr_img)
print(f"105.png 拟合 PSNR: {psnr_105:.2f} dB")

# 2. 保存 LUT
lut_path = base_dir / 'fitted_lut_17_from_105.npy'
cube_path = base_dir / 'fitted_lut_17_from_105.cube'
np.save(str(lut_path), lut.astype(np.float32))

# 写 .cube 文件
lut_size = lut.shape[0]
with open(cube_path, 'w') as f:
    f.write('TITLE "Fitted from 105.png"\n')
    f.write(f'LUT_3D_SIZE {lut_size}\n')
    f.write('DOMAIN_MIN 0.0 0.0 0.0\n')
    f.write('DOMAIN_MAX 1.0 1.0 1.0\n')
    for b in range(lut_size):
        for g in range(lut_size):
            for r in range(lut_size):
                v = lut[r, g, b]
                f.write(f'{v[0]:.10f} {v[1]:.10f} {v[2]:.10f}\n')

print(f"LUT 已保存: {lut_path}, {cube_path}")

# 3. 应用到所有 sdr 图片
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

    # 保存输出
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
