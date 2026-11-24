#!/usr/bin/env python3
"""用全局拟合的 3D LUT 处理阳光绿原图目录。"""

import numpy as np
import cv2
from pathlib import Path
from scipy.ndimage import map_coordinates


def apply_3dlut(image, lut):
    """用 scipy 加速的三线性插值"""
    size = lut.shape[0]
    # image: (H, W, 3) RGB float [0, 1]
    H, W = image.shape[:2]

    # 计算 LUT 坐标
    pos = np.clip(image, 0.0, 1.0) * (size - 1)  # (H, W, 3)

    # map_coordinates 需要 (3, N) 格式的坐标
    coords = pos.reshape(-1, 3).T  # (3, H*W)

    # 对 R, G, B 三个输出通道分别插值
    out = np.zeros((H * W, 3), dtype=np.float32)
    for c in range(3):
        # lut[..., c] shape: (S, S, S)
        out[:, c] = map_coordinates(lut[..., c], coords, order=1, mode='nearest')

    return out.reshape(H, W, 3)


IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}

# 选择 LUT 来源：'csv' = 17阶融合CSV, 'npy' = 33阶拟合npy
LUT_SOURCE = 'npy'

if LUT_SOURCE == 'csv':
    # 读取 CSV 格式的 17 阶 3D LUT
    # CSV 是 .cube 排列：B最外层，G中间，R最内层
    raw = np.loadtxt('/mnt/data/wcm/20260711_color_enhance_plants2/sony_rgb_Q10M_0806_2_LINEAR.csv', delimiter=',', dtype=np.float64)
    lut_bgr = raw.reshape(17, 17, 17, 3) / 255.0  # shape: (B, G, R, 3)
    # 转置为 (R, G, B, 3) 以匹配 apply_3dlut 的索引
    lut = lut_bgr.transpose(2, 1, 0, 3)
    print(f"使用 CSV LUT: 17阶融合CSV")
elif LUT_SOURCE == 'npy':
    # 读取 33 阶 npy LUT（已经是 (R, G, B, 3) 顺序）
    lut = np.load('/mnt/data/wcm/20260711_color_enhance_plants2/fitted_lut_33_hue.npy')
    if lut.dtype != np.float64:
        lut = lut.astype(np.float64)
    print(f"使用 NPY LUT: 33阶拟合npy")
else:
    raise ValueError(f"未知 LUT_SOURCE: {LUT_SOURCE}")

print(f"LUT shape: {lut.shape}, range: [{lut.min():.6f}, {lut.max():.6f}]")
# 转成 float32 加速计算
lut = lut.astype(np.float32)
src_dir = Path('/mnt/data/wcm/20260711_color_enhance_plants2/阳光绿原图')
out_dir = Path('/mnt/data/wcm/data/gainmap/阳光绿原图_3dlut')
out_dir.mkdir(exist_ok=True)

files = sorted([f for f in src_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS])
print(f"共 {len(files)} 张图片")

for i, f in enumerate(files):
    img = cv2.imread(str(f), cv2.IMREAD_COLOR)
    if img is None:
        print(f"  跳过（无法读取）: {f.name}")
        continue
    img_rgb = img[..., ::-1].astype(np.float32) / 255.0
    result = apply_3dlut(img_rgb, lut)
    out_bgr = (np.clip(result, 0.0, 1.0) * 255.0).astype(np.uint8)[..., ::-1]
    cv2.imwrite(str(out_dir / f.name), out_bgr)
    if (i + 1) % 50 == 0:
        print(f"  已处理 {i+1}/{len(files)} 张")

print(f"完成！输出目录: {out_dir}")
