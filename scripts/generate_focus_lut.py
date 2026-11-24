#!/usr/bin/env python3
"""将 focus_brighten.py 的逐像素部分导出为 3D LUT (.cube 格式)。

原理：
- focus_brighten.py 完整流程 = 空间操作 + 逐像素操作
  * 空间操作（3D LUT 无法实现，本脚本去掉）：
      - hue_mask 形态学开/闭运算（5x5 核）
      - focus map 高斯模糊（sigma=5.0）
      - 模糊后再次乘以 hue_mask
  * 逐像素操作（3D LUT 可完美查表实现）：
      - BGR->HSV、hue 范围掩码、sat/val 评分与 gamma
      - focus = val^gamma * sqrt(sat_score) * hue_mask
      - YCrCb 高光保护 smoothstep
      - y_out = y * (1 + strength * weight * (1 - protect))
- 本脚本在 LUT 网格点上直接调用 OpenCV 完整执行 focus_brighten.py 的
  逐像素链路（BGR->HSV / BGR->YCrCb / YCrCb->BGR 均与 OpenCV 一致），
  因此除被去掉的空间操作外无近似误差；剩余误差仅来自 LUT 插值量化，
  增大 --size 可减小。

用法：
  python3 scripts/generate_focus_lut.py focus_brighten.cube --size 65
  python3 scripts/generate_focus_lut.py focus_brighten.cube --verify some.png
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from focus_brighten import brighten_with_focus


def pixelwise_focus(image_bgr: np.ndarray,
                    hue_min: int, hue_max: int,
                    saturation_min: int, value_min: int,
                    value_gamma: float) -> np.ndarray:
    """与 vegetation_focus_map 相同的逐像素逻辑，去掉空间操作。

    去掉：形态学开/闭运算、高斯模糊、模糊后的二次 hue_mask。
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue, saturation, value = cv2.split(hsv)

    hue_mask = ((hue >= hue_min) & (hue <= hue_max)).astype(np.float32)

    sat_score = np.clip(
        (saturation - saturation_min) / max(255 - saturation_min, 1),
        0.0, 1.0,
    )
    val_score = np.clip(
        (value - value_min) / max(255 - value_min, 1), 0.0, 1.0
    )
    val_score = np.power(val_score, value_gamma)

    focus = val_score * np.sqrt(sat_score) * hue_mask
    return np.clip(focus * 255.0, 0, 255).astype(np.uint8)


def generate_focus_lut(size: int,
                       strength: float, highlight_clip: float,
                       hue_min: int, hue_max: int,
                       saturation_min: int, value_min: int,
                       value_gamma: float) -> np.ndarray:
    """在 LUT 网格点上执行完整逐像素链路，返回 (size, size, size, 3) [0,1]。"""
    levels = np.linspace(0, 255, size, dtype=np.uint8)
    b, g, r = np.meshgrid(levels, levels, levels, indexing='ij')
    grid = np.stack([b, g, r], axis=-1).reshape(-1, 1, 3)

    # 与 focus_brighten.py 完全相同的逐像素链路
    focus = pixelwise_focus(grid, hue_min, hue_max,
                            saturation_min, value_min, value_gamma)
    out = brighten_with_focus(grid, focus, strength, highlight_clip)

    lut = out.reshape(size, size, size, 3).astype(np.float32) / 255.0
    return lut


def write_cube(lut: np.ndarray, filepath: str, title: str = 'FocusBrighten'):
    """写入 .cube 格式的 3D LUT。

    约定：R 变化最快，然后 G，然后 B（即行序为 B 最慢）。
    非 Adobe 默认约定，配套读取请用 apply_focus_lut.py 的 load_cube。
    """
    size = lut.shape[0]
    with open(filepath, 'w') as f:
        f.write(f'TITLE "{title}"\n')
        f.write('# channel order: R fastest, G, B slowest\n')
        f.write(f'LUT_3D_SIZE {size}\n')
        f.write('DOMAIN_MIN 0.0 0.0 0.0\n')
        f.write('DOMAIN_MAX 1.0 1.0 1.0\n')
        f.write('\n')
        for b_idx in range(size):
            for g_idx in range(size):
                for r_idx in range(size):
                    r, g, b = lut[r_idx, g_idx, b_idx]
                    f.write(f'{r:.6f} {g:.6f} {b:.6f}\n')


def apply_lut(image_bgr: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """三线性插值应用 LUT，lut 索引顺序为 [b, g, r]。"""
    size = lut.shape[0]
    img = image_bgr.astype(np.float32) / 255.0 * (size - 1)
    b, g, r = cv2.split(img)

    r0 = np.clip(np.floor(r).astype(np.int32), 0, size - 2)
    g0 = np.clip(np.floor(g).astype(np.int32), 0, size - 2)
    b0 = np.clip(np.floor(b).astype(np.int32), 0, size - 2)
    fr, fg, fb = r - r0, g - g0, b - b0
    fr = fr[..., None].astype(np.float32)
    fg = fg[..., None].astype(np.float32)
    fb = fb[..., None].astype(np.float32)

    def lerp(v0, v1, t):
        return v0 * (1 - t) + v1 * t

    r1, g1, b1 = r0 + 1, g0 + 1, b0 + 1
    c00 = lerp(lut[b0, g0, r0], lut[b0, g0, r1], fr)
    c01 = lerp(lut[b0, g1, r0], lut[b0, g1, r1], fr)
    c10 = lerp(lut[b1, g0, r0], lut[b1, g0, r1], fr)
    c11 = lerp(lut[b1, g1, r0], lut[b1, g1, r1], fr)
    c0 = c00 * (1 - fg) + c01 * fg
    c1 = c10 * (1 - fg) + c11 * fg
    out = c0 * (1 - fb) + c1 * fb
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def verify(image_path: str, lut: np.ndarray,
           args: argparse.Namespace) -> None:
    """对比 LUT 输出与逐像素参考输出（无空间操作版）的 PSNR。"""
    data = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        print(f'无法读取图片: {image_path}')
        return

    focus = pixelwise_focus(img, args.hue_min, args.hue_max,
                            args.saturation_min, args.value_min,
                            args.value_gamma)
    ref = brighten_with_focus(img, focus, args.strength, args.highlight_clip)
    out = apply_lut(img, lut)

    diff = out.astype(np.float32) - ref.astype(np.float32)
    mse = np.mean(diff * diff)
    psnr = 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float('inf')
    print(f'验证图片: {image_path}')
    print(f'LUT vs 逐像素参考: PSNR = {psnr:.2f} dB, 最大像素差 = {np.abs(diff).max():.0f}')


def main():
    ap = argparse.ArgumentParser(
        description='将 focus_brighten.py 逐像素逻辑导出为 3D LUT (.cube)')
    ap.add_argument('output', help='输出 .cube 文件路径')
    ap.add_argument('--size', type=int, default=65,
                    help='LUT 尺寸（默认 65，越大插值误差越小）')
    ap.add_argument('--strength', type=float, default=0.4,
                    help='最大提亮比例，与 focus_brighten.py 默认一致')
    ap.add_argument('--highlight-clip', type=float, default=230.0,
                    help='高光保护阈值，与 focus_brighten.py 默认一致')
    ap.add_argument('--hue-min', type=int, default=25)
    ap.add_argument('--hue-max', type=int, default=95)
    ap.add_argument('--saturation-min', type=int, default=35)
    ap.add_argument('--value-min', type=int, default=35)
    ap.add_argument('--value-gamma', type=float, default=0.5,
                    help='与 focus_brighten.py 默认一致')
    ap.add_argument('--verify', default=None, help='用一张图片验证 LUT 精度')
    a = ap.parse_args()

    lut = generate_focus_lut(
        a.size, a.strength, a.highlight_clip,
        a.hue_min, a.hue_max, a.saturation_min, a.value_min, a.value_gamma)

    title = f'FocusBrighten_s{a.strength}_hc{a.highlight_clip}'
    write_cube(lut, a.output, title=title)

    print(f'已生成 3D LUT: {a.output}')
    print(f'尺寸: {a.size}x{a.size}x{a.size}')
    print(f'参数: strength={a.strength}, highlight_clip={a.highlight_clip}, '
          f'hue=[{a.hue_min},{a.hue_max}], sat_min={a.saturation_min}, '
          f'value_min={a.value_min}, value_gamma={a.value_gamma}')
    print('已去掉的空间操作: 形态学开/闭、高斯模糊(sigma=5.0)、模糊后二次掩码')

    if a.verify:
        verify(a.verify, lut, a)


if __name__ == '__main__':
    main()
