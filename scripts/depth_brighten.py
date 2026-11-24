#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 MiDaS 景深 mask 作为权重，按比例提亮 Y 通道（带高光保护）。

流程: MiDaS 深度估计 → 归一化权重 → Y通道乘以 (1 + strength * weight * (1 - protect))

深度说明:
    MiDaS 输出的是视差，值越大 = 越近。
    默认提亮近处（weight = depth_norm），--invert 则提亮远处。

环境要求：torch, torchvision, timm
MiDaS 仓库缓存：~/.cache/torch/hub/isl-org_MiDaS_master

用法:
    # 单张图
    python depth_brighten.py input.png output.png

    # 批量（目录）
    python depth_brighten.py ./images ./output --strength 0.4

    # 提亮远处
    python depth_brighten.py ./images ./output --invert
"""
import argparse
import sys
import os
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from depth import load_dpt_hybrid, estimate_depth


def green_color_weight(image_bgr, hue_min=25, hue_max=95, sat_min=35, hue_soft=5):
    """计算黄绿色调权重图 [0,1]，边界 smoothstep 平滑过渡。

    hue_min/hue_max: OpenCV 色相范围(0-179)
    sat_min: 最小饱和度(低于此值不增强)
    hue_soft: 色相边界平滑宽度
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)

    def ss(x):
        x = np.clip(x, 0., 1.)
        return x * x * (3. - 2. * x)

    # 色相权重: [hue_min, hue_max] 内为 1，边界 hue_soft 宽度平滑过渡
    w_hue = np.ones_like(h)
    below = h < hue_min
    above = h > hue_max
    if hue_soft > 0:
        w_hue[below] = ss((h[below] - (hue_min - hue_soft)) / max(hue_soft, 1e-6))
        w_hue[above] = ss(((hue_max + hue_soft) - h[above]) / max(hue_soft, 1e-6))
    else:
        w_hue[below] = 0.
        w_hue[above] = 0.

    # 饱和度权重: 低于 sat_min 不增强，平滑上升
    w_sat = ss((s - sat_min) / max(60. - sat_min, 1.))

    return (w_hue * w_sat).astype(np.float32)


def brighten_with_depth(
    image_bgr: np.ndarray,
    depth_u8: np.ndarray,
    strength: float = 0.4,
    highlight_clip: float = 230.0,
    invert: bool = False,
    hue_min: int = 25,
    hue_max: int = 95,
    sat_min: float = 35,
    hue_soft: float = 5,
) -> np.ndarray:
    """用景深图作为权重提亮图像，只对黄绿色调生效，带高光保护。

    depth_u8: [0, 255] 灰度图，MiDaS 归一化视差（大值=近）
    strength: 最大提亮比例，0.4 表示 weight=1 的区域提亮 40%
    highlight_clip: 高光保护阈值，Y > highlight_clip 的区域减少提亮
    invert: True 时反转权重（提亮远处）
    hue_min/hue_max/sat_min/hue_soft: 黄绿色调门控参数
    """
    weight = depth_u8.astype(np.float32) / 255.0  # 归一化到 [0, 1]
    if invert:
        weight = 1.0 - weight

    # 黄绿色调门控：只对黄绿色像素提亮
    color_w = green_color_weight(image_bgr, hue_min, hue_max, sat_min, hue_soft)
    weight = weight * color_w

    # 转 YCrCb 提亮 Y 通道
    ycc = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y = ycc[:, :, 0]

    # 高光保护: Y > highlight_clip 时减弱提亮（smoothstep 平滑过渡）
    protect = np.clip((y - highlight_clip) / max(255 - highlight_clip, 1e-6), 0, 1)
    protect = protect * protect * (3 - 2 * protect)

    # 最终增益: strength 随高光减弱
    effective_gain = strength * weight * (1.0 - protect)

    # 提亮: y_out = y * (1 + effective_gain)
    gain = 1.0 + effective_gain
    y_out = np.clip(y * gain, 0, 255)

    ycc[:, :, 0] = y_out
    return cv2.cvtColor(ycc.astype(np.uint8), cv2.COLOR_YCrCb2BGR)


def main():
    ap = argparse.ArgumentParser(description='景深 mask 引导提亮')
    ap.add_argument('input', help='输入图片路径或目录')
    ap.add_argument('output', help='输出图片路径或目录')
    ap.add_argument('--strength', type=float, default=0.4,
                    help='最大提亮比例(0.4=提亮40%%)')
    ap.add_argument('--highlight-clip', type=float, default=230.0,
                    help='高光保护阈值(Y>此值减弱提亮)')
    ap.add_argument('--invert', action='store_true',
                    help='反转深度权重(默认提亮近处，加此参数提亮远处)')
    # 黄绿色调门控参数
    ap.add_argument('--hue-min', type=int, default=25,
                    help='黄绿色相下限(OpenCV 0-179)')
    ap.add_argument('--hue-max', type=int, default=95,
                    help='黄绿色相上限(OpenCV 0-179)')
    ap.add_argument('--sat-min', type=float, default=35,
                    help='最小饱和度(低于此值不增强)')
    ap.add_argument('--hue-soft', type=float, default=5,
                    help='色相边界平滑宽度')
    ap.add_argument('--device', default='cpu', help='设备：cpu 或 cuda')
    ap.add_argument('--model', default='DPT_Hybrid',
                    choices=['DPT_Large', 'DPT_Hybrid', 'MiDaS_small'],
                    help='MiDaS 模型类型')
    ap.add_argument('--depth-cache', default=None,
                    help='深度图缓存目录(已有深度图则直接读取，避免重复推理)')
    ap.add_argument('--debug-dir', default=None, help='保存深度图和gain图')
    a = ap.parse_args()

    input_path = Path(a.input)

    # 收集图片
    if input_path.is_dir():
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
        files = sorted(p for p in input_path.rglob('*')
                       if p.is_file() and p.suffix.lower() in exts)
        output_dir = Path(a.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        files = [input_path]
        output_dir = None

    if not files:
        raise SystemExit(f'没有找到图片: {a.input}')
    print(f'共 {len(files)} 张图片')

    # 加载 MiDaS 模型（一次加载，全部图片复用）
    print(f'加载 {a.model} 模型...')
    model, transform, device = load_dpt_hybrid(a.device, a.model)
    print(f'  设备: {device}')

    depth_cache = Path(a.depth_cache) if a.depth_cache else None
    if depth_cache:
        depth_cache.mkdir(parents=True, exist_ok=True)

    for i, f in enumerate(files):
        try:
            img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f'跳过（读取失败）: {f.name}')
                continue
            if img.dtype != np.uint8:
                print(f'跳过（非8-bit）: {f.name}')
                continue

            # 读取或计算深度图
            cache_path = depth_cache / (f.stem + '_depth.png') if depth_cache else None
            if cache_path and cache_path.exists():
                depth_u8 = cv2.imread(str(cache_path), cv2.IMREAD_GRAYSCALE)
            else:
                _, depth_u8 = estimate_depth(model, transform, device, str(f))
                if cache_path:
                    cv2.imwrite(str(cache_path), depth_u8)

            # 保证深度图与原图尺寸一致
            if depth_u8.shape[:2] != img.shape[:2]:
                depth_u8 = cv2.resize(depth_u8, (img.shape[1], img.shape[0]),
                                      interpolation=cv2.INTER_LINEAR)

            out = brighten_with_depth(img, depth_u8, a.strength,
                                      a.highlight_clip, a.invert,
                                      a.hue_min, a.hue_max, a.sat_min, a.hue_soft)

            # 保存
            if output_dir:
                rel = f.relative_to(input_path)
                out_path = output_dir / rel
                out_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                out_path = Path(a.output)
                out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), out)

            # Debug 输出
            if a.debug_dir:
                dbg = Path(a.debug_dir)
                dbg.mkdir(parents=True, exist_ok=True)
                name = f.stem
                cv2.imwrite(str(dbg / f'{name}_depth.png'), depth_u8)
                weight = depth_u8.astype(np.float32) / 255.0
                if a.invert:
                    weight = 1.0 - weight
                # 叠加黄绿色调门控后的实际权重
                weight = weight * green_color_weight(
                    img, a.hue_min, a.hue_max, a.sat_min, a.hue_soft)
                cv2.imwrite(str(dbg / f'{name}_gain.png'),
                            np.rint(weight * 255).astype(np.uint8))

            if (i + 1) % 10 == 0 or len(files) <= 10:
                print(f'[{i+1}/{len(files)}] 已处理: {f.name}')

        except Exception as e:
            print(f'错误 {f.name}: {e}')

    print(f'完成！输出: {a.output}')


if __name__ == '__main__':
    main()
