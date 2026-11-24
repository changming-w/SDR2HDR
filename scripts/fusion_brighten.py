#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""融合 focus_map（绿植关注区）与 MiDaS 景深 mask 作为权重提亮 Y 通道。

融合语义:
    越是前景(近) 且 越是绿植(focus) 的区域，提亮越多。
    weight = focus_weight ^ fp * depth_weight ^ dp
    默认乘法融合(fp=dp=1)，等价于两个权重都要高才强提亮。

流程: focus_map + MiDaS 深度 → 融合权重 → Y × (1 + strength × weight × (1 - protect))

环境要求：torch, torchvision, timm
MiDaS 仓库缓存：~/.cache/torch/hub/isl-org_MiDaS_master

用法:
    # 单张图
    python fusion_brighten.py input.png output.png

    # 批量（目录）
    python fusion_brighten.py ./images ./output --strength 0.4

    # 深度缓存 + debug
    python fusion_brighten.py ./images ./output --depth-cache ./dc --debug-dir ./dbg
"""
import argparse
import sys
import os
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import CubicSpline

sys.path.insert(0, os.path.dirname(__file__))
from focus_map import vegetation_focus_map
from depth import load_dpt_hybrid, estimate_depth


def smoothstep(x):
    x = np.clip(x, 0., 1.)
    return x * x * (3. - 2. * x)


def dynamic_strength(y: np.ndarray, peak: float = 1.0) -> np.ndarray:
    """提亮强度随像素值变化的平滑曲线（三次样条插值，精确通过所有锚点）。

    锚点（相对于峰值的比例）:
    (0,0%) → (40,5%) → (50,10%) → (60,20%) → (80,70%) → (100,100%) → (128,95%) → (180,40%) → (200,20%) → (255,0%)

    Args:
        y: [H, W] Y 通道，范围 0-255
        peak: 峰值强度（1.0 = 100%）
    Returns:
        [H, W] 每像素的提亮强度
    """
    t = y.astype(np.float32) / 255.0

    # 锚点（归一化到 0-1，值为百分比小数，最终由 peak 缩放）
    xs = np.array([0.0, 40.0, 50.0, 60.0, 80.0, 100.0, 128.0, 180.0, 200.0, 255.0]) / 255.0
    ys = np.array([0, 0.05, 0.1, 0.20, 0.70, 1.00, 0.95, 0.40, 0.20, 0.00])

    # 三次样条插值（自然边界条件，二阶导数为0，起止平滑且有弧度）
    cs = CubicSpline(xs, ys, bc_type='natural')
    result = cs(t)

    return peak * np.clip(result, 0.0, None)


def fuse_weights(
    focus_w: np.ndarray,
    depth_w: np.ndarray,
    mode: str = 'mul',
    alpha: float = 0.5,
    fp: float = 1.0,
    dp: float = 1.0,
) -> np.ndarray:
    """融合 focus 权重与深度权重，均在 [0,1]。

    mode:
        'mul'  : weight = focus^fp * depth^dp  (两者都强才强)
        'add'  : weight = alpha*focus + (1-alpha)*depth
        'min'  : weight = min(focus, depth)
    """
    focus_w = np.clip(focus_w, 0., 1.)
    depth_w = np.clip(depth_w, 0., 1.)
    if mode == 'mul':
        return (np.power(focus_w, fp) * np.power(depth_w, dp)).astype(np.float32)
    if mode == 'add':
        return (alpha * focus_w + (1. - alpha) * depth_w).astype(np.float32)
    if mode == 'min':
        return np.minimum(focus_w, depth_w).astype(np.float32)
    raise ValueError(f'未知融合模式: {mode}')


def brighten_with_weight(
    image_bgr: np.ndarray,
    weight: np.ndarray,
    strength: float = 1.0,
    highlight_clip: float = 230.0,
) -> np.ndarray:
    """用融合权重提亮图像，提亮强度随像素值动态变化（三次样条曲线）。

    最终增益 = 动态强度(Y) × 融合权重 × (1 - 高光保护)
    - 动态强度：暗部/亮部弱，Y=100 处最强
    - 融合权重：绿植 × 前景
    - 高光保护：曲线已在 Y=255 归零，此处作近白区额外抑制（可选）

    weight: [0,1] 融合权重图
    strength: 峰值提亮比例（1.0 = 100%）
    highlight_clip: 高光保护阈值，设为 >=255 可关闭
    """
    ycc = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y = ycc[:, :, 0]

    # 动态强度：随 Y 值呈样条曲线（暗部/亮部弱，Y=100 处达到峰值）
    strength_map = dynamic_strength(y, strength)

    # 融合权重
    effective_gain = strength_map * weight

    # 额外高光保护（穹顶已归零，这里进一步抑制近白，避免溢出）
    if highlight_clip < 255:
        protect = smoothstep((y - highlight_clip) / max(255. - highlight_clip, 1e-6))
        effective_gain = effective_gain * (1.0 - protect)

    y_out = np.clip(y * (1.0 + effective_gain), 0, 255)

    ycc[:, :, 0] = y_out
    return cv2.cvtColor(ycc.astype(np.uint8), cv2.COLOR_YCrCb2BGR)


def main():
    ap = argparse.ArgumentParser(description='Focus + 景深 融合权重提亮')
    ap.add_argument('input', help='输入图片路径或目录')
    ap.add_argument('output', help='输出图片路径或目录')
    ap.add_argument('--strength', type=float, default=0.6,
                    help='峰值提亮比例(1.0=提亮100%%)')
    ap.add_argument('--highlight-clip', type=float, default=230.0,
                    help='高光保护阈值(Y>此值额外减弱提亮)，设为255关闭')

    # 融合方式
    ap.add_argument('--fuse-mode', choices=['mul', 'add', 'min'], default='mul',
                    help='权重融合方式: mul=乘法(默认,两者都强才提亮) / add=加权平均 / min=取最小')
    ap.add_argument('--alpha', type=float, default=0.6,
                    help='add模式下focus权重占比')
    ap.add_argument('--focus-power', type=float, default=1.0, dest='fp',
                    help='mul模式下focus权重的指数')
    ap.add_argument('--depth-power', type=float, default=1.0, dest='dp',
                    help='mul模式下深度权重的指数')
    ap.add_argument('--invert-depth', action='store_true',
                    help='反转深度权重(默认近处权重高，加此参数远处权重高)')

    # focus_map 参数
    ap.add_argument('--hue-min', type=int, default=25)
    ap.add_argument('--hue-max', type=int, default=95)
    ap.add_argument('--saturation-min', type=int, default=15)
    ap.add_argument('--value-min', type=int, default=15)
    ap.add_argument('--blur-sigma', type=float, default=5.0)
    ap.add_argument('--value-gamma', type=float, default=0.5)

    # MiDaS 参数
    ap.add_argument('--device', default='cpu', help='设备：cpu 或 cuda')
    ap.add_argument('--model', default='DPT_Hybrid',
                    choices=['DPT_Large', 'DPT_Hybrid', 'MiDaS_small'],
                    help='MiDaS 模型类型')
    ap.add_argument('--depth-cache', default=None,
                    help='深度图缓存目录(已有深度图则直接读取)')
    ap.add_argument('--debug-dir', default=None,
                    help='保存 focus/depth/融合权重/gain 图')
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

    # 加载 MiDaS 模型
    print(f'加载 {a.model} 模型...')
    model, transform, device = load_dpt_hybrid(a.device, a.model)
    print(f'  设备: {device}, 融合模式: {a.fuse_mode}')

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

            h, w = img.shape[:2]

            # 1. focus 权重
            focus_u8 = vegetation_focus_map(
                img, a.hue_min, a.hue_max,
                a.saturation_min, a.value_min,
                a.blur_sigma, a.value_gamma
            )
            if focus_u8.shape[:2] != (h, w):
                focus_u8 = cv2.resize(focus_u8, (w, h), interpolation=cv2.INTER_LINEAR)
            focus_w = focus_u8.astype(np.float32) / 255.0

            # 2. 深度权重
            cache_path = depth_cache / (f.stem + '_depth.png') if depth_cache else None
            if cache_path and cache_path.exists():
                depth_u8 = cv2.imread(str(cache_path), cv2.IMREAD_GRAYSCALE)
            else:
                _, depth_u8 = estimate_depth(model, transform, device, str(f))
                if cache_path:
                    cv2.imwrite(str(cache_path), depth_u8)
            if depth_u8.shape[:2] != (h, w):
                depth_u8 = cv2.resize(depth_u8, (w, h), interpolation=cv2.INTER_LINEAR)
            depth_w = depth_u8.astype(np.float32) / 255.0
            if a.invert_depth:
                depth_w = 1.0 - depth_w

            # 3. 融合
            weight = fuse_weights(focus_w, depth_w, a.fuse_mode, a.alpha, a.fp, a.dp)

            # 4. 提亮
            out = brighten_with_weight(img, weight, a.strength, a.highlight_clip)

            # 保存
            if output_dir:
                rel = f.relative_to(input_path)
                out_path = output_dir / rel
                out_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                out_path = Path(a.output)
                out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), out)

            # Debug
            if a.debug_dir:
                dbg = Path(a.debug_dir)
                dbg.mkdir(parents=True, exist_ok=True)
                name = f.stem
                cv2.imwrite(str(dbg / f'{name}_focus.png'), focus_u8)
                cv2.imwrite(str(dbg / f'{name}_depth.png'), depth_u8)
                cv2.imwrite(str(dbg / f'{name}_weight.png'),
                            np.rint(np.clip(weight, 0, 1) * 255).astype(np.uint8))

            if (i + 1) % 10 == 0 or len(files) <= 10:
                print(f'[{i+1}/{len(files)}] 已处理: {f.name}')

        except Exception as e:
            print(f'错误 {f.name}: {e}')

    print(f'完成！输出: {a.output}')


if __name__ == '__main__':
    main()
