#!/usr/bin/env python3
"""用 focus_map 识别绿植区域，按权重比例提亮 Y 通道。

流程: focus_map → 归一化权重 → Y通道乘以 (1 + strength * weight)
"""
import argparse, sys, glob, os
from pathlib import Path
import cv2
import numpy as np
from scipy.interpolate import CubicSpline

sys.path.insert(0, os.path.dirname(__file__))
from focus_map import vegetation_focus_map, read_image, write_image


def smoothstep(x):
    x = np.clip(x, 0., 1.)
    return x * x * (3. - 2. * x)


def dynamic_strength(y: np.ndarray, peak: float = 1.0) -> np.ndarray:
    """提亮强度随像素值变化的平滑曲线（三次样条插值，精确通过所有锚点）。

    锚点（相对于峰值的比例）:
    (0,0%) → (40,10%) → (60,70%) → (80,90%) → (100,100%) → (128,90%) → (180,50%) → (200,20%) → (255,0%)

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


def brighten_with_focus(
    image_bgr: np.ndarray,
    focus: np.ndarray,
    strength: float = 1.0,
    highlight_clip: float = 230.0,
) -> np.ndarray:
    """用 focus map 作为权重提亮图像，提亮强度随像素值动态变化（6 段平滑曲线）。

    focus: [0, 255] 灰度图，0=非关注区域，255=最强关注
    strength: 峰值提亮比例（0.5 = 50%）
    highlight_clip: 额外高光保护阈值（辅助安全阀）
    """
    weight = focus.astype(np.float32) / 255.0  # 归一化到 [0, 1]

    # 转 YCrCb 提亮 Y 通道
    ycc = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y = ycc[:, :, 0]

    # 动态强度：6 段平滑曲线，精确控制关键点
    strength_map = dynamic_strength(y, strength)

    # 融合权重
    effective_gain = strength_map * weight

    # 额外高光保护（high_cut 已限制，这里进一步抑制近白避免溢出）
    if highlight_clip < 255:
        protect = smoothstep((y - highlight_clip) / max(255. - highlight_clip, 1e-6))
        effective_gain = effective_gain * (1.0 - protect)

    y_out = np.clip(y * (1.0 + effective_gain), 0, 255)

    ycc[:, :, 0] = y_out
    return cv2.cvtColor(ycc.astype(np.uint8), cv2.COLOR_YCrCb2BGR)


def main():
    ap = argparse.ArgumentParser(description='Focus map 引导提亮')
    ap.add_argument('input', help='输入图片路径或目录')
    ap.add_argument('output', help='输出图片路径或目录')
    ap.add_argument('--strength', type=float, default=0.2,
                    help='峰值提亮比例(1.0=提亮100%%)')
    ap.add_argument('--highlight-clip', type=float, default=230.0,
                    help='高光保护阈值(Y>此值额外减弱提亮)')
    ap.add_argument('--hue-min', type=int, default=25)
    ap.add_argument('--hue-max', type=int, default=95)
    ap.add_argument('--saturation-min', type=int, default=15)
    ap.add_argument('--value-min', type=int, default=15)
    ap.add_argument('--blur-sigma', type=float, default=5.0)
    ap.add_argument('--value-gamma', type=float, default=0.5)
    ap.add_argument('--debug-dir', default=None, help='保存focus map和gain图')
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

    print(f'共 {len(files)} 张图片')

    for i, f in enumerate(files):
        try:
            img = read_image(f)
            focus = vegetation_focus_map(
                img, a.hue_min, a.hue_max,
                a.saturation_min, a.value_min,
                a.blur_sigma, a.value_gamma
            )
            out = brighten_with_focus(img, focus, a.strength, a.highlight_clip)

            # 保存
            if output_dir:
                rel = f.relative_to(input_path)
                out_path = output_dir / rel
                write_image(out_path, out)
            else:
                out_path = Path(a.output)
                write_image(out_path, out)

            # Debug 输出
            if a.debug_dir:
                dbg = Path(a.debug_dir)
                dbg.mkdir(parents=True, exist_ok=True)
                name = f.stem
                cv2.imwrite(str(dbg / f'{name}_focus.png'), focus)
                gain_map = np.clip(
                    (1.0 + a.strength * focus.astype(np.float32) / 255.0 - 1.0)
                    / max(a.strength, 1e-6) * 255, 0, 255
                ).astype(np.uint8)
                cv2.imwrite(str(dbg / f'{name}_gain.png'), gain_map)

            if (i + 1) % 20 == 0 or len(files) <= 20:
                print(f'[{i+1}/{len(files)}] 已处理: {f.name}')

        except Exception as e:
            print(f'错误 {f.name}: {e}')

    print(f'完成！输出: {a.output}')


if __name__ == '__main__':
    main()
