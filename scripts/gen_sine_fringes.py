#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成不同空间频率、相同对比度的正弦条纹图像。

依赖: pip install numpy opencv-python matplotlib

正弦条纹公式:
    I(x) = mean + 0.5 * amplitude * cos(2π * f * x)

其中:
    mean     — 平均亮度 (0~1)
    amplitude— 调制幅度，由目标对比度决定
    f        — 空间频率 (周期/图像宽度)
    x        — 水平方向坐标

对比度定义 (Michelson):
    C = (I_max - I_min) / (I_max + I_min)

所有条纹共享相同的 mean 和 amplitude，因此对比度一致，
仅空间频率不同。
"""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ──────────────────── 参数解析 ────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="生成不同空间频率、相同对比度的正弦条纹"
    )
    p.add_argument("--width",  type=int, default=3840,
                   help="图像宽度 (像素, 默认 3840)")
    p.add_argument("--height", type=int, default=2160,
                   help="图像高度 (像素, 默认 2160)")
    p.add_argument("--mean",   type=float, default=0.5,
                   help="平均亮度 0~1 (默认 0.5)")
    p.add_argument("--contrast", type=float, default=0.8,
                   help="Michelson 对比度 0~1 (默认 0.8)")
    p.add_argument("--direction", choices=["horizontal", "vertical", "both"],
                   default="horizontal",
                   help="条纹方向: horizontal(竖直条纹)/vertical(水平条纹)/both(默认 horizontal)")
    p.add_argument("--freqs", type=str, default="1,2,4,8,16,32",
                   help="空间频率列表，逗号分隔 (周期数/图像宽度, 默认 1,2,4,8,16,32)")
    p.add_argument("--bitdepth", type=int, default=8, choices=[8, 16, 32],
                   help="输出位深: 8(uint8) / 16(uint16) / 32(float32, 默认 8)")
    p.add_argument("-o", "--outdir", type=str, default="sine_fringes",
                   help="输出目录 (默认 sine_fringes)")
    p.add_argument("--prefix", type=str, default="sine",
                   help="文件名前缀 (默认 sine)")
    p.add_argument("--preview", action="store_true",
                   help="生成预览拼图 (需要 matplotlib)")
    return p.parse_args()


# ──────────────────── 核心生成 ────────────────────
def generate_sine_fringe(width: int, height: int,
                         mean: float, contrast: float,
                         freq: float, direction: str) -> np.ndarray:
    """
    生成单幅正弦条纹 (float64, 范围 [0, 1])。

    Parameters
    ----------
    width, height : int
        图像尺寸。
    mean : float
        平均亮度，0~1。
    contrast : float
        Michelson 对比度，0~1。
        amplitude = contrast * 2 * mean  (保证 I_min = mean*(1-C) >= 0)
    freq : float
        空间频率，即完整周期数 / 图像宽度(或高度)。
    direction : str
        "horizontal" → 条纹沿竖直方向排列 (灰度沿 x 变化)
        "vertical"   → 条纹沿水平方向排列 (灰度沿 y 变化)

    Returns
    -------
    img : ndarray, shape (height, width), dtype float64, 范围 [0, 1]
    """
    # 由 Michelson 对比度反推调制幅度
    # C = (I_max - I_min) / (I_max + I_min)
    # I_max = mean + A,  I_min = mean - A
    # => C = A / mean  =>  A = C * mean
    amplitude = contrast * mean

    if direction == "horizontal":
        x = np.linspace(0, width - 1, width, dtype=np.float64)
        phase = 2.0 * math.pi * freq * x / width
        profile = mean + amplitude * np.cos(phase)          # 1-D
        img = np.broadcast_to(profile[np.newaxis, :], (height, width)).copy()
    else:
        y = np.linspace(0, height - 1, height, dtype=np.float64)
        phase = 2.0 * math.pi * freq * y / height
        profile = mean + amplitude * np.cos(phase)
        img = np.broadcast_to(profile[:, np.newaxis], (height, width)).copy()

    return np.clip(img, 0.0, 1.0)


def to_output_dtype(img: np.ndarray, bitdepth: int):
    """将 [0,1] float64 图像转换为目标位深。"""
    if bitdepth == 8:
        return (img * 255).round().astype(np.uint8)
    elif bitdepth == 16:
        return (img * 65535).round().astype(np.uint16)
    else:
        return img.astype(np.float32)


# ──────────────────── 主流程 ────────────────────
def main():
    args = parse_args()

    # 解析频率列表
    try:
        freqs = [float(f.strip()) for f in args.freqs.split(",") if f.strip()]
    except ValueError:
        raise SystemExit(f"频率列表格式错误: {args.freqs}")

    if not freqs:
        raise SystemExit("频率列表不能为空")

    # 参数校验
    if not (0.0 < args.mean < 1.0):
        raise SystemExit(f"mean 必须在 (0,1) 开区间, 当前: {args.mean}")
    if not (0.0 <= args.contrast <= 1.0):
        raise SystemExit(f"contrast 必须在 [0,1], 当前: {args.contrast}")
    # 确保 I_min >= 0: mean - amplitude >= 0  =>  contrast <= 1 (已满足)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    directions = []
    if args.direction in ("horizontal", "both"):
        directions.append("horizontal")
    if args.direction in ("vertical", "both"):
        directions.append("vertical")

    saved = []   # 记录保存路径，用于预览拼图

    for d in directions:
        for freq in freqs:
            img_f = generate_sine_fringe(
                width=args.width, height=args.height,
                mean=args.mean, contrast=args.contrast,
                freq=freq, direction=d,
            )
            img_out = to_output_dtype(img_f, args.bitdepth)

            dir_tag = "h" if d == "horizontal" else "v"
            fname = f"{args.prefix}_{dir_tag}_f{freq:g}.png"
            fpath = outdir / fname
            cv2.imwrite(str(fpath), img_out)

            # 验证实际对比度
            i_max = img_f.max()
            i_min = img_f.min()
            actual_c = (i_max - i_min) / (i_max + i_min) if (i_max + i_min) > 0 else 0.0

            print(f"[OK] {fpath}  "
                  f"freq={freq:>6g} cyc/img  "
                  f"Imin={i_min:.4f}  Imax={i_max:.4f}  "
                  f"C_actual={actual_c:.4f}")
            saved.append((fpath, freq, d, actual_c))

    # ── 预览拼图 ──
    if args.preview and HAS_MPL and len(saved) > 1:
        n = len(saved)
        cols = min(n, 6)
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols,
                                 figsize=(3 * cols, 3 * rows),
                                 squeeze=False)
        for idx, (fpath, freq, d, c_actual) in enumerate(saved):
            r, c = divmod(idx, cols)
            ax = axes[r][c]
            img = cv2.imread(str(fpath), cv2.IMREAD_UNCHANGED)
            if img.dtype == np.uint16:
                img = (img / 256).astype(np.uint8)
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            dir_label = "H" if d == "horizontal" else "V"
            ax.set_title(f"f={freq:g} cyc  C={c_actual:.3f}", fontsize=9)
            ax.axis("off")
        # 隐藏多余子图
        for idx in range(n, rows * cols):
            r, c = divmod(idx, cols)
            axes[r][c].axis("off")
        fig.suptitle(
            f"Sine Fringes  |  mean={args.mean}, contrast={args.contrast}, "
            f"{args.width}×{args.height}",
            fontsize=11,
        )
        fig.tight_layout()
        preview_path = outdir / f"{args.prefix}_preview.png"
        fig.savefig(str(preview_path), dpi=150)
        plt.close(fig)
        print(f"\n[预览拼图] {preview_path}")

    print(f"\n完成，共生成 {len(saved)} 幅图像 -> {outdir}/")


if __name__ == "__main__":
    main()
