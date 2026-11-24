#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绿色 80% 亮度色块 -> 画面中央竖排
==================================
在图像中检测绿色像素，取其中亮度处于 ~80% 档的像素求代表色，生成 100x100 纯色块，
在画面水平居中、竖直方向从上往下依次排列 N 个（默认 6）。

用法:
    python green_center_blocks.py --image in.png
    python green_center_blocks.py --image in.png --out out.png --count 6 --size 100 --gap 20
    python green_center_blocks.py --image in.png --pct 80          # 80% 亮度档(绿色像素亮度分位)

亮度档说明(--luma-ref):
    pct   : 取绿色像素亮度的第 --pct 百分位(默认, 抗噪, "绿色里 80% 亮度")
    range : 取绿色像素亮度区间 [min,max] 内 --pct% 处的值
"""

import os
import argparse

import numpy as np
import cv2


# ---------------------------------------------------------------- 基础工具
def load_bgr01(path):
    """读图为 float32 BGR，值域 [0,1]，支持 8/16bit 与浮点。"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = img[:, :, :3]
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        denom = 1023.0 if img.max() <= 1023 else 65535.0
        return img.astype(np.float32) / denom
    return np.clip(img.astype(np.float32), 0.0, 1.0)


def luma709(bgr):
    """BT.709 亮度，值域随输入。"""
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def green_mask(bgr, s_min=0.20, v_min=0.02, dom=0.03, h_lo=35, h_hi=90):
    """绿色像素检测：HSV 绿色相区间 + G 通道主导，双判据取交集。"""
    u8 = np.clip(bgr * 255.0, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(u8, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0]
    s = hsv[..., 1] / 255.0
    v = hsv[..., 2] / 255.0
    m_hsv = (h >= h_lo) & (h <= h_hi) & (s >= s_min) & (v >= v_min)
    g = bgr[..., 1]
    m_dom = (g - np.maximum(bgr[..., 0], bgr[..., 2])) > dom
    return m_hsv & m_dom


# ---------------------------------------------------------------- 代表色选取
def pick_green_color(bgr, mask, pct=80.0, luma_ref="maxrel",
                     max_pct=99.5, luma_tol=6.0):
    """
    在绿色像素中选取目标亮度档的代表色，返回 (B,G,R) uint8 及统计信息。

    目标亮度 target 由 --luma-ref 决定：
        maxrel(默认): target = (pct/100) * 最亮绿色亮度   -> "当前图最亮绿色的 80%"
                       最亮绿色亮度用第 --max-pct 百分位(默认 99.5，抗热点噪声；
                       设 100 即真正最大值)。
        range       : target = min + (pct/100) * (max-min)
        pct         : target = 绿色亮度的第 pct 百分位
    再取亮度落在 [target-tol, target+tol]（0-255）内的绿色像素，对其颜色取中位数(抗噪)；
    命中过少时自动放宽 tol。
    """
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        raise SystemExit("[error] 未检出绿色像素，请放宽 --s-min / --dom / 调整 --h-lo/--h-hi")

    Y = luma709(bgr)[ys, xs] * 255.0             # 绿色像素亮度 [0,255]
    brightest = float(np.percentile(Y, max_pct))
    if luma_ref == "maxrel":
        target = (pct / 100.0) * brightest
    elif luma_ref == "range":
        lo, hi = float(Y.min()), float(Y.max())
        target = lo + (pct / 100.0) * (hi - lo)
    else:  # pct
        target = float(np.percentile(Y, pct))

    tol = float(luma_tol)
    while True:
        sel = np.abs(Y - target) <= tol
        if int(sel.sum()) >= 20 or tol >= 64.0:
            break
        tol += 2.0                                # 命中过少则放宽窗口

    cols = bgr[ys[sel], xs[sel]]                  # (k,3) BGR in [0,1]
    med = np.median(cols, axis=0)
    color_bgr = tuple(int(round(v * 255)) for v in np.clip(med, 0, 1))
    info = dict(
        n_green=int(len(ys)),
        n_selected=int(sel.sum()),
        brightest_green=round(brightest, 1),
        target_luma=round(target, 1),
        used_tol=round(tol, 1),
        green_luma_range=[round(float(Y.min()), 1), round(float(Y.max()), 1)],
    )
    return color_bgr, info


# ---------------------------------------------------------------- 绘制竖排色块
def draw_blocks(img_u8, color_bgr, count=6, size=100, layout="even", gap=20, outline=False):
    """在画面水平居中排列 count 个 size×size 纯色块。

    layout:
        even (默认): 把画面高度 H 等分为 count 段，每块居于所在段的中心
                     -> 从上往下"均匀"分布(按 2160 平均)。
        stack      : 整列竖直居中，块间固定间距 gap。
    """
    H, W = img_u8.shape[:2]
    if size > W:
        raise SystemExit(f"[error] 色块宽 {size} 超过图像宽 {W}")
    x0 = (W - size) // 2                          # 水平居中

    if layout == "even":
        seg = H / count                            # 每段高度(H=2160 时为 360)
        if size > seg:
            print(f"[warn] 色块高 {size} > 等分段 {seg:.0f}，相邻块会重叠")
        y0s = [int(round((i + 0.5) * seg - size / 2.0)) for i in range(count)]
    else:  # stack
        total_h = count * size + (count - 1) * gap
        if total_h > H:
            raise SystemExit(f"[error] {count} 个色块总高 {total_h} 超过图像高 {H}")
        y_start = (H - total_h) // 2
        y0s = [y_start + i * (size + gap) for i in range(count)]

    out = img_u8.copy()
    boxes = []
    for y0 in y0s:
        y0 = max(0, min(y0, H - size))             # 保证整块落在画内
        out[y0:y0 + size, x0:x0 + size] = color_bgr
        if outline:
            cv2.rectangle(out, (x0, y0), (x0 + size - 1, y0 + size - 1), (0, 0, 255), 2)
        boxes.append((x0, y0, x0 + size, y0 + size))
    return out, boxes


def draw_row(img_u8, color_bgr, cols, size, y_center, outline=False):
    """在给定纵向中心 y_center 处，横向按画面宽度均匀分布 cols 个 size×size 色块。"""
    H, W = img_u8.shape[:2]
    if size > H:
        raise SystemExit(f"[error] 色块高 {size} 超过图像高 {H}")
    seg = W / cols                                 # 每段宽度(W=3840,cols=8 时为 480)
    if size > seg:
        print(f"[warn] 色块宽 {size} > 等分段 {seg:.0f}，相邻块会重叠")
    y0 = max(0, min(int(round(y_center - size / 2.0)), H - size))
    out = img_u8.copy()
    boxes = []
    for j in range(cols):
        x0 = int(round((j + 0.5) * seg - size / 2.0))
        x0 = max(0, min(x0, W - size))
        out[y0:y0 + size, x0:x0 + size] = color_bgr
        if outline:
            cv2.rectangle(out, (x0, y0), (x0 + size - 1, y0 + size - 1), (0, 0, 255), 2)
        boxes.append((x0, y0, x0 + size, y0 + size))
    return out, boxes


# ---------------------------------------------------------------- 单图处理
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def process_one(in_path, out_path, args):
    """处理单张图片：resize -> 检测绿色 -> 取目标亮度代表色 -> 均匀竖排色块 -> 保存。"""
    bgr = load_bgr01(in_path)
    oh, ow = bgr.shape[:2]
    tw, th = args.width, args.height
    if (ow, oh) != (tw, th):                       # 处理前统一 resize 到目标尺寸
        interp = cv2.INTER_AREA if (tw * th) < (ow * oh) else cv2.INTER_CUBIC
        bgr = cv2.resize(bgr, (tw, th), interpolation=interp)
        print(f"[info] {os.path.basename(in_path)}  resize {ow}x{oh} -> {tw}x{th}")
    H, W = bgr.shape[:2]
    mask = green_mask(bgr, args.s_min, args.v_min, args.dom, args.h_lo, args.h_hi)
    color_bgr, info = pick_green_color(bgr, mask, args.pct, args.luma_ref,
                                       args.max_pct, args.luma_tol)

    r, g, b = color_bgr[2], color_bgr[1], color_bgr[0]
    print(f"[info] {os.path.basename(in_path)}  {W}x{H}, 绿色像素 {info['n_green']} "
          f"(亮度 {info['green_luma_range'][0]}~{info['green_luma_range'][1]}, "
          f"最亮绿 {info['brightest_green']})")
    print(f"[info]   {args.pct:.0f}% 档 target={info['target_luma']} "
          f"(±{info['used_tol']}, 选中 {info['n_selected']} 像素) -> "
          f"RGB=({r},{g},{b}) luma={0.2126*r+0.7152*g+0.0722*b:.1f}")

    img_u8 = np.clip(bgr * 255, 0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    if args.layout == "rows":
        # 每张输入 -> count 张输出，每张横排 cols 个色块，仅纵向位置不同(2160 均分)
        seg_h = H / args.count
        base, ext = os.path.splitext(out_path)
        for i in range(args.count):
            yc = (i + 0.5) * seg_h
            out_img, _ = draw_row(img_u8, color_bgr, args.cols, args.size, yc, args.outline)
            cv2.imwrite(f"{base}_r{i}{ext}", out_img)
        print(f"[done] {args.count} 张(每张横排 {args.cols} 块) -> "
              f"{base}_r0..{args.count-1}{ext}")
    else:
        out_img, _ = draw_blocks(img_u8, color_bgr, args.count, args.size,
                                 args.layout, args.gap, args.outline)
        cv2.imwrite(out_path, out_img)
        print(f"[done] -> {out_path}")

    if args.swatch:
        sw = np.full((args.size, args.size, 3), color_bgr, np.uint8)
        cv2.imwrite(os.path.splitext(out_path)[0] + "_swatch.png", sw)
    return color_bgr, info


# ---------------------------------------------------------------- 主流程
def main():
    p = argparse.ArgumentParser(description="绿色最亮值 80% 亮度色块 -> 画面中央竖排 N 个")
    p.add_argument("--image", required=True, help="输入图片路径或目录(目录=批量处理)")
    p.add_argument("--out", default=None,
                   help="输出：单图=文件路径(默认 <输入>_greenblocks.png)；"
                        "目录=输出目录(默认 <输入目录>_greenblocks)")
    p.add_argument("--count", type=int, default=6,
                   help="色块数量；rows 模式下为输出图张数/纵向档数(默认 6)")
    p.add_argument("--cols", type=int, default=8,
                   help="rows 模式每张图横向色块数(默认 8)")
    p.add_argument("--size", type=int, default=100, help="色块边长(默认 100)")
    p.add_argument("--width", type=int, default=3840, help="处理前 resize 宽(默认 3840)")
    p.add_argument("--height", type=int, default=2160, help="处理前 resize 高(默认 2160)")
    p.add_argument("--layout", default="even", choices=["even", "stack", "rows"],
                   help="even=竖列按高度均匀分布(默认) / stack=竖直居中定距 / "
                        "rows=每图横排 cols 块、生成 count 张(仅纵向位置不同,按高度均分)")
    p.add_argument("--gap", type=int, default=20, help="stack 模式相邻色块间距(默认 20)")
    # 亮度档
    p.add_argument("--pct", type=float, default=80.0,
                   help="目标亮度百分比(默认 80，即最亮绿色的 80%%)")
    p.add_argument("--luma-ref", default="maxrel", choices=["maxrel", "range", "pct"],
                   help="maxrel=最亮绿色×pct%%(默认) / range=亮度区间比例 / pct=亮度分位")
    p.add_argument("--max-pct", type=float, default=99.5,
                   help="最亮绿色取第几百分位(默认 99.5 抗噪；100=真正最大值)")
    p.add_argument("--luma-tol", type=float, default=6.0,
                   help="目标亮度选取窗口 ±tol(0-255，默认 6，命中不足自动放宽)")
    # 绿色判据
    p.add_argument("--s-min", type=float, default=0.20, help="绿色最低饱和度")
    p.add_argument("--v-min", type=float, default=0.02, help="绿色最低明度")
    p.add_argument("--dom", type=float, default=0.03, help="G 通道主导阈值")
    p.add_argument("--h-lo", type=int, default=35, help="绿色色相下界(OpenCV 0-179)")
    p.add_argument("--h-hi", type=int, default=90, help="绿色色相上界")
    p.add_argument("--outline", action="store_true", help="给色块画红框")
    p.add_argument("--swatch", action="store_true", help="额外保存单个色块 swatch.png")
    args = p.parse_args()

    if os.path.isdir(args.image):
        # 目录批量模式
        out_dir = args.out or (os.path.normpath(args.image) + "_greenblocks")
        files = sorted(f for f in os.listdir(args.image)
                       if os.path.splitext(f)[1].lower() in IMAGE_EXTS
                       and "_greenblocks" not in f and "_swatch" not in f)
        if not files:
            raise SystemExit(f"[error] 目录中无图片: {args.image}")
        print(f"[info] 批量处理 {len(files)} 张 -> {out_dir}")
        ok, failed = 0, []
        for i, name in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] {name}")
            in_path = os.path.join(args.image, name)
            out_path = os.path.join(out_dir, os.path.splitext(name)[0] + "_greenblocks.png")
            try:
                process_one(in_path, out_path, args)
                ok += 1
            except SystemExit as e:
                print(f"[skip] {name}: {e}")
                failed.append(name)
            except Exception as e:
                print(f"[error] {name}: {e}")
                failed.append(name)
        print(f"\n[done] 成功 {ok}/{len(files)}，输出目录 {out_dir}")
        if failed:
            print(f"[fail] {len(failed)} 张: {', '.join(failed)}")
    else:
        # 单图模式
        out_path = args.out or (os.path.splitext(args.image)[0] + "_greenblocks.png")
        process_one(args.image, out_path, args)


if __name__ == "__main__":
    main()
