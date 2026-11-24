#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
色相-亮度采样脚本
===================
从图像中随机采样不同色相和亮度的点，生成 48 个色块（6 色相 × 8 亮度）。

采样策略：
- 色相：0-180 度分 6 区间（每 30 度），每区间随机选点
- 亮度：每色相区间内，按 V 通道从暗到亮分 8 档
- 约束：色块间不重叠（间距 >= gap），且距边界 >= margin

用法:
    # 单图模式
    python hue_brightness_sampler.py --input image.png --out ./out

    # 目录模式（批量处理）
    python hue_brightness_sampler.py --input ./image_dir --out ./out
"""

import os
import csv
import json
import copy
import re
import shutil
import argparse

import numpy as np
import cv2


# ==================== 基础工具函数 ====================

def box_range(cy, cx, size, anchor="floor"):
    """以 (cy, cx) 为中心计算 size x size 区域的坐标范围。"""
    if size % 2 == 1:
        up = left = size // 2
    else:
        up = left = size // 2 if anchor == "floor" else size // 2 - 1
    return cy - up, cy - up + size, cx - left, cx - left + size, up, left


def load_img(path):
    """加载图片为 float32 BGR 格式，值域 [0, 1]。"""
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


def luma(bgr, coef="709"):
    """计算 BT.709 或 BT.601 亮度。"""
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    if coef == "601":
        return 0.299 * r + 0.587 * g + 0.114 * b
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


# ==================== 采样逻辑 ====================

def sample_by_hue_brightness(img_bgr, n_hue=6, n_bright=8, rng=None,
                              block_size=100, margin=50):
    """
    按色相和亮度采样。
    
    策略：
    - 色相：取图像实际 min/max hue，平均分为 n_hue 区间
    - 亮度：每个色相区间内取实际 min/max V，平均分为 n_bright 档位
    - 约束：色块不重叠（中心距离在 x 或 y 任一方向 >= block_size），边距 >= margin
    """
    if rng is None:
        rng = np.random.default_rng()
    
    H, W = img_bgr.shape[:2]
    img_uint8 = np.clip(img_bgr * 255, 0, 255).astype(np.uint8)
    img_hsv = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2HSV)
    
    # 有效像素（饱和度 > 10，排除灰度）
    valid_mask = img_hsv[:, :, 1] > 10
    valid_ys, valid_xs = np.where(valid_mask)
    if len(valid_ys) == 0:
        print("[warn] 图像中没有有效彩色像素")
        return []
    
    hue_vals = img_hsv[valid_ys, valid_xs, 0]
    hue_min, hue_max = int(hue_vals.min()), int(hue_vals.max())
    print(f"[info] 图像色相范围: {hue_min} ~ {hue_max}")
    
    # 色相区间：按实际范围平均划分
    hue_edges = np.linspace(hue_min, hue_max, n_hue + 1).astype(int)
    hue_bins = [(hue_edges[i], hue_edges[i+1]) for i in range(n_hue)]
    
    samples = []
    placed = []
    
    def is_valid(y, x):
        half = block_size // 2
        # 边距：色块完整在图内
        if y - half < margin or y + half + (block_size % 2) > H - margin:
            return False
        if x - half < margin or x + half + (block_size % 2) > W - margin:
            return False
        # 不重叠：与所有已放置色块，x 和 y 方向都不能同时太近
        for py, px in placed:
            if abs(y - py) < block_size and abs(x - px) < block_size:
                return False
        return True
    
    for hi, (h_lo, h_hi) in enumerate(hue_bins):
        # 找出色相在此区间的像素
        mask = (img_hsv[:, :, 0] >= h_lo) & (img_hsv[:, :, 0] < h_hi) & (img_hsv[:, :, 1] > 10)
        ys, xs = np.where(mask)
        
        if len(ys) < n_bright:
            print(f"[warn] 色相区间 [{h_lo},{h_hi}) 像素不足 {n_bright}，实际 {len(ys)}")
            if len(ys) == 0:
                continue
        
        # 此区间内的亮度范围
        v_vals = img_hsv[ys, xs, 2]
        v_min, v_max = int(v_vals.min()), int(v_vals.max())
        
        # 亮度档位：按实际范围平均划分
        v_edges = np.linspace(v_min, v_max, n_bright + 1).astype(int)
        
        # 按亮度排序
        sorted_idx = np.argsort(v_vals)
        ys_sorted = ys[sorted_idx]
        xs_sorted = xs[sorted_idx]
        v_sorted = v_vals[sorted_idx]
        
        for bi in range(n_bright):
            v_lo, v_hi = v_edges[bi], v_edges[bi + 1]
            # 找亮度在此档位的像素
            in_bin = np.where((v_sorted >= v_lo) & (v_sorted < v_hi))[0]
            if len(in_bin) == 0:
                in_bin = np.where(v_sorted == v_lo)[0]
            if len(in_bin) == 0:
                print(f"[warn] 色相区间 [{h_lo},{h_hi}) 亮度档 [{v_lo},{v_hi}) 无像素，跳过")
                continue
            
            # 随机选一个满足约束的位置
            rng.shuffle(in_bin)
            picked = False
            for idx in in_bin:
                y, x = int(ys_sorted[idx]), int(xs_sorted[idx])
                if is_valid(y, x):
                    bgr_val = img_bgr[y, x]
                    v_val = int(img_hsv[y, x, 2])
                    placed.append((y, x))
                    samples.append({
                        "hue_bin": hi, "bright_bin": bi,
                        "hue_range": f"[{h_lo},{h_hi})",
                        "v_range": f"[{v_lo},{v_hi})",
                        "y": y, "x": x, "bgr": bgr_val, "v": v_val,
                    })
                    picked = True
                    break
            
            if not picked:
                print(f"[warn] 色相区间 [{h_lo},{h_hi}) 亮度档 {bi} 找不到满足约束的位置，跳过")
    
    return samples


# ==================== 主流程 ====================

def run(a):
    """处理单张图片的主流程。"""
    os.makedirs(a.out, exist_ok=True)
    S = a.size
    rng = np.random.default_rng(a.seed)

    # 1. 加载图像
    img = load_img(a.input)
    
    # 2. Resize 到 4K (3840x2160)
    target_h, target_w = 2160, 3840
    orig_h, orig_w = img.shape[:2]
    if (orig_h, orig_w) != (target_h, target_w):
        print(f"[info] Resize {orig_w}x{orig_h} -> {target_w}x{target_h}")
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    H, W = img.shape[:2]

    # 3. 采样（带不重叠约束）
    samples = sample_by_hue_brightness(img, n_hue=a.hue_bins, n_bright=a.bright_bins,
                                       rng=rng, block_size=S, margin=a.margin)
    n_expected = a.hue_bins * a.bright_bins
    print(f"[info] 图像 {W}x{H}, 采样 {len(samples)}/{n_expected} 个色块")
    print(f"[info] 色块尺寸: {S}x{S}, 边距: {a.margin}px")

    # 4. 绘制色块
    img_out = np.clip(img * 255, 0, 255).astype(np.uint8).copy()
    Y = luma(img, a.coef) * 255.0
    
    ce_list = []
    rows_data = []

    for s in samples:
        y, x = s["y"], s["x"]
        bgr = s["bgr"]
        color = tuple(int(round(v * 255)) for v in np.clip(bgr, 0, 1))
        y_val = float(Y[y, x])
        
        # 绘制色块
        y0, y1, x0, x1, _, _ = box_range(y, x, S, a.anchor)
        
        if y0 < 0 or y1 > H or x0 < 0 or x1 > W:
            print(f"[warn] 色块 ({s['hue_bin']},{s['bright_bin']}) 超出边界，跳过")
            continue
        
        img_out[y0:y1, x0:x1] = color
        
        if a.outline:
            cv2.rectangle(img_out, (x0, y0), (x0 + S - 1, y0 + S - 1), (0, 0, 255), 1)
        
        if a.mark_center:
            cv2.drawMarker(img_out, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 9, 1)
        
        ce_list.append(np.full((S, S, 3), color, np.uint8))
        rows_data.append(dict(
            row=s["hue_bin"],
            col=s["bright_bin"],
            cy=int(y), cx=int(x),
            box_y0=int(y0), box_x0=int(x0), box_y1=int(y1 - 1), box_x1=int(x1 - 1),
            Y=round(y_val, 2),
            R=color[2], G=color[1], B=color[0]
        ))

    # 5. 保存结果
    patch_png = os.path.join(a.out, "image_with_patches.png")
    cv2.imwrite(patch_png, img_out)

    # 一致性校验
    written = cv2.imread(patch_png, cv2.IMREAD_UNCHANGED)[:, :, :3]
    mismatch = 0
    for s in samples:
        y, x = s["y"], s["x"]
        if y < 0 or y >= H or x < 0 or x >= W:
            continue
        color = tuple(int(round(v * 255)) for v in np.clip(s["bgr"], 0, 1))
        got = tuple(int(v) for v in written[y, x])
        if got != color:
            mismatch += 1
    if mismatch:
        print(f"[warn] {mismatch} 个色块中心像素与记录不一致")
    else:
        print(f"[check] 色块中心像素校验完成")
    
    # 色卡
    if ce_list:
        cv2.imwrite(os.path.join(a.out, "colorchart.png"), np.hstack(ce_list))
    
    # CSV
    if rows_data:
        with open(os.path.join(a.out, "patches.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_data[0].keys()))
            w.writeheader()
            w.writerows(rows_data)
    
    # JSON
    with open(os.path.join(a.out, "patches.json"), "w", encoding="utf-8") as f:
        json.dump(dict(
            grid=dict(rows=a.hue_bins, cols=a.bright_bins),
            size=S,
            patches=rows_data
        ), f, indent=2, ensure_ascii=False)

    # 打印结果表格
    print(f"\n Row Col    Y   (R, G, B)      center")
    for r_data in rows_data:
        print(f" {r_data['row']:>2d}  {r_data['col']:>2d}  {r_data['Y']:>5.1f}  "
              f"({r_data['R']:>3},{r_data['G']:>3},{r_data['B']:>3})  "
              f"({r_data['cx']:>5},{r_data['cy']:>5})")
    
    print(f"\n[done] -> {a.out}")
    return rows_data


# ==================== 批量处理 ====================

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def make_stem(idx, filename):
    """规范化图片名。"""
    base = os.path.splitext(os.path.basename(filename))[0]
    m = re.match(r"\D*(\d+)", base)
    num = m.group(1) if m else ""
    return f"{idx:03d}_{num}" if num else f"{idx:03d}"


def items_from_rows(rows_data, idx, image, a):
    """生成 dataset.json items。"""
    pid = f"{idx:03d}"
    items = []
    for r_data in rows_data:
        items.append({
            "image": image,
            "id": f"{pid}-R{r_data['row']}C{r_data['col']}",
            "type": "grid",
            "x": int(r_data["cx"]),
            "y": int(r_data["cy"]),
            "rgb": [int(r_data["R"]), int(r_data["G"]), int(r_data["B"])],
        })
    return items


def write_dataset(out_dir, items, a):
    """写出 dataset.json。"""
    dataset = {
        "version": 1,
        "name": a.dataset_name,
        "width": 3840,
        "height": 2160,
        "patch_size": a.size,
        "grid": {"rows": a.hue_bins, "cols": a.bright_bins},
        "items": items,
    }
    path = os.path.join(out_dir, a.dataset_json)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    return path


def main():
    p = argparse.ArgumentParser(
        description="色相-亮度采样：按色相区间和亮度档位随机采样生成色块")

    # 输入输出
    p.add_argument("--input", required=True, help="输入图片路径或目录")
    p.add_argument("--out", default="./out", help="输出根目录")

    # 采样参数
    p.add_argument("--hue-bins", type=int, default=6, help="色相区间数（默认 6，每区间 30 度）")
    p.add_argument("--bright-bins", type=int, default=8, help="每色相区间的亮度档位数（默认 8）")
    p.add_argument("--size", type=int, default=100, help="色块边长（默认 100）")
    p.add_argument("--anchor", default="floor", choices=["floor", "ceil"],
                   help="偶数尺寸时中心像素偏左上(floor)或右下(ceil)")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--margin", type=int, default=50, help="色块中心到图像边界的最小距离")

    # 其他参数
    p.add_argument("--coef", default="709", choices=["709", "601"], help="亮度权重")

    # 可视化选项
    p.add_argument("--outline", action="store_true", help="画红框")
    p.add_argument("--mark-center", action="store_true", help="中心画十字")

    # dataset.json 输出
    p.add_argument("--dataset-name", default="hue-brightness-samples", help="dataset.json 的 name 字段")
    p.add_argument("--dataset-json", default="dataset.json", help="dataset.json 文件名")

    args = p.parse_args()

    # 判断单图还是目录模式
    is_dir = os.path.isdir(args.input)

    if not is_dir:
        # 单图模式
        rows_data = run(args)
        stem = make_stem(1, args.input)
        img_dir = os.path.join(args.out, "images")
        os.makedirs(img_dir, exist_ok=True)
        src = os.path.join(args.out, "image_with_patches.png")
        if os.path.exists(src):
            shutil.copy(src, os.path.join(img_dir, stem + ".png"))
        items = items_from_rows(rows_data, 1, f"images/{stem}.png", args)
        path = write_dataset(args.out, items, args)
        print(f"[out]  dataset.json -> {path}  ({len(items)} 条)")
    else:
        # 目录模式
        files = sorted([f for f in os.listdir(args.input)
                        if os.path.splitext(f)[1].lower() in IMAGE_EXTS])

        if not files:
            raise SystemExit(f"[error] 目录中没有图片: {args.input}")

        print(f"[info] 找到 {len(files)} 张图片")

        # 创建输出目录
        img_dir = os.path.join(args.out, "images")
        os.makedirs(img_dir, exist_ok=True)

        success, failed, all_items = 0, [], []
        for i, name in enumerate(files, 1):
            print(f"\n{'='*60}")
            print(f"[{i}/{len(files)}] 处理: {name}")
            print(f"{'='*60}")

            sub_args = copy.copy(args)
            sub_args.input = os.path.join(args.input, name)
            sub_args.out = os.path.join(args.out, "_tmp_" + os.path.splitext(name)[0])

            try:
                rows_data = run(sub_args)

                stem = make_stem(i, name)
                tmp_img = os.path.join(sub_args.out, "image_with_patches.png")
                if os.path.exists(tmp_img):
                    shutil.copy(tmp_img, os.path.join(img_dir, stem + ".png"))
                    shutil.move(tmp_img, os.path.join(args.out, name))

                all_items += items_from_rows(rows_data, i, f"images/{stem}.png", args)
                shutil.rmtree(sub_args.out, ignore_errors=True)
                success += 1
            except Exception as e:
                print(f"[error] {name}: {e}")
                failed.append((name, str(e)))

        if all_items:
            ds_path = write_dataset(args.out, all_items, args)
        print(f"\n{'='*60}")
        print(f"[done] 完成 {success}/{len(files)} 张")
        print(f"[out]  images -> {img_dir}")
        if all_items:
            print(f"[out]  dataset.json -> {ds_path}  ({len(all_items)} 条)")
        if failed:
            print(f"[fail] {len(failed)} 张失败:")
            for name, err in failed:
                print(f"  {name}: {err}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
