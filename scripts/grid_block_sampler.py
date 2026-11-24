#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
网格分块采样脚本
=================
将图像分成 rows x cols 的网格，选取每个区域中心的像素生成纯色块。

用法:
    # 单图模式
    python grid_block_sampler.py --input image.png --out ./out

    # 目录模式（批量处理）
    python grid_block_sampler.py --input ./image_dir --out ./out

    # 自定义网格大小
    python grid_block_sampler.py --input image.png --out ./out --rows 4 --cols 6
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
    """
    以 (cy, cx) 为中心计算 size x size 区域的坐标范围。
    """
    if size % 2 == 1:
        up = left = size // 2
    else:
        up = left = size // 2 if anchor == "floor" else size // 2 - 1
    return cy - up, cy - up + size, cx - left, cx - left + size, up, left


def load_img(path):
    """
    加载图片为 float32 BGR 格式，值域 [0, 1]。
    """
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


# ==================== 主流程 ====================

def run(a):
    """处理单张图片的主流程。"""
    os.makedirs(a.out, exist_ok=True)
    S = a.size

    # 1. 加载图像
    img = load_img(a.input)
    
    # 2. Resize 到 4K (3840x2160)
    target_h, target_w = 2160, 3840
    orig_h, orig_w = img.shape[:2]
    if (orig_h, orig_w) != (target_h, target_w):
        print(f"[info] Resize {orig_w}x{orig_h} -> {target_w}x{target_h}")
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    H, W = img.shape[:2]
    rows, cols = a.rows, a.cols

    # 3. 计算每个网格区域的中心点
    cell_h = H / rows
    cell_w = W / cols
    
    centers = []
    for r in range(rows):
        for c in range(cols):
            # 区域中心
            cy = int((r + 0.5) * cell_h)
            cx = int((c + 0.5) * cell_w)
            centers.append((r, c, cy, cx))

    print(f"[info] 图像 {W}x{H}, 网格 {rows}x{cols}={rows*cols} 个区域")
    print(f"[info] 单元格大小: {cell_w:.1f}x{cell_h:.1f} 像素")
    print(f"[info] 色块尺寸: {S}x{S}")

    # 4. 采样并绘制色块
    img_out = np.clip(img * 255, 0, 255).astype(np.uint8).copy()
    Y = luma(img, a.coef) * 255.0
    
    picks = []
    ce_list = []
    rows_data = []

    for idx, (r, c, cy, cx) in enumerate(centers):
        # 采样像素
        if a.stabilize:
            pixel = np.median(img[cy-1:cy+2, cx-1:cx+2].reshape(-1, 3), axis=0)
        else:
            pixel = img[cy, cx]
        
        color = tuple(int(round(v * 255)) for v in np.clip(pixel, 0, 1))
        y_val = float(Y[cy, cx])
        
        # 绘制色块
        y0, y1, x0, x1, _, _ = box_range(cy, cx, S, a.anchor)
        
        # 确保色块在图像范围内
        if y0 < 0 or y1 > H or x0 < 0 or x1 > W:
            print(f"[warn] 区域 ({r},{c}) 中心 ({cy},{cx}) 色块超出边界，跳过")
            continue
        
        img_out[y0:y1, x0:x1] = color
        
        # 可选：画红框
        if a.outline:
            cv2.rectangle(img_out, (x0, y0), (x0 + S - 1, y0 + S - 1), (0, 0, 255), 1)
        
        # 可选：标注行列号
        if a.label:
            text = f"{r},{c}"
            text_x = max(5, x0 - 40)
            text_y = y0 + S // 2 + 8
            cv2.putText(img_out, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(img_out, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 可选：中心画十字
        if a.mark_center:
            cv2.drawMarker(img_out, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 9, 1)
        
        ce_list.append(np.full((S, S, 3), color, np.uint8))
        picks.append((r, c, cy, cx, color, y_val))
        rows_data.append(dict(
            row=r, col=c,
            cy=int(cy), cx=int(cx),
            box_y0=int(y0), box_x0=int(x0), box_y1=int(y1 - 1), box_x1=int(x1 - 1),
            Y=round(y_val, 2),
            R=color[2], G=color[1], B=color[0]
        ))

    # 5. 保存结果
    patch_png = os.path.join(a.out, "image_with_patches.png")
    cv2.imwrite(patch_png, img_out)

    # 一致性校验：重新读回落盘图像，逐块比对色块中心像素与记录值（BGR），
    # 不一致直接报错，保证图上色块与 patches.csv/json、dataset.json 完全对应。
    written = cv2.imread(patch_png, cv2.IMREAD_UNCHANGED)[:, :, :3]
    for (r, c, cy, cx, color, _) in picks:
        got = tuple(int(v) for v in written[cy, cx])
        if got != color:
            raise RuntimeError(
                f"色块不一致: 区域({r},{c}) 中心({cy},{cx}) "
                f"记录(B,G,R)={color} 图像(B,G,R)={got}")
    print(f"[check] {len(picks)} 个色块中心像素与 json 记录全部一致")
    
    # 色卡
    sep = np.full((8, S * len(ce_list), 3), 255, np.uint8)
    cv2.imwrite(os.path.join(a.out, "colorchart.png"), np.hstack(ce_list))
    
    # CSV
    with open(os.path.join(a.out, "patches.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_data[0].keys()))
        w.writeheader()
        w.writerows(rows_data)
    
    # JSON
    with open(os.path.join(a.out, "patches.json"), "w", encoding="utf-8") as f:
        json.dump(dict(
            grid=dict(rows=rows, cols=cols),
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
        "grid": {"rows": a.rows, "cols": a.cols},
        "items": items,
    }
    path = os.path.join(out_dir, a.dataset_json)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    return path


def main():
    p = argparse.ArgumentParser(
        description="网格分块采样：将图像分成 rows x cols 网格，每格中心生成纯色块")

    # 输入输出
    p.add_argument("--input", required=True, help="输入图片路径或目录")
    p.add_argument("--out", default="./out", help="输出根目录")

    # 网格参数
    p.add_argument("--rows", type=int, default=6, help="网格行数（默认 6）")
    p.add_argument("--cols", type=int, default=8, help="网格列数（默认 8）")
    p.add_argument("--size", type=int, default=100, help="色块边长（默认 100）")
    p.add_argument("--anchor", default="floor", choices=["floor", "ceil"],
                   help="偶数尺寸时中心像素偏左上(floor)或右下(ceil)")

    # 其他参数
    p.add_argument("--coef", default="709", choices=["709", "601"], help="亮度权重")
    p.add_argument("--stabilize", action="store_true", help="用 3x3 中值代替单像素")

    # 可视化选项
    p.add_argument("--outline", action="store_true", help="画红框")
    p.add_argument("--label", action="store_true", help="标注行列号")
    p.add_argument("--mark-center", action="store_true", help="中心画十字")

    # dataset.json 输出
    p.add_argument("--dataset-name", default="grid-samples", help="dataset.json 的 name 字段")
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
