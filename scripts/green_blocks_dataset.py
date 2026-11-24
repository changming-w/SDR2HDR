#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绿色色块位置 -> dataset.json
=============================
复用 green_center_blocks 的绿色检测 / 80% 亮度取色 / 横竖排布逻辑，对每张图:
    1) resize 到 3840x2160；
    2) 检测绿色像素，取"最亮绿色 x 80%"这一个亮度档的代表色(单一 rgb)；
    3) 生成竖排(even，count 块)与横排(rows，count 行 x cols 块)的色块图；
    4) 把每个色块的左上角 (x,y) 与该单一 rgb，按 dataset.json 格式写入一个 json。

所有色块同色(一个亮度档位)。image 字段统一写成 images/<文件名>，
方便你把生成的图片放到 images/ 下后直接用。

用法:
    python green_blocks_dataset.py --image /mnt/data/wcm/data/jiali \
        --out /mnt/data/wcm/data/jiali_dataset_100 --size 100
"""

import os
import sys
import re
import json
import argparse

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import green_center_blocks as g            # 复用检测/取色/排布


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def imwrite_u(path, img):
    """写图，兼容中文路径(cv2.imencode + tofile)。"""
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


def resize_to(bgr, tw, th):
    oh, ow = bgr.shape[:2]
    if (ow, oh) != (tw, th):
        interp = cv2.INTER_AREA if (tw * th) < (ow * oh) else cv2.INTER_CUBIC
        bgr = cv2.resize(bgr, (tw, th), interpolation=interp)
        print(f"[info]   resize {ow}x{oh} -> {tw}x{th}")
    return bgr


def make_stem(idx, filename):
    """规范化图片名前缀：{序号}_{原名开头数字}，无数字则仅用序号。无中文、无短横线。"""
    base = os.path.splitext(filename)[0]
    m = re.match(r"\D*(\d+)", base)
    num = m.group(1) if m else ""
    return f"{idx:03d}_{num}" if num else f"{idx:03d}"


def process_one(in_path, idx, args, items, save_dir):
    """处理单图：取代表色 + 生成横竖色块图 + 收集 dataset items。"""
    name = os.path.splitext(os.path.basename(in_path))[0]
    stem = make_stem(idx, os.path.basename(in_path))
    bgr = g.load_bgr01(in_path)
    bgr = resize_to(bgr, args.width, args.height)
    H, W = bgr.shape[:2]

    mask = g.green_mask(bgr, args.s_min, args.v_min, args.dom, args.h_lo, args.h_hi)
    color_bgr, info = g.pick_green_color(bgr, mask, args.pct, args.luma_ref,
                                         args.max_pct, args.luma_tol)
    R, G, B = color_bgr[2], color_bgr[1], color_bgr[0]
    rgb = [int(R), int(G), int(B)]
    print(f"[{idx:03d}] {name} -> {stem}  最亮绿 {info['brightest_green']} -> 80% 档 "
          f"target={info['target_luma']}  代表色 RGB={rgb}")

    img_u8 = np.clip(bgr * 255, 0, 255).astype(np.uint8)
    pid = f"{idx:03d}"

    # ---- 竖排(even):count 个块，水平居中，按高度均分 ----
    out_v, boxes_v = g.draw_blocks(img_u8, color_bgr, count=args.count,
                                   size=args.size, layout="even")
    v_img = f"{stem}_v.png"
    if not args.no_images:
        imwrite_u(os.path.join(save_dir, v_img), out_v)
    for i, (x0, y0, x1, y1) in enumerate(boxes_v):
        items.append({
            "image": f"images/{v_img}",
            "id": f"{pid}-vert-{i}",
            "type": "green-vert",
            "x": int(x0), "y": int(y0),
            "rgb": rgb,
        })

    # ---- 横排(rows):count 行(纵向均分)，每行 cols 个块 ----
    seg_h = H / args.count
    for r in range(args.count):
        yc = (r + 0.5) * seg_h
        out_h, boxes_h = g.draw_row(img_u8, color_bgr, args.cols, args.size, yc)
        h_img = f"{stem}_h{r}.png"
        if not args.no_images:
            imwrite_u(os.path.join(save_dir, h_img), out_h)
        for j, (x0, y0, x1, y1) in enumerate(boxes_h):
            items.append({
                "image": f"images/{h_img}",
                "id": f"{pid}-row{r}-{j}",
                "type": "green-row",
                "x": int(x0), "y": int(y0),
                "rgb": rgb,
            })

    return color_bgr


def main():
    p = argparse.ArgumentParser(description="绿色色块横竖排布 -> dataset.json(单一亮度档)")
    p.add_argument("--image", required=True, help="输入图片目录(或单张图片)")
    p.add_argument("--out", default=None, help="输出目录(含 images/ 与 dataset.json)")
    p.add_argument("--json-name", default="dataset.json", help="输出 json 文件名")
    p.add_argument("--name", default="green-scenes-sdr", help="数据集名称")
    p.add_argument("--count", type=int, default=6, help="竖排块数 / 横排行数(默认 6)")
    p.add_argument("--cols", type=int, default=8, help="横排每行块数(默认 8)")
    p.add_argument("--size", type=int, default=100, help="色块边长(默认 100)")
    p.add_argument("--width", type=int, default=3840, help="处理前 resize 宽")
    p.add_argument("--height", type=int, default=2160, help="处理前 resize 高")
    p.add_argument("--no-images", action="store_true", help="只出 dataset.json，不生成图片")
    # 亮度档 / 绿色判据(默认与 green_center_blocks 一致：最亮绿的 80%)
    p.add_argument("--pct", type=float, default=80.0)
    p.add_argument("--luma-ref", default="maxrel", choices=["maxrel", "range", "pct"])
    p.add_argument("--max-pct", type=float, default=99.5)
    p.add_argument("--luma-tol", type=float, default=6.0)
    p.add_argument("--s-min", type=float, default=0.20)
    p.add_argument("--v-min", type=float, default=0.02)
    p.add_argument("--dom", type=float, default=0.03)
    p.add_argument("--h-lo", type=int, default=35)
    p.add_argument("--h-hi", type=int, default=90)
    args = p.parse_args()

    if os.path.isdir(args.image):
        files = sorted(f for f in os.listdir(args.image)
                       if os.path.splitext(f)[1].lower() in IMAGE_EXTS
                       and "_greenblocks" not in f and "_swatch" not in f
                       and not f.endswith("_v.png")
                       and "_h" not in os.path.splitext(f)[0][-3:])
        in_paths = [os.path.join(args.image, f) for f in files]
        default_out = os.path.normpath(args.image) + "_dataset"
    else:
        in_paths = [args.image]
        default_out = os.path.splitext(args.image)[0] + "_dataset"

    if not in_paths:
        raise SystemExit(f"[error] 未找到图片: {args.image}")

    out_dir = args.out or default_out
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    print(f"[info] 处理 {len(in_paths)} 张 -> {out_dir}  "
          f"(size={args.size}, 竖排 {args.count} 块 / 横排 {args.count} 行 x {args.cols} 块)")

    items, ok, failed = [], 0, []
    for i, in_path in enumerate(in_paths, 1):
        try:
            process_one(in_path, i, args, items, img_dir)
            ok += 1
        except SystemExit as e:
            print(f"[skip] {os.path.basename(in_path)}: {e}")
            failed.append(os.path.basename(in_path))
        except Exception as e:
            print(f"[error] {os.path.basename(in_path)}: {e}")
            failed.append(os.path.basename(in_path))

    dataset = {
        "version": 1,
        "name": args.name,
        "width": args.width,
        "height": args.height,
        "patch_size": args.size,
        "items": items,
    }
    json_path = os.path.join(out_dir, args.json_name)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    n_img = 0 if args.no_images else ok * (1 + args.count)
    print(f"\n[done] 成功 {ok}/{len(in_paths)} 张，items {len(items)} 条 "
          f"(每张 {args.count} 竖 + {args.count}x{args.cols} 横 = {args.count + args.count*args.cols})")
    print(f"[out]  dataset.json -> {json_path}")
    if not args.no_images:
        print(f"[out]  图片({n_img} 张)-> {img_dir}")
    if failed:
        print(f"[fail] {len(failed)} 张: {', '.join(failed)}")


if __name__ == "__main__":
    main()
