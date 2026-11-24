#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绿色色块位置 + 深度值 -> dataset.json
======================================
在 green_blocks_dataset.py 基础上，为每个色块坐标附加深度值(使用 MiDaS DPT_Hybrid 估计)。

深度值说明:
    - depth: 原始深度值(视差，大值=近)
    - depth_norm: 归一化到 0-1 的深度值(0=远, 1=近)

用法:
    python green_blocks_dataset_with_depth.py --image /mnt/data/wcm/data/jiali \
        --out /mnt/data/wcm/data/jiali_dataset_100 --size 100
"""

import os
import sys
import re
import json
import argparse

import numpy as np
import cv2
import torch

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


# =============================================================================
# MiDaS 深度估计
# =============================================================================
def load_midas_model(device_str="cpu", model_type="DPT_Hybrid"):
    """加载 MiDaS 模型和变换。"""
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    hub_dir = os.path.expanduser("~/.cache/torch/hub")
    midas_repo = os.path.join(hub_dir, "isl-org_MiDaS_master")
    if not os.path.isdir(midas_repo):
        alt_repo = os.path.join(hub_dir, "intel-isl_MiDaS_master")
        if os.path.isdir(alt_repo):
            midas_repo = alt_repo
        else:
            raise RuntimeError(f"Local MiDaS repo not found under {hub_dir}.")

    model = torch.hub.load(midas_repo, model_type, source="local")
    model.to(device)
    model.eval()

    midas_transforms = torch.hub.load(midas_repo, "transforms", source="local")
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    return model, transform, device


def estimate_depth(model, transform, device, bgr_img):
    """
    对单张 BGR 图像做景深估计。

    Returns:
        depth: [H, W] float64 原始深度值（视差，大值=近）
        depth_norm: [H, W] float64 归一化到 0-1 的深度值(0=远, 1=近)
    """
    img_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    input_batch = transform(img_rgb).to(device)
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(H, W),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()

    # 归一化到 0-1
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-10)

    return depth, depth_norm


def get_depth_at_point(depth_norm, x, y, patch_size):
    """获取色块区域的平均深度值。

    Args:
        depth_norm: [H, W] 归一化深度图
        x, y: 色块左上角坐标
        patch_size: 色块边长

    Returns:
        float: 该区域的平均深度值(0-1)
    """
    H, W = depth_norm.shape
    # 取色块中心区域(避免边缘)
    cx, cy = x + patch_size // 2, y + patch_size // 2
    r = patch_size // 4  # 取中心 1/4 区域

    x0 = max(0, cx - r)
    x1 = min(W, cx + r)
    y0 = max(0, cy - r)
    y1 = min(H, cy + r)

    if x0 >= x1 or y0 >= y1:
        # fallback: 取单点
        return float(depth_norm[min(cy, H-1), min(cx, W-1)])

    return float(np.mean(depth_norm[y0:y1, x0:x1]))


# =============================================================================
# 主处理逻辑
# =============================================================================
def process_one(in_path, idx, args, items, save_dir, depth_model, depth_transform, depth_device):
    """处理单图：取代表色 + 深度估计 + 生成横竖色块图 + 收集 dataset items。"""
    name = os.path.splitext(os.path.basename(in_path))[0]
    stem = make_stem(idx, os.path.basename(in_path))
    bgr = g.load_bgr01(in_path)
    bgr = resize_to(bgr, args.width, args.height)
    H, W = bgr.shape[:2]

    # 绿色检测
    mask = g.green_mask(bgr, args.s_min, args.v_min, args.dom, args.h_lo, args.h_hi)
    color_bgr, info = g.pick_green_color(bgr, mask, args.pct, args.luma_ref,
                                         args.max_pct, args.luma_tol)
    R, G, B = color_bgr[2], color_bgr[1], color_bgr[0]
    rgb = [int(R), int(G), int(B)]
    print(f"[{idx:03d}] {name} -> {stem}  最亮绿 {info['brightest_green']} -> 80% 档 "
          f"target={info['target_luma']}  代表色 RGB={rgb}")

    # 深度估计
    print(f"[{idx:03d}] 估计深度图...")
    img_u8 = np.clip(bgr * 255, 0, 255).astype(np.uint8)
    _, depth_norm = estimate_depth(depth_model, depth_transform, depth_device, bgr)
    print(f"[{idx:03d}] 深度范围: [{depth_norm.min():.3f}, {depth_norm.max():.3f}]")

    pid = f"{idx:03d}"

    # ---- 竖排(even):count 个块，水平居中，按高度均分 ----
    out_v, boxes_v = g.draw_blocks(img_u8, color_bgr, count=args.count,
                                   size=args.size, layout="even")
    v_img = f"{stem}_v.png"
    if not args.no_images:
        imwrite_u(os.path.join(save_dir, v_img), out_v)
    for i, (x0, y0, x1, y1) in enumerate(boxes_v):
        depth_val = get_depth_at_point(depth_norm, x0, y0, args.size)
        items.append({
            "image": f"images/{v_img}",
            "id": f"{pid}-vert-{i}",
            "type": "green-vert",
            "x": int(x0), "y": int(y0),
            "depth": round(depth_val, 4),
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
            depth_val = get_depth_at_point(depth_norm, x0, y0, args.size)
            items.append({
                "image": f"images/{h_img}",
                "id": f"{pid}-row{r}-{j}",
                "type": "green-row",
                "x": int(x0), "y": int(y0),
                "depth": round(depth_val, 4),
                "rgb": rgb,
            })

    return color_bgr


def main():
    p = argparse.ArgumentParser(description="绿色色块横竖排布 + 深度值 -> dataset.json")
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

    # 深度模型参数
    p.add_argument("--depth-model", default="DPT_Hybrid",
                   choices=["DPT_Large", "DPT_Hybrid", "MiDaS_small"],
                   help="深度模型类型")
    p.add_argument("--device", default="cpu", help="深度估计设备：cpu 或 cuda")

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

    # 加载深度模型
    print(f"[info] 加载 {args.depth_model} 深度模型...")
    depth_model, depth_transform, depth_device = load_midas_model(args.device, args.depth_model)
    print(f"[info] 深度设备: {depth_device}")

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
            process_one(in_path, i, args, items, img_dir,
                        depth_model, depth_transform, depth_device)
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
