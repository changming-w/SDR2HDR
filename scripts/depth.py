#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 DPT_Hybrid (MiDaS) 批量生成景深图。

环境要求：torch, torchvision, timm
MiDaS 仓库缓存：~/.cache/torch/hub/isl-org_MiDaS_master

用法：
    # 单张图
    python run_dpt_depth.py --input /path/to/image.png

    # 批量（目录下所有图片）
    python run_dpt_depth.py --input /path/to/images/ --output /path/to/depth/
"""

import os
import glob
import argparse

import cv2
import numpy as np
import torch


def load_dpt_hybrid(device="cpu", model_type="DPT_Hybrid"):
    """加载 MiDaS 模型和变换。

    model_type: "DPT_Large" | "DPT_Hybrid" | "MiDaS_small"
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

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


def estimate_depth(model, transform, device, img_path, resize_to=None):
    """
    对单张图像做景深估计。

    Args:
        resize_to: 元组 (W, H) 或 None，处理前是否缩放图像

    Returns:
        depth: [H, W] float64 原始深度值（视差，大值=近）
        depth_u8: [H, W] uint8 归一化到 0-255 的灰度图
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {img_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 如果指定了缩放分辨率，先缩放
    if resize_to is not None:
        w, h = resize_to
        img_rgb = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_LANCZOS4)

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

    # 归一化到 0-255
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-10)
    depth_u8 = (depth_norm * 255).astype(np.uint8)

    return depth, depth_u8


def main():
    parser = argparse.ArgumentParser(description="MiDaS 批量景深估计")
    parser.add_argument("--input", required=True, help="输入图像路径或目录")
    parser.add_argument("--output", default=None, help="输出目录，默认在输入目录下创建 depth/")
    parser.add_argument("--device", default="cpu", help="设备：cpu 或 cuda")
    parser.add_argument("--model", default="DPT_Hybrid",
                        choices=["DPT_Large", "DPT_Hybrid", "MiDaS_small"],
                        help="模型类型：DPT_Large(最大最精确) | DPT_Hybrid(默认) | MiDaS_small(最小最快)")
    parser.add_argument("--resize", default=None, help="缩放分辨率，格式 WxH，如 3840x2160；不设置则按原分辨率")
    args = parser.parse_args()

    # 加载模型
    print(f"加载 {args.model} 模型...")
    model, transform, device = load_dpt_hybrid(args.device, args.model)
    print(f"  设备: {device}")

    # 收集图像
    input_path = args.input
    if os.path.isdir(input_path):
        exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(input_path, ext)))
            files.extend(glob.glob(os.path.join(input_path, ext.upper())))
        files = sorted(set(files))
        output_dir = args.output or os.path.join(input_path, "depth")
    else:
        files = [input_path]
        if args.output:
            output_dir = args.output
        else:
            output_dir = os.path.join(os.path.dirname(input_path), "depth")

    os.makedirs(output_dir, exist_ok=True)
    print(f"找到 {len(files)} 张图像")
    print(f"输出目录: {output_dir}")

    # 解析缩放分辨率
    resize_to = None
    if args.resize:
        parts = args.resize.lower().split('x')
        if len(parts) == 2:
            try:
                resize_to = (int(parts[0]), int(parts[1]))
                print(f"缩放分辨率: {resize_to}")
            except ValueError:
                print(f"警告：无法解析 --resize '{args.resize}'，将按原分辨率处理")

    for i, img_path in enumerate(files):
        try:
            depth, depth_u8 = estimate_depth(model, transform, device, img_path, resize_to=resize_to)
        except Exception as e:
            print(f"[{i+1}/{len(files)}] 跳过 {os.path.basename(img_path)}: {e}")
            continue

        out_name = os.path.splitext(os.path.basename(img_path))[0] + "_depth.png"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, depth_u8)

        print(f"[{i+1}/{len(files)}] {os.path.basename(img_path)} -> {out_name} "
              f"| depth=[{depth.min():.2f}, {depth.max():.2f}]")

    print("完成！")


if __name__ == "__main__":
    main()
