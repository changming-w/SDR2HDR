#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结合景深信息应用 3D LUT。

越近的区域 LUT 效果越强，越远的区域越接近原图。
使用 MiDaS DPT_Hybrid 估计景深。

用法：
    # 单张图
    python scripts/apply_lut_depth.py \
        --input /path/to/image.png \
        --lut /path/to/lut.npy \
        --output /path/to/output.png

    # 批量处理
    python scripts/apply_lut_depth.py \
        --input /path/to/images/ \
        --lut /path/to/lut.npy \
        --output /path/to/output_dir/

    # 调整参数
    python scripts/apply_lut_depth.py \
        --input /path/to/images/ \
        --lut /path/to/lut.npy \
        --output /path/to/output_dir/ \
        --near-strength 1.0 --far-strength 0.0 \
        --depth-blur 21 --gamma 1.0
"""

import os
import glob
import argparse

import cv2
import numpy as np
import torch


# ==================== 3D LUT 三线性插值 ====================
def apply_3dlut(img_rgb, lut):
    """
    对图像应用 3D LUT（三线性插值）。

    Args:
        img_rgb: (H, W, 3) float64, [0, 1], RGB
        lut: (S, S, S, 3) float64, LUT 数据

    Returns:
        out: (H, W, 3) float64, [0, 1], RGB
    """
    S = lut.shape[0]
    H, W, _ = img_rgb.shape

    # 映射到 LUT 坐标
    pos = np.clip(img_rgb, 0.0, 1.0) * (S - 1)
    base = np.floor(pos).astype(np.int32)
    base = np.minimum(base, S - 2)
    frac = pos - base

    # 8 个顶点的权重
    r0, g0, b0 = base[..., 0], base[..., 1], base[..., 2]
    r1, g1, b1 = r0 + 1, g0 + 1, b0 + 1
    fr, fg, fb = frac[..., 0], frac[..., 1], frac[..., 2]

    # 三线性插值
    w000 = (1 - fr) * (1 - fg) * (1 - fb)
    w001 = (1 - fr) * (1 - fg) * fb
    w010 = (1 - fr) * fg * (1 - fb)
    w011 = (1 - fr) * fg * fb
    w100 = fr * (1 - fg) * (1 - fb)
    w101 = fr * (1 - fg) * fb
    w110 = fr * fg * (1 - fb)
    w111 = fr * fg * fb

    out = (
        lut[r0, g0, b0] * w000[..., None] +
        lut[r0, g0, b1] * w001[..., None] +
        lut[r0, g1, b0] * w010[..., None] +
        lut[r0, g1, b1] * w011[..., None] +
        lut[r1, g0, b0] * w100[..., None] +
        lut[r1, g0, b1] * w101[..., None] +
        lut[r1, g1, b0] * w110[..., None] +
        lut[r1, g1, b1] * w111[..., None]
    )

    return np.clip(out, 0.0, 1.0)


# ==================== MiDaS 景深估计 ====================
def load_midas(device="cpu", model_type="DPT_Hybrid"):
    """加载 MiDaS 模型。"""
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    hub_dir = os.path.expanduser("~/.cache/torch/hub")
    midas_repo = os.path.join(hub_dir, "isl-org_MiDaS_master")
    if not os.path.isdir(midas_repo):
        alt_repo = os.path.join(hub_dir, "intel-isl_MiDaS_master")
        if os.path.isdir(alt_repo):
            midas_repo = alt_repo
        else:
            raise RuntimeError(f"MiDaS 仓库未找到: {hub_dir}")

    model = torch.hub.load(midas_repo, model_type, source="local")
    model.to(device)
    model.eval()

    midas_transforms = torch.hub.load(midas_repo, "transforms", source="local")
    if model_type in ("DPT_Large", "DPT_Hybrid"):
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    return model, transform, device


def estimate_depth(model, transform, device, img_bgr):
    """
    估计景深。

    Returns:
        depth_norm: (H, W) float64, [0, 1], 大值=近
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
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

    # 归一化到 [0, 1]
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-10)
    return depth_norm


# ==================== 主流程 ====================
def process_image(img_path, lut, depth_model, depth_transform, depth_device,
                  near_strength=1.0, far_strength=0.0,
                  depth_blur=21, gamma=1.0, save_depth=False, output_dir=None,
                  color_viz=False):
    """
    处理单张图片。

    Args:
        near_strength: 近处 LUT 强度 (0-1), 1.0 = 全 LUT
        far_strength:  远处 LUT 强度 (0-1), 0.0 = 全原图
        depth_blur:    景深图高斯模糊核大小 (奇数), 0=不模糊
        gamma:         景深 gamma 校正, >1 让近处影响更大, <1 让远处影响更大
        color_viz:     景深/权重图用彩色显示
    """
    img_bgr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img_bgr is None:
        raise FileNotFoundError(img_path)

    # 读取为 float64 RGB [0, 1]
    if img_bgr.dtype == np.uint8:
        img_f = img_bgr.astype(np.float64) / 255.0
    elif img_bgr.dtype == np.uint16:
        denom = 1023.0 if img_bgr.max() <= 1023 else 65535.0
        img_f = img_bgr.astype(np.float64) / denom
    else:
        img_f = np.clip(img_bgr.astype(np.float64), 0.0, 1.0)

    img_rgb = img_f[:, :, :3]
    # BGR -> RGB
    img_rgb = img_rgb[:, :, ::-1]

    H, W = img_rgb.shape[:2]

    # 1. 估计景深
    print(f"  估计景深...")
    img_bgr_u8 = np.clip(img_f * 255, 0, 255).astype(np.uint8)
    depth_norm = estimate_depth(depth_model, depth_transform, depth_device, img_bgr_u8)

    # 景深模糊（让过渡更自然）
    if depth_blur > 0 and depth_blur % 2 == 1:
        depth_norm = cv2.GaussianBlur(depth_norm, (depth_blur, depth_blur), 0)

    # Gamma 校正
    if gamma != 1.0:
        depth_norm = np.power(depth_norm, gamma)

    # 2. 应用 3D LUT
    print(f"  应用 3D LUT...")
    lut_result = apply_3dlut(img_rgb, lut)

    # 3. 按景深融合
    # depth_norm: 大值=近, 小值=远
    # 近处: alpha -> near_strength
    # 远处: alpha -> far_strength
    alpha = depth_norm  # [0, 1], 大=近
    # 混合权重: far_strength + (near_strength - far_strength) * alpha
    weight = far_strength + (near_strength - far_strength) * alpha
    weight = weight[:, :, None]  # (H, W, 1)

    # 融合: weight * lut_result + (1 - weight) * original
    blended = weight * lut_result + (1.0 - weight) * img_rgb
    blended = np.clip(blended, 0.0, 1.0)

    # 转回 BGR uint8
    out_bgr = (blended[:, :, ::-1] * 255).astype(np.uint8)

    # 保存结果到不同子文件夹，文件名保持不变
    basename = os.path.basename(img_path)

    results = {"output": out_bgr}

    if output_dir:
        out_dir = output_dir
        # 输出子文件夹
        result_dir = os.path.join(out_dir, "result")
        depth_dir = os.path.join(out_dir, "depth")
        weight_dir = os.path.join(out_dir, "weight")
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(weight_dir, exist_ok=True)

        # 结果图（保持原文件名）
        out_path = os.path.join(result_dir, basename)
        cv2.imwrite(out_path, out_bgr)
        results["output_path"] = out_path

        if save_depth:
            # 景深图
            depth_u8 = (depth_norm * 255).astype(np.uint8)
            if color_viz:
                depth_out = cv2.applyColorMap(depth_u8, cv2.COLORMAP_MAGMA)
            else:
                depth_out = depth_u8  # 灰度，白=近，黑=远
            depth_path = os.path.join(depth_dir, basename)
            cv2.imwrite(depth_path, depth_out)
            results["depth_path"] = depth_path

            # 权重图
            weight_u8 = (np.clip(weight.squeeze(), 0, 1) * 255).astype(np.uint8)
            if color_viz:
                weight_out = cv2.applyColorMap(weight_u8, cv2.COLORMAP_JET)
            else:
                weight_out = weight_u8  # 灰度，白=LUT强，黑=原图多
            weight_path = os.path.join(weight_dir, basename)
            cv2.imwrite(weight_path, weight_out)
            results["weight_path"] = weight_path

    return results


def main():
    parser = argparse.ArgumentParser(
        description="结合景深信息应用 3D LUT，近处效果强，远处效果弱")

    parser.add_argument("--input", required=True,
                        help="输入图片路径或目录")
    parser.add_argument("--lut", required=True,
                        help="3D LUT .npy 文件路径 (shape: [S,S,S,3])")
    parser.add_argument("--output", default=None,
                        help="输出目录，默认在输入目录下创建 lut_depth/")

    parser.add_argument("--near-strength", type=float, default=1.0,
                        help="近处 LUT 强度 (0-1), 默认 1.0 = 全 LUT")
    parser.add_argument("--far-strength", type=float, default=0.0,
                        help="远处 LUT 强度 (0-1), 默认 0.0 = 全原图")
    parser.add_argument("--depth-blur", type=int, default=21,
                        help="景深图高斯模糊核大小 (奇数), 0=不模糊, 默认 21")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="景深 gamma 校正, >1 近处影响更大, <1 远处影响更大, 默认 1.0")

    parser.add_argument("--depth-model", default="DPT_Hybrid",
                        choices=["DPT_Large", "DPT_Hybrid", "MiDaS_small"],
                        help="景深估计模型, 默认 DPT_Hybrid")
    parser.add_argument("--device", default="cuda",
                        help="设备: cpu 或 cuda")
    parser.add_argument("--save-depth", action="store_true",
                        help="同时保存景深图和权重可视化")
    parser.add_argument("--color-viz", action="store_true",
                        help="景深/权重图用彩色显示（默认灰度）")

    args = parser.parse_args()

    # 加载 LUT
    print(f"加载 3D LUT: {args.lut}")
    lut = np.load(args.lut)
    print(f"  LUT shape: {lut.shape}, dtype: {lut.dtype}")
    assert lut.shape[-1] == 3, f"LUT 最后一维必须是 3 (RGB), 得到 {lut.shape[-1]}"
    assert lut.ndim == 4, f"LUT 必须是 4 维 (S,S,S,3), 得到 {lut.ndim} 维"
    S = lut.shape[0]
    print(f"  LUT 尺寸: {S}x{S}x{S}")

    # 加载景深模型
    print(f"加载 {args.depth_model} 景深模型...")
    depth_model, depth_transform, depth_device = load_midas(args.device, args.depth_model)
    print(f"  设备: {depth_device}")

    # 收集图像
    input_path = args.input
    if os.path.isdir(input_path):
        exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(input_path, ext)))
            files.extend(glob.glob(os.path.join(input_path, ext.upper())))
        files = sorted(set(files))
        output_dir = args.output or os.path.join(input_path.rstrip("/"), "lut_depth")
    else:
        files = [input_path]
        output_dir = args.output or os.path.join(os.path.dirname(input_path), "lut_depth")

    os.makedirs(output_dir, exist_ok=True)
    print(f"找到 {len(files)} 张图像")
    print(f"输出目录: {output_dir}")
    print(f"  result/  - LUT+景深融合结果")
    if args.save_depth:
        print(f"  depth/   - 景深可视化")
        print(f"  weight/  - 混合权重可视化")
    print(f"参数: near={args.near_strength}, far={args.far_strength}, "
          f"blur={args.depth_blur}, gamma={args.gamma}")

    for i, img_path in enumerate(files):
        try:
            print(f"\n[{i+1}/{len(files)}] {os.path.basename(img_path)}")
            results = process_image(
                img_path, lut, depth_model, depth_transform, depth_device,
                near_strength=args.near_strength,
                far_strength=args.far_strength,
                depth_blur=args.depth_blur,
                gamma=args.gamma,
                save_depth=args.save_depth,
                output_dir=output_dir,
                color_viz=args.color_viz,
            )
            print(f"  -> result/{os.path.basename(img_path)}")
            if args.save_depth:
                print(f"  -> depth/{os.path.basename(img_path)}")
                print(f"  -> weight/{os.path.basename(img_path)}")
        except Exception as e:
            print(f"  跳过: {e}")
            continue

    print(f"\n完成！结果保存到: {output_dir}")
    print(f"  result/  - {len(files)} 张融合结果")
    if args.save_depth:
        print(f"  depth/   - {len(files)} 张景深图")
        print(f"  weight/  - {len(files)} 张权重图")


if __name__ == "__main__":
    main()
