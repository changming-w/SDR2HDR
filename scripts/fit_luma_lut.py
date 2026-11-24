#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
亮度 1D LUT 拟合脚本（RGB 等比例缩放）。

与 fit_3dlut.py 的区别：
    - fit_3dlut.py：拟合完整 3D LUT，R/G/B 三通道独立映射，可改变色相/饱和度
    - 本脚本：只拟合亮度（Y）→ 亮度缩放因子，RGB 等比例乘同一个系数，不改变色相

原理：
    1. 将 src/dst 图片转为 YUV，提取 Y 通道
    2. 对每个亮度 bin，计算 dst_Y / src_Y 的平均比值作为该亮度的缩放因子
    3. 输出 1D LUT：输入 Y → 输出 scale factor
    4. 应用时：out_RGB = src_RGB * lut(src_Y)

用法：
    python scripts/fit_luma_lut.py \
        --src /path/to/src_images \
        --dst /path/to/dst_images \
        --out fitted_luma_lut \
        --bins 256
"""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp",
    ".tif", ".tiff", ".webp", ".exr"
}


def scan_images(root):
    """以不包含扩展名的相对路径作为匹配键。"""
    root = Path(root)

    if not root.exists():
        raise RuntimeError(f"目录不存在：{root}")

    if not root.is_dir():
        raise RuntimeError(f"路径不是目录：{root}")

    images = {}

    for image_path in root.rglob("*"):
        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        key = (
            image_path
            .relative_to(root)
            .with_suffix("")
            .as_posix()
            .lower()
        )

        if key in images:
            raise RuntimeError(
                f"目录中存在相同相对路径、不同扩展名的重复图片：\n"
                f"  {images[key]}\n"
                f"  {image_path}"
            )

        images[key] = image_path

    return images


def find_image_pairs(src_dir, dst_dir):
    src_images = scan_images(src_dir)
    dst_images = scan_images(dst_dir)

    common_keys = sorted(set(src_images) & set(dst_images))

    if not common_keys:
        raise RuntimeError(
            "没有找到可匹配的图片。\n"
            "请确保调色前后图片具有相同的相对路径和文件名，"
            "扩展名可以不同。"
        )

    src_only = sorted(set(src_images) - set(dst_images))
    dst_only = sorted(set(dst_images) - set(src_images))

    print(f"调色前图片数量：{len(src_images)}")
    print(f"调色后图片数量：{len(dst_images)}")
    print(f"成功匹配图片数量：{len(common_keys)}")

    if src_only:
        print(f"警告：有 {len(src_only)} 张调色前图片没有匹配结果。")

    if dst_only:
        print(f"警告：有 {len(dst_only)} 张调色后图片没有匹配结果。")

    return [(src_images[key], dst_images[key]) for key in common_keys]


def read_rgb_float(path):
    """读取图片为 float64 RGB [0, 1]。"""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise RuntimeError(f"无法读取图片：{path}")

    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)

    if image.ndim != 3:
        raise RuntimeError(f"不支持的图片维度：{path}, shape={image.shape}")

    channels = image.shape[2]

    if channels == 1:
        image = np.repeat(image, 3, axis=2)
    elif channels == 3:
        image = image[..., ::-1]  # BGR -> RGB
    elif channels == 4:
        image = image[..., :3][..., ::-1]  # BGRA -> RGB
    else:
        raise RuntimeError(f"不支持的图片通道数：{path}, shape={image.shape}")

    if np.issubdtype(image.dtype, np.integer):
        max_value = np.iinfo(image.dtype).max
        image = image.astype(np.float64) / float(max_value)
    elif np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float64)
        finite_mask = np.isfinite(image)
        if not finite_mask.any():
            raise RuntimeError(f"图片中不存在有效浮点像素：{path}")
        image = np.clip(image, 0.0, 1.0)
        if (~finite_mask).any():
            image[~finite_mask] = np.nan
        return image
    else:
        raise RuntimeError(f"不支持的图片数据类型：{path}, dtype={image.dtype}")

    return np.clip(image, 0.0, 1.0)


def rgb_to_y_bt709(rgb):
    """BT.709 亮度公式：Y = 0.2126*R + 0.7152*G + 0.0722*B"""
    return (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    )


def collect_luma_samples(pairs, max_samples, seed):
    """收集 (src_Y, dst_Y) 像素对。"""
    rng = np.random.default_rng(seed)

    per_image_limit = math.ceil(max_samples / len(pairs))

    all_src_y = []
    all_dst_y = []

    for index, (src_path, dst_path) in enumerate(pairs, start=1):
        print(f"[{index}/{len(pairs)}] 读取：{src_path.name} -> {dst_path.name}")

        src = read_rgb_float(src_path)
        dst = read_rgb_float(dst_path)

        if src.shape != dst.shape:
            raise RuntimeError(
                f"图片尺寸不一致：\n"
                f"  输入：{src_path}, shape={src.shape}\n"
                f"  输出：{dst_path}, shape={dst.shape}"
            )

        src_y = rgb_to_y_bt709(src).ravel()
        dst_y = rgb_to_y_bt709(dst).ravel()

        pixel_count = src_y.shape[0]
        take_count = min(pixel_count, per_image_limit)

        if take_count < pixel_count:
            indices = rng.choice(pixel_count, size=take_count, replace=False)
            src_y = src_y[indices]
            dst_y = dst_y[indices]

        valid = np.isfinite(src_y) & np.isfinite(dst_y)
        all_src_y.append(src_y[valid])
        all_dst_y.append(dst_y[valid])

    src_y_all = np.concatenate(all_src_y)
    dst_y_all = np.concatenate(all_dst_y)

    if src_y_all.shape[0] == 0:
        raise RuntimeError("没有收集到有效像素，无法拟合。")

    print(f"候选像素数量：{src_y_all.shape[0]}")

    # 如果超过 max_samples，随机采样
    if src_y_all.shape[0] > max_samples:
        indices = rng.choice(src_y_all.shape[0], size=max_samples, replace=False)
        src_y_all = src_y_all[indices]
        dst_y_all = dst_y_all[indices]

    print(f"最终拟合像素数量：{src_y_all.shape[0]}")

    return src_y_all, dst_y_all


def fit_luma_lut(src_y, dst_y, bins, smooth_weight, clip_output):
    """
    拟合亮度 1D LUT。

    对每个亮度 bin，计算 dst_Y / src_Y 的加权平均比值。
    输出 shape: (bins,)，表示输入 Y ∈ [0,1] → 缩放因子 scale。

    应用方式：out_RGB = src_RGB * lut(round(src_Y * (bins-1)))
    """
    print(f"\n拟合 {bins}-bin 亮度 LUT……")

    # 将 src_Y 量化到 bin
    bin_indices = np.minimum(
        (np.clip(src_y, 0.0, 1.0) * (bins - 1)).astype(np.int32),
        bins - 1
    )

    # 避免除以零：src_Y 极小时用加法模型
    epsilon = 1e-6

    # 每个 bin 累加 dst_Y 和 src_Y
    sum_dst = np.zeros(bins, dtype=np.float64)
    sum_src = np.zeros(bins, dtype=np.float64)
    count = np.zeros(bins, dtype=np.float64)

    np.add.at(sum_dst, bin_indices, dst_y)
    np.add.at(sum_src, bin_indices, src_y)
    np.add.at(count, bin_indices, 1.0)

    # 计算每个 bin 的平均缩放因子
    # scale = mean(dst_Y) / mean(src_Y) = sum_dst / sum_src
    lut = np.ones(bins, dtype=np.float64)  # 默认 identity

    valid_bins = sum_src > epsilon
    lut[valid_bins] = sum_dst[valid_bins] / sum_src[valid_bins]

    # 对于没有数据的 bin，用最近邻插值填充
    if not valid_bins.all():
        from scipy.ndimage import uniform_filter1d
        # 先用线性插值填充空洞
        filled = lut.copy()
        invalid = ~valid_bins
        if valid_bins.any():
            # 用有效值的均值填充无效区域
            valid_indices = np.where(valid_bins)[0]
            invalid_indices = np.where(invalid)[0]
            for idx in invalid_indices:
                nearest = valid_indices[np.argmin(np.abs(valid_indices - idx))]
                filled[idx] = lut[nearest]
        lut = filled

    # 平滑
    if smooth_weight > 0:
        from scipy.ndimage import gaussian_filter1d
        sigma = smooth_weight * bins / 256.0
        lut = gaussian_filter1d(lut, sigma=sigma, mode='nearest')
        print(f"高斯平滑 sigma={sigma:.4f}")

    # 打印统计
    print(f"LUT 范围：[{lut.min():.6f}, {lut.max():.6f}]")
    print(f"LUT 均值：{lut.mean():.6f}")

    # 可选裁剪
    if clip_output:
        lut = np.clip(lut, 0.0, 10.0)  # 缩放因子不应太大
        print("已裁剪到 [0, 10]")

    # 计算拟合误差
    predicted_scale = lut[bin_indices]
    predicted_dst_y = src_y * predicted_scale
    error = predicted_dst_y - dst_y

    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))

    print(f"\n拟合结果：")
    print(f"  MAE：  {mae:.8f}，约 {mae * 255.0:.4f}/255")
    print(f"  RMSE： {rmse:.8f}，约 {rmse * 255.0:.4f}/255")

    return lut


def write_1d_cube(path, lut, title="Fitted Luma 1D LUT"):
    """写入 1D LUT 的 .cube 文件。"""
    path = Path(path)
    bins = len(lut)

    safe_title = str(title).replace('"', "'")

    with path.open("w", encoding="utf-8") as f:
        f.write(f'TITLE "{safe_title}"\n')
        f.write(f"LUT_1D_SIZE {bins}\n")
        f.write("DOMAIN_MIN 0.0\n")
        f.write("DOMAIN_MAX 1.0\n")

        for i in range(bins):
            # 1D LUT: R=G=B=scale，因为是等比例缩放
            f.write(f"{lut[i]:.17g} {lut[i]:.17g} {lut[i]:.17g}\n")


def luma_lut_to_3d(lut_1d, size=33):
    """
    将 1D 亮度缩放 LUT 转换为对角线 3D LUT。

    原理：RGB 等比例缩放意味着 out_R = R * scale(Y),
    其中 Y = 0.2126*R + 0.7152*G + 0.0722*B。
    3D LUT 在每个 (R,G,B) 网格点上预计算该点的 Y，
    然后查 1D LUT 得到 scale，输出 = (R*scale, G*scale, B*scale)。

    Args:
        lut_1d: shape (bins,) 的 1D 亮度缩放因子
        size: 3D LUT 每轴节点数

    Returns:
        shape (size, size, size, 3) 的 3D LUT
    """
    axis = np.linspace(0.0, 1.0, size, dtype=np.float64)
    r_grid, g_grid, b_grid = np.meshgrid(axis, axis, axis, indexing='ij')

    # BT.709 亮度
    y_grid = 0.2126 * r_grid + 0.7152 * g_grid + 0.0722 * b_grid

    # 用三线性插值从 1D LUT 查表
    from scipy.interpolate import interp1d
    bins = len(lut_1d)
    x_1d = np.linspace(0.0, 1.0, bins, dtype=np.float64)
    interpolator = interp1d(x_1d, lut_1d, kind='linear',
                            bounds_error=False,
                            fill_value=(lut_1d[0], lut_1d[-1]))

    scale = interpolator(y_grid.ravel()).reshape(y_grid.shape)

    lut_3d = np.stack([
        r_grid * scale,
        g_grid * scale,
        b_grid * scale,
    ], axis=-1)

    return lut_3d.astype(np.float64)


def write_3d_cube(path, lut, title="Fitted Luma 3D LUT"):
    """
    写入标准 3D LUT .cube 文件。

    Numpy LUT 排列：lut[R, G, B, output_channel]
    .cube 文件排列：R 索引变化最快；G 次之；B 最慢。
    """
    path = Path(path)
    lut_size = lut.shape[0]

    safe_title = str(title).replace('"', "'")

    with path.open("w", encoding="utf-8") as f:
        f.write(f'TITLE "{safe_title}"\n')
        f.write(f"LUT_3D_SIZE {lut_size}\n")
        f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
        f.write("DOMAIN_MAX 1.0 1.0 1.0\n")

        for b in range(lut_size):
            for g in range(lut_size):
                for r in range(lut_size):
                    value = lut[r, g, b]
                    f.write(f"{value[0]:.17g} {value[1]:.17g} {value[2]:.17g}\n")


def main():
    parser = argparse.ArgumentParser(
        description="拟合亮度 1D LUT（RGB 等比例缩放），不改变色相/饱和度"
    )

    parser.add_argument("--src", required=True, help="调色前图片目录")
    parser.add_argument("--dst", required=True, help="调色后图片目录")
    parser.add_argument("--out", default="fitted_luma_lut", help="输出文件前缀")
    parser.add_argument(
        "--bins", type=int, default=256,
        help="LUT bin 数量，默认 256"
    )
    parser.add_argument(
        "--max_samples", type=int, default=5000000,
        help="最大采样像素数，默认 5000000"
    )
    parser.add_argument(
        "--smooth", type=float, default=1.0,
        help="高斯平滑强度（相对于 256 bin 的 sigma），默认 1.0；设为 0 不平滑"
    )
    parser.add_argument(
        "--clip_output", action="store_true",
        help="将缩放因子裁剪到 [0, 10]"
    )
    parser.add_argument(
        "--mode", choices=["1d", "3d"], default="1d",
        help="输出模式：1d 输出 1D LUT，3d 输出对角线 3D LUT（默认 1d）"
    )
    parser.add_argument(
        "--size", type=int, default=33,
        help="3D LUT 每轴节点数（仅 --mode 3d 时有效），默认 33"
    )
    parser.add_argument("--seed", type=int, default=1234, help="随机种子")

    args = parser.parse_args()

    pairs = find_image_pairs(args.src, args.dst)

    src_y, dst_y = collect_luma_samples(
        pairs=pairs,
        max_samples=args.max_samples,
        seed=args.seed
    )

    lut = fit_luma_lut(
        src_y=src_y,
        dst_y=dst_y,
        bins=args.bins,
        smooth_weight=args.smooth,
        clip_output=args.clip_output
    )

    output_base = Path(args.out)

    if output_base.suffix.lower() in {".npy", ".cube"}:
        output_base = output_base.with_suffix("")

    output_base.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "3d":
        # 生成对角线 3D LUT
        print(f"\n将 1D 亮度 LUT 转换为 {args.size}³ 3D LUT……")
        lut_3d = luma_lut_to_3d(lut, size=args.size)

        npy_path = Path(f"{output_base}_3d.npy")
        cube_path = Path(f"{output_base}_3d.cube")

        np.save(npy_path, lut_3d)
        write_3d_cube(cube_path, lut_3d, title=output_base.name)

        # 同时保存原始 1D LUT
        npy_1d_path = Path(f"{output_base}_1d.npy")
        np.save(npy_1d_path, np.asarray(lut, dtype=np.float64))

        print(f"\n输出完成（3D 模式）：")
        print(f"  3D NPY：  {npy_path}")
        print(f"  3D CUBE： {cube_path}")
        print(f"  1D NPY：  {npy_1d_path}")
        print(f"  3D LUT shape：{lut_3d.shape}")
        print(f"  含义：输入 RGB → 输出 RGB（亮度等比例缩放）")
    else:
        npy_path = Path(f"{output_base}.npy")
        cube_path = Path(f"{output_base}.cube")

        np.save(npy_path, np.asarray(lut, dtype=np.float64))
        write_1d_cube(cube_path, lut, title=output_base.name)

        print(f"\n输出完成（1D 模式）：")
        print(f"  NPY：  {npy_path}")
        print(f"  CUBE： {cube_path}")
        print(f"  LUT shape：{lut.shape}")
        print(f"  LUT dtype：{lut.dtype}")
        print(f"  含义：输入 Y → 缩放因子 scale，应用时 out_RGB = src_RGB * scale")


if __name__ == "__main__":
    main()
