#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import cv2
import numpy as np

from scipy.optimize import minimize, Bounds
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import splu


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".exr"
}


# ============================================================
# 图片扫描与配对
# ============================================================

def scan_images(root):
    """
    扫描目录中的所有图片。

    使用“不带扩展名的相对路径”作为匹配键。例如：

        before/scene01/0001.png
        after /scene01/0001.tif

    可以正常匹配。
    """
    root = Path(root)

    if not root.exists():
        raise FileNotFoundError(f"目录不存在：{root}")

    images = {}

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        relative_path = path.relative_to(root)
        key = relative_path.with_suffix("").as_posix().lower()

        if key in images:
            raise RuntimeError(
                "发现具有相同相对路径和主文件名的重复图片：\n"
                f"  {images[key]}\n"
                f"  {path}"
            )

        images[key] = path

    return images


def find_image_pairs(src_dir, dst_dir):
    """
    查找调色前后图片对。
    """
    src_images = scan_images(src_dir)
    dst_images = scan_images(dst_dir)

    common_keys = sorted(
        set(src_images.keys())
        & set(dst_images.keys())
    )

    if not common_keys:
        raise RuntimeError(
            "没有找到匹配的图片。\n"
            "请确保调色前后图片具有相同的相对路径和主文件名。"
        )

    src_only = sorted(
        set(src_images.keys())
        - set(dst_images.keys())
    )

    dst_only = sorted(
        set(dst_images.keys())
        - set(src_images.keys())
    )

    print("=" * 72)
    print(f"调色前图片数量：{len(src_images)}")
    print(f"调色后图片数量：{len(dst_images)}")
    print(f"成功匹配数量：  {len(common_keys)}")

    if src_only:
        print(
            f"警告：{len(src_only)} 张调色前图片没有匹配结果。"
        )

    if dst_only:
        print(
            f"警告：{len(dst_only)} 张调色后图片没有匹配结果。"
        )

    print("=" * 72)

    pairs = [
        (src_images[key], dst_images[key])
        for key in common_keys
    ]

    return pairs


# ============================================================
# 图片读取
# ============================================================

def read_rgb_float64(path):
    """
    使用 OpenCV 读取图片。

    返回：
        H×W×3
        RGB 顺序
        float64
        数值范围 [0,1]

    说明：
        OpenCV 不执行 ICC Profile 色彩管理。
        Alpha 通道会被忽略。
    """
    path = Path(path)

    image = cv2.imread(
        str(path),
        cv2.IMREAD_UNCHANGED
    )

    if image is None:
        raise RuntimeError(f"无法读取图片：{path}")

    # 灰度图
    if image.ndim == 2:
        image = np.repeat(
            image[..., None],
            3,
            axis=2
        )

    if image.ndim != 3:
        raise RuntimeError(
            f"不支持的图片维度：{path}, shape={image.shape}"
        )

    channels = image.shape[2]

    if channels == 1:
        image = np.repeat(
            image,
            3,
            axis=2
        )

    elif channels == 2:
        # 灰度 + Alpha，仅保留灰度
        gray = image[..., 0]
        image = np.repeat(
            gray[..., None],
            3,
            axis=2
        )

    elif channels >= 3:
        # OpenCV 是 BGR/BGRA
        # 只取 BGR 三通道，然后转换为 RGB
        image = image[..., :3][..., ::-1]

    else:
        raise RuntimeError(
            f"不支持的通道数：{path}, shape={image.shape}"
        )

    if np.issubdtype(image.dtype, np.unsignedinteger):
        max_value = float(
            np.iinfo(image.dtype).max
        )

        image = (
            image.astype(np.float64)
            / max_value
        )

    elif np.issubdtype(image.dtype, np.signedinteger):
        raise RuntimeError(
            f"不支持有符号整数图片：{path}, dtype={image.dtype}"
        )

    elif np.issubdtype(image.dtype, np.floating):
        image = image.astype(
            np.float64,
            copy=False
        )

        min_value = float(
            np.nanmin(image)
        )

        max_value = float(
            np.nanmax(image)
        )

        if (
            min_value < -1e-8
            or max_value > 1.0 + 1e-8
        ):
            raise RuntimeError(
                f"浮点图片必须位于 [0,1]：{path}\n"
                f"实际范围：[{min_value}, {max_value}]"
            )

    else:
        raise RuntimeError(
            f"不支持的数据类型：{path}, dtype={image.dtype}"
        )

    return np.clip(
        image,
        0.0,
        1.0
    )


# ============================================================
# 三线性插值矩阵
# ============================================================

def build_trilinear_matrix(rgb, lut_size):
    """
    根据输入 RGB 构建三线性插值稀疏矩阵 A。

    对任意 LUT：

        output = A @ lut.reshape(-1, 3)

    LUT 索引顺序：

        lut[R_index, G_index, B_index, output_channel]

    展平顺序：

        index = (R * size + G) * size + B
    """
    rgb = np.asarray(
        rgb,
        dtype=np.float64
    )

    rgb = np.clip(
        rgb,
        0.0,
        1.0
    )

    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(
            f"rgb 必须是 N×3，当前 shape={rgb.shape}"
        )

    sample_count = rgb.shape[0]

    # 映射到 LUT 网格坐标
    position = rgb * (lut_size - 1)

    index0 = np.floor(
        position
    ).astype(np.int32)

    # 输入为 1.0 时：
    # index0 使用 size-2，fraction 使用 1.0
    index0 = np.minimum(
        index0,
        lut_size - 2
    )

    fraction = position - index0

    corners = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=np.int32
    )

    # N×8×3
    nodes = (
        index0[:, None, :]
        + corners[None, :, :]
    )

    # LUT 节点展平索引
    columns = (
        (
            nodes[:, :, 0] * lut_size
            + nodes[:, :, 1]
        )
        * lut_size
        + nodes[:, :, 2]
    )

    # 每个维度的线性权重
    axis_weights = np.where(
        corners[None, :, :] == 1,
        fraction[:, None, :],
        1.0 - fraction[:, None, :]
    )

    # 三个维度权重相乘
    weights = np.prod(
        axis_weights,
        axis=2
    )

    rows = np.repeat(
        np.arange(
            sample_count,
            dtype=np.int32
        ),
        8
    )

    matrix = csr_matrix(
        (
            weights.reshape(-1),
            (
                rows,
                columns.reshape(-1)
            )
        ),
        shape=(
            sample_count,
            lut_size ** 3
        ),
        dtype=np.float64
    )

    matrix.sum_duplicates()
    matrix.eliminate_zeros()

    return matrix


# ============================================================
# LUT 二阶平滑矩阵
# ============================================================

def build_curvature_matrix(lut_size):
    """
    构建二阶平滑矩阵，包括：

    1. R 方向纯二阶差分；
    2. G 方向纯二阶差分；
    3. B 方向纯二阶差分；
    4. R-G 混合二阶差分；
    5. R-B 混合二阶差分；
    6. G-B 混合二阶差分。

    该约束不会惩罚全局仿射颜色变换，
    主要用于约束没有训练样本覆盖的 LUT 区域。
    """
    rows = []
    columns = []
    values = []

    row_index = 0

    def node_index(r, g, b):
        return (
            (r * lut_size + g)
            * lut_size
            + b
        )

    def add_row(node_indices, coefficients):
        nonlocal row_index

        rows.extend(
            [row_index] * len(node_indices)
        )
        columns.extend(node_indices)
        values.extend(coefficients)

        row_index += 1

    # R 方向纯二阶差分
    for r in range(1, lut_size - 1):
        for g in range(lut_size):
            for b in range(lut_size):
                add_row(
                    [
                        node_index(r - 1, g, b),
                        node_index(r, g, b),
                        node_index(r + 1, g, b)
                    ],
                    [1.0, -2.0, 1.0]
                )

    # G 方向纯二阶差分
    for r in range(lut_size):
        for g in range(1, lut_size - 1):
            for b in range(lut_size):
                add_row(
                    [
                        node_index(r, g - 1, b),
                        node_index(r, g, b),
                        node_index(r, g + 1, b)
                    ],
                    [1.0, -2.0, 1.0]
                )

    # B 方向纯二阶差分
    for r in range(lut_size):
        for g in range(lut_size):
            for b in range(1, lut_size - 1):
                add_row(
                    [
                        node_index(r, g, b - 1),
                        node_index(r, g, b),
                        node_index(r, g, b + 1)
                    ],
                    [1.0, -2.0, 1.0]
                )

    # R-G 混合二阶差分
    for r in range(lut_size - 1):
        for g in range(lut_size - 1):
            for b in range(lut_size):
                add_row(
                    [
                        node_index(r, g, b),
                        node_index(r + 1, g, b),
                        node_index(r, g + 1, b),
                        node_index(r + 1, g + 1, b)
                    ],
                    [1.0, -1.0, -1.0, 1.0]
                )

    # R-B 混合二阶差分
    for r in range(lut_size - 1):
        for g in range(lut_size):
            for b in range(lut_size - 1):
                add_row(
                    [
                        node_index(r, g, b),
                        node_index(r + 1, g, b),
                        node_index(r, g, b + 1),
                        node_index(r + 1, g, b + 1)
                    ],
                    [1.0, -1.0, -1.0, 1.0]
                )

    # G-B 混合二阶差分
    for r in range(lut_size):
        for g in range(lut_size - 1):
            for b in range(lut_size - 1):
                add_row(
                    [
                        node_index(r, g, b),
                        node_index(r, g + 1, b),
                        node_index(r, g, b + 1),
                        node_index(r, g + 1, b + 1)
                    ],
                    [1.0, -1.0, -1.0, 1.0]
                )

    matrix = csr_matrix(
        (
            np.asarray(
                values,
                dtype=np.float64
            ),
            (
                np.asarray(
                    rows,
                    dtype=np.int32
                ),
                np.asarray(
                    columns,
                    dtype=np.int32
                )
            )
        ),
        shape=(
            row_index,
            lut_size ** 3
        ),
        dtype=np.float64
    )

    return matrix


# ============================================================
# 全量像素高精度拟合
# ============================================================

def fit_lut_all_pixels(
    pairs,
    lut_size=17,
    chunk_pixels=500000,
    smooth_weight=1e-6,
    prior_weight=1e-6,
    numerical_jitter=1e-12,
    max_iterations=10000
):
    """
    使用全部前后图像像素拟合 3D LUT。

    优化问题：

        min_L mean(||A L - Y||²)
              + smooth_weight * 二阶平滑项
              + prior_weight * 仿射先验项

        subject to:
              0 <= L <= 1

    其中 A 是三线性插值矩阵。
    """
    if lut_size < 2:
        raise ValueError("lut_size 必须大于等于 2。")

    if chunk_pixels <= 0:
        raise ValueError("chunk_pixels 必须大于 0。")

    node_count = lut_size ** 3

    # A^T A
    data_hessian = csr_matrix(
        (node_count, node_count),
        dtype=np.float64
    )

    # A^T Y
    data_rhs = np.zeros(
        (node_count, 3),
        dtype=np.float64
    )

    # sum(Y²)，用于拟合后直接计算完整 MSE
    target_square_sum = np.zeros(
        3,
        dtype=np.float64
    )

    # 全局仿射模型统计量
    affine_xtx = np.zeros(
        (4, 4),
        dtype=np.float64
    )

    affine_xty = np.zeros(
        (4, 3),
        dtype=np.float64
    )

    total_pixels = 0

    print("\n开始累积全量像素正规方程……")

    for pair_index, (src_path, dst_path) in enumerate(
        pairs,
        start=1
    ):
        print("\n" + "-" * 72)
        print(
            f"[{pair_index}/{len(pairs)}]\n"
            f"输入：{src_path}\n"
            f"目标：{dst_path}"
        )

        src = read_rgb_float64(src_path)
        dst = read_rgb_float64(dst_path)

        if src.shape != dst.shape:
            raise RuntimeError(
                "前后图片尺寸或通道数不一致：\n"
                f"输入：{src_path}, shape={src.shape}\n"
                f"目标：{dst_path}, shape={dst.shape}"
            )

        src_pixels = src.reshape(-1, 3)
        dst_pixels = dst.reshape(-1, 3)

        image_pixel_count = src_pixels.shape[0]

        for start in range(
            0,
            image_pixel_count,
            chunk_pixels
        ):
            end = min(
                start + chunk_pixels,
                image_pixel_count
            )

            x = src_pixels[start:end]
            y = dst_pixels[start:end]

            valid = (
                np.isfinite(x).all(axis=1)
                & np.isfinite(y).all(axis=1)
            )

            x = x[valid]
            y = y[valid]

            valid_count = x.shape[0]

            if valid_count == 0:
                continue

            interpolation_matrix = (
                build_trilinear_matrix(
                    x,
                    lut_size
                )
            )

            chunk_hessian = (
                interpolation_matrix.T
                @ interpolation_matrix
            ).tocsr()

            data_hessian = (
                data_hessian
                + chunk_hessian
            ).tocsr()

            data_rhs += np.asarray(
                interpolation_matrix.T @ y
            )

            target_square_sum += np.sum(
                y * y,
                axis=0
            )

            # 累积全局仿射映射：
            # [R,G,B,1] @ coefficient = output_RGB
            x_augmented = np.empty(
                (valid_count, 4),
                dtype=np.float64
            )

            x_augmented[:, :3] = x
            x_augmented[:, 3] = 1.0

            affine_xtx += (
                x_augmented.T
                @ x_augmented
            )

            affine_xty += (
                x_augmented.T
                @ y
            )

            total_pixels += valid_count

            print(
                f"\r  当前图片：{end}/{image_pixel_count}，"
                f"累计有效像素：{total_pixels}",
                end="",
                flush=True
            )

        print()

    if total_pixels == 0:
        raise RuntimeError("没有找到有效像素。")

    print("\n" + "=" * 72)
    print(f"全量有效像素数量：{total_pixels}")
    print("=" * 72)

    # 转换为均值形式
    inverse_total = 1.0 / total_pixels

    data_hessian = (
        data_hessian * inverse_total
    ).tocsr()

    data_rhs *= inverse_total

    target_square_mean = (
        target_square_sum
        * inverse_total
    )

    data_hessian.sum_duplicates()
    data_hessian.eliminate_zeros()

    # ========================================================
    # 拟合全局仿射先验
    # ========================================================

    print("\n拟合全局仿射颜色先验……")

    identity_affine = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64
    )

    affine_ridge = 1e-12

    affine_matrix = (
        affine_xtx * inverse_total
        + affine_ridge
        * np.eye(4, dtype=np.float64)
    )

    affine_rhs = (
        affine_xty * inverse_total
        + affine_ridge
        * identity_affine
    )

    affine_coefficients = np.linalg.solve(
        affine_matrix,
        affine_rhs
    )

    axis = np.linspace(
        0.0,
        1.0,
        lut_size,
        dtype=np.float64
    )

    grid_r, grid_g, grid_b = np.meshgrid(
        axis,
        axis,
        axis,
        indexing="ij"
    )

    grid_rgb = np.stack(
        [grid_r, grid_g, grid_b],
        axis=-1
    ).reshape(-1, 3)

    grid_augmented = np.concatenate(
        [
            grid_rgb,
            np.ones(
                (node_count, 1),
                dtype=np.float64
            )
        ],
        axis=1
    )

    affine_prior = (
        grid_augmented
        @ affine_coefficients
    )

    affine_prior = np.clip(
        affine_prior,
        0.0,
        1.0
    )

    # ========================================================
    # 构建极弱平滑约束
    # ========================================================

    print("构建 LUT 二阶平滑约束……")

    curvature_matrix = build_curvature_matrix(
        lut_size
    )

    curvature_hessian = (
        curvature_matrix.T
        @ curvature_matrix
    ) / curvature_matrix.shape[0]

    identity_matrix = eye(
        node_count,
        dtype=np.float64,
        format="csc"
    )

    prior_coefficient = (
        prior_weight / node_count
    )

    # jitter 同样锚定到 affine_prior，
    # 避免没有样本覆盖的节点被拉向 0
    total_prior_coefficient = (
        prior_coefficient
        + numerical_jitter
    )

    regularized_hessian = (
        data_hessian
        + smooth_weight
        * curvature_hessian
        + total_prior_coefficient
        * identity_matrix
    ).tocsc()

    regularized_rhs = (
        data_rhs
        + total_prior_coefficient
        * affine_prior
    )

    # ========================================================
    # 无约束闭式解作为初始值
    # ========================================================

    print("计算无约束双精度初始解……")

    direct_solver = splu(
        regularized_hessian
    )

    initial_lut = direct_solver.solve(
        regularized_rhs
    )

    if not np.isfinite(initial_lut).all():
        print(
            "警告：无约束初始解包含非有限值，"
            "改用仿射先验作为初始值。"
        )
        initial_lut = affine_prior.copy()

    initial_lut = np.clip(
        initial_lut,
        0.0,
        1.0
    )

    # ========================================================
    # L-BFGS-B 有界优化
    # ========================================================

    lower_bound = np.zeros(
        node_count,
        dtype=np.float64
    )

    upper_bound = np.ones(
        node_count,
        dtype=np.float64
    )

    bounds = Bounds(
        lower_bound,
        upper_bound
    )

    lut_flat = np.empty(
        (node_count, 3),
        dtype=np.float64
    )

    channel_names = ["R", "G", "B"]

    for channel in range(3):
        print("\n" + "-" * 72)
        print(
            f"开始优化输出 {channel_names[channel]} 通道……"
        )

        rhs_channel = regularized_rhs[:, channel]

        def objective_and_gradient(values):
            hessian_values = (
                regularized_hessian @ values
            )

            objective = (
                0.5
                * np.dot(
                    values,
                    hessian_values
                )
                - np.dot(
                    rhs_channel,
                    values
                )
            )

            gradient = (
                hessian_values
                - rhs_channel
            )

            return objective, gradient

        result = minimize(
            objective_and_gradient,
            initial_lut[:, channel],
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={
                "maxiter": max_iterations,
                "maxcor": 100,
                "ftol": 1e-15,
                "gtol": 1e-12,
                "maxls": 100,
                "disp": False
            }
        )

        lut_flat[:, channel] = result.x

        print(f"success：       {result.success}")
        print(f"iterations：    {result.nit}")
        print(f"function value：{result.fun:.16e}")
        print(f"message：       {result.message}")

        if result.jac is not None:
            print(
                "max |gradient|："
                f"{np.max(np.abs(result.jac)):.16e}"
            )

    # ========================================================
    # 使用正规方程直接计算全量训练集 PSNR
    # ========================================================

    print("\n" + "=" * 72)
    print("计算全量训练集拟合误差……")

    channel_mse = np.zeros(
        3,
        dtype=np.float64
    )

    for channel in range(3):
        values = lut_flat[:, channel]

        mse = (
            np.dot(
                values,
                data_hessian @ values
            )
            - 2.0
            * np.dot(
                values,
                data_rhs[:, channel]
            )
            + target_square_mean[channel]
        )

        # 避免浮点舍入产生 -1e-16 之类的结果
        channel_mse[channel] = max(
            float(mse),
            0.0
        )

    rgb_mse = float(
        np.mean(channel_mse)
    )

    for channel, name in enumerate(
        channel_names
    ):
        mse = channel_mse[channel]

        if mse > 0:
            psnr = 10.0 * np.log10(
                1.0 / mse
            )
        else:
            psnr = float("inf")

        rmse_255 = np.sqrt(mse) * 255.0

        print(
            f"{name} 通道："
            f"MSE={mse:.16e}, "
            f"RMSE={rmse_255:.8f}/255, "
            f"PSNR={psnr:.8f} dB"
        )

    if rgb_mse > 0:
        rgb_psnr = 10.0 * np.log10(
            1.0 / rgb_mse
        )
    else:
        rgb_psnr = float("inf")

    print("-" * 72)
    print(f"RGB MSE： {rgb_mse:.16e}")
    print(
        f"RGB RMSE："
        f"{np.sqrt(rgb_mse) * 255.0:.8f}/255"
    )
    print(f"RGB PSNR：{rgb_psnr:.8f} dB")
    print("=" * 72)

    lut = lut_flat.reshape(
        lut_size,
        lut_size,
        lut_size,
        3
    )

    return lut


# ============================================================
# 全量第二遍验证
# ============================================================

def evaluate_lut_on_all_pairs(
    pairs,
    lut,
    chunk_pixels=500000
):
    """
    重新读取全部图片，实际执行 LUT 三线性插值，
    计算逐图片和整体 PSNR。

    该函数可用于确认：
        正规方程计算结果与实际 LUT 应用结果一致。
    """
    lut_size = lut.shape[0]

    lut_flat = lut.reshape(
        -1,
        3
    ).astype(
        np.float64,
        copy=False
    )

    global_square_error = np.zeros(
        3,
        dtype=np.float64
    )

    global_absolute_error = np.zeros(
        3,
        dtype=np.float64
    )

    global_max_error = np.zeros(
        3,
        dtype=np.float64
    )

    global_pixel_count = 0

    print("\n" + "=" * 72)
    print("开始第二遍全量实际验证……")
    print("=" * 72)

    for pair_index, (src_path, dst_path) in enumerate(
        pairs,
        start=1
    ):
        src = read_rgb_float64(src_path)
        dst = read_rgb_float64(dst_path)

        if src.shape != dst.shape:
            raise RuntimeError(
                f"验证时图片尺寸不一致：{src_path}"
            )

        src_pixels = src.reshape(-1, 3)
        dst_pixels = dst.reshape(-1, 3)

        image_square_error = np.zeros(
            3,
            dtype=np.float64
        )

        image_absolute_error = np.zeros(
            3,
            dtype=np.float64
        )

        image_max_error = np.zeros(
            3,
            dtype=np.float64
        )

        image_valid_pixels = 0

        image_pixel_count = src_pixels.shape[0]

        for start in range(
            0,
            image_pixel_count,
            chunk_pixels
        ):
            end = min(
                start + chunk_pixels,
                image_pixel_count
            )

            x = src_pixels[start:end]
            y = dst_pixels[start:end]

            valid = (
                np.isfinite(x).all(axis=1)
                & np.isfinite(y).all(axis=1)
            )

            x = x[valid]
            y = y[valid]

            if x.shape[0] == 0:
                continue

            interpolation_matrix = (
                build_trilinear_matrix(
                    x,
                    lut_size
                )
            )

            prediction = np.asarray(
                interpolation_matrix @ lut_flat
            )

            error = prediction - y

            image_square_error += np.sum(
                error * error,
                axis=0
            )

            image_absolute_error += np.sum(
                np.abs(error),
                axis=0
            )

            image_max_error = np.maximum(
                image_max_error,
                np.max(
                    np.abs(error),
                    axis=0
                )
            )

            image_valid_pixels += x.shape[0]

        if image_valid_pixels == 0:
            continue

        image_rgb_mse = (
            np.sum(image_square_error)
            / (image_valid_pixels * 3)
        )

        if image_rgb_mse > 0:
            image_psnr = (
                10.0
                * np.log10(
                    1.0 / image_rgb_mse
                )
            )
        else:
            image_psnr = float("inf")

        image_rgb_mae = (
            np.sum(image_absolute_error)
            / (image_valid_pixels * 3)
        )

        print(
            f"[{pair_index}/{len(pairs)}] "
            f"{src_path.name}: "
            f"PSNR={image_psnr:.8f} dB, "
            f"MAE={image_rgb_mae * 255.0:.8f}/255"
        )

        global_square_error += image_square_error
        global_absolute_error += image_absolute_error
        global_max_error = np.maximum(
            global_max_error,
            image_max_error
        )
        global_pixel_count += image_valid_pixels

    if global_pixel_count == 0:
        raise RuntimeError(
            "验证阶段没有有效像素。"
        )

    global_rgb_mse = (
        np.sum(global_square_error)
        / (global_pixel_count * 3)
    )

    global_rgb_mae = (
        np.sum(global_absolute_error)
        / (global_pixel_count * 3)
    )

    if global_rgb_mse > 0:
        global_rgb_psnr = (
            10.0
            * np.log10(
                1.0 / global_rgb_mse
            )
        )
    else:
        global_rgb_psnr = float("inf")

    print("\n" + "=" * 72)
    print("全量实际应用结果")
    print("-" * 72)
    print(f"像素数量：{global_pixel_count}")
    print(
        f"RGB PSNR：{global_rgb_psnr:.8f} dB"
    )
    print(
        f"RGB RMSE："
        f"{np.sqrt(global_rgb_mse) * 255.0:.8f}/255"
    )
    print(
        f"RGB MAE： "
        f"{global_rgb_mae * 255.0:.8f}/255"
    )
    print(
        "每通道最大绝对误差："
        f"R={global_max_error[0] * 255.0:.8f}, "
        f"G={global_max_error[1] * 255.0:.8f}, "
        f"B={global_max_error[2] * 255.0:.8f}"
    )
    print("=" * 72)

    return global_rgb_psnr


# ============================================================
# LUT 应用函数
# ============================================================

def apply_lut_to_rgb_image(
    image_rgb,
    lut,
    chunk_pixels=500000
):
    """
    对 RGB float 图片应用本脚本拟合的 LUT。

    image_rgb：
        H×W×3
        RGB 顺序
        范围 [0,1]

    lut：
        [R,G,B,output_RGB]

    返回：
        RGB float64
        范围 [0,1]
    """
    image_rgb = np.asarray(
        image_rgb,
        dtype=np.float64
    )

    height, width, channels = image_rgb.shape

    if channels != 3:
        raise ValueError(
            f"图片必须为三通道，当前 shape={image_rgb.shape}"
        )

    input_pixels = image_rgb.reshape(
        -1,
        3
    )

    output_pixels = np.empty_like(
        input_pixels,
        dtype=np.float64
    )

    lut_flat = lut.reshape(
        -1,
        3
    ).astype(
        np.float64,
        copy=False
    )

    lut_size = lut.shape[0]

    for start in range(
        0,
        input_pixels.shape[0],
        chunk_pixels
    ):
        end = min(
            start + chunk_pixels,
            input_pixels.shape[0]
        )

        matrix = build_trilinear_matrix(
            input_pixels[start:end],
            lut_size
        )

        output_pixels[start:end] = (
            matrix @ lut_flat
        )

    output = output_pixels.reshape(
        height,
        width,
        3
    )

    return np.clip(
        output,
        0.0,
        1.0
    )


# ============================================================
# CUBE 输出
# ============================================================

def write_cube(
    path,
    lut,
    title="Fitted 3D LUT"
):
    """
    输出标准 .cube 文件。

    NPY 内部排列：
        lut[R,G,B,output_RGB]

    CUBE 文件排列：
        R 索引变化最快；
        G 次之；
        B 最慢。
    """
    path = Path(path)

    lut_size = lut.shape[0]

    with path.open(
        "w",
        encoding="utf-8"
    ) as file:
        file.write(
            f'TITLE "{title}"\n'
        )

        file.write(
            f"LUT_3D_SIZE {lut_size}\n"
        )

        file.write(
            "DOMAIN_MIN 0.0 0.0 0.0\n"
        )

        file.write(
            "DOMAIN_MAX 1.0 1.0 1.0\n"
        )

        for b in range(lut_size):
            for g in range(lut_size):
                for r in range(lut_size):
                    value = lut[r, g, b]

                    file.write(
                        f"{value[0]:.12f} "
                        f"{value[1]:.12f} "
                        f"{value[2]:.12f}\n"
                    )


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "根据严格配准的 Photoshop 调色前后图片，"
            "使用全量像素高精度拟合 3D LUT。"
        )
    )

    parser.add_argument(
        "--src",
        required=True,
        help="调色前图片目录"
    )

    parser.add_argument(
        "--dst",
        required=True,
        help="调色后图片目录"
    )

    parser.add_argument(
        "--out",
        default="fitted_lut_17",
        help=(
            "输出文件前缀。"
            "默认 fitted_lut_17"
        )
    )

    parser.add_argument(
        "--size",
        type=int,
        default=17,
        help="LUT 尺寸，默认 17"
    )

    parser.add_argument(
        "--chunk_pixels",
        type=int,
        default=500000,
        help=(
            "每次处理的像素数。"
            "默认 500000。"
            "内存不足时可以改为 100000。"
        )
    )

    parser.add_argument(
        "--smooth",
        type=float,
        default=1e-6,
        help=(
            "二阶平滑权重。"
            "默认 1e-6。"
            "颜色覆盖充分时可改为 1e-8。"
        )
    )

    parser.add_argument(
        "--prior",
        type=float,
        default=1e-6,
        help=(
            "仿射先验权重。"
            "默认 1e-6。"
            "主要约束无样本覆盖的区域。"
        )
    )

    parser.add_argument(
        "--jitter",
        type=float,
        default=1e-12,
        help="数值稳定项，默认 1e-12"
    )

    parser.add_argument(
        "--max_iterations",
        type=int,
        default=10000,
        help="L-BFGS-B 最大迭代次数"
    )

    parser.add_argument(
        "--skip_evaluation",
        action="store_true",
        help="跳过拟合后的第二遍全量验证"
    )

    args = parser.parse_args()

    pairs = find_image_pairs(
        args.src,
        args.dst
    )

    lut = fit_lut_all_pixels(
        pairs=pairs,
        lut_size=args.size,
        chunk_pixels=args.chunk_pixels,
        smooth_weight=args.smooth,
        prior_weight=args.prior,
        numerical_jitter=args.jitter,
        max_iterations=args.max_iterations
    )

    output_base = Path(args.out)

    if output_base.suffix.lower() in {
        ".npy",
        ".cube"
    }:
        output_base = output_base.with_suffix("")

    output_base.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    npy_path = output_base.with_suffix(
        ".npy"
    )

    cube_path = output_base.with_suffix(
        ".cube"
    )

    # 为保留最大精度，NPY 使用 float64
    np.save(
        npy_path,
        lut.astype(np.float64)
    )

    write_cube(
        cube_path,
        lut,
        title=output_base.stem
    )

    print("\n" + "=" * 72)
    print("LUT 输出完成")
    print("-" * 72)
    print(f"NPY：  {npy_path}")
    print(f"CUBE： {cube_path}")
    print(f"shape：{lut.shape}")
    print(f"dtype：float64")
    print("维度顺序：[R, G, B, output_RGB]")
    print("=" * 72)

    if not args.skip_evaluation:
        evaluate_lut_on_all_pairs(
            pairs=pairs,
            lut=lut,
            chunk_pixels=args.chunk_pixels
        )


if __name__ == "__main__":
    main()