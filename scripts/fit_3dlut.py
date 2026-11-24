#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import LinearOperator, cg, splu


IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp",
    ".tif", ".tiff", ".webp", ".exr"
}

SUPPORTED_LUT_SIZES = (17, 33, 64, 256)

# 命令行中的 smooth 始终表示 17³ LUT 下的参考平滑强度
REFERENCE_LUT_SIZE = 17


def scan_images(root):
    """
    以不包含扩展名的相对路径作为匹配键。

    例如：
        before/scene01/frame001.png
        after/scene01/frame001.tif

    两者可以正确配对。
    """
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

    common_keys = sorted(
        set(src_images) & set(dst_images)
    )

    if not common_keys:
        raise RuntimeError(
            "没有找到可匹配的图片。\n"
            "请确保调色前后图片具有相同的相对路径和文件名，"
            "扩展名可以不同。"
        )

    src_only = sorted(
        set(src_images) - set(dst_images)
    )

    dst_only = sorted(
        set(dst_images) - set(src_images)
    )

    print(f"调色前图片数量：{len(src_images)}")
    print(f"调色后图片数量：{len(dst_images)}")
    print(f"成功匹配图片数量：{len(common_keys)}")

    if src_only:
        print(
            f"警告：有 {len(src_only)} 张调色前图片"
            f"没有匹配结果。"
        )

    if dst_only:
        print(
            f"警告：有 {len(dst_only)} 张调色后图片"
            f"没有匹配结果。"
        )

    return [
        (src_images[key], dst_images[key])
        for key in common_keys
    ]


def read_rgb_float(path):
    """
    使用 OpenCV 读取图片，不执行 ICC 色彩管理转换。

    返回：
        float64 RGB，范围 [0, 1]

    注意：
        1. 浮点 EXR 必须已经归一化到 [0,1]；
        2. BGRA 图片会忽略 Alpha；
        3. 所有整数输入均直接转换为 float64，避免中间 float32
           量化影响拟合精度。
    """
    image = cv2.imread(
        str(path),
        cv2.IMREAD_UNCHANGED
    )

    if image is None:
        raise RuntimeError(
            f"无法读取图片：{path}"
        )

    # 灰度图转三通道
    if image.ndim == 2:
        image = np.repeat(
            image[..., None],
            3,
            axis=2
        )

    if image.ndim != 3:
        raise RuntimeError(
            f"不支持的图片维度："
            f"{path}, shape={image.shape}"
        )

    channels = image.shape[2]

    if channels == 1:
        image = np.repeat(
            image,
            3,
            axis=2
        )

    elif channels == 3:
        # OpenCV BGR -> RGB
        image = image[..., ::-1]

    elif channels == 4:
        # OpenCV BGRA -> RGB，明确忽略 Alpha
        image = image[..., :3][..., ::-1]

    else:
        raise RuntimeError(
            f"不支持的图片通道数："
            f"{path}, shape={image.shape}"
        )

    if np.issubdtype(
        image.dtype,
        np.integer
    ):
        if np.issubdtype(
            image.dtype,
            np.signedinteger
        ):
            if np.min(image) < 0:
                raise RuntimeError(
                    f"不支持包含负值的有符号整数图片："
                    f"{path}, dtype={image.dtype}"
                )

        max_value = np.iinfo(
            image.dtype
        ).max

        image = (
            image.astype(np.float64)
            / float(max_value)
        )

    elif np.issubdtype(
        image.dtype,
        np.floating
    ):
        image = image.astype(
            np.float64
        )

        finite_mask = np.isfinite(image)

        if not finite_mask.any():
            raise RuntimeError(
                f"图片中不存在有效浮点像素：{path}"
            )

        finite_values = image[finite_mask]

        min_value = float(
            finite_values.min()
        )

        max_value = float(
            finite_values.max()
        )

        if (
            min_value < -1e-6
            or max_value > 1.0 + 1e-6
        ):
            raise RuntimeError(
                f"浮点图片不在 [0,1] 范围内："
                f"{path}, "
                f"range=[{min_value}, {max_value}]"
            )

        invalid_mask = ~finite_mask

        image = np.clip(
            image,
            0.0,
            1.0
        )

        # 保持非法像素为 NaN，后续按像素对过滤
        if invalid_mask.any():
            image[invalid_mask] = np.nan

        return image

    else:
        raise RuntimeError(
            f"不支持的图片数据类型："
            f"{path}, dtype={image.dtype}"
        )

    return np.clip(
        image,
        0.0,
        1.0
    )


def stratified_subsample(
    x,
    y,
    max_samples,
    bins,
    rng
):
    """
    根据输入 RGB 在三维颜色空间中分层采样。

    目的：
        避免自然图像中的灰色、肤色等高频颜色完全主导拟合。
    """
    sample_count = x.shape[0]

    if sample_count <= max_samples:
        return x, y

    quantized = np.minimum(
        (
            np.clip(x, 0.0, 1.0)
            * bins
        ).astype(np.int32),
        bins - 1
    )

    cell_ids = (
        (
            quantized[:, 0] * bins
            + quantized[:, 1]
        ) * bins
        + quantized[:, 2]
    )

    # 先随机打乱，再按颜色格子稳定排序，
    # 保证格子内部采样是随机的
    permutation = rng.permutation(
        sample_count
    )

    order_in_permutation = np.argsort(
        cell_ids[permutation],
        kind="stable"
    )

    ordered_indices = permutation[
        order_in_permutation
    ]

    ordered_cells = cell_ids[
        ordered_indices
    ]

    unique_cells, starts, counts = np.unique(
        ordered_cells,
        return_index=True,
        return_counts=True
    )

    occupied_cells = len(
        unique_cells
    )

    if occupied_cells == 0:
        raise RuntimeError(
            "分层采样时没有找到有效颜色格子。"
        )

    print(
        f"候选样本覆盖颜色格子："
        f"{occupied_cells}/{bins ** 3}"
    )

    # 格子数已经超过样本预算：
    # 随机选择格子，每个格子取一个样本
    if occupied_cells >= max_samples:
        selected_groups = rng.choice(
            occupied_cells,
            size=max_samples,
            replace=False
        )

        selected = ordered_indices[
            starts[selected_groups]
        ]

        return (
            x[selected],
            y[selected]
        )

    base_quota = (
        max_samples
        // occupied_cells
    )

    selected_parts = []

    selected_mask = np.zeros(
        sample_count,
        dtype=bool
    )

    for start, count in zip(
        starts,
        counts
    ):
        take_count = min(
            base_quota,
            count
        )

        indices = ordered_indices[
            start:start + take_count
        ]

        selected_parts.append(
            indices
        )

        selected_mask[indices] = True

    selected = np.concatenate(
        selected_parts
    )

    remaining_count = (
        max_samples
        - selected.size
    )

    if remaining_count > 0:
        remaining_indices = np.flatnonzero(
            ~selected_mask
        )

        if (
            remaining_indices.size
            > remaining_count
        ):
            extra = rng.choice(
                remaining_indices,
                size=remaining_count,
                replace=False
            )
        else:
            extra = remaining_indices

        selected = np.concatenate(
            [selected, extra]
        )

    rng.shuffle(selected)

    return (
        x[selected],
        y[selected]
    )


def collect_pixel_samples(
    pairs,
    max_samples,
    oversample,
    lut_size,
    seed
):
    """
    先从每张图片中随机抽取候选像素，
    再在整个 RGB 颜色空间中进行分层采样。
    """
    rng = np.random.default_rng(
        seed
    )

    candidate_limit = (
        max_samples
        * oversample
    )

    per_image_limit = math.ceil(
        candidate_limit
        / len(pairs)
    )

    all_x = []
    all_y = []

    for index, (
        src_path,
        dst_path
    ) in enumerate(
        pairs,
        start=1
    ):
        print(
            f"[{index}/{len(pairs)}] 读取："
            f"{src_path.name} -> "
            f"{dst_path.name}"
        )

        src = read_rgb_float(
            src_path
        )

        dst = read_rgb_float(
            dst_path
        )

        if src.shape != dst.shape:
            raise RuntimeError(
                f"图片尺寸或通道不一致：\n"
                f"  输入：{src_path}, "
                f"shape={src.shape}\n"
                f"  输出：{dst_path}, "
                f"shape={dst.shape}"
            )

        src_pixels = src.reshape(
            -1,
            3
        )

        dst_pixels = dst.reshape(
            -1,
            3
        )

        pixel_count = (
            src_pixels.shape[0]
        )

        take_count = min(
            pixel_count,
            per_image_limit
        )

        if take_count < pixel_count:
            indices = rng.choice(
                pixel_count,
                size=take_count,
                replace=False
            )

            src_pixels = src_pixels[
                indices
            ]

            dst_pixels = dst_pixels[
                indices
            ]

        valid = (
            np.isfinite(
                src_pixels
            ).all(axis=1)
            &
            np.isfinite(
                dst_pixels
            ).all(axis=1)
        )

        src_pixels = src_pixels[
            valid
        ]

        dst_pixels = dst_pixels[
            valid
        ]

        all_x.append(
            src_pixels
        )

        all_y.append(
            dst_pixels
        )

    x = np.concatenate(
        all_x,
        axis=0
    )

    y = np.concatenate(
        all_y,
        axis=0
    )

    if x.shape[0] == 0:
        raise RuntimeError(
            "没有收集到有效像素，无法拟合 LUT。"
        )

    print(
        f"候选像素数量：{x.shape[0]}"
    )

    x, y = stratified_subsample(
        x=x,
        y=y,
        max_samples=max_samples,
        bins=lut_size - 1,
        rng=rng
    )

    if x.shape[0] == 0:
        raise RuntimeError(
            "分层采样后没有有效像素，"
            "无法拟合 LUT。"
        )

    print(
        f"最终拟合像素数量：{x.shape[0]}"
    )

    return (
        np.asarray(
            x,
            dtype=np.float64
        ),
        np.asarray(
            y,
            dtype=np.float64
        )
    )


def build_trilinear_matrix(
    rgb,
    lut_size
):
    """
    为每个输入 RGB 构建三线性插值系数。

    LUT 展平索引：

        index = (R_index * size + G_index) * size + B_index

    每个输入 RGB 只与相邻的 8 个 LUT 节点有关。
    """
    rgb = np.clip(
        np.asarray(
            rgb,
            dtype=np.float64
        ),
        0.0,
        1.0
    )

    sample_count = rgb.shape[0]

    position = rgb * (
        lut_size - 1
    )

    base = np.floor(
        position
    ).astype(np.int32)

    base = np.minimum(
        base,
        lut_size - 2
    )

    fraction = (
        position
        - base
    )

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

    nodes = (
        base[:, None, :]
        + corners[None, :, :]
    )

    columns = (
        (
            nodes[:, :, 0]
            * lut_size
            + nodes[:, :, 1]
        ) * lut_size
        + nodes[:, :, 2]
    )

    weights_per_axis = np.where(
        corners[None, :, :] == 1,
        fraction[:, None, :],
        1.0 - fraction[:, None, :]
    )

    weights = np.prod(
        weights_per_axis,
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

    matrix.eliminate_zeros()

    return matrix


def build_curvature_matrix(
    lut_size
):
    """
    构建 LUT 的二阶平滑约束，包括：

    1. R/G/B 三个方向上的纯二阶差分；
    2. RG/RB/GB 三组混合二阶差分。

    此处保留原始离散网格差分，不直接乘网格尺度。
    LUT 尺寸补偿由 effective_smooth_weight 完成。
    """
    size = lut_size

    node_count = (
        size ** 3
    )

    def node_index_vec(
        r,
        g,
        b
    ):
        return (
            (r * size + g)
            * size
            + b
        )

    parts_rows = []
    parts_cols = []
    parts_data = []

    row_offset = 0

    def append_pure_difference(
        column0,
        column1,
        column2
    ):
        nonlocal row_offset

        column0 = (
            column0
            .ravel()
            .astype(
                np.int32,
                copy=False
            )
        )

        column1 = (
            column1
            .ravel()
            .astype(
                np.int32,
                copy=False
            )
        )

        column2 = (
            column2
            .ravel()
            .astype(
                np.int32,
                copy=False
            )
        )

        count = column0.size

        row_indices = np.arange(
            row_offset,
            row_offset + count,
            dtype=np.int32
        )

        parts_rows.extend(
            [
                row_indices,
                row_indices,
                row_indices
            ]
        )

        parts_cols.extend(
            [
                column0,
                column1,
                column2
            ]
        )

        parts_data.extend(
            [
                np.ones(
                    count,
                    dtype=np.float64
                ),
                -2.0 * np.ones(
                    count,
                    dtype=np.float64
                ),
                np.ones(
                    count,
                    dtype=np.float64
                )
            ]
        )

        row_offset += count

    def append_mixed_difference(
        column0,
        column1,
        column2,
        column3
    ):
        nonlocal row_offset

        column0 = (
            column0
            .ravel()
            .astype(
                np.int32,
                copy=False
            )
        )

        column1 = (
            column1
            .ravel()
            .astype(
                np.int32,
                copy=False
            )
        )

        column2 = (
            column2
            .ravel()
            .astype(
                np.int32,
                copy=False
            )
        )

        column3 = (
            column3
            .ravel()
            .astype(
                np.int32,
                copy=False
            )
        )

        count = column0.size

        row_indices = np.arange(
            row_offset,
            row_offset + count,
            dtype=np.int32
        )

        parts_rows.extend(
            [
                row_indices,
                row_indices,
                row_indices,
                row_indices
            ]
        )

        parts_cols.extend(
            [
                column0,
                column1,
                column2,
                column3
            ]
        )

        parts_data.extend(
            [
                np.ones(
                    count,
                    dtype=np.float64
                ),
                -np.ones(
                    count,
                    dtype=np.float64
                ),
                -np.ones(
                    count,
                    dtype=np.float64
                ),
                np.ones(
                    count,
                    dtype=np.float64
                )
            ]
        )

        row_offset += count

    # R 方向纯二阶差分
    rr = np.arange(
        1,
        size - 1,
        dtype=np.int32
    )

    gg = np.arange(
        size,
        dtype=np.int32
    )

    bb = np.arange(
        size,
        dtype=np.int32
    )

    r0, g0, b0 = np.meshgrid(
        rr,
        gg,
        bb,
        indexing="ij"
    )

    append_pure_difference(
        node_index_vec(
            r0 - 1,
            g0,
            b0
        ),
        node_index_vec(
            r0,
            g0,
            b0
        ),
        node_index_vec(
            r0 + 1,
            g0,
            b0
        )
    )

    # G 方向纯二阶差分
    rr = np.arange(
        size,
        dtype=np.int32
    )

    gg = np.arange(
        1,
        size - 1,
        dtype=np.int32
    )

    bb = np.arange(
        size,
        dtype=np.int32
    )

    r0, g0, b0 = np.meshgrid(
        rr,
        gg,
        bb,
        indexing="ij"
    )

    append_pure_difference(
        node_index_vec(
            r0,
            g0 - 1,
            b0
        ),
        node_index_vec(
            r0,
            g0,
            b0
        ),
        node_index_vec(
            r0,
            g0 + 1,
            b0
        )
    )

    # B 方向纯二阶差分
    rr = np.arange(
        size,
        dtype=np.int32
    )

    gg = np.arange(
        size,
        dtype=np.int32
    )

    bb = np.arange(
        1,
        size - 1,
        dtype=np.int32
    )

    r0, g0, b0 = np.meshgrid(
        rr,
        gg,
        bb,
        indexing="ij"
    )

    append_pure_difference(
        node_index_vec(
            r0,
            g0,
            b0 - 1
        ),
        node_index_vec(
            r0,
            g0,
            b0
        ),
        node_index_vec(
            r0,
            g0,
            b0 + 1
        )
    )

    # R-G 混合二阶差分
    rr = np.arange(
        size - 1,
        dtype=np.int32
    )

    gg = np.arange(
        size - 1,
        dtype=np.int32
    )

    bb = np.arange(
        size,
        dtype=np.int32
    )

    r0, g0, b0 = np.meshgrid(
        rr,
        gg,
        bb,
        indexing="ij"
    )

    append_mixed_difference(
        node_index_vec(
            r0,
            g0,
            b0
        ),
        node_index_vec(
            r0 + 1,
            g0,
            b0
        ),
        node_index_vec(
            r0,
            g0 + 1,
            b0
        ),
        node_index_vec(
            r0 + 1,
            g0 + 1,
            b0
        )
    )

    # R-B 混合二阶差分
    rr = np.arange(
        size - 1,
        dtype=np.int32
    )

    gg = np.arange(
        size,
        dtype=np.int32
    )

    bb = np.arange(
        size - 1,
        dtype=np.int32
    )

    r0, g0, b0 = np.meshgrid(
        rr,
        gg,
        bb,
        indexing="ij"
    )

    append_mixed_difference(
        node_index_vec(
            r0,
            g0,
            b0
        ),
        node_index_vec(
            r0 + 1,
            g0,
            b0
        ),
        node_index_vec(
            r0,
            g0,
            b0 + 1
        ),
        node_index_vec(
            r0 + 1,
            g0,
            b0 + 1
        )
    )

    # G-B 混合二阶差分
    rr = np.arange(
        size,
        dtype=np.int32
    )

    gg = np.arange(
        size - 1,
        dtype=np.int32
    )

    bb = np.arange(
        size - 1,
        dtype=np.int32
    )

    r0, g0, b0 = np.meshgrid(
        rr,
        gg,
        bb,
        indexing="ij"
    )

    append_mixed_difference(
        node_index_vec(
            r0,
            g0,
            b0
        ),
        node_index_vec(
            r0,
            g0 + 1,
            b0
        ),
        node_index_vec(
            r0,
            g0,
            b0 + 1
        ),
        node_index_vec(
            r0,
            g0 + 1,
            b0 + 1
        )
    )

    rows = np.concatenate(
        parts_rows
    )

    columns = np.concatenate(
        parts_cols
    )

    data = np.concatenate(
        parts_data
    )

    matrix = csr_matrix(
        (
            data,
            (
                rows,
                columns
            )
        ),
        shape=(
            row_offset,
            node_count
        ),
        dtype=np.float64
    )

    matrix.eliminate_zeros()

    return matrix


def fit_affine_prior(
    x,
    y,
    grid_rgb
):
    """
    拟合全局 3x3 颜色矩阵和 bias，
    作为缺少数据区域的弱先验。
    """
    sample_count = x.shape[0]

    x_augmented = np.concatenate(
        [
            x,
            np.ones(
                (
                    sample_count,
                    1
                ),
                dtype=np.float64
            )
        ],
        axis=1
    )

    identity_coefficients = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64
    )

    affine_ridge = 1e-8

    normal_matrix = (
        x_augmented.T
        @ x_augmented
    ) / sample_count

    normal_matrix += (
        affine_ridge
        * np.eye(
            4,
            dtype=np.float64
        )
    )

    normal_rhs = (
        x_augmented.T
        @ y
    ) / sample_count

    normal_rhs += (
        affine_ridge
        * identity_coefficients
    )

    coefficients = np.linalg.solve(
        normal_matrix,
        normal_rhs
    )

    grid_augmented = np.concatenate(
        [
            grid_rgb,
            np.ones(
                (
                    grid_rgb.shape[0],
                    1
                ),
                dtype=np.float64
            )
        ],
        axis=1
    )

    prior = (
        grid_augmented
        @ coefficients
    )

    # prior 只是正则参考，不代表最终 LUT 必须裁剪
    return np.clip(
        prior,
        0.0,
        1.0
    )


def calculate_effective_smooth_weight(
    smooth_weight,
    lut_size,
    reference_size=REFERENCE_LUT_SIZE
):
    """
    将以 reference_size 为基准的平滑权重换算到当前 LUT 尺寸。

    原始二阶差分近似：

        D²f ≈ h² f''

    损失中使用平方惩罚，所以需要按 h^-4 缩放。
    """
    scale = (
        (
            lut_size - 1
        )
        / (
            reference_size - 1
        )
    ) ** 4

    effective_weight = (
        smooth_weight
        * scale
    )

    return (
        effective_weight,
        scale
    )


def relative_residual(
    operator,
    solution,
    rhs
):
    """
    计算真实相对残差：

        ||rhs - Hx|| / ||rhs||
    """
    residual = (
        rhs
        - operator @ solution
    )

    denominator = max(
        np.linalg.norm(rhs),
        1e-30
    )

    value = (
        np.linalg.norm(residual)
        / denominator
    )

    return (
        float(value),
        residual
    )


def solve_lut_direct(
    interpolation_matrix,
    curvature_matrix,
    y,
    affine_prior,
    effective_smooth_weight,
    prior_weight,
    refine_steps=3,
    residual_tolerance=1e-12
):
    """
    Float64 稀疏 LU 求解，并执行迭代改进。

    适用于 17³ 和 33³ LUT。
    """
    sample_count = (
        interpolation_matrix.shape[0]
    )

    node_count = (
        interpolation_matrix.shape[1]
    )

    print("构建显式 Float64 正规方程……")

    data_hessian = (
        interpolation_matrix.T
        @ interpolation_matrix
    ) / sample_count

    smooth_hessian = (
        curvature_matrix.T
        @ curvature_matrix
    ) / curvature_matrix.shape[0]

    identity_matrix = eye(
        node_count,
        dtype=np.float64,
        format="csc"
    )

    prior_coefficient = (
        prior_weight
        / node_count
    )

    # prior > 0 时已经可以保证正定性。
    # 保留极小 jitter，用于防止用户把 prior 设置为 0 后矩阵退化。
    numerical_jitter = 1e-14

    hessian = (
        data_hessian
        + effective_smooth_weight
        * smooth_hessian
        + prior_coefficient
        * identity_matrix
        + numerical_jitter
        * identity_matrix
    ).tocsc()

    rhs = (
        interpolation_matrix.T
        @ y
    ) / sample_count

    rhs += (
        prior_coefficient
        * affine_prior
    )

    rhs = np.asarray(
        rhs,
        dtype=np.float64
    )

    print("执行 Float64 稀疏 LU 分解……")

    solver = splu(
        hessian,
        permc_spec="COLAMD"
    )

    solution = solver.solve(
        rhs
    )

    rhs_norm = np.maximum(
        np.linalg.norm(
            rhs,
            axis=0
        ),
        1e-30
    )

    for step in range(
        refine_steps + 1
    ):
        residual = (
            rhs
            - hessian @ solution
        )

        relative = (
            np.linalg.norm(
                residual,
                axis=0
            )
            / rhs_norm
        )

        print(
            f"  LU 残差，第 {step} 次检查："
            f"R={relative[0]:.6e}, "
            f"G={relative[1]:.6e}, "
            f"B={relative[2]:.6e}"
        )

        if np.max(relative) <= residual_tolerance:
            break

        if step >= refine_steps:
            raise RuntimeError(
                "稀疏 LU 在迭代改进后仍未达到"
                f"残差要求 {residual_tolerance:.3e}，"
                f"当前最大相对残差为 "
                f"{np.max(relative):.6e}。"
            )

        correction = solver.solve(
            residual
        )

        solution += correction

    if not np.isfinite(
        solution
    ).all():
        raise RuntimeError(
            "稀疏 LU 求解结果包含 NaN 或 Inf。"
        )

    return solution


def sparse_gram_diagonal(
    matrix
):
    """
    精确计算：

        diag(matrix.T @ matrix)

    但不显式构建 Gram 矩阵。
    """
    squared_data = np.square(
        matrix.data
    )

    diagonal = np.bincount(
        matrix.indices,
        weights=squared_data,
        minlength=matrix.shape[1]
    )

    return diagonal.astype(
        np.float64,
        copy=False
    )


def run_cg(
    operator,
    rhs,
    x0,
    preconditioner,
    tolerance,
    maxiter,
    callback
):
    """
    兼容不同 SciPy 版本的 CG 参数名称。
    """
    try:
        return cg(
            operator,
            rhs,
            x0=x0,
            M=preconditioner,
            rtol=tolerance,
            atol=0.0,
            maxiter=maxiter,
            callback=callback
        )

    except TypeError:
        return cg(
            operator,
            rhs,
            x0=x0,
            M=preconditioner,
            tol=tolerance,
            atol=0.0,
            maxiter=maxiter,
            callback=callback
        )


def solve_lut_cg(
    interpolation_matrix,
    curvature_matrix,
    y,
    affine_prior,
    effective_smooth_weight,
    prior_weight,
    tolerance=1e-11,
    maxiter=5000,
    refinement_steps=2
):
    """
    Float64 矩阵自由预条件 CG。

    适用于 64³ LUT：

    1. 不显式构建 H = A.T A + C.T C；
    2. 不执行会产生巨量 fill-in 的稀疏 LU；
    3. 使用精确的 Jacobi 对角预条件器；
    4. 使用真实残差检查；
    5. 对残差方程执行额外校正求解。
    """
    sample_count = (
        interpolation_matrix.shape[0]
    )

    node_count = (
        interpolation_matrix.shape[1]
    )

    curvature_count = (
        curvature_matrix.shape[0]
    )

    smooth_coefficient = (
        effective_smooth_weight
        / curvature_count
    )

    prior_coefficient = (
        prior_weight
        / node_count
    )

    numerical_jitter = 1e-14

    def hessian_matvec(vector):
        data_term = (
            interpolation_matrix.T
            @ (
                interpolation_matrix
                @ vector
            )
        ) / sample_count

        smooth_term = (
            curvature_matrix.T
            @ (
                curvature_matrix
                @ vector
            )
        ) * smooth_coefficient

        diagonal_term = (
            prior_coefficient
            + numerical_jitter
        ) * vector

        result = (
            data_term
            + smooth_term
            + diagonal_term
        )

        return np.asarray(
            result,
            dtype=np.float64
        ).reshape(-1)

    hessian_operator = LinearOperator(
        shape=(
            node_count,
            node_count
        ),
        matvec=hessian_matvec,
        dtype=np.float64
    )

    rhs = (
        interpolation_matrix.T
        @ y
    ) / sample_count

    rhs += (
        prior_coefficient
        * affine_prior
    )

    rhs = np.asarray(
        rhs,
        dtype=np.float64
    )

    print(
        "构建精确 Jacobi 对角预条件器……"
    )

    data_diagonal = (
        sparse_gram_diagonal(
            interpolation_matrix
        )
        / sample_count
    )

    smooth_diagonal = (
        sparse_gram_diagonal(
            curvature_matrix
        )
        * smooth_coefficient
    )

    hessian_diagonal = (
        data_diagonal
        + smooth_diagonal
        + prior_coefficient
        + numerical_jitter
    )

    hessian_diagonal = np.maximum(
        hessian_diagonal,
        1e-30
    )

    def preconditioner_matvec(
        vector
    ):
        return (
            vector
            / hessian_diagonal
        )

    preconditioner = LinearOperator(
        shape=(
            node_count,
            node_count
 ),
        matvec=preconditioner_matvec,
        dtype=np.float64
    )

    solution_rgb = np.empty(
        (
            node_count,
            3
        ),
        dtype=np.float64
    )

    # SciPy CG 的内部停止准则和真实非预条件残差可能略有差异，
    # 因此使用稍宽于 rtol 的真实残差验收阈值。
    accepted_residual = max(
        10.0 * tolerance,
        1e-12
    )

    for channel in range(3):
        print(
            f"求解通道 {channel + 1}/3……"
        )

        rhs_channel = rhs[
            :,
            channel
        ]

        # 仿射先验作为高质量初值
        solution = affine_prior[
            :,
            channel
        ].copy()

        total_iterations = 0
        converged = False

        for refinement in range(
            refinement_steps + 1
        ):
            if refinement == 0:
                solve_rhs = rhs_channel
                initial_guess = solution
            else:
                _, solve_rhs = relative_residual(
                    hessian_operator,
                    solution,
                    rhs_channel
                )

                initial_guess = np.zeros(
                    node_count,
                    dtype=np.float64
                )

            iteration_counter = [0]

            def callback(_):
                iteration_counter[0] += 1

            correction_or_solution, info = run_cg(
                operator=hessian_operator,
                rhs=solve_rhs,
                x0=initial_guess,
                preconditioner=preconditioner,
                tolerance=tolerance,
                maxiter=maxiter,
                callback=callback
            )

            total_iterations += (
                iteration_counter[0]
            )

            if info < 0:
                raise RuntimeError(
                    f"CG 求解通道 {channel + 1} "
                    f"时发生错误，info={info}。"
                )

            if refinement == 0:
                solution = (
                    correction_or_solution
                )
            else:
                solution += (
                    correction_or_solution
                )

            current_residual, _ = relative_residual(
                hessian_operator,
                solution,
                rhs_channel
            )

            print(
                f"  第 {refinement + 1} 轮，"
                f"本轮迭代={iteration_counter[0]}，"
                f"累计迭代={total_iterations}，"
                f"真实相对残差="
                f"{current_residual:.6e}"
            )

            if (
                current_residual
                <= accepted_residual
            ):
                converged = True
                break

            if info > 0:
                print(
                    f"  提示：本轮 CG 达到最大迭代次数，"
                    f"继续执行残差校正。"
                )

        if not converged:
            final_residual, _ = relative_residual(
                hessian_operator,
                solution,
                rhs_channel
            )

            raise RuntimeError(
                f"通道 {channel + 1} 未达到精度要求。"
                f"目标真实相对残差 <= "
                f"{accepted_residual:.3e}，"
                f"实际为 {final_residual:.6e}。"
                f"请增大 --cg_maxiter，"
                f"或适当增大 --prior。"
            )

        if not np.isfinite(
            solution
        ).all():
            raise RuntimeError(
                f"CG 求解通道 {channel + 1} "
                f"得到 NaN 或 Inf。"
            )

        solution_rgb[
            :,
            channel
        ] = solution

    return solution_rgb


def fit_lut(
    x,
    y,
    lut_size,
    smooth_weight,
    prior_weight,
    solver_name,
    cg_tolerance,
    cg_maxiter,
    cg_refinement_steps,
    clip_output
):
    if lut_size not in SUPPORTED_LUT_SIZES:
        raise ValueError(
            f"不支持的 LUT 尺寸：{lut_size}，"
            f"支持的尺寸为 "
            f"{SUPPORTED_LUT_SIZES}。"
        )

    node_count = (
        lut_size ** 3
    )

    print(
        "\n构建三线性插值矩阵……"
    )

    interpolation_matrix = (
        build_trilinear_matrix(
            x,
            lut_size
        )
    )

    print(
        "构建 LUT 二阶平滑矩阵……"
    )

    curvature_matrix = (
        build_curvature_matrix(
            lut_size
        )
    )

    print(
        f"插值矩阵："
        f"shape={interpolation_matrix.shape}, "
        f"nnz={interpolation_matrix.nnz}"
    )

    print(
        f"平滑矩阵："
        f"shape={curvature_matrix.shape}, "
        f"nnz={curvature_matrix.nnz}"
    )

    grid_axis = np.linspace(
        0.0,
        1.0,
        lut_size,
        dtype=np.float64
    )

    grid_r, grid_g, grid_b = np.meshgrid(
        grid_axis,
        grid_axis,
        grid_axis,
        indexing="ij"
    )

    grid_rgb = np.stack(
        [
            grid_r,
            grid_g,
            grid_b
        ],
        axis=-1
    ).reshape(
        -1,
        3
    )

    print(
        "拟合全局仿射颜色先验……"
    )

    affine_prior = fit_affine_prior(
        x=x,
        y=y,
        grid_rgb=grid_rgb
    )

    (
        effective_smooth_weight,
        smooth_scale
    ) = calculate_effective_smooth_weight(
        smooth_weight=smooth_weight,
        lut_size=lut_size
    )

    print(
        f"平滑权重：\n"
        f"  参考尺寸："
        f"{REFERENCE_LUT_SIZE}³\n"
        f"  参考权重："
        f"{smooth_weight:.12g}\n"
        f"  尺寸缩放："
        f"{smooth_scale:.12g}\n"
        f"  有效权重："
        f"{effective_smooth_weight:.12g}"
    )

    if solver_name == "auto":
        if lut_size <= 33:
            selected_solver = "direct"
        else:
            selected_solver = "cg"
    else:
        selected_solver = (
            solver_name
        )

    if (
        lut_size == 64
        and selected_solver == "direct"
    ):
        print(
            "警告：64³ LUT 使用直接 LU "
            "可能产生巨量 fill-in 并导致 OOM。"
        )

    print(
        f"求解器：{selected_solver}"
    )

    print(
        f"求解 {lut_size}³ LUT，"
        f"共 {node_count} 个节点……"
    )

    if selected_solver == "direct":
        lut_flat = solve_lut_direct(
            interpolation_matrix=(
                interpolation_matrix
            ),
            curvature_matrix=(
                curvature_matrix
            ),
            y=y,
            affine_prior=affine_prior,
            effective_smooth_weight=(
                effective_smooth_weight
            ),
            prior_weight=prior_weight
        )

    elif selected_solver == "cg":
        lut_flat = solve_lut_cg(
            interpolation_matrix=(
                interpolation_matrix
            ),
            curvature_matrix=(
                curvature_matrix
            ),
            y=y,
            affine_prior=affine_prior,
            effective_smooth_weight=(
                effective_smooth_weight
            ),
            prior_weight=prior_weight,
            tolerance=cg_tolerance,
            maxiter=cg_maxiter,
            refinement_steps=(
                cg_refinement_steps
            )
        )

    else:
        raise ValueError(
            f"不支持的求解器："
            f"{selected_solver}"
        )

    lut_flat = np.asarray(
        lut_flat,
        dtype=np.float64
    )

    if not np.isfinite(
        lut_flat
    ).all():
        raise RuntimeError(
            "LUT 求解结果包含 NaN 或 Inf。"
        )

    raw_min = float(
        lut_flat.min()
    )

    raw_max = float(
        lut_flat.max()
    )

    out_of_range = (
        (lut_flat < 0.0)
        |
        (lut_flat > 1.0)
    )

    clip_ratio = float(
        np.mean(out_of_range)
    )

    print(
        f"求解后 LUT 范围："
        f"[{raw_min:.12g}, "
        f"{raw_max:.12g}]"
    )

    print(
        f"超出 [0,1] 的节点通道比例："
        f"{clip_ratio * 100.0:.8f}%"
    )

    if clip_output:
        print(
            "根据 --clip_output 将 LUT "
            "裁剪到 [0,1]。"
        )

        lut_flat = np.clip(
            lut_flat,
            0.0,
            1.0
        )
    else:
        print(
            "未裁剪 LUT，以保留正规方程的"
            "高精度最优解。"
        )

    print(
        "计算拟合样本误差……"
    )

    prediction = (
        interpolation_matrix
        @ lut_flat
    )

    error = (
        prediction
        - y
    )

    mae = np.mean(
        np.abs(error)
    )

    rmse = np.sqrt(
        np.mean(
            error ** 2
        )
    )

    p95 = np.percentile(
        np.abs(error),
        95
    )

    if rmse > 0:
        psnr = (
            20.0
            * np.log10(
                1.0 / rmse
            )
        )
    else:
        psnr = float("inf")

    print(
        "\n拟合结果（拟合样本集）："
    )

    print(
        f"  MAE：  "
        f"{mae:.12g}，约 "
        f"{mae * 255.0:.8f}/255"
    )

    print(
        f"  RMSE： "
        f"{rmse:.12g}，约 "
        f"{rmse * 255.0:.8f}/255"
    )

    print(
        f"  P95：  "
        f"{p95:.12g}，约 "
        f"{p95 * 255.0:.8f}/255"
    )

    print(
        f"  PSNR： "
        f"{psnr:.8f} dB"
    )

    return lut_flat.reshape(
        lut_size,
        lut_size,
        lut_size,
        3
    )


def write_cube(
    path,
    lut,
    title="Fitted 3D LUT"
):
    """
    写入标准 .cube 文件。

    Numpy LUT 排列：
        lut[R, G, B, output_channel]

    .cube 文件排列：
        R 索引变化最快；
        G 次之；
        B 最慢。

    使用 17 位有效数字写出，最大程度保留 Float64 精度。
    """
    path = Path(path)

    lut_size = (
        lut.shape[0]
    )

    safe_title = str(
        title
    ).replace(
        '"',
        "'"
    )

    with path.open(
        "w",
        encoding="utf-8"
    ) as file:
        file.write(
            f'TITLE "{safe_title}"\n'
        )

        file.write(
            f"LUT_3D_SIZE "
            f"{lut_size}\n"
        )

        file.write(
            "DOMAIN_MIN "
            "0.0 0.0 0.0\n"
        )

        file.write(
            "DOMAIN_MAX "
            "1.0 1.0 1.0\n"
        )

        for b in range(
            lut_size
        ):
            for g in range(
                lut_size
            ):
                for r in range(
                    lut_size
                ):
                    value = lut[
                        r,
                        g,
                        b
                    ]

                    file.write(
                        f"{value[0]:.17g} "
                        f"{value[1]:.17g} "
                        f"{value[2]:.17g}\n"
                    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "根据严格配准的调色前后图片，"
            "高精度拟合 17/33/64 阶 3D LUT"
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
        default="fitted_lut",
        help=(
            "输出文件前缀，"
            "默认 fitted_lut"
        )
    )

    parser.add_argument(
        "--size",
        type=int,
        choices=SUPPORTED_LUT_SIZES,
        default=17,
        help=(
            "LUT 每轴节点数，"
            "可选 17、33、64，"
            "默认 17"
        )
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=10000000,
        help=(
            "最终用于拟合的最大像素数量，"
            "默认 500000；"
            "64 阶建议至少 1000000"
        )
    )

    parser.add_argument(
        "--oversample",
        type=int,
        default=4,
        help=(
            "分层采样前的候选样本倍数，"
            "默认 4"
        )
    )

    parser.add_argument(
        "--smooth",
        type=float,
        default=0.01,
        help=(
            "以 17³ LUT 为参考的二阶平滑权重，"
            "默认 0.01；"
            "33/64 阶会自动缩放"
        )
    )

    parser.add_argument(
        "--prior",
        type=float,
        default=0.001,
        help=(
            "未覆盖颜色区域的仿射先验权重，"
            "默认 0.001"
        )
    )

    parser.add_argument(
        "--solver",
        choices=(
            "auto",
            "direct",
            "cg"
        ),
        default="auto",
        help=(
            "求解器；"
            "auto 对 17/33 使用 Float64 LU，"
            "对 64 使用 Float64 CG"
        )
    )

    parser.add_argument(
        "--cg_tol",
        type=float,
        default=1e-11,
        help=(
            "CG 内部相对收敛阈值，"
            "默认 1e-11"
        )
    )

    parser.add_argument(
        "--cg_maxiter",
        type=int,
        default=5000,
        help=(
            "每轮 CG 最大迭代次数，"
            "默认 5000"
        )
    )

    parser.add_argument(
        "--cg_refine_steps",
        type=int,
        default=2,
        help=(
            "CG 残差校正轮数，"
            "默认 2"
        )
    )

    parser.add_argument(
        "--clip_output",
        action="store_true",
        help=(
            "将最终 LUT 裁剪到 [0,1]；"
            "默认不裁剪，以保留高精度最优解"
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="随机种子"
    )

    args = parser.parse_args()

    if args.max_samples <= 0:
        raise ValueError(
            "max_samples 必须大于 0。"
        )

    if args.oversample < 1:
        raise ValueError(
            "oversample 必须大于等于 1。"
        )

    if args.smooth < 0:
        raise ValueError(
            "smooth 必须大于等于 0。"
        )

    if args.prior < 0:
        raise ValueError(
            "prior 必须大于等于 0。"
        )

    if args.cg_tol <= 0:
        raise ValueError(
            "cg_tol 必须大于 0。"
        )

    if args.cg_maxiter <= 0:
        raise ValueError(
            "cg_maxiter 必须大于 0。"
        )

    if args.cg_refine_steps < 0:
        raise ValueError(
            "cg_refine_steps 必须大于等于 0。"
        )

    if (
        args.size == 64
        and args.max_samples < 1000000
    ):
        print(
            "提示：64³ LUT 有 262144 个节点，"
            "当前 max_samples 小于 1000000。"
            "如果颜色覆盖不足，建议提高样本数量。"
        )

    pairs = find_image_pairs(
        args.src,
        args.dst
    )

    x, y = collect_pixel_samples(
        pairs=pairs,
        max_samples=args.max_samples,
        oversample=args.oversample,
        lut_size=args.size,
        seed=args.seed
    )

    lut = fit_lut(
        x=x,
        y=y,
        lut_size=args.size,
        smooth_weight=args.smooth,
        prior_weight=args.prior,
        solver_name=args.solver,
        cg_tolerance=args.cg_tol,
        cg_maxiter=args.cg_maxiter,
        cg_refinement_steps=(
            args.cg_refine_steps
        ),
        clip_output=args.clip_output
    )

    output_base = Path(
        args.out
    )

    if output_base.suffix.lower() in {
        ".npy",
        ".cube"
    }:
        output_base = (
            output_base
            .with_suffix("")
        )

    output_base.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # 使用字符串追加扩展名，
    # 避免 result.v1 被 with_suffix 截断
    npy_path = Path(
        f"{output_base}.npy"
    )

    cube_path = Path(
        f"{output_base}.cube"
    )

    # 使用 Float64 保存，不降低求解精度
    np.save(
        npy_path,
        np.asarray(
            lut,
            dtype=np.float64
        )
    )

    write_cube(
        cube_path,
        lut,
        title=output_base.name
    )

    print(
        "\n输出完成："
    )

    print(
        f"  NPY：  {npy_path}"
    )

    print(
        f"  CUBE： {cube_path}"
    )

    print(
        f"  LUT shape：{lut.shape}"
    )

    print(
        f"  NPY dtype：{lut.dtype}"
    )

    print(
        "  NPY 维度顺序："
        "[R, G, B, output_RGB]"
    )

    print(
        f"  CUBE LUT_3D_SIZE："
        f"{args.size}"
    )


if __name__ == "__main__":
    main()