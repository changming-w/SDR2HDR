"""
LUT Bank 构建 + 测试脚本

原理：
  离线：用模型处理大量图片 → 每张图得到 (D, 12) 系数 → K-Means 聚类成 K 个中心
        → 对每个中心生成 33×33×33×3 的纯 3D LUT → 存为 LUT Bank
  部署：跑模型得到 (D, 12) → 在 LUT Bank 中找最近的 K 个 → 加权插值 → 纯 3D LUT → 查表

用法：
  # 构建 LUT Bank
  python scripts/build_lut_bank.py --mode build \
    --checkpoint experiments/train_hdrnet_edr_original/models/net_g_latest.pth \
    --image_dir /mnt/data/wcm/20260711_color_enhance_plants2/sdr \
    --output_dir results/lut_bank_k128 \
    --K 128 --lut_size 33

  # 用 LUT Bank 测试
  python scripts/build_lut_bank.py --mode test \
    --lut_bank_dir results/lut_bank_k128 \
    --image_dir /mnt/data/wcm/20260711_color_enhance_plants2/阳光绿原图 \
    --output_dir results/lut_bank_test_output
"""

import sys
sys.path.insert(0, '/mnt/data/wcm/HDRNet')

import argparse
import os
import glob
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import time


# =====================================================================
# 模型加载
# =====================================================================
def load_model(checkpoint_path, device='cuda'):
    from basicsr.archs.hdrnet_original_arch import HDRnetOrig
    model = HDRnetOrig(
        num_in_ch=3, num_out_ch=3,
        grid_H=32, grid_W=32, grid_D=16,
        base_channels=16, num_downsample=3, num_residual=2,
        fixed_ds_size=512,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt['params'] if 'params' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"模型加载成功: {checkpoint_path}")
    return model


# =====================================================================
# 从图像提取 (D, 12) 系数
# =====================================================================
def extract_coeffs(model, image_paths, device='cuda'):
    """对每张图跑模型，得到空间平均后的 (D, 12) 系数"""
    coeffs_list = []
    valid_paths = []

    print(f"从 {len(image_paths)} 张图像提取系数...")
    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            print(f"  跳过: {path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_t = img_t.unsqueeze(0).to(device)

        with torch.no_grad():
            x_low = F.interpolate(img_t, size=(512, 512), mode='bilinear', align_corners=False)
            grid = model.lowres_net(x_low)  # (1, D, Hg, Wg, 12)
            coeff = grid.squeeze(0).mean(dim=(1, 2)).cpu().numpy()  # (D, 12)

        coeffs_list.append(coeff)
        valid_paths.append(path)

        if (i + 1) % 50 == 0:
            print(f"  已处理 {i+1}/{len(image_paths)}")

    print(f"有效图像: {len(coeffs_list)} 张")
    return np.stack(coeffs_list, axis=0), valid_paths  # (N, D, 12)


# =====================================================================
# K-Means 聚类
# =====================================================================
def kmeans_clustering(data, K, max_iter=100, tol=1e-4):
    """
    对 (N, D, 12) 数据做 K-Means 聚类
    返回: centers (K, D, 12), labels (N,)
    """
    N = data.shape[0]
    # 展平: (N, D*12)
    data_flat = data.reshape(N, -1)

    # 随机初始化中心
    np.random.seed(42)
    indices = np.random.choice(N, K, replace=False)
    centers = data_flat[indices].copy()  # (K, D*12)

    for iteration in range(max_iter):
        # 计算距离 (N, K)
        dists = np.linalg.norm(data_flat[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)  # (N,)

        # 更新中心
        new_centers = np.zeros_like(centers)
        for k in range(K):
            mask = labels == k
            if mask.sum() > 0:
                new_centers[k] = data_flat[mask].mean(axis=0)
            else:
                new_centers[k] = centers[k]

        # 检查收敛
        shift = np.linalg.norm(new_centers - centers)
        centers = new_centers
        if shift < tol:
            print(f"  K-Means 在第 {iteration+1} 轮收敛")
            break

    # 统计每个 cluster 的样本数
    unique, counts = np.unique(labels, return_counts=True)
    print(f"  聚类分布: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")
    empty = K - len(unique)
    if empty > 0:
        print(f"  警告: {empty} 个空 cluster")

    centers = centers.reshape(K, data.shape[1], data.shape[2])  # (K, D, 12)
    return centers, labels


# =====================================================================
# 从 (D, 12) 系数生成纯 3D LUT
# =====================================================================
def rgb_to_luminance(rgb):
    """BT.709 亮度"""
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def coeffs_to_3dlut(coeffs, size=33):
    """
    从 (D, 12) 仿射系数生成 (size, size, size, 3) 的纯 3D LUT
    """
    D = coeffs.shape[0]

    # 创建 RGB 网格
    r = np.linspace(0, 1, size)
    g = np.linspace(0, 1, size)
    b = np.linspace(0, 1, size)
    rr, gg, bb = np.meshgrid(r, g, b, indexing='ij')
    input_rgb = np.stack([rr, gg, bb], axis=-1)  # (S, S, S, 3)
    input_flat = input_rgb.reshape(-1, 3)  # (N, 3)

    # BT.709 亮度 → 深度
    luminance = rgb_to_luminance(input_rgb)  # (S, S, S)
    z = luminance * (D - 1)
    z_flat = z.reshape(-1)

    # 沿深度轴插值
    z0 = np.floor(z_flat).clip(0, D - 1).astype(int)
    z1 = np.clip(z0 + 1, 0, D - 1)
    dz = z_flat - z0

    coeff0 = coeffs[z0]  # (N, 12)
    coeff1 = coeffs[z1]
    coeff = coeff0 * (1 - dz[:, None]) + coeff1 * dz[:, None]  # (N, 12)

    # 仿射变换: y = A @ x + b
    A = coeff[:, :9].reshape(-1, 3, 3)  # (N, 3, 3)
    bias = coeff[:, 9:]  # (N, 3)

    output_flat = np.einsum('nij,nj->ni', A, input_flat) + bias  # (N, 3)
    output_flat = output_flat.clip(0, 1)

    return output_flat.reshape(size, size, size, 3)


# =====================================================================
# 三线性插值查 3D LUT
# =====================================================================
def apply_3dlut(img, lut_cube):
    """
    用 3D LUT 处理图像（三线性插值查表）
    img: (H, W, 3) uint8
    lut_cube: (S, S, S, 3) float [0, 1]
    返回: (H, W, 3) uint8
    """
    H, W, C = img.shape
    size = lut_cube.shape[0]

    img_float = img.astype(np.float32) / 255.0

    r = img_float[..., 0] * (size - 1)
    g = img_float[..., 1] * (size - 1)
    b = img_float[..., 2] * (size - 1)

    r0 = np.floor(r).clip(0, size - 2).astype(int)
    g0 = np.floor(g).clip(0, size - 2).astype(int)
    b0 = np.floor(b).clip(0, size - 2).astype(int)
    r1 = r0 + 1
    g1 = g0 + 1
    b1 = b0 + 1

    dr = (r - r0)[..., None]
    dg = (g - g0)[..., None]
    db = (b - b0)[..., None]

    c000 = lut_cube[r0, g0, b0]
    c001 = lut_cube[r0, g0, b1]
    c010 = lut_cube[r0, g1, b0]
    c011 = lut_cube[r0, g1, b1]
    c100 = lut_cube[r1, g0, b0]
    c101 = lut_cube[r1, g0, b1]
    c110 = lut_cube[r1, g1, b0]
    c111 = lut_cube[r1, g1, b1]

    c00 = c000 * (1 - db) + c001 * db
    c01 = c010 * (1 - db) + c011 * db
    c10 = c100 * (1 - db) + c101 * db
    c11 = c110 * (1 - db) + c111 * db

    c0 = c00 * (1 - dg) + c01 * dg
    c1 = c10 * (1 - dg) + c11 * dg

    c = c0 * (1 - dr) + c1 * dr

    return (c * 255).clip(0, 255).astype(np.uint8)


# =====================================================================
# 构建 LUT Bank
# =====================================================================
def build_lut_bank(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载模型
    model = load_model(args.checkpoint, device)

    # 2. 收集图像
    image_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
        image_paths.extend(glob.glob(f'{args.image_dir}/{ext}'))
    image_paths = sorted(image_paths)
    print(f"找到 {len(image_paths)} 张图像")

    # 3. 提取系数
    coeffs_all, valid_paths = extract_coeffs(model, image_paths, device)
    # coeffs_all: (N, D, 12)

    # 4. K-Means 聚类
    K = args.K
    print(f"\nK-Means 聚类 (K={K})...")
    centers, labels = kmeans_clustering(coeffs_all, K)
    # centers: (K, D, 12)

    # 5. 对每个中心生成 3D LUT
    lut_size = args.lut_size
    print(f"\n生成 {K} 个 3D LUT ({lut_size}³)...")
    lut_bank = np.zeros((K, lut_size, lut_size, lut_size, 3), dtype=np.float32)

    for k in range(K):
        lut_bank[k] = coeffs_to_3dlut(centers[k], size=lut_size)
        if (k + 1) % 20 == 0:
            print(f"  已生成 {k+1}/{K}")

    # 6. 保存
    # LUT Bank (uint8 节省空间)
    lut_bank_uint8 = (lut_bank * 255).clip(0, 255).astype(np.uint8)
    np.save(os.path.join(args.output_dir, 'lut_bank_uint8.npy'), lut_bank_uint8)

    # 中心系数 (float32)
    np.save(os.path.join(args.output_dir, 'centers.npy'), centers)

    # 元信息
    meta = {
        'K': K,
        'lut_size': lut_size,
        'grid_D': centers.shape[1],
        'num_images': len(valid_paths),
    }
    np.save(os.path.join(args.output_dir, 'meta.npy'), meta)

    # 统计
    storage_mb = lut_bank_uint8.nbytes / (1024 * 1024)
    print(f"\nLUT Bank 构建完成:")
    print(f"  K = {K}")
    print(f"  LUT 尺寸 = {lut_size}³×3")
    print(f"  存储大小 = {storage_mb:.1f} MB (uint8)")
    print(f"  保存目录 = {args.output_dir}")


# =====================================================================
# 用 LUT Bank 测试
# =====================================================================
def test_with_lut_bank(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载 LUT Bank
    lut_bank_uint8 = np.load(os.path.join(args.lut_bank_dir, 'lut_bank_uint8.npy'))
    centers = np.load(os.path.join(args.lut_bank_dir, 'centers.npy'))
    meta = np.load(os.path.join(args.lut_bank_dir, 'meta.npy'), allow_pickle=True).item()

    K = meta['K']
    lut_size = meta['lut_size']
    lut_bank = lut_bank_uint8.astype(np.float32) / 255.0
    print(f"LUT Bank 加载成功: K={K}, lut_size={lut_size}")

    # 2. 加载模型（用于提取测试图的系数）
    model = load_model(args.checkpoint, device)

    # 3. 收集测试图
    image_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.webp']:
        image_paths.extend(glob.glob(f'{args.image_dir}/{ext}'))
    image_paths = sorted(image_paths)
    print(f"找到 {len(image_paths)} 张测试图")

    # 4. 对每张图：提取系数 → 查 LUT Bank → 生成 3D LUT → 查表
    top_k = args.top_k
    print(f"使用 top-{top_k} 加权插值")

    times = []
    for i, path in enumerate(image_paths):
        t0 = time.time()

        img = cv2.imread(path)
        if img is None:
            print(f"  跳过: {path}")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 提取系数
        img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_t = img_t.unsqueeze(0).to(device)

        with torch.no_grad():
            x_low = F.interpolate(img_t, size=(512, 512), mode='bilinear', align_corners=False)
            grid = model.lowres_net(x_low)
            coeff = grid.squeeze(0).mean(dim=(1, 2)).cpu().numpy()  # (D, 12)

        # 在 LUT Bank 中找最近的 top-K 个中心
        coeff_flat = coeff.reshape(-1)  # (D*12,)
        centers_flat = centers.reshape(K, -1)  # (K, D*12)
        dists = np.linalg.norm(centers_flat - coeff_flat[None, :], axis=1)  # (K,)

        # top-K 最近邻
        top_indices = np.argsort(dists)[:top_k]
        top_dists = dists[top_indices]

        # 加权：距离倒数
        weights = 1.0 / (top_dists + 1e-8)
        weights = weights / weights.sum()

        # 加权平均 LUT
        lut_frame = np.zeros((lut_size, lut_size, lut_size, 3), dtype=np.float32)
        for j, idx in enumerate(top_indices):
            lut_frame += weights[j] * lut_bank[idx]

        # 查表
        out_rgb = apply_3dlut(img_rgb, lut_frame)

        # 保存
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(args.output_dir, os.path.basename(path))
        cv2.imwrite(out_path, out_bgr)

        elapsed = time.time() - t0
        times.append(elapsed)

        if (i + 1) % 50 == 0:
            print(f"  已处理 {i+1}/{len(image_paths)}, 平均 {np.mean(times)*1000:.0f} ms/张")

    print(f"\n完成！")
    print(f"  处理 {len(times)} 张图")
    print(f"  平均耗时: {np.mean(times)*1000:.0f} ms/张")
    print(f"  输出目录: {args.output_dir}")


# =====================================================================
# 主入口
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description='LUT Bank 构建 + 测试')
    parser.add_argument('--mode', type=str, required=True, choices=['build', 'test'],
                        help='build=构建 LUT Bank, test=用 LUT Bank 测试')

    # 构建参数
    parser.add_argument('--checkpoint', type=str,
                        default='experiments/train_hdrnet_edr_original/models/net_g_latest.pth')
    parser.add_argument('--image_dir', type=str,
                        default='/mnt/data/wcm/20260711_color_enhance_plants2/sdr')
    parser.add_argument('--output_dir', type=str, default='results/lut_bank_k128')
    parser.add_argument('--K', type=int, default=128, help='LUT Bank 大小')
    parser.add_argument('--lut_size', type=int, default=33, help='3D LUT 尺寸')

    # 测试参数
    parser.add_argument('--lut_bank_dir', type=str, default='results/lut_bank_k128')
    parser.add_argument('--top_k', type=int, default=8, help='插值用的近邻数')

    args = parser.parse_args()

    if args.mode == 'build':
        build_lut_bank(args)
    else:
        test_with_lut_bank(args)


if __name__ == '__main__':
    main()
