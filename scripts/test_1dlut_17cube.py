"""
从 1D LUT 模型推理 → 生成 17 阶 3D LUT (CSV) → 应用 3D LUT 输出 EDR

流程:
1. 加载模型 (net_g_latest.pth)
2. 对每张图: LowResNet → (4096, 12) → 生成 17³ × 3 的 3D LUT
3. 保存 3D LUT 为 CSV (列: R, G, B, R_out, G_out, B_out)
4. 三线性插值应用 3D LUT → 输出 EDR
"""

import os
import sys
import csv
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, '/mnt/data/wcm/HDRNet')
from basicsr.archs.hdrnet_1dlut_arch import HDRnet1DLUT


def generate_3dlut_from_coeffs(coeffs, size=17):
    """
    从 (4096, 12) 系数生成 3D LUT

    Args:
        coeffs: (4096, 12) 亮度 bin 的仿射系数
        size: 3D LUT 阶数 (默认 17)

    Returns:
        lut: (size, size, size, 3) 的 3D LUT
    """
    num_bins = coeffs.shape[0]

    r_axis = np.linspace(0, 1, size)
    g_axis = np.linspace(0, 1, size)
    b_axis = np.linspace(0, 1, size)

    lut = np.zeros((size, size, size, 3), dtype=np.float32)

    for i, rv in enumerate(r_axis):
        for j, gv in enumerate(g_axis):
            for k, bv in enumerate(b_axis):
                # BT.709 亮度
                y = 0.2126 * rv + 0.7152 * gv + 0.0722 * bv

                # 找最近的 bin
                bin_idx = int(np.clip(y * (num_bins - 1), 0, num_bins - 1))
                coeff = coeffs[bin_idx]  # (12,)

                # 仿射变换: y = A @ x + b
                A = coeff[:9].reshape(3, 3)
                bias = coeff[9:12]

                rgb_in = np.array([rv, gv, bv])
                rgb_out = A @ rgb_in + bias

                lut[i, j, k] = np.clip(rgb_out, 0, 1)

    return lut


def save_3dlut_csv(lut, output_path, size=17):
    """
    保存 3D LUT 为 CSV 文件

    格式: R, G, B, R_out, G_out, B_out
    共 size³ 行
    """
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['R', 'G', 'B', 'R_out', 'G_out', 'B_out'])

        for i in range(size):
            for j in range(size):
                for k in range(size):
                    r_in = i / (size - 1)
                    g_in = j / (size - 1)
                    b_in = k / (size - 1)
                    r_out, g_out, b_out = lut[i, j, k]
                    writer.writerow([
                        f'{r_in:.6f}', f'{g_in:.6f}', f'{b_in:.6f}',
                        f'{r_out:.6f}', f'{g_out:.6f}', f'{b_out:.6f}'
                    ])


def apply_3dlut(image, lut):
    """
    用三线性插值应用 3D LUT

    Args:
        image: (H, W, 3) RGB 图像 [0, 1]
        lut: (size, size, size, 3) 的 3D LUT

    Returns:
        output: (H, W, 3) 处理后的图像 [0, 1]
    """
    size = lut.shape[0]
    h, w = image.shape[:2]

    # 向量化三线性插值
    rgb_scaled = image * (size - 1)  # (H, W, 3)

    r = rgb_scaled[:, :, 0]
    g = rgb_scaled[:, :, 1]
    b = rgb_scaled[:, :, 2]

    r0 = np.floor(r).astype(np.int32)
    g0 = np.floor(g).astype(np.int32)
    b0 = np.floor(b).astype(np.int32)
    r1 = np.minimum(r0 + 1, size - 1)
    g1 = np.minimum(g0 + 1, size - 1)
    b1 = np.minimum(b0 + 1, size - 1)

    dr = (r - r0)[..., np.newaxis]
    dg = (g - g0)[..., np.newaxis]
    db = (b - b0)[..., np.newaxis]

    c000 = lut[r0, g0, b0]
    c001 = lut[r0, g0, b1]
    c010 = lut[r0, g1, b0]
    c011 = lut[r0, g1, b1]
    c100 = lut[r1, g0, b0]
    c101 = lut[r1, g0, b1]
    c110 = lut[r1, g1, b0]
    c111 = lut[r1, g1, b1]

    c00 = c000 * (1 - dr) + c100 * dr
    c01 = c001 * (1 - dr) + c101 * dr
    c10 = c010 * (1 - dr) + c110 * dr
    c11 = c011 * (1 - dr) + c111 * dr

    c0 = c00 * (1 - dg) + c10 * dg
    c1 = c01 * (1 - dg) + c11 * dg

    output = c0 * (1 - db) + c1 * db

    return np.clip(output, 0, 1).astype(np.float32)


def extract_coeffs_from_model(model, image_path, device='cuda'):
    """
    从模型提取 (4096, 12) 系数
    """
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_t = img_t.unsqueeze(0).to(device)

    img_low = F.interpolate(img_t, size=(512, 512), mode='bilinear', align_corners=False)

    with torch.no_grad():
        coeffs = model.lowres_net(img_low)  # (1, 4096, 12)
        coeffs = coeffs.squeeze(0).cpu().numpy()  # (4096, 12)

    return coeffs


def main():
    # 配置
    checkpoint_path = '/mnt/data/wcm/HDRNet/experiments/train_hdrnet_1dlut/train_hdrnet_1dlut/models/net_g_latest.pth'
    input_dir = '/mnt/data/wcm/20260711_color_enhance_plants2/阳光绿原图'
    output_dir = '/mnt/data/wcm/20260711_color_enhance_plants2/阳光绿原图_1dlut_17cube'
    lut_csv_dir = os.path.join(output_dir, 'lut_csv')
    lut_size = 17

    os.makedirs(lut_csv_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载模型
    print('加载模型...')
    model = HDRnet1DLUT(num_bins=4096, base_channels=16, num_downsample=4, fixed_ds_size=512)
    state_dict = torch.load(checkpoint_path, map_location=device)
    if 'params_ema' in state_dict:
        state_dict = state_dict['params_ema']
    elif 'params' in state_dict:
        state_dict = state_dict['params']
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    print(f'模型已加载: {checkpoint_path}')

    # 获取所有图像
    img_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f'找到 {len(img_files)} 张图像')

    for img_name in tqdm(img_files, desc='处理中'):
        img_path = os.path.join(input_dir, img_name)
        base_name = os.path.splitext(img_name)[0]

        # 1. 提取系数
        coeffs = extract_coeffs_from_model(model, img_path, device)  # (4096, 12)

        # 2. 生成 17 阶 3D LUT
        lut = generate_3dlut_from_coeffs(coeffs, size=lut_size)  # (17, 17, 17, 3)

        # 3. 保存 CSV
        csv_path = os.path.join(lut_csv_dir, f'{base_name}_3dlut_17.csv')
        save_3dlut_csv(lut, csv_path, size=lut_size)

        # 4. 读取原图并应用 3D LUT
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        output_rgb = apply_3dlut(img_rgb, lut)

        # 5. 保存 EDR 输出
        output_bgr = cv2.cvtColor((output_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        output_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_path, output_bgr)

    print(f'完成!')
    print(f'  EDR 输出: {output_dir}')
    print(f'  3D LUT CSV: {lut_csv_dir}')


if __name__ == '__main__':
    main()
