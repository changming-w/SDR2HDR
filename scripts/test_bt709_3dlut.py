"""
从 BT.709 模型提取 grid 系数 → 生成纯 3D LUT → 处理测试图

流程：
  1. 加载 BT.709 模型
  2. 对多张图提取 grid → 空间平均 → (D, 12) → 取平均
  3. 遍历所有 RGB 采样点 → 亮度→深度→插值系数→仿射 → 生成 3D LUT
  4. 对测试图做三线性插值查表
"""

import sys
sys.path.insert(0, '/mnt/data/wcm/HDRNet')

import os
import glob
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import time


def load_model(checkpoint_path, device='cuda'):
    from basicsr.archs.hdrnet_bt709_arch import HDRnetBT709
    model = HDRnetBT709(
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


def extract_coeffs(model, image_paths, device='cuda'):
    """对多张图提取 grid 系数并空间平均"""
    coeffs_list = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_t = img_t.unsqueeze(0).to(device)
        with torch.no_grad():
            x_low = F.interpolate(img_t, size=(512, 512), mode='bilinear', align_corners=False)
            grid = model.lowres_net(x_low)  # (1, D, Hg, Wg, 12)
            coeff = grid.squeeze(0).mean(dim=(1, 2)).cpu().numpy()  # (D, 12)
        coeffs_list.append(coeff)
    print(f"提取了 {len(coeffs_list)} 张图的系数")
    return np.mean(np.stack(coeffs_list, axis=0), axis=0)  # (D, 12)


def coeffs_to_3dlut(coeffs, size=64):
    """从 (D, 12) 生成 (size, size, size, 3) 的纯 3D LUT"""
    D = coeffs.shape[0]
    r = np.linspace(0, 1, size)
    g = np.linspace(0, 1, size)
    b = np.linspace(0, 1, size)
    rr, gg, bb = np.meshgrid(r, g, b, indexing='ij')
    input_rgb = np.stack([rr, gg, bb], axis=-1).reshape(-1, 3)  # (N, 3)

    # BT.709 亮度 → 深度
    luminance = 0.2126 * input_rgb[:, 0] + 0.7152 * input_rgb[:, 1] + 0.0722 * input_rgb[:, 2]
    z = luminance * (D - 1)

    z0 = np.floor(z).clip(0, D - 1).astype(int)
    z1 = np.clip(z0 + 1, 0, D - 1)
    dz = z - z0

    coeff0 = coeffs[z0]  # (N, 12)
    coeff1 = coeffs[z1]
    coeff = coeff0 * (1 - dz[:, None]) + coeff1 * dz[:, None]

    A = coeff[:, :9].reshape(-1, 3, 3)
    bias = coeff[:, 9:]
    output = np.einsum('nij,nj->ni', A, input_rgb) + bias
    output = output.clip(0, 1).reshape(size, size, size, 3)
    return output


def apply_3dlut(img, lut):
    """三线性插值查表"""
    size = lut.shape[0]
    img_f = img.astype(np.float32) / 255.0
    r = img_f[..., 0] * (size - 1)
    g = img_f[..., 1] * (size - 1)
    b = img_f[..., 2] * (size - 1)

    r0 = np.floor(r).clip(0, size - 2).astype(int)
    g0 = np.floor(g).clip(0, size - 2).astype(int)
    b0 = np.floor(b).clip(0, size - 2).astype(int)
    dr = (r - r0)[..., None]
    dg = (g - g0)[..., None]
    db = (b - b0)[..., None]

    c000 = lut[r0, g0, b0]; c001 = lut[r0, g0, b0+1]
    c010 = lut[r0, g0+1, b0]; c011 = lut[r0, g0+1, b0+1]
    c100 = lut[r0+1, g0, b0]; c101 = lut[r0+1, g0, b0+1]
    c110 = lut[r0+1, g0+1, b0]; c111 = lut[r0+1, g0+1, b0+1]

    c00 = c000*(1-db) + c001*db; c01 = c010*(1-db) + c011*db
    c10 = c100*(1-db) + c101*db; c11 = c110*(1-db) + c111*db
    c0 = c00*(1-dg) + c01*dg; c1 = c10*(1-dg) + c11*dg
    c = c0*(1-dr) + c1*dr
    return (c * 255).clip(0, 255).astype(np.uint8)


def extract_single_coeffs(model, image, device='cuda'):
    """对单张图提取 grid 系数并空间平均 → (D, 12)"""
    img_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    img_t = img_t.unsqueeze(0).to(device)
    with torch.no_grad():
        x_low = F.interpolate(img_t, size=(512, 512), mode='bilinear', align_corners=False)
        grid = model.lowres_net(x_low)  # (1, D, Hg, Wg, 12)
        coeff = grid.squeeze(0).mean(dim=(1, 2)).cpu().numpy()  # (D, 12)
    return coeff


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = 'experiments/train_hdrnet_bt709/models/net_g_22000.pth'
    test_image_dir = '/mnt/data/wcm/20260711_color_enhance_plants2/阳光绿原图'
    output_dir = 'results/bt709_3dlut_per_image_output'
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载模型
    model = load_model(checkpoint, device)

    # 2. 收集测试图
    test_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.webp']:
        test_paths.extend(glob.glob(f'{test_image_dir}/{ext}'))
    test_paths = sorted(test_paths)
    print(f"共 {len(test_paths)} 张测试图，每张图独立生成 3D LUT...")

    t0 = time.time()
    for i, path in enumerate(test_paths):
        img = cv2.imread(path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 每张图：提取系数 → 生成 3D LUT → 处理
        coeffs = extract_single_coeffs(model, img_rgb, device)
        lut = coeffs_to_3dlut(coeffs, size=64)
        out_rgb = apply_3dlut(img_rgb, lut)
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(output_dir, os.path.basename(path))
        cv2.imwrite(out_path, out_bgr)

        if (i + 1) % 50 == 0:
            print(f"  已处理 {i+1}/{len(test_paths)}")

    elapsed = time.time() - t0
    print(f"\n完成！")
    print(f"  处理 {len(test_paths)} 张图，耗时 {elapsed:.1f}s")
    print(f"  平均 {elapsed/len(test_paths)*1000:.0f} ms/张")
    print(f"  输出目录: {output_dir}")


if __name__ == '__main__':
    main()
