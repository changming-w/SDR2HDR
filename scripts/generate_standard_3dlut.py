"""
生成标准 3D LUT (RGB -> RGB)

原理：
  - 加载压缩后的 3D LUT 系数: (16, 12)
  - 对每个输入 RGB 值 (0-255):
    1. 计算亮度 -> 映射到深度 bin
    2. 插值仿射系数
    3. 应用变换: output = A @ input + b
  - 输出: 标准 3D LUT cube

用法：
  python scripts/generate_standard_3dlut.py \
    --lut_coeffs results/3dlut_compressed/lut_3d_16x12.npy \
    --output results/3dlut_compressed/lut_3d_cube_64.npy \
    --size 64
"""

import sys
sys.path.insert(0, '/mnt/data/wcm/HDRNet')

import argparse
import os
import numpy as np
import torch


def rgb_to_luminance(rgb):
    """RGB -> 亮度 (BT.709)"""
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def generate_3dlut_cube(lut_coeffs, size=64, device='cpu'):
    """
    生成标准 3D LUT cube
    
    lut_coeffs: (D, 12) numpy array - 仿射系数
    size: LUT 尺寸 (默认 64)
    
    返回: (size, size, size, 3) numpy array
    """
    D = lut_coeffs.shape[0]
    
    # 创建 RGB 网格
    r = np.linspace(0, 1, size)
    g = np.linspace(0, 1, size)
    b = np.linspace(0, 1, size)
    
    # 创建 3D 网格
    rr, gg, bb = np.meshgrid(r, g, b, indexing='ij')
    
    # 输入 RGB: (size, size, size, 3)
    input_rgb = np.stack([rr, gg, bb], axis=-1)
    input_flat = input_rgb.reshape(-1, 3)  # (N, 3)
    
    # 计算亮度 -> 深度坐标
    luminance = rgb_to_luminance(input_rgb)  # (size, size, size)
    z = luminance * (D - 1)  # 映射到 [0, D-1]
    z_flat = z.reshape(-1)
    
    # 下取整 / 上取整
    z0 = np.floor(z_flat).clip(0, D - 1).astype(int)
    z1 = np.clip(z0 + 1, 0, D - 1)
    dz = z_flat - z0
    
    # 插值系数
    # lut_coeffs: (D, 12)
    coeff0 = lut_coeffs[z0]  # (N, 12)
    coeff1 = lut_coeffs[z1]
    coeff = coeff0 * (1 - dz[:, None]) + coeff1 * dz[:, None]  # (N, 12)
    
    # 应用仿射变换: output = A @ input + b
    # A: 3x3 matrix, b: 3 vector
    A = coeff[:, :9].reshape(-1, 3, 3)  # (N, 3, 3)
    b = coeff[:, 9:]  # (N, 3)
    
    # output = A @ input + b
    input_flat_t = torch.from_numpy(input_flat).float().unsqueeze(-1)  # (N, 3, 1)
    A_t = torch.from_numpy(A).float()
    b_t = torch.from_numpy(b).float()
    
    output_flat = torch.bmm(A_t, input_flat_t).squeeze(-1) + b_t  # (N, 3)
    output_flat = output_flat.numpy().clip(0, 1)
    
    # reshape 回 3D cube
    output_cube = output_flat.reshape(size, size, size, 3)
    
    return output_cube


def save_3dlut_cube(cube, output_path, size):
    """保存 3D LUT cube"""
    # 保存为 numpy 文件 (float)
    np.save(output_path, cube)
    
    # 保存为 .cube 格式（标准 3D LUT 格式）
    cube_path = output_path.replace('.npy', '.cube')
    with open(cube_path, 'w') as f:
        f.write(f"TITLE \"HDRNet 3D LUT\"\n")
        f.write(f"LUT_3D_SIZE {size}\n")
        f.write(f"DOMAIN_MIN 0.0 0.0 0.0\n")
        f.write(f"DOMAIN_MAX 1.0 1.0 1.0\n")
        
        # .cube 格式: R 变化最快，然后 G，然后 B
        for b_idx in range(size):
            for g_idx in range(size):
                for r_idx in range(size):
                    rgb = cube[r_idx, g_idx, b_idx]
                    f.write(f"{rgb[0]:.6f} {rgb[1]:.6f} {rgb[2]:.6f}\n")
    
    # 保存为 uint8 版本（方便部署）
    cube_uint8 = (cube * 255).clip(0, 255).astype(np.uint8)
    uint8_path = output_path.replace('.npy', '_uint8.npy')
    np.save(uint8_path, cube_uint8)
    
    print(f"\n3D LUT cube 已保存:")
    print(f"  Float:  {output_path}")
    print(f"  .cube:  {cube_path}")
    print(f"  Uint8:  {uint8_path}")
    print(f"  尺寸:   {cube.shape}")


def main():
    parser = argparse.ArgumentParser(description='生成标准 3D LUT cube')
    parser.add_argument('--lut_coeffs', type=str,
                        default='results/3dlut_compressed/lut_3d_16x12.npy')
    parser.add_argument('--output', type=str,
                        default='results/3dlut_compressed/lut_3d_cube_64.npy')
    parser.add_argument('--size', type=int, default=64,
                        help='LUT 尺寸 (默认 64)')
    args = parser.parse_args()
    
    # 加载系数
    lut_coeffs = np.load(args.lut_coeffs)  # (16, 12)
    print(f"加载系数: {args.lut_coeffs}, shape: {lut_coeffs.shape}")
    
    # 生成 3D LUT cube
    print(f"生成 {args.size}x{args.size}x{args.size} 3D LUT...")
    cube = generate_3dlut_cube(lut_coeffs, size=args.size)
    
    # 保存
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_3dlut_cube(cube, args.output, args.size)
    
    print("\n完成！")


if __name__ == '__main__':
    main()
