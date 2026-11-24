"""
用标准 3D LUT cube 处理测试图

原理：
  - 加载 3D LUT cube: (size, size, size, 3)
  - 对每个输入像素，做三线性插值查表
  - 输出增强后的图像
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


def apply_3dlut_cube(img, lut_cube, device='cuda'):
    """
    用 3D LUT cube 处理图像
    
    img: (H, W, 3) numpy array in [0, 255] uint8
    lut_cube: (size, size, size, 3) numpy array in [0, 1] float
    
    返回: (H, W, 3) numpy array in [0, 255] uint8
    """
    H, W, C = img.shape
    size = lut_cube.shape[0]
    
    # 转成 float [0, 1]
    img_float = img.astype(np.float32) / 255.0
    
    # 映射到 LUT 坐标 [0, size-1]
    r = img_float[..., 0] * (size - 1)
    g = img_float[..., 1] * (size - 1)
    b = img_float[..., 2] * (size - 1)
    
    # 下取整 / 上取整
    r0 = np.floor(r).clip(0, size - 2).astype(int)
    g0 = np.floor(g).clip(0, size - 2).astype(int)
    b0 = np.floor(b).clip(0, size - 2).astype(int)
    r1 = r0 + 1
    g1 = g0 + 1
    b1 = b0 + 1
    
    # 插值权重
    dr = r - r0
    dg = g - g0
    db = b - b0
    
    # 三线性插值查表
    # 8 个顶点
    c000 = lut_cube[r0, g0, b0]
    c001 = lut_cube[r0, g0, b1]
    c010 = lut_cube[r0, g1, b0]
    c011 = lut_cube[r0, g1, b1]
    c100 = lut_cube[r1, g0, b0]
    c101 = lut_cube[r1, g0, b1]
    c110 = lut_cube[r1, g1, b0]
    c111 = lut_cube[r1, g1, b1]
    
    # 权重
    dr = dr[..., None]
    dg = dg[..., None]
    db = db[..., None]
    
    # 插值
    c00 = c000 * (1 - db) + c001 * db
    c01 = c010 * (1 - db) + c011 * db
    c10 = c100 * (1 - db) + c101 * db
    c11 = c110 * (1 - db) + c111 * db
    
    c0 = c00 * (1 - dg) + c01 * dg
    c1 = c10 * (1 - dg) + c11 * dg
    
    c = c0 * (1 - dr) + c1 * dr
    
    # 转回 uint8
    output = (c * 255).clip(0, 255).astype(np.uint8)
    
    return output


def main():
    parser = argparse.ArgumentParser(description='用 3D LUT cube 处理测试图')
    parser.add_argument('--lut_cube', type=str,
                        default='results/3dlut_compressed/lut_3d_cube_64.npy')
    parser.add_argument('--image_dir', type=str,
                        default='/mnt/data/wcm/20260711_color_enhance_plants2/阳光绿原图')
    parser.add_argument('--output_dir', type=str,
                        default='results/3dlut_cube_test_output')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载 3D LUT cube
    lut_cube = np.load(args.lut_cube)
    print(f"3D LUT cube 加载成功: {args.lut_cube}, shape: {lut_cube.shape}")
    
    # 收集图像（支持多种格式）
    image_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.webp']:
        image_paths.extend(glob.glob(f'{args.image_dir}/{ext}'))
    image_paths = sorted(image_paths)
    print(f"找到 {len(image_paths)} 张图像")
    
    # 处理每张图像
    for i, path in enumerate(image_paths):
        # 读取图像
        img = cv2.imread(path)
        if img is None:
            print(f"  跳过: {path}")
            continue
        
        # BGR -> RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 用 3D LUT 处理
        out_rgb = apply_3dlut_cube(img_rgb, lut_cube)
        
        # RGB -> BGR 保存
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(args.output_dir, os.path.basename(path))
        cv2.imwrite(out_path, out_bgr)
        
        if (i + 1) % 50 == 0:
            print(f"  已处理 {i+1}/{len(image_paths)}")
    
    print(f"\n完成！输出保存在: {args.output_dir}")


if __name__ == '__main__':
    main()
