"""
用生成的 3D LUT 跑测试图

原理：
  - 加载 3D LUT: (16, 12)
  - 加载模型（只用 GuideNet 生成 guide map）
  - 对每个像素，根据 guide 值在 3D LUT 的深度轴上做线性插值
  - 应用仿射变换: y = A(z) @ x + b(z)
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

from basicsr.archs.hdrnet_original_arch import HDRnetOrig


def load_model_and_lut(checkpoint_path, lut_path, device='cuda'):
    """加载模型和 3D LUT"""
    # 加载模型
    model = HDRnetOrig(
        num_in_ch=3, num_out_ch=3,
        grid_H=32, grid_W=32, grid_D=16,
        base_channels=16, num_downsample=3, num_residual=2,
        fixed_ds_size=512,
    ).to(device)
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    if 'params' in ckpt:
        state_dict = ckpt['params']
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"模型加载成功: {checkpoint_path}")
    
    # 加载 3D LUT
    lut = np.load(lut_path)  # (16, 12)
    lut_tensor = torch.from_numpy(lut).float().to(device)
    print(f"3D LUT 加载成功: {lut_path}, shape: {lut.shape}")
    
    return model, lut_tensor


def apply_3dlut(model, lut, img_tensor, device='cuda'):
    """
    用 3D LUT 处理图像
    
    img_tensor: (1, 3, H, W) in [0, 1]
    lut: (D, 12) tensor
    """
    B, C, H, W = img_tensor.shape
    D = lut.shape[0]
    
    # 生成 guide map
    with torch.no_grad():
        guide = model.guide_net(img_tensor)  # (1, 1, H, W) in [0, 1]
    
    # 深度坐标 z ∈ [0, D-1]
    z = guide * (D - 1)  # (1, 1, H, W)
    
    # 下取整 / 上取整
    z0 = torch.floor(z).clamp(0, D - 1).long()
    z1 = torch.clamp(z0 + 1, max=D - 1)
    dz = z - z0.float()
    
    # 从 LUT 中 gather 系数
    # lut: (D, 12) -> (D, 12)
    # z0, z1: (1, 1, H, W)
    
    # gather z0 的系数
    z0_flat = z0.view(-1)  # (H*W,)
    coeff0 = lut[z0_flat]  # (H*W, 12)
    coeff0 = coeff0.view(1, H, W, 12).permute(0, 3, 1, 2)  # (1, 12, H, W)
    
    # gather z1 的系数
    z1_flat = z1.view(-1)
    coeff1 = lut[z1_flat]
    coeff1 = coeff1.view(1, H, W, 12).permute(0, 3, 1, 2)
    
    # 线性插值
    coeff = coeff0 * (1 - dz) + coeff1 * dz  # (1, 12, H, W)
    
    # 仿射变换: y = A @ x + b
    A = coeff[:, :9, :, :].reshape(1, 3, 3, H, W)
    b = coeff[:, 9:, :, :].reshape(1, 3, 1, H, W)
    x_vec = img_tensor.reshape(1, 3, 1, H, W)
    out = (A * x_vec).sum(dim=2) + b.squeeze(2)
    
    return torch.clamp(out, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description='用 3D LUT 处理测试图')
    parser.add_argument('--checkpoint', type=str,
                        default='experiments/train_hdrnet_edr_original/models/net_g_latest.pth')
    parser.add_argument('--lut_path', type=str,
                        default='results/3dlut_compressed/lut_3d_16x12.npy')
    parser.add_argument('--image_dir', type=str,
                        default='/mnt/data/wcm/20260711_color_enhance_plants2/阳光绿原图')
    parser.add_argument('--output_dir', type=str,
                        default='results/3dlut_test_output')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载模型和 LUT
    model, lut = load_model_and_lut(args.checkpoint, args.lut_path, device)
    
    # 收集图像
    image_paths = sorted(glob.glob(f'{args.image_dir}/*.png'))
    if not image_paths:
        image_paths = sorted(glob.glob(f'{args.image_dir}/*.jpg'))
    print(f"找到 {len(image_paths)} 张图像")
    
    # 处理每张图像
    for i, path in enumerate(image_paths):
        # 读取图像
        img = cv2.imread(path)
        if img is None:
            print(f"  跳过: {path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # 用 3D LUT 处理
        with torch.no_grad():
            out = apply_3dlut(model, lut, img_tensor, device)
        
        # 保存结果
        out_img = (out[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(args.output_dir, os.path.basename(path))
        cv2.imwrite(out_path, out_img)
        
        if (i + 1) % 50 == 0:
            print(f"  已处理 {i+1}/{len(image_paths)}")
    
    print(f"\n完成！输出保存在: {args.output_dir}")


if __name__ == '__main__':
    main()
