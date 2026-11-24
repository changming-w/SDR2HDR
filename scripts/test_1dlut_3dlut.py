"""
从 1D LUT 模型生成 3D LUT 并处理测试集

流程:
1. 加载模型
2. 对每张图: LowResNet → (4096, 12) → 生成 3D LUT (64³ × 3)
3. 用 3D LUT 处理图像
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, '/mnt/data/wcm/HDRNet')
from basicsr.archs.hdrnet_1dlut_arch import HDRnet1DLUT


def generate_3dlut_from_coeffs(coeffs, size=64):
    """
    从 (4096, 12) 系数生成 3D LUT
    
    Args:
        coeffs: (4096, 12) 亮度 bin 的仿射系数
        size: 3D LUT 尺寸 (默认 64)
    
    Returns:
        lut: (size, size, size, 3) 的 3D LUT
    """
    num_bins = coeffs.shape[0]
    
    # 生成所有 RGB 采样点
    r = np.linspace(0, 1, size)
    g = np.linspace(0, 1, size)
    b = np.linspace(0, 1, size)
    
    lut = np.zeros((size, size, size, 3), dtype=np.float32)
    
    for i, rv in enumerate(r):
        for j, gv in enumerate(g):
            for k, bv in enumerate(b):
                # BT.709 亮度
                y = 0.2126 * rv + 0.7152 * gv + 0.0722 * bv
                
                # 找最近的 bin
                bin_idx = int(np.clip(y * (num_bins - 1), 0, num_bins - 1))
                coeff = coeffs[bin_idx]  # (12,)
                
                # 仿射变换
                A = coeff[:9].reshape(3, 3)
                bias = coeff[9:12]
                
                rgb_in = np.array([rv, gv, bv])
                rgb_out = A @ rgb_in + bias
                
                lut[i, j, k] = np.clip(rgb_out, 0, 1)
    
    return lut


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
    
    # 归一化到 [0, size-1]
    rgb_scaled = image * (size - 1)
    
    # 三线性插值
    output = np.zeros_like(image)
    
    for i in range(h):
        for j in range(w):
            r, g, b = rgb_scaled[i, j]
            
            # 找到周围的 8 个点
            r0, g0, b0 = int(np.floor(r)), int(np.floor(g)), int(np.floor(b))
            r1, g1, b1 = min(r0 + 1, size - 1), min(g0 + 1, size - 1), min(b0 + 1, size - 1)
            
            # 插值权重
            dr, dg, db = r - r0, g - g0, b - b0
            
            # 三线性插值
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
            
            output[i, j] = c0 * (1 - db) + c1 * db
    
    return np.clip(output, 0, 1)


def extract_coeffs_from_model(model, image_path, device='cuda'):
    """
    从模型提取 (4096, 12) 系数
    
    Args:
        model: HDRnet1DLUT 模型
        image_path: 图像路径
        device: 设备
    
    Returns:
        coeffs: (4096, 12) numpy array
    """
    # 读取图像
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 预处理
    img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_t = img_t.unsqueeze(0).to(device)
    
    # 下采样到 512x512
    img_low = F.interpolate(img_t, size=(512, 512), mode='bilinear', align_corners=False)
    
    # 提取系数
    with torch.no_grad():
        coeffs = model.lowres_net(img_low)  # (1, 4096, 12)
        coeffs = coeffs.squeeze(0).cpu().numpy()  # (4096, 12)
    
    return coeffs


def main():
    # 配置
    checkpoint_path = '/mnt/data/wcm/HDRNet/experiments/train_hdrnet_1dlut/train_hdrnet_1dlut/models/net_g_8000.pth'
    input_dir = '/mnt/data/wcm/20260711_color_enhance_plants2/阳光绿原图'
    output_dir = '/mnt/data/wcm/20260711_color_enhance_plants2/阳光绿原图_1dlut_3dlut_8000'
    lut_size = 64  # 3D LUT 尺寸
    
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载模型
    print('加载模型...')
    model = HDRnet1DLUT(num_bins=4096, base_channels=16, num_downsample=4, fixed_ds_size=512)
    state_dict = torch.load(checkpoint_path, map_location=device)
    # 处理 EMA 权重
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
    
    # 处理每张图
    for img_name in tqdm(img_files, desc='处理中'):
        img_path = os.path.join(input_dir, img_name)
        
        # 提取系数
        coeffs = extract_coeffs_from_model(model, img_path, device)  # (4096, 12)
        
        # 生成 3D LUT
        lut = generate_3dlut_from_coeffs(coeffs, size=lut_size)  # (64, 64, 64, 3)
        
        # 读取原图
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # 应用 3D LUT
        output_rgb = apply_3dlut(img_rgb, lut)
        
        # 保存
        output_bgr = cv2.cvtColor((output_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        output_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_path, output_bgr)
    
    print(f'完成! 结果保存在: {output_dir}')


if __name__ == '__main__':
    main()
