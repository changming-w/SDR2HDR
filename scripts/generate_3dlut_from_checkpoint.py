"""
从训练好的 HDRNet checkpoint 生成 3D LUT（后处理压缩）

原理：
  - 原模型 grid: (D, Hg, Wg, 12) = (16, 32, 32, 12) 有空间变化
  - 压缩后 3D LUT: (D, 12) = (16, 12) 只保留亮度维度
  - 压缩方式：对空间维度 (Hg, Wg) 取均值

用法：
  python scripts/generate_3dlut_from_checkpoint.py \
    --checkpoint experiments/train_hdrnet_edr_original/models/net_g_latest.pth \
    --output_dir results/3dlut_compressed
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


def load_model(checkpoint_path, device='cuda'):
    """加载模型"""
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
    return model


def analyze_spatial_variation(grids):
    """分析 grid 的空间变化程度"""
    # grids: list of (D, Hg, Wg, 12) tensors
    stacked = torch.stack(grids, dim=0)  # (N, D, Hg, Wg, 12)
    
    # 空间维度的标准差
    spatial_std = stacked.std(dim=(2, 3))  # (N, D, 12)
    spatial_mean = stacked.mean(dim=(2, 3))  # (N, D, 12)
    
    # 相对变化率 = std / mean
    relative_var = spatial_std / (spatial_mean.abs() + 1e-8)
    
    print("\n=== 空间变化分析 ===")
    print(f"Grid 形状: {stacked.shape[1:]}")
    print(f"空间标准差均值: {spatial_std.mean():.6f}")
    print(f"空间均值绝对值: {spatial_mean.abs().mean():.6f}")
    print(f"相对变化率: {relative_var.mean():.4f} ({relative_var.mean()*100:.2f}%)")
    
    if relative_var.mean() < 0.1:
        print("结论: 空间变化较小，压缩到 3D LUT 损失不大")
    elif relative_var.mean() < 0.3:
        print("结论: 空间变化中等，压缩会有一定损失")
    else:
        print("结论: 空间变化较大，压缩损失可能明显")
    
    return relative_var


def generate_3dlut(model, image_paths, device='cuda'):
    """从多张图像生成 3D LUT（取空间均值）"""
    grids = []
    
    print(f"\n处理 {len(image_paths)} 张图像...")
    for i, path in enumerate(image_paths):
        # 读取图像
        img = cv2.imread(path)
        if img is None:
            print(f"  跳过: {path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0).to(device)
        
        # 下采样到固定尺寸
        x_low = F.interpolate(img, size=(512, 512), mode='bilinear', align_corners=False)
        
        # 生成 grid
        with torch.no_grad():
            grid = model.lowres_net(x_low)  # (1, D, Hg, Wg, 12)
        
        grids.append(grid.squeeze(0).cpu())  # (D, Hg, Wg, 12)
        
        if (i + 1) % 20 == 0:
            print(f"  已处理 {i+1}/{len(image_paths)}")
    
    print(f"共处理 {len(grids)} 张图像")
    
    # 分析空间变化
    analyze_spatial_variation(grids)
    
    # 压缩到 3D LUT: 对空间维度取均值
    # grids: list of (D, Hg, Wg, 12)
    luts = []
    for grid in grids:
        lut = grid.mean(dim=(1, 2))  # (D, 12)
        luts.append(lut)
    
    # 多张图取平均
    avg_lut = torch.stack(luts, dim=0).mean(dim=0)  # (D, 12)
    
    return avg_lut


def save_lut(lut, output_path):
    """保存 3D LUT"""
    lut_np = lut.numpy()
    
    # 保存为 numpy 文件
    np.save(output_path.replace('.npy', '') + '.npy', lut_np)
    
    # 保存为文本文件（方便查看）
    txt_path = output_path.replace('.npy', '.txt')
    with open(txt_path, 'w') as f:
        f.write("# 3D LUT: (D=16, C=12)\n")
        f.write("# 每行: depth_bin, a00, a01, a02, a10, a11, a12, a20, a21, a22, b0, b1, b2\n")
        for d in range(lut_np.shape[0]):
            coeffs = lut_np[d]
            f.write(f"{d}: " + " ".join(f"{c:.6f}" for c in coeffs) + "\n")
    
    print(f"\n3D LUT 已保存:")
    print(f"  NumPy: {output_path}")
    print(f"  Text:  {txt_path}")
    
    return lut_np


def visualize_lut(lut_np):
    """可视化 3D LUT"""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 3x3 矩阵的 9 个系数
        matrix_names = ['a00', 'a01', 'a02', 'a10', 'a11', 'a12', 'a20', 'a21', 'a22']
        for i, name in enumerate(matrix_names):
            ax = axes[i // 3, i % 3]
            ax.plot(lut_np[:, i], 'b-', linewidth=2)
            ax.set_xlabel('Depth bin')
            ax.set_ylabel(name)
            ax.set_title(f'{name} vs Depth')
            ax.grid(True)
            ax.set_ylim(-2, 2)
        
        # bias
        ax = axes[1, 0]
        for i, name in enumerate(['b0', 'b1', 'b2']):
            ax.plot(lut_np[:, 9 + i], linewidth=2, label=name)
        ax.set_xlabel('Depth bin')
        ax.set_title('Bias vs Depth')
        ax.legend()
        ax.grid(True)
        
        # 隐藏多余的子图
        axes[1, 1].axis('off')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        vis_path = 'results/3dlut_compressed/lut_visualization.png'
        plt.savefig(vis_path, dpi=150)
        print(f"  可视化: {vis_path}")
        
    except Exception as e:
        print(f"可视化失败: {e}")


def main():
    parser = argparse.ArgumentParser(description='从 checkpoint 生成 3D LUT')
    parser.add_argument('--checkpoint', type=str, 
                        default='experiments/train_hdrnet_edr_original/models/net_g_latest.pth')
    parser.add_argument('--image_dir', type=str,
                        default='/mnt/data/wcm/20260711_color_enhance_plants2/val/sdr')
    parser.add_argument('--output_dir', type=str, default='results/3dlut_compressed')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载模型
    model = load_model(args.checkpoint, device)
    
    # 收集图像
    image_paths = sorted(glob.glob(f'{args.image_dir}/*.png'))[:50]  # 最多 50 张
    if not image_paths:
        image_paths = sorted(glob.glob(f'{args.image_dir}/*.jpg'))[:50]
    print(f"找到 {len(image_paths)} 张图像")
    
    # 生成 3D LUT
    lut = generate_3dlut(model, image_paths, device)
    
    # 保存
    output_path = os.path.join(args.output_dir, 'lut_3d_16x12.npy')
    lut_np = save_lut(lut, output_path)
    
    # 可视化
    visualize_lut(lut_np)
    
    print("\n完成！")


if __name__ == '__main__':
    main()
