#!/usr/bin/env python3
"""将 adaptive_highlight_pooling_lut.py 的增强效果导出为 3D LUT (.cube 格式)。

原理：
- 原算法是空间相关的（金字塔 mask 依赖图像内容）
- 3D LUT 是固定映射 (R,G,B) → (R',G',B')
- 近似方法：将金字塔增强效果简化为 Y 依赖的增益曲线
  gain(y) = smoothstep((y*255 - threshold) / soft) * avg_contrast
  其中 avg_contrast 是 ltmo_contrast 的加权平均

用法：
  python3 scripts/generate_3dlut.py output.cube --size 33
  python3 scripts/generate_3dlut.py output.cube --mode ratio --ltmo-contrast 1.4 1.4 1.2 1.3 1.3 1.2 1.0
"""
import argparse
import numpy as np


def smoothstep(x):
    x = np.clip(x, 0., 1.)
    return x * x * (3. - 2. * x)


def compute_y_curve(threshold, soft, ltmo_contrast, mode='ratio', gamma=0.62):
    """计算 Y 通道的增益曲线。
    
    返回一个函数 f(y) -> y'，其中 y, y' 在 [0, 1] 范围。
    """
    # 计算平均对比度增强（简化版）
    # ltmo_contrast 中 > 1.0 的值会被应用
    contrasts = [c for c in ltmo_contrast if c > 1.0]
    if contrasts:
        avg_ratio = np.mean(contrasts)
    else:
        avg_ratio = 1.0
    
    def apply(y):
        y = np.asarray(y, dtype=np.float32)
        y_uint8 = np.clip(y * 255, 0, 255)
        
        # 像素门控：暗区不增强
        pixel_mask = smoothstep((y_uint8 - threshold) / max(soft, 1e-6))
        
        if mode == 'ratio':
            # 线性提亮
            mapped = np.clip(y * avg_ratio, 0, 1)
            gain = pixel_mask  # 只增强亮区
            out = y + gain * (mapped - y)
        else:
            # gamma 模式
            mapped = np.power(np.clip(y, 0, 1), gamma)
            gain = pixel_mask * (avg_ratio - 1.0)  # 混合强度
            out = y + gain * (mapped - y)
        
        return np.clip(out, 0, 1)
    
    return apply


def generate_3dlut(y_curve, size=33, hue_min=25, hue_max=95, sat_min=35, hue_soft=5):
    """生成 3D LUT，只对黄绿色区域生效。
    
    y_curve: Y 通道的增益函数
    size: LUT 尺寸（默认 33）
    hue_min/hue_max: 黄绿色相范围（OpenCV 0-179）
    sat_min: 最小饱和度（低于此值不增强）
    hue_soft: 色相边界平滑宽度（OpenCV 0-179）
    
    返回 shape=(size, size, size, 3) 的数组，值为 [0, 1]。
    """
    # 生成 RGB 网格
    x = np.linspace(0, 1, size, dtype=np.float32)
    r, g, b = np.meshgrid(x, x, x, indexing='ij')
    
    # 转 HSV 计算色相权重
    # RGB -> HSV (手动计算，避免依赖 OpenCV)
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin
    
    # Hue (0-360 degrees)
    hue = np.zeros_like(r)
    mask_r = (cmax == r) & (delta > 0)
    mask_g = (cmax == g) & (delta > 0)
    mask_b = (cmax == b) & (delta > 0)
    hue[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
    hue[mask_g] = 60 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
    hue[mask_b] = 60 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)
    
    # 转为 OpenCV 的 hue 范围 (0-179)
    hue_cv = hue / 2.0
    
    # Saturation (0-255)
    sat = np.zeros_like(r)
    sat[cmax > 0] = (delta[cmax > 0] / cmax[cmax > 0]) * 255
    
    # 色相权重：黄绿色区域内=1，区域外=0，边界平滑
    def smooth_hue_weight(h, h_min, h_max, h_soft):
        """在 [h_min, h_max] 内为 1，边界 h_soft 宽度平滑过渡"""
        w = np.ones_like(h)
        # 下边界
        below = h < h_min
        if h_soft > 0:
            t = np.clip((h[below] - (h_min - h_soft)) / max(h_soft, 1e-6), 0, 1)
            w[below] = t * t * (3 - 2 * t)
        else:
            w[below] = 0
        # 上边界
        above = h > h_max
        if h_soft > 0:
            t = np.clip(((h_max + h_soft) - h[above]) / max(h_soft, 1e-6), 0, 1)
            w[above] = t * t * (3 - 2 * t)
        else:
            w[above] = 0
        return w
    
    hue_weight = smooth_hue_weight(hue_cv, hue_min, hue_max, hue_soft)
    
    # 饱和度权重
    sat_weight = np.clip((sat - sat_min) / max(255 - sat_min, 1), 0, 1)
    
    # 综合颜色权重
    color_weight = hue_weight * sat_weight  # [0, 1]
    
    # 转 YCrCb（BT.601）
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = 0.500 * r - 0.419 * g - 0.081 * b + 0.5
    cb = -0.169 * r - 0.331 * g + 0.500 * b + 0.5
    
    # 对 Y 应用增益
    y_out = y_curve(y)
    
    # 按颜色权重混合：增强后的 Y 和原始 Y
    y_final = y + color_weight * (y_out - y)
    
    # 转回 RGB
    r_out = y_final + 1.403 * (cr - 0.5)
    g_out = y_final - 0.344 * (cb - 0.5) - 0.714 * (cr - 0.5)
    b_out = y_final + 1.773 * (cb - 0.5)
    
    # 组合
    lut = np.stack([r_out, g_out, b_out], axis=-1)
    return np.clip(lut, 0, 1)


def write_cube(lut, filepath, title='AdaptiveHighlight'):
    """写入 .cube 格式的 3D LUT。"""
    size = lut.shape[0]
    with open(filepath, 'w') as f:
        f.write(f'TITLE "{title}"\n')
        f.write(f'LUT_3D_SIZE {size}\n')
        f.write('DOMAIN_MIN 0.0 0.0 0.0\n')
        f.write('DOMAIN_MAX 1.0 1.0 1.0\n')
        f.write('\n')
        # .cube 格式：R 变化最快，然后 G，然后 B
        for b_idx in range(size):
            for g_idx in range(size):
                for r_idx in range(size):
                    r, g, b = lut[r_idx, g_idx, b_idx]
                    f.write(f'{r:.6f} {g:.6f} {b:.6f}\n')


def main():
    ap = argparse.ArgumentParser(description='生成 3D LUT (.cube)')
    ap.add_argument('output', help='输出 .cube 文件路径')
    ap.add_argument('--size', type=int, default=33, help='LUT 尺寸（默认 33）')
    ap.add_argument('--threshold', type=float, default=50.)
    ap.add_argument('--soft', type=float, default=20.)
    ap.add_argument('--ltmo-contrast', type=float, nargs=7,
                    default=[1.1, 1.3, 1.5, 1.7, 1.8, 1.4, 1.0])
    ap.add_argument('--mode', choices=['gamma', 'ratio'], default='ratio')
    ap.add_argument('--gamma', type=float, default=0.62)
    ap.add_argument('--hue-min', type=int, default=25, help='黄绿色相下限(OpenCV 0-179)')
    ap.add_argument('--hue-max', type=int, default=95, help='黄绿色相上限(OpenCV 0-179)')
    ap.add_argument('--sat-min', type=float, default=35, help='最小饱和度(低于不增强)')
    ap.add_argument('--hue-soft', type=float, default=5, help='色相边界平滑宽度')
    a = ap.parse_args()
    
    # 计算 Y 增益曲线
    y_curve = compute_y_curve(
        a.threshold, a.soft, a.ltmo_contrast,
        mode=a.mode, gamma=a.gamma
    )
    
    # 生成 3D LUT
    lut = generate_3dlut(y_curve, size=a.size,
                         hue_min=a.hue_min, hue_max=a.hue_max,
                         sat_min=a.sat_min, hue_soft=a.hue_soft)
    
    # 写入文件
    title = f'AdaptiveHighlight_{a.mode}_t{a.threshold}_s{a.soft}'
    write_cube(lut, a.output, title=title)
    
    print(f'已生成 3D LUT: {a.output}')
    print(f'尺寸: {a.size}x{a.size}x{a.size}')
    print(f'模式: {a.mode}')
    print(f'参数: threshold={a.threshold}, soft={a.soft}')
    print(f'ltmo_contrast: {a.ltmo_contrast}')
    print(f'色相范围: [{a.hue_min}, {a.hue_max}], sat_min={a.sat_min}, hue_soft={a.hue_soft}')


if __name__ == '__main__':
    main()
