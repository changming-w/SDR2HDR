"""用 BT.709 模型直接推理处理测试集（支持大图 tiling）"""
import sys
sys.path.insert(0, '/mnt/data/wcm/HDRNet')

import os
import glob
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import time

device = 'cuda'
checkpoint = 'experiments/train_hdrnet_bt709/models/net_g_22000.pth'
test_image_dir = '/mnt/data/wcm/20260711_color_enhance_plants2/阳光绿原图'
output_dir = 'results/bt709_model_direct_output'
os.makedirs(output_dir, exist_ok=True)

# 加载模型
from basicsr.archs.hdrnet_bt709_arch import HDRnetBT709
model = HDRnetBT709(
    num_in_ch=3, num_out_ch=3,
    grid_H=32, grid_W=32, grid_D=16,
    base_channels=16, num_downsample=3, num_residual=2,
    fixed_ds_size=512,
).to(device)
ckpt = torch.load(checkpoint, map_location=device)
model.load_state_dict(ckpt['params'] if 'params' in ckpt else ckpt, strict=True)
model.eval()
print(f"模型加载成功: {checkpoint}")


def process_image(model, img_rgb, tile_size=1024, overlap=64):
    """处理单张图，大图用 tiling"""
    h, w = img_rgb.shape[:2]
    
    # 小图直接处理
    if max(h, w) <= tile_size:
        img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_t = img_t.unsqueeze(0).to(device)
        with torch.no_grad():
            out_t = model(img_t)
        return (out_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    
    # 大图 tiling
    # 先计算 grid（全图低分辨率）
    img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_t = img_t.unsqueeze(0).to(device)
    with torch.no_grad():
        x_low = F.interpolate(img_t, size=(512, 512), mode='bilinear', align_corners=False)
        grid = model.lowres_net(x_low)
        guide_full = model.guide_net(img_t)  # (1, 1, H, W)
    
    # 分块处理
    out = np.zeros_like(img_rgb, dtype=np.float32)
    step = tile_size - overlap
    
    for y in range(0, h, step):
        for x in range(0, w, step):
            y1, x1 = min(y + tile_size, h), min(x + tile_size, w)
            y0, x0 = max(0, y1 - tile_size), max(0, x1 - tile_size)
            
            tile_rgb = img_rgb[y0:y1, x0:x1]
            tile_t = torch.from_numpy(tile_rgb).permute(2, 0, 1).float() / 255.0
            tile_t = tile_t.unsqueeze(0).to(device)
            tile_guide = guide_full[:, :, y0:y1, x0:x1]
            
            with torch.no_grad():
                out_tile = model.slice_apply(grid, tile_guide, tile_t)
            
            out_tile_np = (out_tile.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255)
            out[y0:y1, x0:x1] = out_tile_np
    
    return out.astype(np.uint8)


# 收集测试图
test_paths = []
for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.webp']:
    test_paths.extend(glob.glob(f'{test_image_dir}/{ext}'))
test_paths = sorted(test_paths)
print(f"共 {len(test_paths)} 张测试图")

t0 = time.time()
for i, path in enumerate(test_paths):
    img = cv2.imread(path)
    if img is None:
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out_rgb = process_image(model, img_rgb)
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
