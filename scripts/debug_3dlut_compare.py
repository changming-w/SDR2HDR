"""对比：模型直接推理 vs 3D LUT 查表"""
import sys
sys.path.insert(0, '/mnt/data/wcm/HDRNet')

import cv2
import numpy as np
import torch
import torch.nn.functional as F

device = 'cuda'
checkpoint = 'experiments/train_hdrnet_bt709/models/net_g_22000.pth'

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

# 测试图
img_path = '/mnt/data/wcm/20260711_color_enhance_plants2/阳光绿原图/100阳光绿.png'
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
img_t = img_t.unsqueeze(0).to(device)

# === 方法1: 模型直接推理 ===
with torch.no_grad():
    out_model = model(img_t)
out_model_np = (out_model.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

# === 方法2: 3D LUT 查表 ===
with torch.no_grad():
    x_low = F.interpolate(img_t, size=(512, 512), mode='bilinear', align_corners=False)
    grid = model.lowres_net(x_low)  # (1, D, Hg, Wg, 12)
    coeff = grid.squeeze(0).mean(dim=(1, 2)).cpu().numpy()  # (D, 12)

# 生成 64^3 LUT
D = coeff.shape[0]
size = 64
r = np.linspace(0, 1, size)
g = np.linspace(0, 1, size)
b = np.linspace(0, 1, size)
rr, gg, bb = np.meshgrid(r, g, b, indexing='ij')
input_rgb = np.stack([rr, gg, bb], axis=-1).reshape(-1, 3)

luminance = 0.2126 * input_rgb[:, 0] + 0.7152 * input_rgb[:, 1] + 0.0722 * input_rgb[:, 2]
z = luminance * (D - 1)
z0 = np.floor(z).clip(0, D - 1).astype(int)
z1 = np.clip(z0 + 1, 0, D - 1)
dz = z - z0

coeff_interp = coeff[z0] * (1 - dz[:, None]) + coeff[z1] * dz[:, None]
A = coeff_interp[:, :9].reshape(-1, 3, 3)
bias = coeff_interp[:, 9:]
output = np.einsum('nij,nj->ni', A, input_rgb) + bias
output = output.clip(0, 1).reshape(size, size, size, 3)

# 三线性插值查表
img_f = img_rgb.astype(np.float32) / 255.0
r_idx = img_f[..., 0] * (size - 1)
g_idx = img_f[..., 1] * (size - 1)
b_idx = img_f[..., 2] * (size - 1)
r0 = np.floor(r_idx).clip(0, size - 2).astype(int)
g0 = np.floor(g_idx).clip(0, size - 2).astype(int)
b0 = np.floor(b_idx).clip(0, size - 2).astype(int)
dr = (r_idx - r0)[..., None]
dg = (g_idx - g0)[..., None]
db = (b_idx - b0)[..., None]

c000 = output[r0, g0, b0]; c001 = output[r0, g0, b0+1]
c010 = output[r0, g0+1, b0]; c011 = output[r0, g0+1, b0+1]
c100 = output[r0+1, g0, b0]; c101 = output[r0+1, g0, b0+1]
c110 = output[r0+1, g0+1, b0]; c111 = output[r0+1, g0+1, b0+1]
c00 = c000*(1-db) + c001*db; c01 = c010*(1-db) + c011*db
c10 = c100*(1-db) + c101*db; c11 = c110*(1-db) + c111*db
c0 = c00*(1-dg) + c01*dg; c1 = c10*(1-dg) + c11*dg
c = c0*(1-dr) + c1*dr
out_lut = (c * 255).clip(0, 255).astype(np.uint8)

# === 对比 ===
diff = np.abs(out_model_np.astype(float) - out_lut.astype(float))
print(f"模型推理 vs 3D LUT 差异:")
print(f"  平均差异: {diff.mean():.2f}")
print(f"  最大差异: {diff.max():.2f}")
print(f"  模型输出范围: [{out_model_np.min()}, {out_model_np.max()}]")
print(f"  LUT输出范围:  [{out_lut.min()}, {out_lut.max()}]")
print(f"  输入范围:     [{img_rgb.min()}, {img_rgb.max()}]")

# 保存对比
cv2.imwrite('/tmp/debug_model.png', cv2.cvtColor(out_model_np, cv2.COLOR_RGB2BGR))
cv2.imwrite('/tmp/debug_lut.png', cv2.cvtColor(out_lut, cv2.COLOR_RGB2BGR))
cv2.imwrite('/tmp/debug_input.png', img)
print("已保存: /tmp/debug_model.png, /tmp/debug_lut.png, /tmp/debug_input.png")
