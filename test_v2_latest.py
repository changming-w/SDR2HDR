import torch, torch.nn as nn, torch.nn.functional as F
import cv2, os, glob
import numpy as np

def srgb_to_linear(x):
    x = x.clamp(1e-6, 1.0)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(x):
    x = x.clamp(1e-6, 1.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1.0 / 2.4)) - 0.055)

def conv_lrelu(in_ch, out_ch, k, s, p, bias=True):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, k, s, p, bias=bias), nn.LeakyReLU(0.2, inplace=True))

class LowResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            conv_lrelu(3, 16, 3, 2, 1), conv_lrelu(16, 32, 3, 2, 1),
            conv_lrelu(32, 64, 3, 2, 1), conv_lrelu(64, 64, 3, 2, 1),
            conv_lrelu(64, 64, 3, 1, 1))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 16*32*32*1)
    def forward(self, x):
        B = x.size(0)
        feat = self.trunk(x)
        pooled = self.pool(feat).view(B, -1)
        return self.fc(pooled).view(B, 16, 32, 32, 1)

class GuideNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = conv_lrelu(3, 16, 1, 1, 0)
        self.conv2 = conv_lrelu(16, 16, 3, 1, 1)
        self.conv3 = conv_lrelu(16, 16, 3, 1, 1)
        self.conv_out = nn.Conv2d(16, 1, 1)
    def forward(self, x):
        return torch.sigmoid(self.conv_out(self.conv3(self.conv2(self.conv1(x)))))

class SliceScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.D, self.Hg, self.Wg = 16, 32, 32
    def forward(self, grid, guide, x):
        B, D, Hg, Wg, _ = grid.shape
        _, _, H, W = x.shape
        dev = x.device
        x_lin = srgb_to_linear(x)
        z = guide * (D - 1)
        ys = torch.linspace(0, Hg-1, H, device=dev).view(1,1,H,1).expand(B,1,H,W)
        xs = torch.linspace(0, Wg-1, W, device=dev).view(1,1,1,W).expand(B,1,H,W)
        x0,y0,z0 = torch.floor(xs),torch.floor(ys),torch.floor(z)
        x1,y1,z1 = torch.clamp(x0+1,max=Wg-1),torch.clamp(y0+1,max=Hg-1),torch.clamp(z0+1,max=D-1)
        dx,dy,dz = xs-x0, ys-y0, z-z0
        x0,y0,z0,x1,y1,z1 = [t.long() for t in [x0,y0,z0,x1,y1,z1]]
        gf = grid.view(B, D*Hg*Wg, 1)
        def idx(z_,y_,x_): return (z_*Hg+y_)*Wg+x_
        def g(idx_):
            f = idx_.view(B,-1).unsqueeze(-1).expand(-1,-1,1)
            return torch.gather(gf,1,f).permute(0,2,1).contiguous().view(B,1,H,W)
        w = lambda a: 1-a
        s = (g(idx(z0,y0,x0))*w(dx)*w(dy)*w(dz) + g(idx(z0,y0,x1))*dx*w(dy)*w(dz) +
             g(idx(z0,y1,x0))*w(dx)*dy*w(dz) + g(idx(z0,y1,x1))*dx*dy*w(dz) +
             g(idx(z1,y0,x0))*w(dx)*w(dy)*dz + g(idx(z1,y0,x1))*dx*w(dy)*dz +
             g(idx(z1,y1,x0))*w(dx)*dy*dz + g(idx(z1,y1,x1))*dx*dy*dz)
        return linear_to_srgb(x_lin * s.clamp(0.01, 10.0))

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lowres_net = LowResNet()
        self.guide_net = GuideNet()
        self.slice_apply = SliceScale()
    def forward(self, x):
        x_low = F.interpolate(x, size=(512,512), mode='bilinear', align_corners=False)
        grid = self.lowres_net(x_low)
        guide = self.guide_net(x)
        return torch.clamp(self.slice_apply(grid, guide, x), 0, 1)

model = Model().cuda().eval()
ckpt = torch.load('experiments/train_hdrnet_simple_v2/models/net_g_latest.pth', map_location='cuda')
model.load_state_dict(ckpt['params'], strict=True)
print('模型加载成功')

test_dir = '/mnt/data/wcm/20260711_color_enhance_plants2/阳光绿原图'
out_dir = 'results/test_hdrnet_simple_v2_latest'
os.makedirs(out_dir, exist_ok=True)

files = sorted(glob.glob(f'{test_dir}/*.png'))
print(f'共 {len(files)} 张测试图')

for i, f in enumerate(files):
    img = cv2.imread(f)
    img_t = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float()/255.0
    with torch.no_grad():
        out = model(img_t.cuda()).cpu()
    out_img = (out[0].permute(1,2,0).numpy()*255).clip(0,255).astype(np.uint8)
    cv2.imwrite(f'{out_dir}/{os.path.basename(f)}', out_img)
    if (i+1) % 50 == 0:
        print(f'已处理 {i+1}/{len(files)}')

print(f'完成！输出保存在 {out_dir}')
