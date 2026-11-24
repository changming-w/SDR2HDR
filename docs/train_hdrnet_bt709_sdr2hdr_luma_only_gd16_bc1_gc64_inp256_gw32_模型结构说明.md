# HDRnetBT709PaperLumaOnly（gc64 / inp256 / grid 32×32×16）模型结构说明

> 对应训练配置：`options/train/Custom/train_hdrnet_bt709_sdr2hdr_luma_only_gd16_bc1_gc64_inp256_gw32.yml`
> 代码实现：`basicsr/archs/hdrnet_bt709_paper_luma_only_arch.py`
> （GuideNet 位于 `basicsr/archs/hdrnet_bt709_arch.py`）
> 损失函数：`basicsr/losses/basic_loss.py::LumaL1Loss`
> 数据集：`basicsr/data/hdr_paired_dataset.py::HDRPairedImageDataset`
> 整理日期：2026-11-03（参数量与形状均在 `hdrnet` conda 环境实测）

---

## 1. 概述

本模型是 HDRNet（Gharbi et al., *Deep Bilateral Learning for Real-Time Image Enhancement*, SIGGRAPH 2017）的 **luma-only（只改亮度）** 变体，用于 **SDR→HDR 的局部色调映射**：输入 BT.709 SDR 的非线性码值 RGB，输出"更接近 HDR 观感"的 RGB。

与原始 HDRNet / 本仓库 12 系数版本相比，核心改动：

| 项目 | 原版 HDRNet | 本模型 |
|---|---|---|
| 每个格点系数 | 12 个（3×4 仿射矩阵，可改色彩） | **2 个（亮度 scale、offset）** |
| Guide（引导图） | 可学习逐像素网络 | **固定 BT.709 亮度公式，零参数** |
| 输出方式 | 对 RGB 做仿射变换 | 先求目标亮度 Y'，再对 RGB 三通道**等比缩放**（色度严格不变） |
| 运行分辨率 | 低分辨率分支固定尺寸 | 输入统一缩放到 256×256 再进网络（本配置） |

由此得到四个关键特性：

1. **色度不变**：输出 = 输入 × 每像素一个标量，R:G:B 比例不变，色相/饱和度/白点不动（仅推理 clamp 触顶时例外）。
2. **局部自适应**：亮度变换随像素位置与亮度变化，等价于一张"随空间位置变化的色调曲线"。
3. **计算与输出分辨率解耦**：CNN 只在 256×256 缩略图上跑，全分辨率只做三线性查表 + 乘法。
4. **极轻量**：**513,856 参数（≈0.514M）**，全模型 MACs ≈ **0.137 GMAC**（256×256 输入），适合端侧/实时部署。

---

## 2. 配置解析

```yaml
network_g:
  type: HDRnetBT709PaperLumaOnly
  num_in_ch: 3
  num_out_ch: 3
  grid_H: 32          # bilateral grid 空间高
  grid_W: 32          # bilateral grid 空间宽
  grid_D: 16          # bilateral grid 亮度维档数
  base_channels: 1    # 通道基数 cm
  fixed_ds_size: 256  # 低分辨率分支输入尺寸
  global_channels: 64 # 全局/局部特征通道数 gc
```

| 参数 | 值 | 作用 |
|---|---|---|
| `grid_H × grid_W` | 32 × 32 | grid 空间分辨率。4K 输入时每格覆盖 120×67.5 像素；1080p 时 60×33.75 像素 |
| `grid_D` | 16 | 亮度维档数（Y∈[0,1] 分 16 个节点，间隔 1/15），同时是 splat 通道倍率 |
| `base_channels` | 1 | splat 第 i 层输出通道 = `cm·2^i·gd`，即 16 / 32 / 64 |
| `fixed_ds_size` | 256 | 任意尺寸输入先双线性缩放到 256×256，再进低分辨率分支 |
| `global_channels` | 64 | 全局分支、局部分支、融合特征的通道数 gc |

由配置推导：

- **splat 下采样层数** = `log2(fixed_ds_size / grid_H)` = `log2(256/32)` = **3 层**（256→128→64→32）
- **全局分支卷积层数** = `max(1, log2(grid_H / 4))` = `log2(32/4)` = **3 层**（32→16→8→4）
- 全局 FC：`gc·4·4 → gc·4 → gc·2 → gc`（1024→256→128→64）

> 注意：输入被缩放到 **正方形 256×256**，16:9 画面在低分辨率分支中被纵向拉伸；切片时 grid 坐标按整幅图线性铺开，二者在拓扑上一致，不影响"空间平滑"性质，但局部特征对应的是拉伸后的坐标。

---

## 3. 整体结构

```
                        输入 x (B,3,H,W)，[0,1]，BT.709 非线性码值
                                   │
            ┌──────────────────────┼──────────────────────────┐
            │                      │                          │
   双线性缩放 256×256          GuideNetBT709               （原图 x 直通）
            │            Y = 0.2126R + 0.7152G + 0.0722B       │
            ▼                      │                          │
   ┌──────────────────┐            │                          │
   │ LowResNet        │            │                          │
   │ splat → global   │            │                          │
   │ → local → fusion │            │                          │
   │ → coeff (1×1)    │            │                          │
   └────────┬─────────┘            │                          │
            ▼                      ▼                          │
     grid (B,16,32,32,2)      guide Y (B,1,H,W)                │
            │                      │                          │
            └──────────┬───────────┘                          │
                       ▼                                      │
        三线性插值 (x, y, Y) → 每像素 (scale, offset)          │
                       ▼                                      │
              Y' = scale·Y + offset                           │
              ratio = Y' / (Y + 1e-6)                         │
                       └──────────────────┬───────────────────┘
                                          ▼
                             out = x · ratio（RGB 等比缩放）
                                          ▼
                     strength 混合（可选）→ 推理 clamp(0,1)
```

三个子模块：

| 子模块 | 类名 | 作用 | 参数量 |
|---|---|---|---|
| 低分辨率系数网络 | `LowResNetBT709PaperLumaOnly` | 从缩略图预测 bilateral grid | **513,856** |
| 引导图 | `GuideNetBT709` | 计算 BT.709 亮度 Y | 0（固定 buffer） |
| 切片与应用 | `BilateralSliceApplyLumaOnly` | 三线性插值 + 亮度变换 + RGB 等比缩放 | 0 |

---

## 4. 低分辨率系数网络 LowResNet

输入 `(B, 3, 256, 256)`，输出 `(B, 16, 32, 32, 2)`。
所有卷积后接 `LeakyReLU(0.2)`（`coeff_conv` 除外）。

### 4.1 逐层结构与参数量（实测）

| 阶段 | 层 | 结构 | 输出尺寸 | 参数量 |
|---|---|---|---|---|
| Splat | splat.0 | Conv 3×3, s2, p1, 3→16 | 16×128×128 | 448 |
| | splat.1 | Conv 3×3, s2, 16→32 | 32×64×64 | 4,640 |
| | splat.2 | Conv 3×3, s2, 32→64 | 64×32×32 | 18,496 |
| 全局分支 | global_trunk.0 | Conv 3×3, s2, 64→64 | 64×16×16 | 36,928 |
| | global_trunk.1 | Conv 3×3, s2, 64→64 | 64×8×8 | 36,928 |
| | global_trunk.2 | Conv 3×3, s2, 64→64 | 64×4×4 | 36,928 |
| | 展平 | — | 1024 | — |
| | global_fc1 | Linear 1024→256 + LReLU | 256 | 262,400 |
| | global_fc2 | Linear 256→128 + LReLU | 128 | 32,896 |
| | global_fc3 | Linear 128→64（无激活） | 64×1×1 | 8,256 |
| 局部分支 | local_proj | **Identity**（splat 输出已是 64 通道） | 64×32×32 | 0 |
| | local_conv1 | Conv 3×3, s1, 64→64 | 64×32×32 | 36,928 |
| | local_conv2 | Conv 3×3, s1, 64→64 | 64×32×32 | 36,928 |
| 融合 | fusion | `ReLU(local + global)`，global 空间广播 | 64×32×32 | 0 |
| 系数输出 | coeff_conv | Conv 1×1, 64→32（= D×2） | 32×32×32 | 2,080 |
| 重排 | view + permute | `(B,32,32,32)` → `(B,16,2,32,32)` → `(B,16,32,32,2)` | 16×32×32×2 | — |
| **合计** | | | | **513,856** |

参数分布：全局 FC 占 **59.1%**（其中 fc1 一项占 51.1%），global_trunk 占 21.6%，local 卷积占 14.4%，splat 占 4.6%，coeff 占 0.4%。

### 4.2 各分支职责

- **Splat**：3 次 stride-2 卷积把 256×256 缩到 32×32，与 grid 空间分辨率对齐；通道按 `16→32→64` 递增。
- **全局分支**：继续下采样到 4×4（保留粗空间布局，**不做全局池化**），展平成 1024 维后经三级 FC 降到 64 维。它编码整图亮度分布/场景类型等全局信息，并广播加到每个空间位置。
  > "保留 4×4 再 FC"是本 paper 版与前代实现的区别之一；也正因此它要求输入固定 256×256，不能直接吃任意分辨率缩略图（否则 FC 维度不匹配）。
- **局部分支**：在 32×32 上做两层 3×3 卷积，提取局部亮度/对比度/边缘信息。
- **融合与输出**：`ReLU(local + global)` 后用 1×1 卷积输出 32 通道，重排为 16 个亮度档 × 2 个系数。

### 4.3 Grid 的含义

grid 形状 `(B, D=16, Hg=32, Wg=32, 2)`：

- **空间维 32×32**：把全图划分为 32×32 个区域；
- **亮度维 16**：Y∈[0,1] 均匀分 16 个节点；
- **系数维 2**：`scale` 与 `offset`。

即：每个空间区域拥有"16 个亮度节点"定义的一条分段亮度变换曲线，节点 i 处 `Y' = scale_i·Y + offset_i`。

---

## 5. 引导图 GuideNetBT709

```python
guide = 0.2126 * R + 0.7152 * G + 0.0722 * B
```

- 系数以 `register_buffer` 固定，**不参与训练**；
- 直接在**非线性码值**上计算，不做线性化（模型训练时即如此）；
- 输出 `(B,1,H,W)`，同时用作：① grid 亮度维查表坐标；② 亮度变换的输入 Y。

---

## 6. 切片与应用 BilateralSliceApplyLumaOnly

### 6.1 三线性插值坐标

对全分辨率像素 \(p=(u,v)\)：

\[
x_g = u\cdot\frac{W_g-1}{W-1},\qquad
y_g = v\cdot\frac{H_g-1}{H-1},\qquad
z_g = \mathrm{clamp}(Y(p),0,1)\cdot(D-1)
\]

空间坐标由 `torch.linspace(0, Wg-1, W)` 均匀铺开，图像四角对齐 grid 四角；亮度坐标由 guide 决定。

### 6.2 插值系数

取包围 \((x_g,y_g,z_g)\) 的 8 个格点，用三线性权重加权得到该像素的 \((s(p), o(p))\)。

### 6.3 亮度变换与等比缩放

\[
Y'(p)=s(p)\cdot Y(p)+o(p)
\]
\[
r(p)=\frac{Y'(p)}{Y(p)+10^{-6}},\qquad
\mathrm{out}(p)=x(p)\cdot r(p)
\]

- \(Y(p)\le10^{-6}\)（纯黑）时 \(r=1\)，像素保持原样，避免除零；
- R/G/B 乘同一标量 \(r(p)\)，**R:G:B 比例严格不变**；
- \(s,o\) 随位置和亮度变化 ⇒ \(r(p)\) 是内容/位置/亮度的非线性函数。

### 6.4 后处理

```python
if strength < 1.0:                      # 训练/推理都可选
    out = strength * out + (1 - strength) * x
if self.training:
    return out                          # 训练不截断，允许 >1 的亮度预测
return torch.clamp(out, 0.0, 1.0)       # 推理截断到 [0,1]
```

`strength` 默认 1.0。推理 clamp 是"色度不变"的唯一例外：增益 >1 的高光像素触顶后 RGB 比例会轻微改变。

---

## 7. 前向伪代码

```python
def forward(x, strength=1.0):                    # x: (B,3,H,W), [0,1]
    x_low = bilinear_resize(x, 256, 256)         # 固定尺寸
    grid  = lowres_net(x_low)                    # (B,16,32,32,2)
    Y     = 0.2126*R + 0.7152*G + 0.0722*B       # (B,1,H,W)
    s, o  = trilinear_grid_sample(grid, x, y, Y)
    ratio = (s*Y + o) / (Y + 1e-6)               # Y≈0 时 ratio=1
    out   = x * ratio
    out   = strength*out + (1-strength)*x
    return clamp(out, 0, 1)                      # 仅推理
```

训练时的张量形状（`gt_size=512`，batch=64）：输入 `(64,3,512,512)` → 低分辨率分支 `(64,3,256,256)` → grid `(64,16,32,32,2)` → guide `(64,1,512,512)` → 输出 `(64,3,512,512)`。

---

## 8. 参数量与计算量

| 项目 | 数值 |
|---|---|
| 总参数量 | **513,856（0.514 M）** |
| 其中可训练参数 | 全部 513,856（GuideNet 无参数） |
| 模型文件大小（fp32） | ≈ 2.0 MB |
| 低分辨率分支 MACs | ≈ **0.137 GMAC**（256×256 输入，与输出分辨率无关） |
| 全分辨率开销 | 每像素：8 次 gather ×2 系数 + 三线性加权 + 1 次除法 + 3 次乘法 |

各层 MAC 估算：splat ≈ 44.8M，global_trunk ≈ 12.4M，全局 FC ≈ 0.3M，local 卷积 ≈ 75.5M，coeff ≈ 2.1M。

实测（RTX 4090，训练 512×512 crop、batch 64）：约 **0.32 s/iter**（与另一任务共享 GPU 时），1M iter 共 **3 天 16.5 小时**。
实测（PyTorch CPU，4K 单帧推理）：约 0.38 s（未优化）。

---

## 9. 训练配置（本 yml）

### 9.1 数据

| 项目 | 值 |
|---|---|
| 数据集类型 | `HDRPairedImageDataset`（SDR→HDR 配对） |
| 训练对 | 1492 对，1080p；`dump_sdr`(8bit PNG) → `dump_hdr`(16bit PNG) |
| 验证对 | 50 对，同一采集 |
| 读取方式 | GT 用 `IMREAD_UNCHANGED`/65535；LQ 8bit/255（BGR→RGB） |
| 增强 | 随机裁 512×512 + 水平翻转 + 90° 旋转 |
| 加载 | `load_to_memory: true`，1492 对解码后常驻内存（约 28 GB） |
| batch / workers | 64 / 16 |

### 9.2 损失

`LumaL1Loss(target_colorspace: bt2020)`：

```python
pred_y   = 0.2126*R + 0.7152*G + 0.0722*B     # pred 为 BT.709
target_y = 0.2627*R + 0.6780*G + 0.0593*B     # GT 为 BT.2020
loss = L1(pred_y, target_y)
```

等价于"先把 BT.2020 GT 转到 BT.709 再比 Y"，但无需矩阵乘法、无 clip 损失。**只监督亮度**，色度由等比缩放结构保证。

### 9.3 优化

| 项目 | 值 |
|---|---|
| 优化器 | AdamW，lr=2e-4，weight_decay=1e-6，betas=(0.9,0.999) |
| 调度 | CosineAnnealingRestartLR，periods=[1,000,000]，eta_min=1e-6，无 warmup |
| 总迭代 | 1,000,000 |
| EMA | 关闭 |
| 验证 | 每 10k iter，全 RGB PSNR / SSIM（crop_border=0，不做 Y 通道） |
| 保存 | 每 5k iter |

---

## 10. 训练结果与同系列对比

本配置训练于 2026-10-30 启动，2026-11-03 07:35 完成（3 天 16.5 小时）：

| 指标 | 最佳值 | 出现位置 | 最终值 |
|---|---|---|---|
| PSNR | **32.2142 dB** | 480k iter | 32.1660 |
| SSIM | **0.9525** | 60k iter | 0.9522 |

与 4 倍参数量的大模型对比（同数据、同损失、同调度）：

| | 本模型（gc64 / inp256） | 大模型（gc128 / inp512） |
|---|---|---|
| 参数量 | **513,856 (0.514M)** | 2,052,800 (2.053M) |
| Best PSNR | **32.2142** @480k | 32.2017 @940k |
| Best SSIM | 0.9525 @60k | **0.9534** @770k |

结论：luma-only 任务的目标函数（主要是随亮度/空间缓变的色调曲线）容量需求低，**global 通道与缩略图分辨率翻 4 倍几乎没有收益**（PSNR 差 0.01 dB = 50 张验证图上的噪声级），小模型还收敛更快（480k 即达 best）。部署优先选本配置。

---

## 11. 模型特性与局限

### 11.1 色度严格保持
对未触顶像素，`out = x · r(p)` ⇒ 色相、饱和度、白点不变；灰阶输出仍为灰阶。

### 11.2 局部、平滑、保边
grid 只有 32×32×16 且切片用三线性插值 ⇒ 亮度变换在空间与亮度方向都平滑，无块效应/振铃；亮度维使 grid 能在边缘两侧给出不同变换，具备保边能力。

### 11.3 与训练数据的关系
模型学习的是"BT.709 SDR 码值 → 目标亮度"的映射。若训练 GT 的 HDR 观感来自某种 tone mapping（本数据为 dump_hdr 16bit），模型会把该风格固化进 LUT；换采集条件/目标风格需要重训。

### 11.4 注意点
- **推理 clamp**：增益 >1 的高光会触顶，比例轻微改变（实测约 0.17% 像素），高光层次也可能受损；必要时可加高光门控或 strength 控制。
- **纯黑保护**：Y≤1e-6 时 ratio=1，黑场不会被抬亮。
- **输入要求**：3 通道 RGB，[0,1]，BT.709 非线性码值；不要先做线性化。
- **固定输入尺寸**：低分辨率分支要求 `fixed_ds_size=256`；换尺寸必须与 checkpoint 一致（`strict_load_g: true`）。
- **色彩空间**：等比缩放与色彩空间矩阵可交换，因此 BT.2020 输入也能用（色度结果一致，增益略有差异）。

---

## 12. 同系列变体

| 模型 | 系数 | 特点 |
|---|---|---|
| `HDRnetBT709Paper` | 12（3×4 仿射） | 可改色度，全色域对齐 |
| `HDRnetBT709PaperLumaOnly` | **2（scale+offset）** | **只改亮度，色度不变（本文档）** |
| `HDRnetBT709PaperMaxRGB` | 2（maxRGB 域） | 以 max(R,G,B) 为引导，高光保护更好 |
| `HDRnetBT709PaperLegacy` | 12 | affine-fix 之前的旧权重兼容实现 |

---

## 13. 使用

### 13.1 训练

```bash
cd /mnt/data/wcm/HDRNet
python basicsr/train.py -opt options/train/Custom/train_hdrnet_bt709_sdr2hdr_luma_only_gd16_bc1_gc64_inp256_gw32.yml
```

### 13.2 推理

```bash
cd /mnt/data/wcm/HDRNet
PYTHONPATH=. python basicsr/test.py \
  -opt options/test/Custom/test_hdrnet_bt709_sdr2hdr_luma_only_gd16_bc1_gc64_inp256_gw32.yml
```

- 输入目录由 `datasets.test_1.dataroot_lq` 指定，输出落在 `results/<name>/visualization/`；
- 权重需与 `grid_H/W/D`、`base_channels`、`fixed_ds_size`、`global_channels` 完全匹配；
- 本配置的 checkpoint：`experiments/train_hdrnet_bt709_sdr2hdr_luma_only_gd16_bc1_gc64_inp256_gw32/models/net_g_*.pth`（best 在 480k）。

---

## 附录 A：形状速查

| 张量 | 训练（512 crop, B=64） | 4K 推理（B=1） |
|---|---|---|
| 输入 x | (64,3,512,512) | (1,3,2160,3840) |
| x_low | (64,3,256,256) | (1,3,256,256) |
| splat 输出 | (64,64,32,32) | (1,64,32,32) |
| global_trunk 输出 | (64,64,4,4) → 1024 | (1,64,4,4) → 1024 |
| global 向量 | (64,64,1,1) | (1,64,1,1) |
| fusion | (64,64,32,32) | (1,64,32,32) |
| coeff_conv 输出 | (64,32,32,32) | (1,32,32,32) |
| grid | (64,16,32,32,2) | (1,16,32,32,2) |
| guide Y | (64,1,512,512) | (1,1,2160,3840) |
| scale / offset / ratio | (64,1,512,512) | (1,1,2160,3840) |
| 输出 | (64,3,512,512) | (1,3,2160,3840) |

## 附录 B：相关文件

- 架构：`basicsr/archs/hdrnet_bt709_paper_luma_only_arch.py`
- GuideNet：`basicsr/archs/hdrnet_bt709_arch.py`
- 损失：`basicsr/losses/basic_loss.py`（`LumaL1Loss`）
- 数据集：`basicsr/data/hdr_paired_dataset.py`（`HDRPairedImageDataset`）
- 训练配置：`options/train/Custom/train_hdrnet_bt709_sdr2hdr_luma_only_gd16_bc1_gc64_inp256_gw32.yml`
- 旧版结构说明（基于测试 yml 的同架构文档）：`docs/HDRnetBT709PaperLumaOnly_模型结构说明.md`
