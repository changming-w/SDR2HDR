# HDRnetBT709PaperLumaOnly 模型结构说明

> 对应配置：`options/test/Custom/test_hdrnet_bt709_sdr2hdr_luma_only_gd16_bc1_gc64_inp256_gw32.yml`
> 代码实现：`basicsr/archs/hdrnet_bt709_paper_luma_only_arch.py`（GuideNet 位于 `basicsr/archs/hdrnet_bt709_arch.py`）
> 权重：`experiments/train_hdrnet_bt709_sdr2hdr_luma_only_gd16_bc1_gc64_inp256_gw32/models/net_g_480000.pth`

---

## 1. 概述

本模型是 HDRNet（Gharbi et al., *Deep Bilateral Learning for Real-Time Image Enhancement*, SIGGRAPH 2017）的**只改亮度（luma-only）**版本，用于 SDR→HDR 的局部色调映射。

与原版 HDRNet 相比，主要改动有三点：

| 项目 | 原版 HDRNet | 本模型 |
|---|---|---|
| 每个格点的系数 | 12 个（3×4 仿射矩阵，可改色彩） | 2 个（亮度 scale、offset） |
| Guide | 可学习的逐像素网络 | 固定公式：BT.709 亮度 Y，无可学习参数 |
| 输出方式 | 对 RGB 做仿射变换 | 先算出目标亮度 Y'，再将 RGB 三通道等比缩放 |

由此得到的特性：

- **色度不变**：输出与输入的 R:G:B 比例相同，色相、饱和度、白点均不改变，只有亮度改变。
- **局部自适应**：亮度变换随像素的位置和亮度变化，相当于一条随空间位置变化的色调曲线。
- **计算量与输出分辨率基本无关**：神经网络只在 256×256 的缩略图上运行，全分辨率部分只做查表插值和乘法。
- **参数量小**：总计 513,856 个参数（约 0.51M）。

---

## 2. 配置参数

```yaml
network_g:
  type: HDRnetBT709PaperLumaOnly
  num_in_ch: 3
  num_out_ch: 3
  grid_H: 32          # bilateral grid 空间高
  grid_W: 32          # bilateral grid 空间宽
  grid_D: 16          # bilateral grid 亮度维（亮度分 16 档）
  base_channels: 1    # 通道基数 cm
  fixed_ds_size: 256  # 低分辨率分支的输入尺寸
  global_channels: 64 # 全局/局部特征通道数 gc
```

| 参数 | 值 | 作用 |
|---|---|---|
| `grid_H × grid_W` | 32×32 | grid 的空间分辨率。4K 输入时每个格子约覆盖 120×68 像素 |
| `grid_D` | 16 | 亮度维的档数，也决定 splat 部分的通道倍数 |
| `base_channels` | 1 | splat 第 i 层的输出通道数为 `cm × 2^i × gd`，即 16 / 32 / 64 |
| `fixed_ds_size` | 256 | 任意尺寸的输入先双线性缩放到 256×256，再送入低分辨率分支 |
| `global_channels` | 64 | 全局分支、局部分支和融合特征的通道数 |

由配置推导出的层数：

- splat 下采样层数：`log2(fixed_ds_size / grid_H) = log2(256/32) = 3`
- 全局分支卷积层数：`log2(grid_H / 4) = log2(32/4) = 3`（下采样到 4×4）

测试配置的其他信息：`model_type: SRModel`，`scale: 1`，`num_gpu: 0`（在 CPU 上推理），`strict_load_g: true`，数据集类型为 `SingleImageDataset`（只有输入，没有 GT）。

---

## 3. 整体结构

```
                    输入 x (B,3,H,W)，[0,1]，非线性码值
                               │
           ┌───────────────────┼─────────────────────────┐
           │                   │                         │
   双线性缩放到 256×256     GuideNetBT709              （原图 x）
           │            Y = 0.2126R+0.7152G+0.0722B      │
           ▼                   │                         │
  ┌─────────────────┐          │                         │
  │ LowResNet       │          │                         │
  │ (splat/global/  │          │                         │
  │  local/fusion)  │          │                         │
  └────────┬────────┘          │                         │
           ▼                   ▼                         │
   grid (B,16,32,32,2)    guide Y (B,1,H,W)              │
           │                   │                         │
           └─────────┬─────────┘                         │
                     ▼                                   │
     三线性插值 (x, y, Y) → 每像素 (scale, offset)        │
                     ▼                                   │
          Y' = scale·Y + offset                          │
          ratio = Y' / Y                                 │
                     └──────────────┬────────────────────┘
                                    ▼
                         out = x · ratio（RGB 等比缩放）
                                    ▼
                    （可选）strength 混合 → 推理时 clamp(0,1)
```

整个模型由三个子模块组成：

| 子模块 | 类名 | 作用 | 参数量 |
|---|---|---|---|
| 低分辨率系数网络 | `LowResNetBT709PaperLumaOnly` | 根据缩略图预测 bilateral grid | 513,856 |
| 引导图 | `GuideNetBT709` | 计算 BT.709 亮度 Y | 0（系数为固定 buffer） |
| 切片与应用 | `BilateralSliceApplyLumaOnly` | 三线性插值，计算亮度变换，RGB 等比缩放 | 0 |

---

## 4. 低分辨率系数网络 LowResNet

输入为 `(B, 3, 256, 256)`，输出为 `(B, 16, 32, 32, 2)`。

### 4.1 逐层结构（本配置的实际值）

所有卷积后都接 `LeakyReLU(0.2)`，`coeff_conv` 除外。

| 阶段 | 层 | 结构 | 输出尺寸 | 参数量 |
|---|---|---|---|---|
| Splat（低层特征） | splat.0 | Conv 3×3, s2, 3→16 | 16×128×128 | 448 |
| | splat.1 | Conv 3×3, s2, 16→32 | 32×64×64 | 4,640 |
| | splat.2 | Conv 3×3, s2, 32→64 | 64×32×32 | 18,496 |
| 全局分支 | global_trunk.0 | Conv 3×3, s2, 64→64 | 64×16×16 | 36,928 |
| | global_trunk.1 | Conv 3×3, s2, 64→64 | 64×8×8 | 36,928 |
| | global_trunk.2 | Conv 3×3, s2, 64→64 | 64×4×4 | 36,928 |
| | 展平 | — | 1024 | — |
| | global_fc1 | Linear 1024→256 + LReLU | 256 | 262,400 |
| | global_fc2 | Linear 256→128 + LReLU | 128 | 32,896 |
| | global_fc3 | Linear 128→64（无激活） | 64×1×1 | 8,256 |
| 局部分支 | local_proj | Identity（splat 输出已是 64 通道） | 64×32×32 | 0 |
| | local_conv1 | Conv 3×3, s1, 64→64 | 64×32×32 | 36,928 |
| | local_conv2 | Conv 3×3, s1, 64→64 | 64×32×32 | 36,928 |
| 融合 | fusion | `ReLU(local + global)`，global 在空间上广播 | 64×32×32 | 0 |
| 系数输出 | coeff_conv | Conv 1×1, 64→32（= D×2） | 32×32×32 | 2,080 |
| 重排 | view + permute | `(B,32,32,32)` → `(B,16,2,32,32)` → `(B,16,32,32,2)` | 16×32×32×2 | — |
| **合计** | | | | **513,856** |

参数主要集中在全局分支的 `global_fc1`（约 51%）和 3 层 `global_trunk` 卷积（约 22%）。

### 4.2 各分支作用

- **Splat**：通过 3 次步长为 2 的下采样，从 256×256 缩到 32×32，与 grid 的空间分辨率一致。
- **全局分支**：继续下采样到 4×4，再经全连接层得到一个 64 维的整图特征向量，描述整张图的亮度分布、场景类型等全局信息。这个向量会加到每个空间位置上，所以同一张图的所有格点共享全局信息。
- **局部分支**：在 32×32 分辨率上做两层 3×3 卷积，提取局部亮度和对比度信息。
- **融合与输出**：两路相加后经 ReLU，再用 1×1 卷积在每个 32×32 位置输出 32 个数，重排为 16 个亮度档 × 2 个系数。

### 4.3 Grid 的含义

grid 形状为 `(B, D=16, Hg=32, Wg=32, 2)`：

- 空间维 32×32：把图像划分成 32×32 个区域。
- 亮度维 16：把 Y ∈ [0,1] 均分为 16 个节点（间隔 1/15）。
- 最后一维的 2 个系数：`scale` 和 `offset`。

可以理解为：每个空间区域有一条由 16 个节点定义的亮度变换曲线，节点 i 处的变换为 `Y' = scale_i · Y + offset_i`。

---

## 5. 引导图 GuideNetBT709

```python
guide = 0.2126 * R + 0.7152 * G + 0.0722 * B
```

- 系数通过 `register_buffer` 固定，不参与训练，也不会被优化器更新。
- 直接在输入的**非线性码值**上计算，不做线性化。得到的是 luma Y'，在本文档中仍记为 Y。
- 输出与输入同分辨率 `(B,1,H,W)`，取值在 [0,1]，用于：
  1. 作为 grid 亮度维的查表坐标；
  2. 作为亮度变换的原始亮度 Y。

---

## 6. 切片与应用 BilateralSliceApplyLumaOnly

对全分辨率图像的每个像素 \(p=(u,v)\)：

### 6.1 三线性插值坐标

\[
x_g = u\cdot\frac{W_g-1}{W-1},\quad
y_g = v\cdot\frac{H_g-1}{H-1},\quad
z_g = \mathrm{clamp}(Y(p),0,1)\cdot(D-1)
\]

空间坐标用 `linspace(0, Wg-1, W)` 均匀映射，图像的四个角对齐 grid 的四个角点。亮度坐标由 guide 决定。

### 6.2 插值得到系数

取包围 \((x_g,y_g,z_g)\) 的 8 个格点，按三线性权重加权，得到该像素的 \((s(p), o(p))\)。

### 6.3 亮度变换与 RGB 等比缩放

\[
Y'(p) = s(p)\cdot Y(p) + o(p)
\]
\[
r(p) = \frac{Y'(p)}{Y(p)+\varepsilon},\qquad \varepsilon=10^{-6}
\]
\[
\mathrm{out}(p) = x(p)\cdot r(p)
\]

- 当 \(Y(p)\le\varepsilon\)（纯黑）时 `r = 1`，像素保持不变，避免除零。
- R、G、B 三通道乘同一个 \(r(p)\)，因此 R:G:B 比例不变。
- 由于 \(s,o\) 依赖位置和亮度，\(r(p)\) 是关于图像内容、位置、亮度的非线性函数。

### 6.4 后处理

```python
if strength < 1.0:
    out = strength * out + (1 - strength) * x   # 与原图线性混合，控制强度
if self.training:
    return out                                   # 训练时不截断
return torch.clamp(out, 0.0, 1.0)                # 推理时截断到 [0,1]
```

- `strength` 默认为 1.0，即完整效果。
- 推理时 clamp 会让触顶像素的 RGB 比例发生轻微变化，这是色度不变性唯一的例外。

---

## 7. 前向流程（伪代码）

```python
def forward(x, strength=1.0):            # x: (B,3,H,W), [0,1]
    x_low = bilinear_resize(x, 256, 256)
    grid  = lowres_net(x_low)            # (B,16,32,32,2)
    Y     = 0.2126*R + 0.7152*G + 0.0722*B   # (B,1,H,W)
    s, o  = trilinear(grid, u, v, Y)     # 每像素 (scale, offset)
    ratio = (s*Y + o) / (Y + 1e-6)       # Y≈0 时 ratio=1
    out   = x * ratio
    out   = strength*out + (1-strength)*x
    return clamp(out, 0, 1)              # 仅推理时
```

---

## 8. 计算量与性能

| 部分 | 计算位置 | 计算量 |
|---|---|---|
| LowResNet | 256×256 缩略图 | 约 0.135 GMAC，与输出分辨率无关 |
| 缩放到 256 | 全分辨率 → 256 | 双线性插值 |
| Guide | 全分辨率 | 每像素 3 次乘加 |
| Slice + Apply | 全分辨率 | 每像素 8 次 gather × 2 个系数 + 三线性权重 + 1 次除法 + 3 次乘法 |

LowResNet 各层 MAC 估算：splat 约 45M，global_trunk 约 12M，FC 约 0.3M，local_conv 约 75M，coeff_conv 约 2M。

实测：PyTorch CPU 单帧 3840×2160 前向约 0.38 s（未做优化）。

---

## 9. 模型特性

### 9.1 色度不变
输出 = 输入 × 每像素同一个数，因此对未触顶的像素：
- R:G:B 比例严格不变，色相、饱和度不变；
- 灰色（R=G=B）输出后仍为灰色，白平衡不变。

实测（10 帧 4K）：未触顶像素的 RGB 比例平均偏差 0.0024；因 clamp 改变比例的像素占 0.168%。

### 9.2 局部色调映射
\(r(p)\) 随位置和亮度变化。实测按输入亮度分段的平均增益如下：中低亮压暗约 20–25%，高光基本保持。

| 输入 Y | 0–0.1 | 0.1–0.2 | 0.2–0.3 | 0.3–0.4 | 0.4–0.5 | 0.5–0.6 | 0.6–0.7 | 0.7–0.8 | 0.8–0.9 | 0.9–1.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| 平均增益 | 0.800 | 0.754 | 0.765 | 0.777 | 0.808 | 0.849 | 0.894 | 0.955 | 1.005 | 1.008 |

### 9.3 空间平滑
grid 只有 32×32×16，切片用三线性插值，所以亮度变换在空间和亮度两个方向上都是平滑的，不会产生块效应或振铃。同时，亮度维的存在使 grid 能在边缘两侧（亮度不同）给出不同的变换，具有保边特性。

### 9.4 与色彩空间的关系
模型不改变码值的编码 gamma，也不涉及显示 EOTF。guide 系数固定为 BT.709，但当输入为 BT.2020 编码时：
- 由于输出是 RGB 等比缩放，而色彩空间转换 \(G\) 与等比缩放可交换，\(G(F(x))\) 与 \(F(G(x))\) 的色度完全一致；
- 两者的差异只来自模型对两种输入施加的增益不同。实测相对差 2.74%，Pearson 0.968，SSIM 0.990，逐像素平均差 0.63–2.27/255（详见色彩空间验证报告）。

---

## 10. 使用说明

### 10.1 推理
```bash
cd /mnt/data/wcm/HDRNet
python basicsr/test.py -opt options/test/Custom/test_hdrnet_bt709_sdr2hdr_luma_only_gd16_bc1_gc64_inp256_gw32.yml
```
输入目录由 `datasets.test_1.dataroot_lq` 指定，输出保存在 `results/<name>/visualization/`。

### 10.2 注意事项
- **配置名与输入路径**：当前 yml 中 `name` 带有 `_bt2020` 后缀，`dataroot_lq` 指向 `model_input_bt2020`，是色彩空间验证时修改的。正常推理 BT.709 数据时请改回对应目录和名称，避免覆盖结果。
- **参数需与 checkpoint 一致**：`grid_H/W/D`、`base_channels`、`fixed_ds_size`、`global_channels` 任一不同都会导致 `strict_load_g: true` 加载失败。代码 docstring 中的示例配置（`fixed_ds_size: 512`、`global_channels: 128`）与本权重不符，不能直接套用。
- **输入要求**：3 通道 RGB，取值 [0,1]，非线性码值（模型在码值上训练，不要先线性化）。
- **触顶**：增益大于 1 的高光像素可能被 clamp，导致局部色度轻微偏移。
- **部署位置**：模型按码值做等比缩放。如果放在屏幕三通道 gamma 校正 LUT 之后，白点可能漂移，建议放在 gamma 校正之前。

---

## 附录：形状速查（B=1，4K 输入）

| 张量 | 形状 |
|---|---|
| 输入 x | (1, 3, 2160, 3840) |
| x_low | (1, 3, 256, 256) |
| splat 输出 | (1, 64, 32, 32) |
| global_trunk 输出 | (1, 64, 4, 4) → 展平 1024 |
| global 向量 | (1, 64, 1, 1) |
| fusion | (1, 64, 32, 32) |
| coeff_conv 输出 | (1, 32, 32, 32) |
| grid | (1, 16, 32, 32, 2) |
| guide Y | (1, 1, 2160, 3840) |
| scale / offset / ratio | (1, 1, 2160, 3840) |
| 输出 | (1, 3, 2160, 3840) |
