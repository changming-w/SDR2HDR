import os
from glob import glob

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ---- 1. 加载 MiDaS/DPT 模型（官方 MiDaS 的简化版本） ----

def load_midas(device=None, model_type="DPT_Large"):
    """
    使用 torch.hub 从 isl-org/MiDaS 加载 DPT/MiDaS 模型和对应 transform。
    model_type: "DPT_Large", "DPT_Hybrid", "MiDaS_small" 等
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 从 MiDaS 官方 repo 拉模型
    model = torch.hub.load("isl-org/MiDaS", model_type)
    model.to(device)
    model.eval()

    # 输入变换
    transforms = torch.hub.load("isl-org/MiDaS", "transforms")
    if "DPT" in model_type:
        transform = transforms.dpt_transform
    else:
        transform = transforms.small_transform

    return model, transform, device


@torch.no_grad()
def infer_depth(model, transform, device, img_bgr):
    """
    输入：BGR uint8, HxWx3
    输出：float32 depth, HxW
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)  # 1x3xH'xW'

    depth = model(input_batch)  # 1x1xH'xW' 或 1xHxW
    if depth.ndim == 3:
        depth = depth.unsqueeze(1)

    depth = F.interpolate(
        depth, size=img_rgb.shape[:2], mode="bicubic", align_corners=False
    )  # 1x1xH xW
    depth = depth[0, 0].cpu().numpy().astype(np.float32)  # HxW

    return depth

def depth_to_fg_mask_soft(depth, invert=False, low=0.3, high=1.0, blur_ksize=31):
    """
    depth: HxW float32
    invert: 是否先 1 - depth
    low, high: 只增强 [low, high] 这段深度，生成一个 0~1 的软 mask
               0~low 这一段基本不增强（背景），越靠近 high 增强越强（前景）
    """
    d = depth.copy()
    d = (d - d.min()) / (d.max() - d.min() + 1e-6)

    if invert:
        d = 1.0 - d

    # 映射到 [0,1]，low 以下→0，high 以上→1，中间线性变化
    d = np.clip((d - low) / (high - low + 1e-6), 0.0, 1.0)

    # 可选：做一次较大的模糊，让过渡更宽
    if blur_ksize > 1:
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        d = cv2.GaussianBlur(d, (k, k), 0)

    # 再归一化一次，避免 blur 后范围变窄
    d = (d - d.min()) / (d.max() - d.min() + 1e-6)

    return d  # 直接用于混合的软 mask，0~1 连续

def depth_to_fg_mask(depth, invert=True, top_ratio=0.4, blur_ksize=15):
    """
    depth: float32, HxW
    invert: True 表示先做 1 - depth，让“近处=高值”
    top_ratio: 取靠近相机的 top_ratio 范围作为前景（0~1）
    """
    d = depth.copy()
    d = (d - d.min()) / (d.max() - d.min() + 1e-6)  # [0,1]
    if invert:
        d = 1.0 - d  # 近处 → 值更大

    # 根据分位数阈值生成粗 mask
    thr = np.quantile(d, 1.0 - top_ratio)  # 比如 top 40% 最“近”的区域
    mask = (d >= thr).astype(np.float32)    # HxW, {0,1}

    # 高斯模糊平滑一下
    if blur_ksize > 1:
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)

    # 再归一化一下
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)  # [0,1]
    return mask  # HxW


# ---- 2. 主体增强：用 mask 提亮/拉对比 ----

def enhance_subject(img_bgr_uint8, mask_float, gain=1.05, contrast=0.05):
    img = img_bgr_uint8.astype(np.float32) / 255.0

    if mask_float.ndim == 2:
        mask = mask_float[..., None]
    else:
        mask = mask_float
    mask = np.clip(mask, 0.0, 1.0)

    # 轻微提亮主体
    subj = img * gain

    # 很轻的对比度调整（对比度太夸张就干脆设成 0）
    if contrast != 0:
        subj = (subj - 0.5) * (1.0 + contrast) + 0.5

    subj = np.clip(subj, 0.0, 1.0)

    mask_blur = cv2.GaussianBlur(mask, (15, 15), 0)
    mask_blur = np.clip(mask_blur, 0.0, 1.0)

    # 扩成 3 通道
    if mask_blur.ndim == 2:
        mask_blur = mask_blur[..., None]
    if mask_blur.shape[2] == 1:
        mask_blur3 = np.repeat(mask_blur, 3, axis=2)
    else:
        mask_blur3 = mask_blur

    out = mask_blur3 * subj + (1.0 - mask_blur3) * img
    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


# ---- 3. 批处理：原 GT -> 深度 -> 前景 mask -> 增强后 GT ----

def batch_enhance_gt_with_dpt(
    gt_dir,
    new_gt_dir,
    mask_vis_dir,
    pattern="*.png",
    model_type="DPT_Large",
):
    os.makedirs(new_gt_dir, exist_ok=True)
    os.makedirs(mask_vis_dir, exist_ok=True)

    model, transform, device = load_midas(model_type=model_type)
    paths = sorted(glob(os.path.join(gt_dir, pattern)))
    print(f"Found {len(paths)} images in {gt_dir}")

    for i, p in enumerate(paths):
        name = os.path.basename(p)
        stem, ext = os.path.splitext(name)

        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            print("[WARN] read fail:", p)
            continue

        # 1) 深度估计
        depth = infer_depth(model, transform, device, img)

        # 2) 从深度生成前景 mask
        mask = depth_to_fg_mask_soft(
            depth,
            invert=False,   # 当前模型近处值大，通常不需要 invert
            low=0.3,        # 0~0.3：当背景/弱增强
            high=1.0,       # 0.3~1.0：越靠近 1 增强越强
            blur_ksize=15,  # 加大模糊核，让过渡区域更宽
        )  # HxW

        # 保存 mask 方便肉眼检查
        cv2.imwrite(
            os.path.join(mask_vis_dir, stem + "_mask.png"),
            (mask * 255).astype(np.uint8),
        )

        # 3) 主体增强，形成新 GT
        new_gt = enhance_subject(
            img,
            mask,
            gain=1.14,
            contrast=0.08,
        )
        cv2.imwrite(os.path.join(new_gt_dir, stem + ext), new_gt)

        if (i + 1) % 10 == 0 or i == len(paths) - 1:
            print(f"[{i+1}/{len(paths)}] processed {name}")


if __name__ == "__main__":
    gt_dir = r"/mnt/data/datasets/HDR/train_set/train_sdr_gt"
    new_gt_dir = r"/mnt/data/datasets/HDR/train_set/gt/gt_subject_enhanced_dpt_0.3_1.14_0.08"
    mask_vis_dir = r"/mnt/data/datasets/HDR/train_set/gt/gt_subject_mask_dpt_0.3_1.14_0.08"

    batch_enhance_gt_with_dpt(
        gt_dir,
        new_gt_dir,
        mask_vis_dir,
        pattern="*.png",
        model_type="DPT_Hybrid",  # 也可以用 "DPT_Large", "MiDaS_small" 等
    )