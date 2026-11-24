import os
import cv2
import torch
import numpy as np
from torchvision import transforms
import torch.nn.functional as F

# -------------------------
# 1. 加载 DPT-Hybrid 深度模型
# -------------------------

def load_dpt_hybrid(device="cuda"):
    """
    使用 torch.hub 加载 DPT-Hybrid 模型（MiDaS v3 系列）。
    也可以改成你本地 clone 的 repo 方式。
    """
    # 第一次运行会自动下载权重
    model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
    model.to(device)
    model.eval()

    # 对应的官方预处理（重要）
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform

    return model, transform


# -------------------------
# 2. 利用 DPT-Hybrid 生成深度图
# -------------------------

@torch.no_grad()
def estimate_depth_dpt(image_bgr, model, transform, device="cuda"):
    """
    输入：BGR uint8 图像 (H, W, 3)
    输出：深度图 depth (H, W)，float32
    """
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # transform 会做 resize / normalization / ToTensor
    input_batch = transform(img_rgb).to(device)  # 1,C,H,W

    # 前向推理
    prediction = model(input_batch)

    # 官方 MiDaS 输出是 [1, H', W']，需要插值到原图大小
    prediction = F.interpolate(
        prediction.unsqueeze(1),
        size=img_rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze(1)

    depth = prediction.squeeze().cpu().numpy().astype("float32")  # (H, W)
    return depth


# -------------------------
# 3. 根据深度图生成前后景权重 mask
# -------------------------

def build_weight_mask_from_depth(depth, t_near=0.2, t_far=0.7, blur_ksize=21, blur_sigma=7):
    """
    输入：
        depth: (H, W) float32，相对深度（数值大约表示远）
    输出：
        w: (H, W) float32, [0,1]，越靠前(近)权重越大

    t_near: 归一化后深度 <= t_near 认为是前景（权重接近 1）
    t_far : 归一化后深度 >= t_far 认为是背景（权重接近 0）
    """

    d = depth.copy()

    # 1) 归一化到 [0,1]
    d_min, d_max = np.min(d), np.max(d)
    if d_max - d_min < 1e-6:
        # 深度退化，直接全 1
        return np.ones_like(d, dtype=np.float32)

    d_norm = (d - d_min) / (d_max - d_min + 1e-8)  # 0~1, 0 近 1 远（注意：MiDaS 其实是相对深度，可以按排序理解）

    # 2) 映射成权重：前景大、背景小
    #    w = 1 - (d_norm - t_near) / (t_far - t_near)
    #    再裁剪到 [0,1]
    w = 1.0 - (d_norm - t_near) / (t_far - t_near + 1e-8)
    w = np.clip(w, 0.0, 1.0).astype(np.float32)

    # 3) 高斯模糊平滑，避免硬边和噪点
    if blur_ksize % 2 == 0:
        blur_ksize += 1  # OpenCV 要求奇数 kernel
    w_blur = cv2.GaussianBlur(w, (blur_ksize, blur_ksize), blur_sigma)

    # 再次裁剪
    w_blur = np.clip(w_blur, 0.0, 1.0).astype(np.float32)
    return w_blur


# -------------------------
# 4. 示例：一个占位的去模糊模型（请替换成你自己的）
# -------------------------

class DummyDeblurModel(torch.nn.Module):
    """
    示例模型：这里只做一个轻微锐化，实际项目中请换成你的去模糊网络。
    """
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, x):
        # x: [B,3,H,W], [0,1]
        # 用一个很简单的锐化核模拟“清晰图”
        # 仅示意，不要在实际任务中使用这个 dummy
        kernel = torch.tensor(
            [[0, -1,  0],
             [-1, 5, -1],
             [0, -1,  0]],
            dtype=torch.float32, device=x.device
        ).view(1, 1, 3, 3)

        x_gray = 0.299 * x[:,0:1] + 0.587 * x[:,1:2] + 0.114 * x[:,2:3]
        x_sharp_gray = F.conv2d(x_gray, kernel, padding=1)
        x_sharp_gray = torch.clamp(x_sharp_gray, 0.0, 1.0)
        x_sharp = x + (x_sharp_gray - x_gray)
        x_sharp = torch.clamp(x_sharp, 0.0, 1.0)
        return x_sharp


# -------------------------
# 5. 图像读写与融合
# -------------------------

def to_tensor_image(image_bgr):
    """
    BGR uint8 -> torch.FloatTensor [1,3,H,W], 归一化到 0~1
    """
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    return tensor


def to_bgr_image(tensor):
    """
    torch.FloatTensor [1,3,H,W], 0~1 -> BGR uint8
    """
    tensor = tensor.squeeze(0).clamp(0.0, 1.0)
    img_rgb = tensor.permute(1, 2, 0).cpu().numpy()
    img_rgb = (img_rgb * 255.0 + 0.5).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr


def fuse_sharp_and_blur(I_blur_bgr, I_sharp_bgr, w):
    """
    I_blur_bgr, I_sharp_bgr: (H,W,3) uint8
    w: (H,W) float32, [0,1]
    输出：融合后的 BGR uint8 图像
    """
    h, w_img, _ = I_blur_bgr.shape
    assert I_sharp_bgr.shape[:2] == (h, w_img), "尺寸不一致"
    assert w.shape == (h, w_img), "mask 尺寸不一致"

    w3 = np.stack([w, w, w], axis=2)  # (H,W,3)
    blur_f = I_blur_bgr.astype(np.float32)
    sharp_f = I_sharp_bgr.astype(np.float32)

    out = w3 * sharp_f + (1.0 - w3) * blur_f
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


# -------------------------
# 6. 主流程：读图 -> 去模糊 -> 深度 -> 权重 -> 融合
# -------------------------

def main(
    input_path,
    output_sharp_path,
    output_depth_path,
    output_mask_path,
    output_fused_path,
    device="cuda"
):
    # 1) 读入原图（I_blur）
    I_blur_bgr = cv2.imread(input_path)
    if I_blur_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")
    h, w = I_blur_bgr.shape[:2]
    print(f"Input image size: {w}x{h}")

    # 2) 加载 DPT-Hybrid 模型
    dpt_model, dpt_transform = load_dpt_hybrid(device=device)

    # 3) 估计深度图
    depth = estimate_depth_dpt(I_blur_bgr, dpt_model, dpt_transform, device=device)
    depth_norm_vis = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_vis = (depth_norm_vis * 255).astype(np.uint8)
    cv2.imwrite(output_depth_path, depth_vis)
    print(f"Saved depth visualization to {output_depth_path}")

    # 4) 根据深度构建权重 mask
    w_mask = build_weight_mask_from_depth(
        depth,
        t_near=0.2,  # 可以根据你画面的实际深度分布微调
        t_far=0.7,
        blur_ksize=21,
        blur_sigma=7
    )
    w_mask_vis = (w_mask * 255).astype(np.uint8)
    cv2.imwrite(output_mask_path, w_mask_vis)
    print(f"Saved weight mask to {output_mask_path}")

    # 5) 去模糊模型推理（这里用 DummyDeblurModel 占位）
    device_torch = torch.device(device if torch.cuda.is_available() else "cpu")
    deblur_model = DummyDeblurModel().to(device_torch)
    deblur_model.eval()

    I_blur_tensor = to_tensor_image(I_blur_bgr).to(device_torch)  # [1,3,H,W]
    with torch.no_grad():
        I_sharp_tensor = deblur_model(I_blur_tensor)
    I_sharp_bgr = to_bgr_image(I_sharp_tensor)
    cv2.imwrite(output_sharp_path, I_sharp_bgr)
    print(f"Saved sharp result to {output_sharp_path}")

    # 6) 用深度权重进行融合
    I_fused_bgr = fuse_sharp_and_blur(I_blur_bgr, I_sharp_bgr, w_mask)
    cv2.imwrite(output_fused_path, I_fused_bgr)
    print(f"Saved fused result to {output_fused_path}")


if __name__ == "__main__":
    # 修改成你自己的路径
    input_image_path = "/mnt/ext_ssd/wangchangming/datasets/jingshen_mix_test/input/5-input.png"               # 原始输入图
    output_sharp_path = "/mnt/ext_ssd/wangchangming/datasets/jingshen_mix_test/test_finetune_aisr_lora_region_loss_vgg_loss_finetune_sepconv-slomo/5-input.png"       # 去模糊结果（模型输出）
    output_depth_path = "output_depth.png"       # 深度可视化
    output_mask_path = "output_mask.png"         # 权重 mask 可视化
    output_fused_path = "output_fused.png"       # 融合后的最终结果

    os.makedirs(os.path.dirname(output_sharp_path) or ".", exist_ok=True)

    main(
        input_image_path,
        output_sharp_path,
        output_depth_path,
        output_mask_path,
        output_fused_path,
        device="cuda"
    )