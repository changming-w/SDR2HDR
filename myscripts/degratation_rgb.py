import cv2
import numpy as np
import math
from basicsr.data.degradations import random_mixed_kernels

def degrade_4k_to_low_1080p(hr_4k, scale=2):
    """
    4K蓝光（80M）→ 低质量1080P（≈5M）：
    核心特征：块效应适中、细节进一步丢失、真实压缩色彩漂移（局部色调轻微割裂）
    """
    h, w = hr_4k.shape[:2]
    h = h - (h % scale)
    w = w - (w % scale)
    hr_4k = hr_4k[:h, :w, :].copy()
    img = hr_4k.astype(np.float32)

    # -------------------------- 1. 压缩强度小幅提升 -------------------------
    jpeg_quality = np.random.randint(12, 29)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    _, enc_img = cv2.imencode('.jpg', img.astype(np.uint8), encode_param)
    img_compressed = cv2.imdecode(enc_img, cv2.IMREAD_COLOR).astype(np.float32)

    # -------------------------- 2. 模糊强度小幅提升 -------------------------
    kernel_size = np.random.choice([7, 9, 11, 13])
    sigma = np.random.uniform(2.2, 4.0)
    img_blurred = cv2.GaussianBlur(img_compressed, (kernel_size, kernel_size), sigmaX=sigma)

    # -------------------------- 3. 下采样保留块效应 -------------------------
    lr_h, lr_w = h // scale, w // scale
    interp_method = cv2.INTER_NEAREST if np.random.random() < 0.7 else cv2.INTER_LINEAR
    lr_1080p = cv2.resize(img_blurred, (lr_w, lr_h), interpolation=interp_method)

    # -------------------------- 4. 块状噪声强度小幅提升 -------------------------
    if np.random.random() < 0.3:
        block_size = 8
        h_lr, w_lr = lr_1080p.shape[:2]
        for i in range(0, h_lr, block_size):
            for j in range(0, w_lr, block_size):
                block_noise = np.random.uniform(-3, 3)
                lr_1080p[i:i+block_size, j:j+block_size, :] += block_noise

    # -------------------------- 5. 真实压缩色彩漂移（核心修改） -------------------------
    # 模拟低码率压缩的“块内色彩偏移+局部色调割裂”，无彩色噪声
    if np.random.random() < 0.9:  # 90%概率出现，贴近真实压缩
        drift_block_size = 16  # 按16x16块漂移（JPEG压缩块的2倍，更自然）
        h_lr, w_lr = lr_1080p.shape[:2]
        for i in range(0, h_lr, drift_block_size):
            for j in range(0, w_lr, drift_block_size):
                # 每个块的三通道独立轻微漂移（模拟压缩时的色彩量化误差）
                # 偏移范围±3，仅轻微割裂，不产生异常色块
                r_drift = np.random.randint(-3, 4)
                g_drift = np.random.randint(-3, 4)
                b_drift = np.random.randint(-3, 4)

                # 块内漂移一致，块间独立，模拟真实压缩的色彩失真
                lr_1080p[i:i+drift_block_size, j:j+drift_block_size, 0] += b_drift  # B通道
                lr_1080p[i:i+drift_block_size, j:j+drift_block_size, 1] += g_drift  # G通道
                lr_1080p[i:i+drift_block_size, j:j+drift_block_size, 2] += r_drift  # R通道

    lr_1080p = np.clip(lr_1080p, 0, 255)
    hr_ref = hr_4k.astype(np.float32)
    return lr_1080p.astype(np.uint8), hr_ref