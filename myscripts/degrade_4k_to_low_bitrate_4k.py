import cv2
import numpy as np

def degrade_4k_to_low_bitrate_4k(hr_4k, scale=2):
    """
    面向“真实低质量 4K 视频”的退化：
    - 有明显细节损失、压缩感、轻块效应和轻噪声
    - 避免极端棋盘格块
    """
    h, w = hr_4k.shape[:2]
    h_aligned = h - (h % scale)
    w_aligned = w - (w % scale)
    hr_4k = hr_4k[:h_aligned, :w_aligned, :].copy()
    img = hr_4k.astype(np.float32)

    # ---------------- 1. 双重 JPEG（但比你现在温和） ----------------
    # 第一次：中等质量（大部分真实 4K 流媒体的水平）
    q1 = np.random.randint(26, 40)  # 原来 20-34，整体抬高
    _, enc1 = cv2.imencode('.jpg', img.astype(np.uint8),
                           [int(cv2.IMWRITE_JPEG_QUALITY), q1])
    img_jpeg = cv2.imdecode(enc1, cv2.IMREAD_COLOR).astype(np.float32)

    # 第二次：稍差一点的质量，但不要太狠，概率也降一点
    if np.random.random() < 0.5:                   # 原 0.8 -> 0.5
        q2 = np.random.randint(18, 32)             # 原 8-24，抬高，避免个位数
        _, enc2 = cv2.imencode('.jpg', img_jpeg.astype(np.uint8),
                               [int(cv2.IMWRITE_JPEG_QUALITY), q2])
        img_jpeg = cv2.imdecode(enc2, cv2.IMREAD_COLOR).astype(np.float32)

    # ---------------- 2. 模糊（轻～中） ----------------
    if np.random.random() < 0.8:                   # 略高，保证细节有损
        kernel_size = np.random.choice([3, 5, 7])
        sigma = np.random.uniform(0.8, 2.0)        # 比你原来的 1.0-2.6 弱一点
        img_blurred = cv2.GaussianBlur(img_jpeg, (kernel_size, kernel_size),
                                       sigmaX=sigma)
    else:
        img_blurred = img_jpeg

    # ---------------- 3. 下采样到 1080p ----------------
    lr_h, lr_w = h_aligned // scale, w_aligned // scale
    r = np.random.random()
    # NEAREST 比例调小，主要靠 AREA/LINEAR，让块感柔和很多
    if r < 0.1:                                    # 原 0.4 -> 0.1
        interp_down = cv2.INTER_NEAREST
    elif r < 0.7:
        interp_down = cv2.INTER_AREA
    else:
        interp_down = cv2.INTER_LINEAR
    lr_1080p = cv2.resize(img_blurred, (lr_w, lr_h), interpolation=interp_down)

    # ---------------- 3.5 1080p 上再 JPEG（可选，弱化） ----------------
    # 真实链路里确实常有“1080p 再编码”，但不必太狠
    if np.random.random() < 0.4:                   # 原 0.8 -> 0.4
        q3 = np.random.randint(22, 36)             # 原 10-28，抬高很多
        _, enc3 = cv2.imencode('.jpg', lr_1080p.astype(np.uint8),
                               [int(cv2.IMWRITE_JPEG_QUALITY), q3])
        lr_1080p = cv2.imdecode(enc3, cv2.IMREAD_COLOR).astype(np.float32)

    # ---------------- 4. 块状噪声（更轻、更少） ----------------
    # 只做一点点块间不连续，不再刻意制造棋盘格
    if np.random.random() < 0.15:                  # 原 0.25 -> 0.15
        block_size = 8                              # 还原到 8，避免 4 像极端伪影
        h_lr, w_lr = lr_1080p.shape[:2]
        for i in range(0, h_lr, block_size):
            for j in range(0, w_lr, block_size):
                block_noise = np.random.uniform(-1.0, 1.0)
                lr_1080p[i:i+block_size, j:j+block_size, :] += block_noise

    # ---------------- 5. 轻度噪声（保守） ----------------
    if np.random.random() < 0.4:
        noise_std = np.random.uniform(0.5, 1.5)    # 上限从 2.0 拉到 1.5
        noise = np.random.randn(*lr_1080p.shape) * noise_std
        lr_1080p = lr_1080p + noise

    # ---------------- 6a. 块级轻微色漂（保持） ----------------
    if np.random.random() < 0.3:                   # 稍微降一点
        drift_block_size = np.random.choice([8, 16])
        h_lr, w_lr = lr_1080p.shape[:2]
        for i in range(0, h_lr, drift_block_size):
            for j in range(0, w_lr, drift_block_size):
                r_drift = np.random.randint(-1, 2) # [-1,1]
                g_drift = np.random.randint(-1, 2)
                b_drift = np.random.randint(-1, 2)
                lr_1080p[i:i+drift_block_size, j:j+drift_block_size, 0] += b_drift
                lr_1080p[i:i+drift_block_size, j:j+drift_block_size, 1] += g_drift
                lr_1080p[i:i+drift_block_size, j:j+drift_block_size, 2] += r_drift

    # ---------------- 6b. 全局亮度 / 对比度微调（保守） ----------------
    if np.random.random() < 0.4:
        alpha = np.random.uniform(0.95, 1.05)      # 原 0.9-1.05，收窄
        beta = np.random.uniform(-3, 3)
        lr_1080p = lr_1080p * alpha + beta

    lr_1080p = np.clip(lr_1080p, 0, 255).astype(np.uint8)

    # ---------------- 7. 上采样回 4K ----------------
    interp_up = cv2.INTER_CUBIC if np.random.random() < 0.7 else cv2.INTER_LINEAR
    lr_4k = cv2.resize(lr_1080p, (w_aligned, h_aligned), interpolation=interp_up)

    hr_ref = hr_4k.astype(np.float32)
    return lr_4k, hr_ref

def degrade_4k_to_low_bitrate_4k_nonoise(hr_4k, scale=2):
    """
    面向“真实低质量 4K 视频”的退化（无显式噪声），顺序：
    4K → 1080p（下采样）→ JPEG/模糊等退化 → 1080p → 上采样回 4K
    """
    h, w = hr_4k.shape[:2]
    h_aligned = h - (h % scale)
    w_aligned = w - (w % scale)
    hr_4k = hr_4k[:h_aligned, :w_aligned, :].copy()
    img = hr_4k.astype(np.float32)

    # ---------------- 1. 先下采样到 1080p（模拟平台内部编码分辨率） ----------------
    lr_h, lr_w = h_aligned // scale, w_aligned // scale
    r = np.random.random()
    # 下采样以 AREA/LINEAR 为主，少量 NEAREST 作轻微块感
    if r < 0.1:
        interp_down = cv2.INTER_NEAREST
    elif r < 0.7:
        interp_down = cv2.INTER_AREA
    else:
        interp_down = cv2.INTER_LINEAR
    lr_1080p = cv2.resize(img, (lr_w, lr_h), interpolation=interp_down)

    # ---------------- 2. 在 1080p 上做 1–2 次 JPEG 压缩（主退化） ----------------
    severity = np.random.rand()
    img_jpeg = lr_1080p

    if severity < 0.5:
        # 轻压缩
        q1 = np.random.randint(30, 45)
        _, enc1 = cv2.imencode('.jpg', img_jpeg.astype(np.uint8),
                               [int(cv2.IMWRITE_JPEG_QUALITY), q1])
        img_jpeg = cv2.imdecode(enc1, cv2.IMREAD_COLOR).astype(np.float32)

        if np.random.random() < 0.3:
            q2 = np.random.randint(24, 36)
            _, enc2 = cv2.imencode('.jpg', img_jpeg.astype(np.uint8),
                                   [int(cv2.IMWRITE_JPEG_QUALITY), q2])
            img_jpeg = cv2.imdecode(enc2, cv2.IMREAD_COLOR).astype(np.float32)
    else:
        # 重压缩
        q1 = np.random.randint(22, 36)
        _, enc1 = cv2.imencode('.jpg', img_jpeg.astype(np.uint8),
                               [int(cv2.IMWRITE_JPEG_QUALITY), q1])
        img_jpeg = cv2.imdecode(enc1, cv2.IMREAD_COLOR).astype(np.float32)

        if np.random.random() < 0.6:
            q2 = np.random.randint(16, 30)
            _, enc2 = cv2.imencode('.jpg', img_jpeg.astype(np.uint8),
                                   [int(cv2.IMWRITE_JPEG_QUALITY), q2])
            img_jpeg = cv2.imdecode(enc2, cv2.IMREAD_COLOR).astype(np.float32)

    # ---------------- 3. 模糊（轻～中） ----------------
    if np.random.random() < 0.8:
        kernel_size = np.random.choice([3, 5, 7])
        sigma = np.random.uniform(0.8, 2.0)
        lr_1080p_degraded = cv2.GaussianBlur(
            img_jpeg, (kernel_size, kernel_size), sigmaX=sigma
        )
    else:
        lr_1080p_degraded = img_jpeg

    # ---------------- 4. 全局亮度 / 对比度轻微调整 ----------------
    if np.random.random() < 0.2:
        alpha = np.random.uniform(0.95, 1.05)
        beta = np.random.uniform(-2, 2)
        lr_1080p_degraded = lr_1080p_degraded * alpha + beta

    lr_1080p_degraded = np.clip(lr_1080p_degraded, 0, 255).astype(np.uint8)

    # ---------------- 5. 上采样回 4K ----------------
    interp_up = cv2.INTER_CUBIC if np.random.random() < 0.7 else cv2.INTER_LINEAR
    lr_4k = cv2.resize(lr_1080p_degraded, (w_aligned, h_aligned),
                       interpolation=interp_up)

    hr_ref = hr_4k.astype(np.float32)
    return lr_4k, hr_ref

def degrade_4k_to_low_bitrate_4k_light(hr_4k, scale=2):
    """
    4K → 1080p → 退化 → 4K
    目标：有明显压缩/轻模糊，但不要糊成一片，保留可增强的细节结构。
    """
    h, w = hr_4k.shape[:2]
    h_aligned = h - (h % scale)
    w_aligned = w - (w % scale)
    hr_4k = hr_4k[:h_aligned, :w_aligned, :].copy()
    img = hr_4k.astype(np.float32)

    # ---------------- 1. 先下采样到 1080p ----------------
    lr_h, lr_w = h_aligned // scale, w_aligned // scale
    r = np.random.random()
    # 减少 NEAREST，避免太糊又带强块
    if r < 0.05:
        interp_down = cv2.INTER_NEAREST
    elif r < 0.75:
        interp_down = cv2.INTER_AREA
    else:
        interp_down = cv2.INTER_LINEAR
    lr_1080p = cv2.resize(img, (lr_w, lr_h), interpolation=interp_down)

    # ---------------- 2. 在 1080p 上做 1~2 次 JPEG 压缩 ----------------
    # 整体比你刚才的版本再温和一点
    severity = np.random.rand()
    img_jpeg = lr_1080p

    if severity < 0.5:
        # 轻压缩
        q1 = np.random.randint(32, 45)      # 32–44
        _, enc1 = cv2.imencode('.jpg', img_jpeg.astype(np.uint8),
                               [int(cv2.IMWRITE_JPEG_QUALITY), q1])
        img_jpeg = cv2.imdecode(enc1, cv2.IMREAD_COLOR).astype(np.float32)

        if np.random.random() < 0.25:
            q2 = np.random.randint(26, 38)  # 26–37
            _, enc2 = cv2.imencode('.jpg', img_jpeg.astype(np.uint8),
                                   [int(cv2.IMWRITE_JPEG_QUALITY), q2])
            img_jpeg = cv2.imdecode(enc2, cv2.IMREAD_COLOR).astype(np.float32)
    else:
        # 中等压缩
        q1 = np.random.randint(26, 38)      # 26–37
        _, enc1 = cv2.imencode('.jpg', img_jpeg.astype(np.uint8),
                               [int(cv2.IMWRITE_JPEG_QUALITY), q1])
        img_jpeg = cv2.imdecode(enc1, cv2.IMREAD_COLOR).astype(np.float32)

        if np.random.random() < 0.5:
            q2 = np.random.randint(20, 32)  # 20–31
            _, enc2 = cv2.imencode('.jpg', img_jpeg.astype(np.uint8),
                                   [int(cv2.IMWRITE_JPEG_QUALITY), q2])
            img_jpeg = cv2.imdecode(enc2, cv2.IMREAD_COLOR).astype(np.float32)

    # ---------------- 3. 模糊（明显减弱） ----------------
    # 这里只做轻模糊，避免压缩+放大后过糊
    if np.random.random() < 0.5:
        kernel_size = np.random.choice([3, 5])      # 不用 7 了
        sigma = np.random.uniform(0.4, 1.2)         # 比之前 0.8–2.0 弱很多
        lr_1080p_degraded = cv2.GaussianBlur(
            img_jpeg, (kernel_size, kernel_size), sigmaX=sigma
        )
    else:
        lr_1080p_degraded = img_jpeg

    # ---------------- 4. 全局亮度 / 对比度轻微调整 ----------------
    if np.random.random() < 0.3:
        alpha = np.random.uniform(0.97, 1.03)   # 再收窄一点
        beta = np.random.uniform(-2, 2)
        lr_1080p_degraded = lr_1080p_degraded * alpha + beta

    lr_1080p_degraded = np.clip(lr_1080p_degraded, 0, 255).astype(np.uint8)

    # ---------------- 5. 上采样回 4K ----------------
    interp_up = cv2.INTER_CUBIC if np.random.random() < 0.8 else cv2.INTER_LINEAR
    lr_4k = cv2.resize(lr_1080p_degraded, (w_aligned, h_aligned),
                       interpolation=interp_up)

    hr_ref = hr_4k.astype(np.float32)
    return lr_4k, hr_ref

def motion_kernel(ksize, angle):
    """
    生成一个线性运动模糊核（归一化）。
    ksize: 核大小，建议奇数，如 5, 7, 9 ...
    angle: 运动方向，单位：度。0 表示水平向右，正角度逆时针旋转。
    """
    if ksize % 2 == 0:
        ksize += 1

    kernel = np.zeros((ksize, ksize), dtype=np.float32)

    theta = (angle % 180) * np.pi / 180.0
    center = ksize // 2
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # 在核中沿着指定方向画一条线
    for t in np.linspace(-center, center, ksize * 2):
        x = center + t * cos_t
        y = center + t * sin_t

        ix = int(round(x))
        iy = int(round(y))

        if 0 <= ix < ksize and 0 <= iy < ksize:
            kernel[iy, ix] = 1.0

    s = kernel.sum()
    if s > 0:
        kernel /= s
    else:
        kernel[:] = 1.0 / (ksize * ksize)

    return kernel


def degrade_for_strong_enhancer(hr):
    """
    针对非常清晰的 GT：
    - 模糊强度相对偏强（轻糊 + 中等糊），让模型恢复能力更强；
    - 高斯模糊为主，运动模糊为辅；
    - 少量轻 JPEG + 轻亮度/对比度/gamma 扰动。
    """
    img = hr.astype(np.float32)

    # --- 1. 模糊：高斯 或 线性运动 ---
    r_blur_type = np.random.rand()
    if r_blur_type < 0.8:
        # 75% 高斯模糊
        r = np.random.rand()
        if r < 0.5:
            # 50%: 轻糊
            k = np.random.choice([3, 5])
            sigma = np.random.uniform(0.5, 1.0)
        else:
            # 50%: 中等糊（比之前更强）
            k = np.random.choice([5, 7])
            sigma = np.random.uniform(1.0, 2.0)
        img = cv2.GaussianBlur(img, (k, k), sigmaX=sigma)
    else:
        # 25% 线性运动模糊
        r = np.random.rand()
        if r < 0.6:
            # 60%：轻运动模糊
            k = np.random.choice([5, 7])
            angle = np.random.uniform(-8, 8)
        else:
            # 40%：稍重运动模糊
            k = np.random.choice([7, 9])
            angle = np.random.uniform(-12, 12)
        kernel = motion_kernel(k, angle)
        img = cv2.filter2D(img, -1, kernel)

    # --- 2. 极轻 JPEG 压缩（提高泛化，不是重度去压缩） ---
    if np.random.rand() < 0.2:
        q = np.random.randint(50, 70)  # 轻压缩
        _, enc = cv2.imencode(
            '.jpg',
            img.clip(0, 255).astype(np.uint8),
            [int(cv2.IMWRITE_JPEG_QUALITY), q]
        )
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR).astype(np.float32)

    # --- 3. 轻亮度/对比度变化 ---
    if np.random.rand() < 0.2:
        alpha = np.random.uniform(0.98, 1.02)   # 对比度
        beta = np.random.uniform(-2.0, 2.0)     # 亮度
        img = img * alpha + beta

    # --- 4. 轻 gamma 变化 ---
    if np.random.rand() < 0.2:
        gamma = np.random.uniform(0.9, 1.1)
        img_norm = np.clip(img / 255.0, 0.0, 1.0)
        img = np.power(img_norm, gamma) * 255.0

    lr = np.clip(img, 0, 255).astype(np.uint8)
    hr_ref = hr.astype(np.float32)
    return lr, hr_ref