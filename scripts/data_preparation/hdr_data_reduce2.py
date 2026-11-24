import os
from glob import glob

import cv2
import numpy as np


# ------------------- 基础模块 -------------------

def apply_tone_curve(img, gamma=1.1, mid=0.5, strength=-0.2, black_protect=0.01):
    x = np.clip(img, 0.0, 1.0)
    y_gamma = np.power(x, gamma)

    y_s = (x - mid) * (1.0 + strength) + mid
    y_s = np.clip(y_s, 0.0, 1.0)

    y = 0.5 * y_gamma + 0.5 * y_s

    # 低亮度保护：对小于 black_protect 的区域做插值，避免被抬太高
    mask = (x < black_protect).astype(np.float32)
    y = mask * (x * (y / (black_protect + 1e-6))) + (1 - mask) * y

    return np.clip(y, 0.0, 1.0)


def add_real_camera_noise(img_lin, read_sigma=0.005, shot_coeff=0.002):
    """
    在线性空间模拟真实相机噪声: 读出噪声 + 光子噪声。
    img_lin: float32, [0,1], 近似线性空间
    read_sigma: 读出噪声标准差
    shot_coeff: 光子噪声系数，越大噪声越重
    """
    img_lin = np.clip(img_lin, 0.0, 1.0)
    # shot noise: std ~ sqrt(I * shot_coeff)
    shot_std = np.sqrt(img_lin * shot_coeff).astype(np.float32)

    noise_shot = np.random.normal(0.0, 1.0, img_lin.shape).astype(np.float32) * shot_std
    noise_read = np.random.normal(0.0, read_sigma, img_lin.shape).astype(np.float32)

    noisy = img_lin + noise_shot + noise_read
    return np.clip(noisy, 0.0, 1.0)


def add_resize_artifact(img_bgr, scale=0.6):
    """
    缩小再放大，模拟低质量缩放带来的模糊/锯齿。
    img_bgr: uint8 BGR
    scale: 0~1, 缩小尺度
    """
    h, w = img_bgr.shape[:2]
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    small = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    back = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return back


def add_chroma_subsampling_and_jpeg(img_bgr, jpeg_quality=80):
    """
    模拟 YUV 4:2:0 色度子采样 + JPEG 压缩伪影。
    img_bgr: uint8 BGR
    jpeg_quality: JPEG 质量 (50~100)，越低伪影越重
    """
    # BGR -> YCrCb
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    # 模拟 4:2:0: 色度下采样再上采样
    cr_small = cv2.resize(cr, (cr.shape[1] // 2, cr.shape[0] // 2), interpolation=cv2.INTER_AREA)
    cb_small = cv2.resize(cb, (cb.shape[1] // 2, cb.shape[0] // 2), interpolation=cv2.INTER_AREA)
    cr_up = cv2.resize(cr_small, (cr.shape[1], cr.shape[0]), interpolation=cv2.INTER_LINEAR)
    cb_up = cv2.resize(cb_small, (cb.shape[1], cb.shape[0]), interpolation=cv2.INTER_LINEAR)

    ycrcb_420 = cv2.merge([y, cr_up, cb_up])
    img_420 = cv2.cvtColor(ycrcb_420, cv2.COLOR_YCrCb2BGR)

    # 再做一次 JPEG 压缩/解压
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    ok, enc = cv2.imencode('.jpg', img_420, encode_param)
    if not ok:
        return img_420
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if dec is None:
        return img_420
    return dec


# ------------------- 整体降质 pipeline -------------------

def degrade_sdr_realistic(
    gt_bgr_uint8: np.ndarray,
    tone_gamma: float = 1.1,
    tone_mid: float = 0.5,
    tone_strength: float = -0.2,
    blur_ksize: int = 5,
    blur_mix: float = 0.4,
    sat_scale: float = 0.9,
    noise_read_sigma: float = 0.005,
    noise_shot_coeff: float = 0.02,
    resize_scale: float = 0.6,
    jpeg_quality: int = 80
) -> np.ndarray:
    """
    对 8bit SDR GT 做“更加真实”的降质：
    tone curve -> 局部模糊 -> 降饱和度 -> 相机噪声 -> 缩放伪影 -> 色度子采样 + JPEG

    gt_bgr_uint8: uint8, HWC, BGR, [0,255]
    返回: uint8, HWC, BGR, [0,255] 的 LQ
    """
    # 转 float32 [0,1]
    img = gt_bgr_uint8.astype(np.float32) / 255.0

    # 1) tone curve：让图整体偏灰/偏暗，降低对比度
    img = apply_tone_curve(img, gamma=tone_gamma, mid=tone_mid, strength=tone_strength)

    # 2) 局部对比下降：高斯模糊 + 原图混合
    if blur_ksize > 1 and blur_mix > 0:
        ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1  # 保证奇数
        blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)
        img = (1.0 - blur_mix) * img + blur_mix * blurred
        img = np.clip(img, 0.0, 1.0)

    # 3) 降饱和度（电视色彩偏保守）
    img_8 = (img * 255.0).astype(np.uint8)
    hsv = cv2.cvtColor(img_8, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= sat_scale
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    img_8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 4) 在线性空间加真实相机噪声
    img_lin = np.power(img_8.astype(np.float32) / 255.0, 2.2)  # 近似反 gamma 得线性
    img_lin_noisy = add_real_camera_noise(
        img_lin,
        read_sigma=noise_read_sigma,
        shot_coeff=noise_shot_coeff
    )
    img_noisy = np.power(np.clip(img_lin_noisy, 0.0, 1.0), 1.0 / 2.2)
    img_noisy_8 = (img_noisy * 255.0).astype(np.uint8)

    # 5) 缩放伪影：缩小再放大
    img_resized = add_resize_artifact(img_noisy_8, scale=resize_scale)

    # 6) 色度子采样 + JPEG 压缩伪影
    img_final = add_chroma_subsampling_and_jpeg(img_resized, jpeg_quality=jpeg_quality)

    return img_final


# ------------------- 批处理脚本 -------------------

def batch_make_lq_from_gt(
    gt_dir: str,
    lq_dir: str,
    pattern: str = '*.png'
):
    os.makedirs(lq_dir, exist_ok=True)
    paths = sorted(glob(os.path.join(gt_dir, pattern)))
    print(f'Found {len(paths)} GT images in {gt_dir}')

    for i, p in enumerate(paths):
        name = os.path.basename(p)
        gt = cv2.imread(p, cv2.IMREAD_COLOR)
        if gt is None:
            print('[WARN] read fail:', p)
            continue

        lq = degrade_sdr_realistic(
            gt,
            # ---- 整体/局部对比度：下降明显 ----
            tone_gamma=1.2,       # 降低全局对比度
            tone_mid=0.50,         # 不动中点，避免整体偏得太暗
            tone_strength=-0.2,   # 对比压得比较狠（-0.4 ~ -0.5）

            blur_ksize=7,          # 模糊核大一点
            blur_mix=0.5,          # 细节明显变软（0.55~0.7 之间）

            # ---- 亮度 / 饱和度：只轻微下降 ----
            sat_scale=0.92,        # 只降一点颜色（0.9~0.98 都行）

            # ---- 噪声：几乎不加 ----
            noise_read_sigma=0.0001,    # 非常小
            noise_shot_coeff=0.0005,    # 非常小，只保留一点点“活感”

            # ---- 缩放伪影：几乎不加 ----
            resize_scale=1.0,         # 接近 1，几乎看不出缩放痕迹

            # ---- JPEG 伪影：基本关闭 ----
            jpeg_quality=100            # 非常高质量，几乎无伪影
        )

        out_path = os.path.join(lq_dir, name)
        cv2.imwrite(out_path, lq)

        if (i + 1) % 50 == 0 or i == len(paths) - 1:
            print(f'[{i + 1}/{len(paths)}] saved {out_path}')


if __name__ == '__main__':
    # 原始“好看 SDR”所在目录
    gt_dir = r'/mnt/data/datasets/HDR/train_set/train_sdr_gt'
    # 生成的 LQ 存放目录
    lq_dir = r'/mnt/data/datasets/HDR/train_set/train_sdr_lq_light1226'

    batch_make_lq_from_gt(gt_dir, lq_dir)