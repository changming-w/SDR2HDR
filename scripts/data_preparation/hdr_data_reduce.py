import os
from glob import glob
import cv2
import numpy as np


def add_noise(img, sigma=3, poisson_scale=0.0):
    """
    在 8bit BGR 上加噪声。
    sigma: 高斯噪声标准差 (0~10) ，越大噪越明显
    poisson_scale: 泊松噪声权重 (0~1)，一般 0~0.3
    """
    img = img.astype(np.float32)

    # 高斯噪声
    if sigma > 0:
        gauss = np.random.normal(0, sigma, img.shape).astype(np.float32)
        img = img + gauss

    # 泊松噪声（基于像素值模拟 photon 噪声）
    if poisson_scale > 0:
        # 归一到 [0,1] 再采样
        img_norm = np.clip(img / 255.0, 0.0, 1.0)
        # scale 决定噪声强度
        noisy = np.random.poisson(img_norm * 255.0) / 255.0
        img = (1 - poisson_scale) * img_norm + poisson_scale * noisy
        img = img * 255.0

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def add_jpeg_artifacts(img, quality=70):
    """
    通过 JPEG 编码/解码一次增加轻微块效应。
    quality: 50~95, 越低伪影越重。
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    # 编码到内存
    ok, enc = cv2.imencode('.jpg', img, encode_param)
    if not ok:
        return img
    # 解码回来
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if dec is None:
        return img
    return dec


def degrade_sdr_for_contrast_task(
    gt_bgr: np.ndarray,
    contrast: float = 0.7,
    brightness_shift: float = -0.03,
    blur_radius: int = 3,
    detail_mix: float = 0.5,
    sat_scale: float = 0.85,
    min_luma: float = 0.05,
    max_luma: float = 0.95,
    gauss_sigma: float = 3.0,
    poisson_scale: float = 0.0,
    jpeg_quality: int = 80
) -> np.ndarray:
    """
    在原来“对比度/局部对比/饱和度下降”的基础上，加入噪声和压缩伪影。
    gt_bgr: uint8, HWC, BGR, [0,255]
    """

    # ---- 1) 对比度 & 亮度 & 亮度范围限制 ----
    img = gt_bgr.astype(np.float32) / 255.0
    img = (img - 0.5) * contrast + 0.5 + brightness_shift

    img = np.clip(img, 0.0, 1.0)

    # 亮度通道压缩在 [min_luma, max_luma]
    luma = 0.114 * img[..., 0] + 0.587 * img[..., 1] + 0.299 * img[..., 2]
    scale = (max_luma - min_luma)
    luma_adj = luma * scale + min_luma
    eps = 1e-6
    ratio = luma_adj / (luma + eps)
    img *= ratio[..., None]
    img = np.clip(img, 0.0, 1.0)

    # ---- 2) 降局部对比：用模糊图混合 ----
    if blur_radius > 0 and detail_mix > 0:
        ksize = blur_radius * 2 + 1
        blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)
        img = (1 - detail_mix) * img + detail_mix * blurred
        img = np.clip(img, 0.0, 1.0)

    # ---- 3) 降饱和度 ----
    img_8 = (img * 255.0).astype(np.uint8)
    hsv = cv2.cvtColor(img_8, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= sat_scale
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    img_8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # ---- 4) 加噪声 ----
    img_8 = add_noise(img_8, sigma=gauss_sigma, poisson_scale=poisson_scale)

    # ---- 5) 加 JPEG 伪影 ----
    if jpeg_quality < 100:
        img_8 = add_jpeg_artifacts(img_8, quality=jpeg_quality)

    return img_8


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

        # 一套比较温和的降质配置
        lq = degrade_sdr_for_contrast_task(
            gt,
            contrast=0.7,          # 全局对比度降低
            brightness_shift=-0.03,# 稍微变暗
            blur_radius=3,
            detail_mix=0.4,        # 局部细节软一点
            sat_scale=0.9,         # 饱和度略降
            min_luma=0.06,
            max_luma=0.95,
            gauss_sigma=3.0,       # 高斯噪声强度（0~5）
            poisson_scale=0.1,     # 一点点泊松噪声，可设 0~0.3
            jpeg_quality=80        # 60~90: 值越低伪影越重，100=无伪影
        )

        out_path = os.path.join(lq_dir, name)
        cv2.imwrite(out_path, lq)

        if (i + 1) % 50 == 0 or i == len(paths) - 1:
            print(f'[{i + 1}/{len(paths)}] saved {out_path}')


if __name__ == '__main__':
    gt_dir = r'/mnt/data/datasets/HDR/train_set/train_sdr_gt'
    lq_dir = r'/mnt/data/datasets/HDR/train_set/train_sdr_lq'
    batch_make_lq_from_gt(gt_dir, lq_dir)