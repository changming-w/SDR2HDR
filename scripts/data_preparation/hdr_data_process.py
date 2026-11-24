import os
from glob import glob

import cv2
import numpy as np


def hdr_to_ldr_drago(hdr16: np.ndarray,
                     gamma: float = 1.0,
                     saturation: float = 1.0,
                     bias: float = 0.8) -> np.ndarray:
    """16bit HDR(uint16) -> 0~1 LDR(float32)，使用 Drago tone-mapping。"""
    if hdr16 is None:
        raise ValueError("hdr16 is None")

    # 灰度图转三通道
    if hdr16.ndim == 2:
        hdr16 = np.stack([hdr16, hdr16, hdr16], axis=-1)

    if hdr16.dtype != np.uint16:
        raise ValueError(f"hdr16 dtype is {hdr16.dtype}, not uint16")

    # 归一化到 [0,1]
    hdr = hdr16.astype(np.float32) / 65535.0
    max_val = hdr.max()

    # 这里不再跳过全黑，只在上层判断 max_val==0 时不调用这个函数
    if max_val <= 0:
        raise ValueError("max_val <= 0 (all black image)")

    hdr = np.clip(hdr, 0.0, 1.0)

    tonemap = cv2.createTonemapDrago(
        gamma=gamma,
        saturation=saturation,
        bias=bias
    )
    ldr = tonemap.process(hdr)
    ldr = np.clip(ldr, 0.0, 1.0)
    return ldr


def local_contrast_boost(ldr: np.ndarray,
                         amount: float = 0.3,
                         radius: int = 3) -> np.ndarray:
    """在 0~1 LDR 图像上做局部对比增强 (unsharp mask)。"""
    ldr = np.clip(ldr, 0.0, 1.0)
    ksize = radius * 2 + 1
    blurred = cv2.GaussianBlur(ldr, (ksize, ksize), 0)
    detail = ldr - blurred
    enhanced = ldr + amount * detail
    enhanced = np.clip(enhanced, 0.0, 1.0)
    return enhanced


def hdr_to_ldr_drago_with_boost(hdr16: np.ndarray,
                                gamma: float = 1.0,
                                saturation: float = 1.0,
                                bias: float = 0.8,
                                boost_amount: float = 0.3,
                                boost_radius: int = 3) -> np.ndarray:
    """Drago tone-mapping + 局部对比增强，输出 8bit SDR。"""
    ldr = hdr_to_ldr_drago(
        hdr16,
        gamma=gamma,
        saturation=saturation,
        bias=bias
    )
    ldr = local_contrast_boost(
        ldr,
        amount=boost_amount,
        radius=boost_radius
    )
    ldr8 = (ldr * 255.0 + 0.5).astype(np.uint8)
    return ldr8


def make_black_8bit_like(hdr16: np.ndarray, value: int = 0) -> np.ndarray:
    """
    对全黑 HDR 图，生成一张同尺寸的 8bit 图。
    默认全黑，也可以把 value 改成 16/32 之类的暗灰。
    """
    if hdr16.ndim == 2:
        h, w = hdr16.shape
        c = 1
    else:
        h, w, c = hdr16.shape
    if c == 1:
        out = np.full((h, w), value, dtype=np.uint8)
    else:
        out = np.full((h, w, c), value, dtype=np.uint8)
    return out


def batch_convert_hdr_to_sdr_drago_boost(
    hdr_dir: str,
    out_dir: str,
    gamma: float = 1.0,
    saturation: float = 1.0,
    bias: float = 0.8,
    boost_amount: float = 0.3,
    boost_radius: int = 3,
    pattern: str = '*.png'
):
    os.makedirs(out_dir, exist_ok=True)

    hdr_paths = sorted(glob(os.path.join(hdr_dir, pattern)))
    print(f'Found {len(hdr_paths)} HDR images in {hdr_dir}')

    black_cnt = 0
    err_cnt = 0

    for i, hdr_path in enumerate(hdr_paths):
        name = os.path.basename(hdr_path)
        out_path = os.path.join(out_dir, name)

        hdr16 = cv2.imread(hdr_path, cv2.IMREAD_UNCHANGED)
        if hdr16 is None:
            print(f'[WARN] Failed to read: {hdr_path}')
            err_cnt += 1
            continue

        if hdr16.dtype != np.uint16:
            print(f'[WARN] Not uint16 image, try force process: {hdr_path}, dtype={hdr16.dtype}')

        # 判断是否全黑
        if hdr16.max() <= 0:
            # 全黑图，直接生成同尺寸全黑 8bit
            black_cnt += 1
            ldr8 = make_black_8bit_like(hdr16, value=0)  # 如果想要暗灰，可改成 value=16/32
        else:
            try:
                ldr8 = hdr_to_ldr_drago_with_boost(
                    hdr16,
                    gamma=gamma,
                    saturation=saturation,
                    bias=bias,
                    boost_amount=boost_amount,
                    boost_radius=boost_radius
                )
            except Exception as e:
                print(f'[ERROR] Failed processing {hdr_path}: {e}')
                err_cnt += 1
                continue

        ok = cv2.imwrite(out_path, ldr8)
        if not ok:
            print(f'[WARN] Failed to save: {out_path}')
            err_cnt += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(hdr_paths):
            print(f'[{i + 1}/{len(hdr_paths)}] Saved: {out_path}')

    print(f'Done. Black images converted: {black_cnt}, errors: {err_cnt}.')


if __name__ == '__main__':
    hdr_dir = r'/mnt/data/datasets/HDR/train_set/train_hdr'
    out_dir = r'/mnt/data/datasets/HDR/train_set/train_hdr2sdr_drago_boost'

    batch_convert_hdr_to_sdr_drago_boost(
        hdr_dir=hdr_dir,
        out_dir=out_dir,
        gamma=1.0,
        saturation=1.0,
        bias=0.85,
        boost_amount=0.3,
        boost_radius=3,
        pattern='*.png'
    )