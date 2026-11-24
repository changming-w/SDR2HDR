import os
import sys
import cv2
import numpy as np
from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from myscripts.degratation_rgb import degrade_4k_to_low_1080p  # 4K→1080P 退化
from myscripts.degrade_4k_to_low_bitrate_4k import degrade_4k_to_low_bitrate_4k, degrade_4k_to_low_bitrate_4k_nonoise, degrade_4k_to_low_bitrate_4k_light, degrade_for_strong_enhancer  # 4K→4K 低码率退化


def list_hr_images(hr_4k_dir):
    """返回 4K HR 图像路径列表"""
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    hr_paths = [
        os.path.join(hr_4k_dir, f)
        for f in os.listdir(hr_4k_dir)
        if os.path.splitext(f)[-1].lower() in extensions
    ]
    return hr_paths


def batch_generate_4k_to_low_1080p(hr_4k_dir, lr_1080p_dir, hr_ref_dir, scale=2):
    """
    批量生成：4K 原图（标签）→ 极低质量 1080P（含大量细节丢失和块效应）
    """
    os.makedirs(lr_1080p_dir, exist_ok=True)
    os.makedirs(hr_ref_dir, exist_ok=True)

    hr_paths = list_hr_images(hr_4k_dir)

    for hr_path in tqdm(hr_paths, desc="生成极低质量1080P（强块效应+细节丢失）"):
        hr_bgr = cv2.imread(hr_path)
        if hr_bgr is None:
            print(f"警告：无法读取 {hr_path}，跳过")
            continue
        hr_rgb = cv2.cvtColor(hr_bgr, cv2.COLOR_BGR2RGB)

        # 4K→极低质量1080P
        lr_rgb, hr_ref_rgb = degrade_4k_to_low_1080p(hr_rgb, scale=scale)

        # 保存低清图（1080P，模型输入）
        lr_bgr = cv2.cvtColor(lr_rgb, cv2.COLOR_RGB2BGR)
        lr_save_path = os.path.join(lr_1080p_dir, os.path.basename(hr_path))
        cv2.imwrite(lr_save_path, lr_bgr)

        # 保存 4K 标签（原始 4K）
        hr_ref_bgr = cv2.cvtColor(hr_ref_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
        hr_ref_save_path = os.path.join(hr_ref_dir, os.path.basename(hr_path))
        cv2.imwrite(hr_ref_save_path, hr_ref_bgr)


def batch_generate_4k_to_low_bitrate_4k(hr_4k_dir, lr_4k_dir):
    """
    批量生成：4K 原图目录 → 4K 低码率目录
    只保存退化后的 4K 低质量图，GT 直接用 hr_4k_dir 里已有的 4K 图。
    """
    os.makedirs(lr_4k_dir, exist_ok=True)
    hr_paths = list_hr_images(hr_4k_dir)

    for hr_path in tqdm(hr_paths, desc="生成4K低码率退化图（4K→4K）"):
        hr_bgr = cv2.imread(hr_path)
        if hr_bgr is None:
            print(f"警告：无法读取 {hr_path}，跳过")
            continue
        hr_rgb = cv2.cvtColor(hr_bgr, cv2.COLOR_BGR2RGB)

        # 4K→4K 低码率退化（GT 不再单独保存）
        lr_rgb, _ = degrade_for_strong_enhancer(hr_rgb)

        # 保存 4K 低码率图（模型输入）
        lr_bgr = cv2.cvtColor(lr_rgb, cv2.COLOR_RGB2BGR)
        lr_save_path = os.path.join(lr_4k_dir, os.path.basename(hr_path))
        cv2.imwrite(lr_save_path, lr_bgr)


if __name__ == "__main__":
    HR_4K_DIR = "/mnt/data/datasets/HDR/train_set/train_sdr_gt"  # 4K 原图目录

    # 任务1：4K→1080P 极低质量（用于超分 + 去压缩）
    # LR_1080P_DIR = "/mnt/ext_ssd/wangchangming/AISR/data/train_lr_extreme_low"
    # HR_REF_4K_FOR_1080P = "/mnt/ext_ssd/wangchangming/AISR/data/train_hr_original"
    # SCALE = 2

    # 任务2：4K→4K 低码率（用于 4K 增强 / 去压缩）
    LR_4K_LOW_BITRATE_DIR = "/mnt/data/datasets/AISR/train_set/train_sdr_lq11"

    # 如果你想 1080P 和 4K 共享同一份 4K GT 标签目录，也可以把
    # HR_REF_4K_FOR_1080P 和 HR_REF_4K_FOR_4K 设成同一个路径

    # 生成 4K→1080P 极低质量
    # batch_generate_4k_to_low_1080p(HR_4K_DIR, LR_1080P_DIR, HR_REF_4K_FOR_1080P, SCALE)
    # print(f"生成完成：极低质量1080P：{LR_1080P_DIR}，4K标签：{HR_REF_4K_FOR_1080P}")

    # 生成 4K→4K 低码率
    batch_generate_4k_to_low_bitrate_4k(HR_4K_DIR, LR_4K_LOW_BITRATE_DIR)
    print(f"生成完成：4K低码率：{LR_4K_LOW_BITRATE_DIR}，4K标签（已存在）：{HR_4K_DIR}")