import os
import cv2
import numpy as np
import shutil
from basicsr.data.transforms import augment
from basicsr.utils import scandir
from tqdm import tqdm

# ===================== 配置参数（根据实际情况修改）=====================
RAW_HR_DIR = "/mnt/ext_ssd/wangchangming/AISR/data/train_hr_original"
RAW_LR_DIR = "/mnt/ext_ssd/wangchangming/AISR/data/train_lr_extreme_low"
PROC_HR_DIR = "/mnt/ext_ssd/wangchangming/BasicSR/datasets/SR/train_hr_patches_30w"
PROC_LR_DIR = "/mnt/ext_ssd/wangchangming/BasicSR/datasets/SR/train_lr_patches_30w"

CROP_SIZE = 256
SCALE = 2
NUM_CROPS_PER_IMG = 200  # 4K-8K图建议值
MAX_TOTAL_SAMPLES = 1000000
# ======================================================================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def advanced_crop(hr_img, lr_img, crop_size, scale):
    h_hr, w_hr = hr_img.shape[:2]
    h_lr, w_lr = lr_img.shape[:2]
    lr_crop_size = crop_size // scale

    if (h_hr != h_lr * scale) or (w_hr != w_lr * scale):
        raise ValueError(f"HR与LR尺寸不匹配！HR: {h_hr}x{w_hr}, LR: {h_lr}x{w_lr}, scale: {scale}")

    regions = [
        (0, 0, h_hr//2, w_hr//2),
        (0, w_hr//2, h_hr//2, w_hr),
        (h_hr//2, 0, h_hr, w_hr//2),
        (h_hr//2, w_hr//2, h_hr, w_hr),
        (h_hr//4, w_hr//4, 3*h_hr//4, 3*w_hr//4)
    ]
    region = regions[np.random.randint(0, len(regions))]
    rh_start, rw_start, rh_end, rw_end = region

    top = np.random.randint(rh_start, rh_end - crop_size + 1)
    left = np.random.randint(rw_start, rw_end - crop_size + 1)
    hr_crop = hr_img[top:top + crop_size, left:left + crop_size, :]

    lr_top = top // scale
    lr_left = left // scale
    lr_crop = lr_img[lr_top:lr_top + lr_crop_size, lr_left:lr_left + lr_crop_size, :]

    # if np.random.random() < 0.5:
    #     hr_crop = cv2.flip(hr_crop, 1)
    #     lr_crop = cv2.flip(lr_crop, 1)

    return hr_crop, lr_crop

def main():
    ensure_dir(PROC_HR_DIR)
    ensure_dir(PROC_LR_DIR)
    print(f"预处理后HR图保存至：{PROC_HR_DIR}")
    print(f"预处理后LR图保存至：{PROC_LR_DIR}")

    # 关键修正：将 suffix 列表改为元组（用 () 替代 []）
    hr_paths = sorted(scandir(RAW_HR_DIR, suffix=('png', 'jpg', 'jpeg', 'bmp'), recursive=True))
    lr_paths = sorted(scandir(RAW_LR_DIR, suffix=('png', 'jpg', 'jpeg', 'bmp'), recursive=True))

    if len(hr_paths) != len(lr_paths):
        print(f"[错误] HR图片数量（{len(hr_paths)}）与LR图片数量（{len(lr_paths)}）不匹配！")
        return

    total_crops = 0
    print(f"开始预处理 {len(hr_paths)} 对4K-8K图片（每张图裁剪{NUM_CROPS_PER_IMG}个patch）...")

    for hr_name, lr_name in tqdm(zip(hr_paths, lr_paths), total=len(hr_paths)):
        hr_path = os.path.join(RAW_HR_DIR, hr_name)
        lr_path = os.path.join(RAW_LR_DIR, lr_name)
        hr_img = cv2.imread(hr_path, cv2.IMREAD_COLOR)
        lr_img = cv2.imread(lr_path, cv2.IMREAD_COLOR)

        if hr_img is None or lr_img is None:
            print(f"[警告] 跳过无法读取的图片对：{hr_name} / {lr_name}")
            continue

        for crop_idx in range(NUM_CROPS_PER_IMG):
            if total_crops >= MAX_TOTAL_SAMPLES:
                print(f"\n已达到最大样本量 {MAX_TOTAL_SAMPLES}，停止裁剪")
                print("预处理完成！")
                return

            try:
                hr_crop, lr_crop = advanced_crop(hr_img, lr_img, CROP_SIZE, SCALE)
            except ValueError as e:
                print(f"[警告] 图片对 {hr_name} 裁剪失败：{str(e)}，跳过该patch")
                continue

            base_name = os.path.splitext(hr_name)[0]
            hr_save_path = os.path.join(PROC_HR_DIR, f"{base_name}_crop{crop_idx}.png")
            lr_save_path = os.path.join(PROC_LR_DIR, f"{base_name}_crop{crop_idx}.png")

            cv2.imwrite(hr_save_path, hr_crop)
            cv2.imwrite(lr_save_path, lr_crop)
            total_crops += 1

    print(f"\n预处理完成！共生成 {total_crops} 个训练样本")

if __name__ == "__main__":
    main()