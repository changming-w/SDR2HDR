import os
import cv2
import numpy as np

# 1. 先检查所有LR图是否能正常读取（找到真实图片名）
VAL_LR_DIR = "/mnt/ext_ssd/wangchangming/AISR/data/val/lq"
VAL_HR_DIR = "/mnt/ext_ssd/wangchangming/AISR/data/val/gt"
scale = 2

# 打印所有能正常读取的LR图，选一个来测试
valid_lr_names = []
for lr_name in os.listdir(VAL_LR_DIR):
    lr_path = os.path.join(VAL_LR_DIR, lr_name)
    lr_img = cv2.imread(lr_path)
    if lr_img is not None:
        valid_lr_names.append(lr_name)
        print(f"✅ 可读取的LR图：{lr_name}")
    else:
        print(f"❌ 无法读取：{lr_name}（可能损坏或格式错误）")

# 2. 选第一个有效图片做对齐测试（不用手动改，自动获取）
if valid_lr_names:
    lr_name = valid_lr_names[0]
    lr_path = os.path.join(VAL_LR_DIR, lr_name)
    hr_path = os.path.join(VAL_HR_DIR, lr_name)  # 假设HR和LR文件名相同

    # 读取图片
    lr = cv2.imread(lr_path)
    hr = cv2.imread(hr_path)

    if hr is None:
        print(f"❌ 对应HR图{lr_name}不存在或无法读取")
    else:
        # 检查尺寸是否匹配（LR×2 == HR）
        h_lr, w_lr = lr.shape[:2]
        h_hr, w_hr = hr.shape[:2]
        print(f"\nLR图尺寸：{h_lr}x{w_lr}，HR图尺寸：{h_hr}x{w_hr}")
        print(f"LR×2是否等于HR：{(h_lr*2 == h_hr) and (w_lr*2 == w_hr)}")
        print(f"LR尺寸是否是4的倍数：{(h_lr%4 == 0) and (w_lr%4 == 0)}")

        # 对齐验证（中心区域对比）
        cx, cy = w_lr//2, h_lr//2
        lr_center = lr[cy-5:cy+5, cx-5:cx+5]
        hr_center = hr[2*(cy-5):2*(cy+5), 2*(cx-5):2*(cx+5)]
        hr_center_down = cv2.resize(hr_center, (10,10), interpolation=cv2.INTER_LINEAR)
        diff = np.mean(np.abs(lr_center - hr_center_down))
        print(f"中心区域差异（越小越对齐）：{diff:.2f}")
else:
    print("❌ 没有找到可正常读取的LR图，请检查目录或图片文件")