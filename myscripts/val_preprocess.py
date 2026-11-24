import os
import cv2

# 配置路径（请再次确认正确）
VAL_LR_DIR = "/mnt/ext_ssd/wangchangming/AISR/data/val/lq"
VAL_HR_DIR = "/mnt/ext_ssd/wangchangming/AISR/data/val/gt"
scale = 2

problem_files = []
for hr_name in os.listdir(VAL_HR_DIR):
    # 读取HR图并获取尺寸
    hr_path = os.path.join(VAL_HR_DIR, hr_name)
    hr_img = cv2.imread(hr_path)
    if hr_img is None:
        continue
    h_hr, w_hr = hr_img.shape[:2]

    # 对应LR图路径
    lr_path = os.path.join(VAL_LR_DIR, hr_name)
    if not os.path.exists(lr_path):
        continue
    lr_img = cv2.imread(lr_path)
    if lr_img is None:
        continue
    h_lr, w_lr = lr_img.shape[:2]

    # 检查超分输出尺寸是否匹配HR图
    if h_lr * scale != h_hr or w_lr * scale != w_hr:
        problem_files.append(hr_name)
        # 直接删除LR和HR图
        os.remove(lr_path)
        os.remove(hr_path)
        print(f"已删除问题图片：{hr_name}")
        print(f"  LR路径：{lr_path}")
        print(f"  HR路径：{hr_path}\n")

if problem_files:
    print(f"✅ 共删除{len(problem_files)}张问题图片")
else:
    print("✅ 未发现尺寸不匹配的图片")