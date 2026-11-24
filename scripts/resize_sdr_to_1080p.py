#!/usr/bin/env python3
# 将 sdr 目录所有图片 resize 到 1080P (1920x1080) 并保存

import cv2
from pathlib import Path

src_dir = Path('/mnt/data/wcm/20260711_color_enhance_plants2/sdr')
out_dir = Path('/mnt/data/wcm/20260711_color_enhance_plants2/sdr_1080p')
out_dir.mkdir(parents=True, exist_ok=True)

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff'}
TARGET_W, TARGET_H = 1920, 1080

count = 0
for img_path in sorted(src_dir.iterdir()):
    if img_path.suffix.lower() not in EXTENSIONS:
        continue

    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"跳过：{img_path.name}")
        continue

    resized = cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LANCZOS4)
    out_path = out_dir / img_path.name
    cv2.imwrite(str(out_path), resized)
    count += 1

print(f"完成，共处理 {count} 张图片")
print(f"输出目录：{out_dir}")
