"""sony_light 4K -> 1080p 下采样：hdr (10bit<<6 16bit) 与 sdr (8bit) 同步处理。

输出: /mnt/data/wcm/data/hdr/train/sony_light/hdr_1080p 与 sdr_1080p
- GT 下采样后直接保存插值结果，不做重量化；
  归一化由 HDRPairedMultiDataset10bitGT 用 /65472 (满量程 1023*64) 完成，无中间量化误差
- SDR 8bit 正常下采样
- INTER_AREA 抗混叠，16 进程并行，幂等可重跑（已存在则跳过）

增量新数据直接重跑本脚本即可。
"""
import os
from multiprocessing import Pool

import cv2

ROOT = '/mnt/data/wcm/data/hdr/train/sony_light'
PAIRS = [('hdr', 'hdr_1080p'), ('sdr', 'sdr_1080p')]
TARGET_SIZE = (1920, 1080)  # (w, h)


def process(args):
    src, dst, filename = args
    src_path = os.path.join(ROOT, src, filename)
    dst_path = os.path.join(ROOT, dst, filename)
    if os.path.exists(dst_path):
        return filename, True, 'skip'
    img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return filename, False, 'read failed'
    img_lr = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    ok = cv2.imwrite(dst_path, img_lr)
    return filename, bool(ok), 'ok' if ok else 'write failed'


def main():
    jobs = []
    for src, dst in PAIRS:
        os.makedirs(os.path.join(ROOT, dst), exist_ok=True)
        for f in sorted(os.listdir(os.path.join(ROOT, src))):
            if f.lower().endswith('.png'):
                jobs.append((src, dst, f))

    print(f'total jobs: {len(jobs)}')
    n_fail = 0
    with Pool(16) as pool:
        for i, (fname, ok, msg) in enumerate(pool.imap_unordered(process, jobs)):
            if not ok:
                n_fail += 1
                print(f'FAIL: {fname} ({msg})')
            if (i + 1) % 200 == 0:
                print(f'progress: {i + 1}/{len(jobs)}')
    print(f'done, failed: {n_fail}')


if __name__ == '__main__':
    main()
