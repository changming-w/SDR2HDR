import os
import sys
from multiprocessing import Pool

import cv2

# 4K (3840x2160) -> 1080p (1920x1080)，精确 2 倍下采样
ROOT = "/mnt/data/wcm/data/hdr/train/数据采集dark_ld开"
PAIRS = [("hdr", "hdr_1080p"), ("sdr", "sdr_1080p")]
TARGET_SIZE = (1920, 1080)  # (w, h)


def process(args):
    src, dst, filename = args
    src_path = os.path.join(ROOT, src, filename)
    dst_path = os.path.join(ROOT, dst, filename)
    if os.path.exists(dst_path):
        return filename, True, "skip"
    img = cv2.imread(src_path, cv2.IMREAD_COLOR)
    if img is None:
        return filename, False, "read failed"
    # INTER_AREA 是下采样首选插值（抗混叠效果最好）
    img_lr = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    ok = cv2.imwrite(dst_path, img_lr)
    return filename, bool(ok), "ok" if ok else "write failed"


def main():
    jobs = []
    for src, dst in PAIRS:
        os.makedirs(os.path.join(ROOT, dst), exist_ok=True)
        for f in sorted(os.listdir(os.path.join(ROOT, src))):
            if f.lower().endswith(".png"):
                jobs.append((src, dst, f))

    print(f"total jobs: {len(jobs)}")
    n_fail = 0
    with Pool(16) as pool:
        for i, (fname, ok, msg) in enumerate(pool.imap_unordered(process, jobs)):
            if not ok:
                n_fail += 1
                print(f"FAIL: {fname} ({msg})")
            if (i + 1) % 200 == 0:
                print(f"progress: {i + 1}/{len(jobs)}")
    print(f"done, failed: {n_fail}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
