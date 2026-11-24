"""批量生成多分辨率副本：指定文件 -> 540p/720p/1080p 各 3 份，hdr/sdr 同步对齐。

用法:
    python3 multi_res_copies.py pair_000008 pair_000056
    python3 multi_res_copies.py pair_000008.png   # 带不带扩展名均可
已在 hdr_1080p / sdr_1080p 处理过的文件会自动跳过（检测到 copy1 已存在）。
"""
import os
import sys

import cv2

LIGHT = '/mnt/data/wcm/data/hdr/train/数据采集light'
DIRS = ['hdr_1080p', 'sdr_1080p']

# 三种目标分辨率 (w, h)，后缀与分辨率对应
SIZES = {
    '_540p': (960, 540),
    '_720p': (1280, 720),
    '_1080p': (1920, 1080),
}
N_COPIES = 3


def process_one(stem: str) -> None:
    ext = '.png'
    for d in DIRS:
        src = os.path.join(LIGHT, d, stem + ext)
        img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
        assert img is not None, f'read failed: {src}'
        for suffix, (w, h) in SIZES.items():
            first = os.path.join(LIGHT, d, f'{stem}{suffix}_copy1{ext}')
            if os.path.exists(first):
                print(f'  [skip] {d}/{stem}{suffix} 已存在')
                continue
            resized = img.copy() if (w, h) == (img.shape[1], img.shape[0]) else \
                cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            for i in range(1, N_COPIES + 1):
                assert cv2.imwrite(os.path.join(LIGHT, d, f'{stem}{suffix}_copy{i}{ext}'), resized)
        print(f'  {d}: done')


def main():
    stems = [os.path.splitext(a)[0] for a in sys.argv[1:]]
    if not stems:
        print(__doc__)
        sys.exit(1)
    for stem in stems:
        print(f'== {stem}')
        process_one(stem)


if __name__ == '__main__':
    main()
