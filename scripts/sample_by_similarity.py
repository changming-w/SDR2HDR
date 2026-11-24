#!/usr/bin/env python3
"""基于相似度的抽帧：把两个对齐帧目录共同缩减到目标帧数（默认 ~500）。

逻辑：
  - 以 dirA（SDR，8-bit）为基准计算帧间相似度：下采样成 480x270 灰度，
    平均绝对差（MAD，0~255）；
  - 贪心保留：第 1 帧必留；之后每帧与「上一个保留帧」比较，
    差异 < 阈值则丢弃（连续相似段只留代表帧）；
  - 二分搜索阈值，使保留帧数尽量接近目标值；
  - 同一套保留/删除方案同步应用到 dirA 和 dirB，删后重编号，
    保证两边仍然逐帧对齐。

用法：
  python sample_by_similarity.py dirA dirB [--target 500] [--apply]
"""

import argparse
import os
import shutil

import cv2
import numpy as np

DS_W, DS_H = 480, 270


def list_png(d: str):
    return sorted(f for f in os.listdir(d) if f.lower().endswith('.png'))


def load_gray_thumbs(d: str, files):
    """全部帧读成下采样灰度小图，便于快速反复比较。"""
    thumbs = []
    for i, name in enumerate(files, 1):
        img = cv2.imread(os.path.join(d, name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f'读取失败: {name}')
        thumbs.append(cv2.resize(img, (DS_W, DS_H),
                                 interpolation=cv2.INTER_AREA))
        if i % 300 == 0:
            print(f'  已读取 {i}/{len(files)}')
    return [t.astype(np.int16) for t in thumbs]


def count_keep(thumbs, thr: float) -> int:
    keep = 1
    last = thumbs[0]
    for t in thumbs[1:]:
        if np.abs(t - last).mean() < thr:
            continue
        keep += 1
        last = t
    return keep


def keep_indices(thumbs, thr: float):
    idx = [0]
    last = thumbs[0]
    for i, t in enumerate(thumbs[1:], 1):
        if np.abs(t - last).mean() < thr:
            continue
        idx.append(i)
        last = t
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dirA')
    ap.add_argument('dirB')
    ap.add_argument('--target', type=int, default=500)
    ap.add_argument('--apply', action='store_true',
                    help='真正执行删除+重编号；默认只报告')
    a = ap.parse_args()

    fa, fb = list_png(a.dirA), list_png(a.dirB)
    if fa != fb:
        raise SystemExit('两个目录文件名不一致，无法对齐抽帧')
    n = len(fa)
    print(f'两目录各 {n} 帧，文件名一致。读取 {a.dirA} 计算相似度 …')
    thumbs = load_gray_thumbs(a.dirA, fa)

    # 二分搜索阈值，使保留帧数最接近 target
    lo, hi, best_thr, best_cnt = 0.0, 50.0, None, None
    for _ in range(30):
        mid = (lo + hi) / 2
        c = count_keep(thumbs, mid)
        if best_cnt is None or abs(c - a.target) < abs(best_cnt - a.target):
            best_thr, best_cnt = mid, c
        if c > a.target:
            lo = mid      # 阈值太小，留太多
        else:
            hi = mid
        if abs(c - a.target) <= 2:
            break
    print(f'最佳阈值 MAD<{best_thr:.3f}，预计保留 {best_cnt} 帧（目标 {a.target}）')

    keep = keep_indices(thumbs, best_thr)
    remove = sorted(set(range(n)) - set(keep))
    print(f'实际：保留 {len(keep)}，删除 {len(remove)}')

    if not a.apply:
        print('（仅报告，未删除；加 --apply 执行）')
        return

    remove_names = {fa[i] for i in remove}
    keep_names = [fa[i] for i in keep]
    for d in (a.dirA, a.dirB):
        tmp = d.rstrip('/') + '_sample_tmp'
        os.makedirs(tmp, exist_ok=True)
        for new_idx, name in enumerate(keep_names, 1):
            shutil.move(os.path.join(d, name),
                        os.path.join(tmp, f'{new_idx:06d}.png'))
        for name in remove_names:
            os.remove(os.path.join(d, name))
        for name in os.listdir(tmp):
            shutil.move(os.path.join(tmp, name), os.path.join(d, name))
        os.rmdir(tmp)
        print(f'{d}: 已抽帧并重编号 -> {len(keep_names)} 帧')
    print('完成')


if __name__ == '__main__':
    main()
