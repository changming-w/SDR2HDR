#!/usr/bin/env python3
"""对两个帧目录做联合去重（帧对齐保持）。

规则：
  - 逐帧比较：第 i 帧与「前一个保留帧」比较（不是仅与 i-1 比，
    处理连续多帧重复的情况）；
  - 只有当该帧在 **两个目录中都** 与前保留帧字节级相同（MD5）时，
    才判定为重复帧并同时从两边删除，保证去重后两边仍然逐帧对齐；
  - 删除后两边统一按剩余顺序重新编号 000001.png 起。

用法：
  python dedup_frames.py dirA dirB [--apply]     # 不带 --apply 只报告
"""

import argparse
import hashlib
import os
import shutil


def list_png(d: str):
    return sorted(f for f in os.listdir(d) if f.lower().endswith('.png'))


def md5_of(path: str) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dirA')
    ap.add_argument('dirB')
    ap.add_argument('--apply', action='store_true',
                    help='真正执行删除+重编号；默认只报告')
    a = ap.parse_args()

    fa, fb = list_png(a.dirA), list_png(a.dirB)
    if fa != fb:
        raise SystemExit('两个目录文件名不一致，无法对齐去重')
    n = len(fa)
    print(f'两目录各 {n} 帧，文件名一致，开始计算 MD5 …')

    # 逐帧比较：与「上一个保留帧」比，连续重复也能全部检出
    keep, dup = [], []
    last_ha = last_hb = None
    for i, name in enumerate(fa, 1):
        ha = md5_of(os.path.join(a.dirA, name))
        hb = md5_of(os.path.join(a.dirB, name))
        if last_ha is not None and ha == last_ha and hb == last_hb:
            dup.append(name)
        else:
            keep.append(name)
            last_ha, last_hb = ha, hb
        if i % 200 == 0:
            print(f'  进度 {i}/{n}  已检出重复 {len(dup)}')

    print(f'完成：保留 {len(keep)} 帧，重复 {len(dup)} 帧')
    if dup:
        print('重复帧示例（前 10）:', ', '.join(dup[:10]))

    if not a.apply:
        print('（仅报告，未删除；加 --apply 执行）')
        return
    if not dup:
        print('无重复，无需处理')
        return

    # 删除重复帧，剩余帧移到暂存目录后重新编号，避免同名覆盖冲突
    for d, files in ((a.dirA, fa), (a.dirB, fb)):
        tmp = d.rstrip('/') + '_dedup_tmp'
        os.makedirs(tmp, exist_ok=True)
        for new_idx, name in enumerate(keep, 1):
            shutil.move(os.path.join(d, name),
                        os.path.join(tmp, f'{new_idx:06d}.png'))
        for name in dup:
            os.remove(os.path.join(d, name))
        # 把暂存内容移回原目录（此时原目录只剩重复帧已删空）
        for name in os.listdir(tmp):
            shutil.move(os.path.join(tmp, name), os.path.join(d, name))
        os.rmdir(tmp)
        print(f'{d}: 已去重并重编号 -> {len(keep)} 帧')
    print('完成')


if __name__ == '__main__':
    main()
