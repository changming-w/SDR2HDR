#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 sdrframes_jianying_daili_hevc_60fps 目录中所有图片合成一个视频。
每张图重复 30 帧 → 每张展示 5 秒 @ 6fps，CRF=10。

用法
----
python scripts/make_sdr_videos_6fps.py
"""

import os
import sys
import glob
import subprocess
import tempfile

# ── 路径配置 ──────────────────────────────────────────────────────────────────
INPUT_DIR = '/mnt/data/wcm/data/hdr/train/light_scene/sdrframes_jianying_daili_hevc_60fps'
OUTPUT_DIR = '/mnt/data/wcm/data/hdr/train/light_scene/sdr_videos_6fps_crf10'

# ── 视频参数 ──────────────────────────────────────────────────────────────────
FPS = 6
REPEAT = 30        # 每张图重复帧数
CRF = 10
PRESET = 'slow'
MAX_IMAGES = 400   # 只处理前 N 张图


def run_cmd(cmd, hint='Command failed'):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f'{hint}\nCMD: {" ".join(cmd)}\n{(r.stderr or "")[-4000:]}')
    return r


def main():
    print('=' * 68)
    print('SDR 视频生成脚本 — 全部图片合成一个视频')
    print('=' * 68)

    # 1. 收集图片
    all_imgs = sorted(glob.glob(os.path.join(INPUT_DIR, '*.png')))
    if not all_imgs:
        all_imgs = sorted(glob.glob(os.path.join(INPUT_DIR, '*.jpg')))
    if not all_imgs:
        raise RuntimeError(f'未找到图片: {INPUT_DIR}')

    total = len(all_imgs)
    total = min(total, MAX_IMAGES)
    all_imgs = all_imgs[:total]
    total_frames = total * REPEAT
    total_duration = total_frames / FPS
    print(f'\n[INFO] 输入目录: {INPUT_DIR}')
    print(f'[INFO] 共 {total} 张图片')
    print(f'[INFO] 每张图重复 {REPEAT} 帧 → {REPEAT/FPS:.0f} 秒')
    print(f'[INFO] 总帧数: {total_frames}, 总时长: {total_duration:.0f} 秒 ({total_duration/60:.1f} 分钟)')
    print(f'[INFO] 编码参数: fps={FPS}, CRF={CRF}, preset={PRESET}')

    # 2. 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, 'all_frames_6fps_crf10.mp4')
    print(f'[INFO] 输出文件: {output_path}')

    # 3. 生成 concat 列表文件（每张图重复 REPEAT 次）
    concat_list = os.path.join(OUTPUT_DIR, '_concat_list.txt')
    with open(concat_list, 'w') as f:
        for img_path in all_imgs:
            # ffmpeg concat demuxer 需要 file 行 + duration 行
            f.write(f"file '{img_path}'\n")
            f.write(f'duration {REPEAT / FPS}\n')
        # 最后一张需要再写一次，否则 concat demuxer 会截掉最后一张
        f.write(f"file '{all_imgs[-1]}'\n")

    print(f'[INFO] concat 列表: {concat_list}')

    # 4. 编码
    cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', concat_list,
        '-an', '-sn', '-dn',
        # ── 编码 ──
        '-c:v', 'libx265',
        '-preset', PRESET,
        '-crf', str(CRF),
        '-pix_fmt', 'yuv420p',
        # ── 输出帧率 ──
        '-r', str(FPS),
        # ── 输出色彩标签 (BT.709) ──
        '-color_primaries', 'bt709',
        '-color_trc', 'bt709',
        '-colorspace', 'bt709',
        '-color_range', 'tv',
        output_path,
    ]

    print(f'\n[INFO] 开始编码...')
    run_cmd(cmd, '编码失败')

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f'\n[OK] 输出文件: {output_path}')
    print(f'[OK] 文件大小: {size_mb:.1f} MB')

    # 清理临时列表
    # os.remove(concat_list)

    print('\n' + '=' * 68)
    print(f'[DONE] 完成！')
    print(f'       输出: {output_path}')
    print(f'       大小: {size_mb:.1f} MB')
    print('=' * 68)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n[WARN] 用户中断')
        sys.exit(130)
    except Exception as e:
        print(f'\n[ERROR] {e}', file=sys.stderr)
        sys.exit(1)
