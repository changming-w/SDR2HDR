#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将目录内的静态图片按顺序合成 30 Hz 视频，每张图片停留 5 秒。

默认处理 数据采集1114/dark_hdr（内容已是 SDR），使用 HEVC CRF 10（高画质）。

运行示例::

    python /mnt/data/wcm/HDRNet/scripts/make_sdr_videos_02fps.py
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path


INPUT_DIR = Path('/mnt/data/wcm/data/hdr/train/数据采集1114/dark_hdr')
OUTPUT_DIR = Path('/mnt/data/wcm/data/hdr/train/数据采集1114/dark_hdr_video_30fps_5s_crf10')
FPS = 30
HOLD_SEC = 5
CRF = 10
PRESET = 'medium'
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def run_cmd(cmd, hint='Command failed'):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f'{hint}\nCMD: {" ".join(cmd)}\n{(r.stderr or "")[-4000:]}')
    return r


def _concat_quote(path):
    """Quote a path for the ffmpeg concat demuxer."""
    value = str(path.resolve()).replace('\\', '\\\\').replace("'", "'\\''")
    return f"'{value}'"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=INPUT_DIR,
                        help=f'图片目录（默认：{INPUT_DIR}）')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR,
                        help=f'视频输出目录（默认：{OUTPUT_DIR}）')
    args = parser.parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    print('=' * 68)
    print('静态图片视频生成 — 30 Hz, 每张图片 5 秒, HEVC CRF 10')
    print('=' * 68)

    if not input_dir.is_dir():
        raise RuntimeError(f'输入目录不存在: {input_dir}')

    all_imgs = sorted(
        (p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
        key=lambda p: p.name,
    )
    if not all_imgs:
        raise RuntimeError(f'未找到图片: {input_dir}')

    total = len(all_imgs)
    total_frames = total * FPS * HOLD_SEC
    total_duration = total * HOLD_SEC
    print(f'\n[INFO] 输入目录: {input_dir}')
    print(f'[INFO] 共 {total} 张图片')
    print(f'[INFO] 每张图展示 {HOLD_SEC} 秒, 输出帧率={FPS} Hz')
    print(f'[INFO] 预计总帧数: {total_frames}, 总时长: {total_duration} 秒 ({total_duration/60:.1f} 分钟)')
    print(f'[INFO] 编码参数: fps={FPS}, CRF={CRF}, preset={PRESET}')

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'dark_hdr_sdr_30hz_5s_crf10.mp4'
    print(f'[INFO] 输出文件: {output_path}')

    # concat demuxer uses per-file durations; duplicate the final file to
    # ensure the final still is held for its full duration.
    concat_list = output_dir / '_concat_list.txt'
    with concat_list.open('w', encoding='utf-8', newline='\n') as f:
        f.write('ffconcat version 1.0\n')
        for img_path in all_imgs:
            f.write(f'file {_concat_quote(img_path)}\n')
            f.write(f'duration {HOLD_SEC}\n')
        f.write(f'file {_concat_quote(all_imgs[-1])}\n')

    print(f'[INFO] concat 列表: {concat_list}')

    cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', str(concat_list),
        '-an', '-sn', '-dn',
        '-c:v', 'libx265',
        '-preset', PRESET,
        '-crf', str(CRF),
        '-pix_fmt', 'yuv420p',
        '-r', str(FPS),
        '-fps_mode', 'cfr',
        '-color_primaries', 'bt709',
        '-color_trc', 'bt709',
        '-colorspace', 'bt709',
        '-color_range', 'tv',
        str(output_path),
    ]

    print(f'\n[INFO] 开始编码...')
    run_cmd(cmd, '编码失败')

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f'\n[OK] 输出文件: {output_path}')
    print(f'[OK] 文件大小: {size_mb:.1f} MB')

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
