#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 hdrframes 目录中挑选 10 张图片，每张生成 30 帧 5 秒的 HDR10 视频。
============================================================================

- 输入：/mnt/data/wcm/data/hdr/train/light_scene/hdrframes/ 中的 PNG 图片
  （3840x2160, rgb48be, PQ / BT.2020）
- 输出：每个视频 30 帧，5 秒（fps=6），HEVC Main10 / yuv420p10le，
  带完整 HDR10 静态元数据（MasteringDisplay + ContentLightLevel）
- 元数据：/mnt/data/wcm/data/hdr/test/SDR/hdr10_metadata_default.json

用法
----
python scripts/make_hdr10_videos.py
"""

import os
import sys
import json
import glob
import subprocess
import numpy as np


# ── 路径配置 ──────────────────────────────────────────────────────────────────
HDR_FRAMES_DIR = '/mnt/data/wcm/data/hdr/train/light_scene/hdrframes'
METADATA_JSON = '/mnt/data/wcm/data/hdr/test/SDR/hdr10_metadata_default.json'
OUTPUT_DIR = '/mnt/data/wcm/data/hdr/train/light_scene/hdr10_videos'

# ── 视频参数 ──────────────────────────────────────────────────────────────────
NUM_VIDEOS = 10       # 挑选图片数量
NUM_FRAMES = 1        # 每个视频的帧数
DURATION = 5          # 每个视频的时长（秒）
FPS_NUM, FPS_DEN = 1, DURATION  # fps = 1/5 = 0.2
FPS = FPS_NUM / FPS_DEN

# ── 编码参数（参考 infer_sdr2hdr.py） ────────────────────────────────────────
CRF = 12.0
PRESET = 'slow'


def run_cmd(cmd, hint='Command failed'):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f'{hint}\nCMD: {" ".join(cmd)}\n{(r.stderr or "")[-4000:]}')
    return r


def parse_ratio(x):
    if x is None or x == 'N/A':
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if '/' in s:
        a, b = s.split('/', 1)
        try:
            return float(a) / float(b) if float(b) != 0 else 0.0
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def parse_hdr10_metadata(metadata_path):
    """
    从 JSON 中解析 HDR10 元数据，返回 (master_display_str, max_cll, max_fall)。
    逻辑与 infer_sdr2hdr.py 中的 parse_hdr10_metadata 一致。
    """
    with open(metadata_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    to50k = lambda v: int(round(parse_ratio(v) * 50000))
    to10k = lambda v: int(round(parse_ratio(v) * 10000))

    # 优先从 streams 解析
    streams = meta.get('streams') or []
    master = None
    max_cll, max_fall = 1000, 400

    if streams:
        for sd in (streams[0].get('side_data_list') or []):
            tp = sd.get('side_data_type')
            if tp == 'Mastering display metadata':
                master = 'G({},{})B({},{})R({},{})WP({},{})L({},{})'.format(
                    to50k(sd.get('green_x')), to50k(sd.get('green_y')),
                    to50k(sd.get('blue_x')),  to50k(sd.get('blue_y')),
                    to50k(sd.get('red_x')),   to50k(sd.get('red_y')),
                    to50k(sd.get('white_point_x')), to50k(sd.get('white_point_y')),
                    to10k(sd.get('max_luminance')),
                    max(1, to10k(sd.get('min_luminance'))),
                )
            elif tp == 'Content light level metadata':
                max_cll = int(sd.get('max_content') or max_cll)
                max_fall = int(sd.get('max_average') or max_fall)

    # 兜底从 ext_meta_data 解析
    if master is None and 'ext_meta_data' in meta:
        e = meta['ext_meta_data']
        master = 'G({},{})B({},{})R({},{})WP({},{})L({},{})'.format(
            int(round(float(e['green_x']) * 50000)), int(round(float(e['green_y']) * 50000)),
            int(round(float(e['blue_x']) * 50000)),  int(round(float(e['blue_y']) * 50000)),
            int(round(float(e['red_x']) * 50000)),   int(round(float(e['red_y']) * 50000)),
            int(round(float(e['white_point_x']) * 50000)),
            int(round(float(e['white_point_y']) * 50000)),
            int(round(float(e.get('max_luma', 1000)) * 10000)),
            max(1, int(round(float(e.get('min_luma', 0.0001)) * 10000))),
        )
        max_cll = int(e.get('max_cll', e.get('max_luma', max_cll)))
        max_fall = int(e.get('max_fall', max_fall))

    if master is None:
        master = 'G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)'

    return master, max_cll, max_fall


def build_x265_params(master_display, max_cll, max_fall):
    """构造 x265 参数字符串，与 infer_sdr2hdr.py 一致。"""
    parts = [
        'hdr10-opt=1',
        'repeat-headers=1',
        'colorprim=bt2020',
        'transfer=smpte2084',
        'colormatrix=bt2020nc',
        f'master-display={master_display}',
        f'max-cll={max_cll},{max_fall}',
        'range=limited',
    ]
    return ':'.join(parts)


def pick_images(img_dir, n=10):
    """从目录中均匀挑选 n 张图片。"""
    all_imgs = sorted(glob.glob(os.path.join(img_dir, '*.png')))
    if not all_imgs:
        all_imgs = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
    if len(all_imgs) < n:
        raise RuntimeError(f'图片不足：找到 {len(all_imgs)} 张，需要 {n} 张')

    # 均匀间隔采样
    indices = np.linspace(0, len(all_imgs) - 1, n, dtype=int)
    selected = [all_imgs[i] for i in indices]
    return selected


def verify_hdr10(path):
    """验证输出视频是否包含 HDR10 SEI。"""
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-read_intervals', '%+#1', '-show_frames',
           '-show_entries', 'frame=side_data_list', '-of', 'json', path]
    try:
        data = json.loads(run_cmd(cmd, 'ffprobe failed').stdout)
    except Exception:
        return False, False

    types = set()
    for fr in data.get('frames', []) or []:
        for sd in (fr.get('side_data_list') or []):
            t = sd.get('side_data_type')
            if t:
                types.add(t)
    return ('Mastering display metadata' in types,
            'Content light level metadata' in types)


def encode_video(img_path, output_path, master_display, max_cll, max_fall):
    """
    将单张 HDR 图片编码为 30 帧 5 秒的 HDR10 视频。

    输入图片是 rgb48be (16-bit RGB, full range, PQ/BT.2020 编码值)，
    需要转换为 yuv420p10le (BT.2020nc, limited range) 进行 HEVC Main10 编码。
    """
    fps_str = f'{FPS_NUM}/{FPS_DEN}'  # 有理数形式，避免浮点精度问题
    x265_params = build_x265_params(master_display, max_cll, max_fall)

    cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y',
        # ── 输入：循环读入同一张图片 ──
        '-loop', '1',
        '-framerate', fps_str,
        '-i', img_path,
        # ── 限制帧数 ──
        '-frames:v', str(NUM_FRAMES),
        '-an', '-sn', '-dn',
        # ── 色彩转换：RGB48(full) -> YUV420P10(BT.2020nc, limited) ──
        '-vf', (
            'scale=in_range=full:out_range=tv:'
            'out_color_matrix=bt2020:'
            'flags=accurate_rnd+full_chroma_int,'
            'format=yuv420p10le'
        ),
        # ── 编码 ──
        '-c:v', 'libx265',
        '-profile:v', 'main10',
        '-preset', PRESET,
        '-crf', f'{CRF:g}',
        '-x265-params', x265_params,
        # ── 输出色彩标签 ──
        '-color_primaries', 'bt2020',
        '-color_trc', 'smpte2084',
        '-colorspace', 'bt2020nc',
        '-color_range', 'tv',
        '-chroma_sample_location', 'left',
        # ── 输出帧率 ──
        '-r', fps_str,
        output_path,
    ]

    run_cmd(cmd, f'编码失败: {img_path} -> {output_path}')


def main():
    print('=' * 68)
    print('HDR10 视频生成脚本')
    print('=' * 68)

    # 1. 解析 HDR10 元数据
    print(f'\n[INFO] 读取元数据: {METADATA_JSON}')
    master_display, max_cll, max_fall = parse_hdr10_metadata(METADATA_JSON)
    print(f'[INFO] MasteringDisplay = {master_display}')
    print(f'[INFO] MaxCLL = {max_cll}, MaxFALL = {max_fall}')

    # 2. 挑选图片
    print(f'\n[INFO] 从 {HDR_FRAMES_DIR} 中均匀挑选 {NUM_VIDEOS} 张图片')
    selected = pick_images(HDR_FRAMES_DIR, NUM_VIDEOS)
    for i, p in enumerate(selected):
        print(f'  [{i+1:2d}] {os.path.basename(p)}')

    # 3. 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f'\n[INFO] 输出目录: {OUTPUT_DIR}')
    print(f'[INFO] 每个视频: {NUM_FRAMES} 帧, {DURATION} 秒, fps={FPS}')

    # 4. 逐个编码
    for i, img_path in enumerate(selected):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(OUTPUT_DIR, f'{basename}_hdr10.mp4')

        print(f'\n[{i+1}/{NUM_VIDEOS}] 编码: {os.path.basename(img_path)} -> {os.path.basename(output_path)}')

        encode_video(img_path, output_path, master_display, max_cll, max_fall)

        # 验证 HDR10 元数据
        has_md, has_cll = verify_hdr10(output_path)
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f'  [OK] 大小: {size_mb:.1f} MB | '
              f'MasteringDisplay: {"OK" if has_md else "MISSING"} | '
              f'ContentLightLevel: {"OK" if has_cll else "MISSING"}')

        if not (has_md and has_cll):
            print(f'  [WARN] 输出缺少 HDR10 元数据！')

    # 5. 汇总
    print('\n' + '=' * 68)
    print(f'[DONE] 全部完成！共生成 {NUM_VIDEOS} 个 HDR10 视频')
    print(f'       输出目录: {OUTPUT_DIR}')
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
