#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成红、绿、蓝、白四张色块图，每张持续 10 秒，拼接为一个 HDR10 视频。
============================================================================

- 画面：3840x2160，黑色背景，中央矩形色块，面积占整幅图像的 20%
- 色块：BT.2020 基色的纯红/纯绿/纯蓝/白，峰值亮度默认 1000 nits，
        线性光经 ST 2084 (PQ) OETF 编码后量化为 16 bit 码值
- 编码：HEVC Main10 / yuv420p10le，BT.2020nc，limited range，附带 HDR10 静态元数据
- 元数据：/mnt/data/wcm/data/hdr/test/SDR/hdr10_metadata_default.json
- 验证：编码完成后用 ffprobe 检查 MasteringDisplay 与 ContentLightLevel 是否存在

用法
----
python scripts/make_color_block_hdr10_video.py
"""

import os
import sys
import json
import math
import shutil
import tempfile
import argparse
import subprocess
import numpy as np


# ── 路径配置 ──────────────────────────────────────────────────────────────────
METADATA_JSON = '/mnt/data/wcm/data/hdr/test/SDR/hdr10_metadata_default.json'
OUTPUT_DIR = '/mnt/data/wcm/data/hdr/test/color_block_hdr10'
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'color_block_rgbw_hdr10_4k_10s.mp4')
IMAGE_DIR = os.path.join(OUTPUT_DIR, 'images')
# 运行时会根据 --peak-nits 更新为带峰值标记的路径

# ── 图像与色块参数 ────────────────────────────────────────────────────────────
WIDTH = 3840
HEIGHT = 2160
AREA_RATIO = 0.20         # 色块面积占整幅图像的比例
PEAK_NITS = 1000.0        # 色块峰值亮度 (nits)
BG_NITS = 0.0             # 背景亮度 (nits)，黑色

# 顺序：红、绿、蓝、白。RGB 为 BT.2020 线性基色系数
COLORS = [
    ('red', (1.0, 0.0, 0.0)),
    ('green', (0.0, 1.0, 0.0)),
    ('blue', (0.0, 0.0, 1.0)),
    ('white', (1.0, 1.0, 1.0)),
]

# ── 视频参数 ──────────────────────────────────────────────────────────────────
DURATION = 10         # 每张图持续时间（秒）
FPS = 30              # 输出帧率（静态图，帧率仅影响容器中的帧数）

# ── 编码参数（参考 make_hdr10_videos.py） ─────────────────────────────────────
CRF = 12.0
PRESET = 'slow'

# ── ST 2084 (PQ) 常量 ─────────────────────────────────────────────────────────
PQ_M1 = 2610.0 / 16384.0
PQ_M2 = 2523.0 / 4096.0 * 128.0
PQ_C1 = 3424.0 / 4096.0
PQ_C2 = 2413.0 / 4096.0 * 32.0
PQ_C3 = 2392.0 / 4096.0 * 32.0


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
    """构造 x265 参数字符串。"""
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


def pq_oetf(nits):
    """
    ST 2084 (PQ) OETF：绝对亮度 (nits) -> PQ 非线性码值 [0, 1]。
    """
    y = np.clip(nits / 10000.0, 0.0, 1.0) ** PQ_M1
    return ((PQ_C1 + PQ_C2 * y) / (1.0 + PQ_C3 * y)) ** PQ_M2


def build_color_block_frame(rgb_weight):
    """
    生成一帧 3840x2160 的 RGB48 little-endian 图像（PQ 码值，BT.2020）。
    色块为中央矩形，宽高等比例缩放，面积 = AREA_RATIO * 图像面积。
    """
    scale = math.sqrt(AREA_RATIO)
    rect_w = int(round(WIDTH * scale))
    rect_h = int(round(HEIGHT * scale))
    x0 = (WIDTH - rect_w) // 2
    y0 = (HEIGHT - rect_h) // 2

    nits = np.full((HEIGHT, WIDTH, 3), BG_NITS, dtype=np.float64)
    nits[y0:y0 + rect_h, x0:x0 + rect_w, :] = np.array(rgb_weight, dtype=np.float64) * PEAK_NITS

    code = pq_oetf(nits)
    frame = np.round(code * 65535.0).astype('<u2')
    actual_ratio = rect_w * rect_h / float(WIDTH * HEIGHT)
    return frame, actual_ratio


def encode_video(raw_paths, output_path, master_display, max_cll, max_fall):
    """
    将每张 rgb48le 原始帧循环 DURATION 秒，在一次 ffmpeg 调用中按顺序拼接为单个 HDR10 视频。
    色彩转换与输出标签与 make_hdr10_videos.py 的 encode_video 保持一致。
    """
    x265_params = build_x265_params(master_display, max_cll, max_fall)
    fps_str = str(FPS)
    n = len(raw_paths)

    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y']
    # ── 输入：每张色块图一路，循环读入同一帧并截取 DURATION 秒 ──
    for raw_path in raw_paths:
        cmd += [
            '-stream_loop', '-1',
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb48le',
            '-video_size', f'{WIDTH}x{HEIGHT}',
            '-framerate', fps_str,
            '-t', str(DURATION),
            '-i', raw_path,
        ]
    # ── 拼接 + 色彩转换：RGB48(full) -> YUV420P10(BT.2020nc, limited) ──
    filter_complex = (
        ''.join(f'[{i}:v]' for i in range(n)) +
        f'concat=n={n}:v=1:a=0,'
        'scale=in_range=full:out_range=tv:'
        'out_color_matrix=bt2020:'
        'flags=accurate_rnd+full_chroma_int,'
        'format=yuv420p10le[v]'
    )
    cmd += [
        '-filter_complex', filter_complex,
        '-map', '[v]',
        '-an', '-sn', '-dn',
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
        '-r', fps_str,
        output_path,
    ]
    run_cmd(cmd, f'编码失败: {output_path}')


def save_png(raw_path, png_path):
    """将 rgb48le 原始帧存为 16 bit PNG（rgb48be，与 hdrframes 格式一致）。"""
    cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y',
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb48le',
        '-video_size', f'{WIDTH}x{HEIGHT}',
        '-i', raw_path,
        '-frames:v', '1',
        '-pix_fmt', 'rgb48be',
        png_path,
    ]
    run_cmd(cmd, f'导出 PNG 失败: {png_path}')


def verify_hdr10(path):
    """验证输出视频是否包含 MasteringDisplay 与 ContentLightLevel 元数据。"""
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


def main():
    global PEAK_NITS, OUTPUT_PATH, IMAGE_DIR

    parser = argparse.ArgumentParser(description='生成红/绿/蓝/白色块 HDR10 视频')
    parser.add_argument('--peak-nits', type=float, default=PEAK_NITS,
                        help='色块峰值亮度 (nits)，默认 %(default)g')
    args = parser.parse_args()

    PEAK_NITS = args.peak_nits
    tag = f'peak{PEAK_NITS:g}nits'
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, f'color_block_rgbw_hdr10_4k_10s_{tag}.mp4')
    IMAGE_DIR = os.path.join(OUTPUT_DIR, f'images_{tag}')

    print('=' * 68)
    print('色块 HDR10 视频生成脚本（红/绿/蓝/白）')
    print('=' * 68)

    # 1. 解析 HDR10 元数据
    print(f'\n[INFO] 读取元数据: {METADATA_JSON}')
    master_display, max_cll, max_fall = parse_hdr10_metadata(METADATA_JSON)
    print(f'[INFO] MasteringDisplay = {master_display}')
    print(f'[INFO] MaxCLL = {max_cll}, MaxFALL = {max_fall}')

    # 2. 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f'\n[INFO] 输出文件: {OUTPUT_PATH}')
    print(f'[INFO] 分辨率 {WIDTH}x{HEIGHT}, 峰值 {PEAK_NITS:g} nits, '
          f'每段 {DURATION} 秒, fps={FPS}')

    os.makedirs(IMAGE_DIR, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix='color_block_', dir=OUTPUT_DIR)
    try:
        # 3. 生成各色块帧（PQ 码值，rgb48le 原始数据）
        raw_paths = []
        for i, (name, weight) in enumerate(COLORS):
            frame, actual_ratio = build_color_block_frame(weight)
            raw_path = os.path.join(tmp_dir, f'{i:02d}_{name}.rgb48le')
            frame.tofile(raw_path)
            raw_paths.append(raw_path)
            png_path = os.path.join(IMAGE_DIR, f'{name}.png')
            save_png(raw_path, png_path)
            print(f'[{i+1}/{len(COLORS)}] {name}: 色块面积占比 = {actual_ratio * 100:.2f}%, 图片: {png_path}')

        # 4. 单次 ffmpeg 调用：四张图按顺序拼接并编码为一个视频
        print('\n[INFO] 编码为单个 HDR10 视频 ...')
        encode_video(raw_paths, OUTPUT_PATH, master_display, max_cll, max_fall)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # 6. ffprobe 验证 HDR10 元数据
    has_md, has_cll = verify_hdr10(OUTPUT_PATH)
    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f'\n[VERIFY] 大小: {size_mb:.1f} MB | '
          f'MasteringDisplay: {"OK" if has_md else "MISSING"} | '
          f'ContentLightLevel: {"OK" if has_cll else "MISSING"}')

    print('\n' + '=' * 68)
    if has_md and has_cll:
        print(f'[DONE] 视频生成完成: {OUTPUT_PATH}')
    else:
        print(f'[WARN] 输出缺少 HDR10 元数据，请检查: {OUTPUT_PATH}')
    print('=' * 68)

    return 0 if (has_md and has_cll) else 2


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('\n[WARN] 用户中断')
        sys.exit(130)
    except Exception as e:
        print(f'\n[ERROR] {e}', file=sys.stderr)
        sys.exit(1)
