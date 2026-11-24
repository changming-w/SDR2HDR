#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create a 30 Hz HDR10 video from still images (HEVC Main10).

The default input is dark_scene_1200/hdr. Each image is shown for five seconds.
HDR10 mastering metadata uses BT.2020 primaries, ST 2084/PQ, and a 4000-nit
mastering-display peak by default.

Example:
    python3 /mnt/data/wcm/HDRNet/scripts/make_hdr10.py
"""

import argparse
import subprocess
import sys
from pathlib import Path


INPUT_DIR = Path('/mnt/data/wcm/data/hdr/train/标准SDR2HDR训练数据/dark_scene_1200/hdr')
OUTPUT_DIR = Path('/mnt/data/wcm/data/hdr/train/标准SDR2HDR训练数据/dark_scene_1200/hdr10_02hz_4000nit')
FPS = 0.2
HOLD_SEC = 5
CRF = 10
PRESET = 'medium'
PEAK_NITS = 4000
MAX_CLL = 4000
MAX_FALL = 400
NUM_FRAMES = 99999
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}

# BT.2020 primaries, D65 white point. Mastering display luminance is in 1/10000 nit.
MASTER_DISPLAY_TEMPLATE = (
    'G(8500,39850)B(6550,2300)R(35400,14600)'
    'WP(15635,16450)L({max_luma},1)'
)


def run_cmd(cmd, hint='Command failed'):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f'{hint}\nCMD: {" ".join(cmd)}\n{(result.stderr or "")[-4000:]}'
        )
    return result


def quote_concat_path(path):
    """Quote a path according to ffmpeg concat-demuxer syntax."""
    escaped = str(path.resolve()).replace('\\', '\\\\').replace("'", "'\\''")
    return f"'{escaped}'"


def build_x265_params(master_display, max_cll, max_fall):
    """Construct x265 HDR10 signaling parameters."""
    return ':'.join([
        'hdr10=1',
        'hdr10-opt=1',
        'repeat-headers=1',
        'colorprim=bt2020',
        'transfer=smpte2084',
        'colormatrix=bt2020nc',
        f'master-display={master_display}',
        f'max-cll={max_cll},{max_fall}',
        'range=limited',
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=INPUT_DIR,
                        help=f'HDR image directory (default: {INPUT_DIR})')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR,
                        help=f'Output directory (default: {OUTPUT_DIR})')
    parser.add_argument('--peak-nits', type=int, default=PEAK_NITS,
                        help=f'Mastering-display peak in nits (default: {PEAK_NITS})')
    parser.add_argument('--hold-sec', type=float, default=HOLD_SEC,
                        help=f'Duration per image in seconds (default: {HOLD_SEC})')
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if args.peak_nits <= 0:
        parser.error('--peak-nits must be positive')
    if args.hold_sec <= 0:
        parser.error('--hold-sec must be positive')
    if not input_dir.is_dir():
        raise RuntimeError(f'Input directory does not exist: {input_dir}')

    images = sorted(
        (path for path in input_dir.iterdir()
         if path.is_file() and path.suffix.lower() in IMAGE_EXTS),
        key=lambda path: path.name,
    )[:NUM_FRAMES]
    if not images:
        raise RuntimeError(f'No images found: {input_dir}')

    master_display = MASTER_DISPLAY_TEMPLATE.format(max_luma=args.peak_nits * 10000)
    total_duration = len(images) * args.hold_sec
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'dark_scene_1200_hdr_{FPS}hz_{args.peak_nits}nit_crf{CRF}.mp4'
    concat_list = output_dir / '_concat_list.txt'

    print('=' * 68)
    print('HDR10 video generation — BT.2020/PQ, HEVC Main10')
    print(f'[INFO] Input: {input_dir}')
    print(f'[INFO] Images: {len(images)}; each held {args.hold_sec:g}s')
    print(f'[INFO] Output: {FPS} Hz, duration {total_duration:g}s')
    print(f'[INFO] Mastering peak: {args.peak_nits} nit; MaxCLL/FALL={MAX_CLL}/{MAX_FALL}')
    print(f'[INFO] Mastering display: {master_display}')
    print(f'[INFO] Encoding: CRF={CRF}, preset={PRESET}')
    print(f'[INFO] Output file: {output_path}')

    with concat_list.open('w', encoding='utf-8', newline='\n') as file:
        file.write('ffconcat version 1.0\n')
        for image in images:
            file.write(f'file {quote_concat_path(image)}\n')
            file.write(f'duration {args.hold_sec:g}\n')
        # Repeat final still so the concat demuxer honors its full duration.
        file.write(f'file {quote_concat_path(images[-1])}\n')

    x265_params = build_x265_params(master_display, MAX_CLL, MAX_FALL)
    command = [
        'ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y',
        '-f', 'concat', '-safe', '0', '-i', str(concat_list),
        '-an', '-sn', '-dn',
        '-vf', (
            'scale=in_range=full:out_range=tv:'
            'out_color_matrix=bt2020:'
            'flags=accurate_rnd+full_chroma_int,'
            'format=yuv420p10le'
        ),
        '-c:v', 'libx265', '-profile:v', 'main10',
        '-preset', PRESET, '-crf', str(CRF),
        '-x265-params', x265_params,
        '-r', str(FPS), '-fps_mode', 'cfr',
        '-color_primaries', 'bt2020',
        '-color_trc', 'smpte2084',
        '-colorspace', 'bt2020nc',
        '-color_range', 'tv',
        '-chroma_sample_location', 'left',
        str(output_path),
    ]

    print('[INFO] Encoding...')
    run_cmd(command, 'HDR10 encoding failed')
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f'[OK] Output: {output_path}')
    print(f'[OK] Size: {size_mb:.1f} MB')
    print('=' * 68)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n[WARN] Interrupted by user')
        sys.exit(130)
    except Exception as error:
        print(f'\n[ERROR] {error}', file=sys.stderr)
        sys.exit(1)
