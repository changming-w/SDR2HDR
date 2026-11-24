#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SDR 帧/视频 -> HDR10 推理脚本（高质量、支持断点续传、支持 mp4/mkv）

示例:
1) SDR 视频 -> HDR10 视频
python scripts/infer_sdr2hdr10.py \
  --input /path/to/sdr.mp4 \
  --output_video /path/to/output_hdr10.mp4 \
  --checkpoint /path/to/model.pth

2) 图片目录 -> HDR10 视频
python scripts/infer_sdr2hdr10.py \
  --input /path/to/sdr_frames \
  --output_video /path/to/output_hdr10.mkv \
  --checkpoint /path/to/model.pth \
  --fps 60

3) 单图测试
python scripts/infer_sdr2hdr10.py \
  --input /path/to/sdr.png \
  --output_img /path/to/hdr_output.png \
  --checkpoint /path/to/model.pth
"""

import os
import glob
import json
import argparse
import tempfile
import subprocess
from typing import List, Optional, Tuple

import cv2
import yaml
import torch
import numpy as np
from tqdm import tqdm

from basicsr.utils.registry import ARCH_REGISTRY


VIDEO_EXTS = {'.mp4', '.mkv', '.mov', '.avi', '.flv', '.wmv', '.ts', '.m4v', '.webm'}
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


# -----------------------------
# 通用工具
# -----------------------------
def run_cmd(cmd: List[str], err_hint: str = 'Command failed') -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        err = result.stderr[-4000:] if result.stderr else ''
        raise RuntimeError(f'{err_hint}\nCMD: {" ".join(cmd)}\n{err}')
    return result


def parse_ratio_to_float(x) -> float:
    if x is None:
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if '/' in s:
        a, b = s.split('/')
        b = float(b)
        if b == 0:
            return 0.0
        return float(a) / b
    return float(s)


def ffprobe_stream(path: str, count_frames: bool = False) -> dict:
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0']
    if count_frames:
        cmd += ['-count_frames']
    cmd += [
        '-show_entries', 'stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames,duration',
        '-of', 'json', path
    ]
    r = run_cmd(cmd, 'ffprobe failed')
    data = json.loads(r.stdout)
    if 'streams' not in data or len(data['streams']) == 0:
        raise RuntimeError(f'No video stream found in {path}')
    return data['streams'][0]


def ratio_str_to_fps(s: str) -> float:
    if not s:
        return 0.0
    return parse_ratio_to_float(s)


def probe_video_info(path: str) -> dict:
    st = ffprobe_stream(path, count_frames=False)
    fps = ratio_str_to_fps(st.get('avg_frame_rate') or st.get('r_frame_rate'))
    duration = parse_ratio_to_float(st.get('duration')) if st.get('duration') is not None else None
    w = int(st['width'])
    h = int(st['height'])
    return {'width': w, 'height': h, 'fps': fps, 'duration': duration}


def probe_frame_count(path: str, fps_hint: Optional[float] = None) -> int:
    """尽量准确地获取视频帧数（用于断点续传）。"""
    if (not os.path.exists(path)) or os.path.getsize(path) <= 0:
        return 0
    try:
        st = ffprobe_stream(path, count_frames=True)
    except Exception:
        return 0

    # 优先 nb_read_frames
    v = st.get('nb_read_frames')
    if v not in (None, 'N/A'):
        try:
            n = int(v)
            if n > 0:
                return n
        except Exception:
            pass

    # 其次 nb_frames
    v = st.get('nb_frames')
    if v not in (None, 'N/A'):
        try:
            n = int(v)
            if n > 0:
                return n
        except Exception:
            pass

    # 最后 duration * fps
    dur = parse_ratio_to_float(st.get('duration')) if st.get('duration') not in (None, 'N/A') else 0.0
    fps = fps_hint if fps_hint else ratio_str_to_fps(st.get('avg_frame_rate') or st.get('r_frame_rate'))
    if dur > 0 and fps > 0:
        return int(round(dur * fps))
    return 0


def detect_input_type(path: str) -> str:
    if os.path.isdir(path):
        return 'dir'
    ext = os.path.splitext(path)[1].lower()
    if ext in VIDEO_EXTS:
        return 'video'
    if ext in IMAGE_EXTS:
        return 'img'
    if os.path.isfile(path):
        # 兜底
        return 'img'
    raise FileNotFoundError(f'Cannot detect input type: {path}')


def ensure_video_output_ext(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext not in {'.mp4', '.mkv'}:
        raise ValueError(f'Only .mp4 / .mkv are supported, got: {path}')


def maybe_resize_hw(h: int, w: int, max_size: Optional[int]) -> Tuple[int, int]:
    if max_size is None or max(h, w) <= max_size:
        return h, w
    scale = max_size / max(h, w)
    return int(round(h * scale)), int(round(w * scale))


# -----------------------------
# HDR 元数据
# -----------------------------
def parse_hdr10_metadata(metadata_json: Optional[str]) -> Tuple[str, int, int]:
    """
    返回:
      master_display: x265 需要的字符串
      max_cll
      max_fall
    """
    # 默认值（你原脚本中的 Bulgaria4KHDR 一套）
    master_display = 'G(12250,35150)B(6899,2600)R(33899,16099)WP(15634,16449)L(10000000,1)'
    max_cll = 1100
    max_fall = 180

    if not metadata_json or (not os.path.exists(metadata_json)):
        return master_display, max_cll, max_fall

    with open(metadata_json, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    def to_50000(v):
        return int(round(parse_ratio_to_float(v) * 50000))

    def to_10000(v):
        return int(round(parse_ratio_to_float(v) * 10000))

    # ffprobe 常见结构: streams[0].side_data_list
    if 'streams' in meta and len(meta['streams']) > 0:
        s0 = meta['streams'][0]
        side_data = s0.get('side_data_list', [])
        for sd in side_data:
            tp = sd.get('side_data_type')
            if tp == 'Mastering display metadata':
                rx, ry = to_50000(sd.get('red_x')), to_50000(sd.get('red_y'))
                gx, gy = to_50000(sd.get('green_x')), to_50000(sd.get('green_y'))
                bx, by = to_50000(sd.get('blue_x')), to_50000(sd.get('blue_y'))
                wx, wy = to_50000(sd.get('white_point_x')), to_50000(sd.get('white_point_y'))
                min_l = to_10000(sd.get('min_luminance'))
                max_l = to_10000(sd.get('max_luminance'))
                min_l = max(min_l, 1)  # 避免 0 导致部分编码器/播放器异常
                master_display = f'G({gx},{gy})B({bx},{by})R({rx},{ry})WP({wx},{wy})L({max_l},{min_l})'
            elif tp == 'Content light level metadata':
                try:
                    max_cll = int(sd.get('max_content', max_cll))
                    max_fall = int(sd.get('max_average', max_fall))
                except Exception:
                    pass
        return master_display, max_cll, max_fall

    # 兼容 ext_meta_data 结构
    if 'ext_meta_data' in meta:
        ext = meta['ext_meta_data']
        rx = int(round(float(ext['red_x']) * 50000))
        ry = int(round(float(ext['red_y']) * 50000))
        gx = int(round(float(ext['green_x']) * 50000))
        gy = int(round(float(ext['green_y']) * 50000))
        bx = int(round(float(ext['blue_x']) * 50000))
        by = int(round(float(ext['blue_y']) * 50000))
        wx = int(round(float(ext['white_point_x']) * 50000))
        wy = int(round(float(ext['white_point_y']) * 50000))
        max_l = int(round(float(ext.get('max_luma', 1000)) * 10000))
        min_l = max(1, int(round(float(ext.get('min_luma', 0.0001)) * 10000)))
        master_display = f'G({gx},{gy})B({bx},{by})R({rx},{ry})WP({wx},{wy})L({max_l},{min_l})'
        max_cll = int(ext.get('max_luma', max_cll))
        max_fall = int(ext.get('max_fall', max_fall))

    return master_display, max_cll, max_fall


# -----------------------------
# 模型
# -----------------------------
def load_model(config_path: str, checkpoint_path: str, device='cuda', fp16=False):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    net_cfg = config['network_g']
    model = ARCH_REGISTRY.get(net_cfg['type'])(
        num_in_ch=net_cfg.get('num_in_ch', 3),
        num_out_ch=net_cfg.get('num_out_ch', 3),
        grid_H=net_cfg.get('grid_H', 16),
        grid_W=net_cfg.get('grid_W', 16),
        grid_D=net_cfg.get('grid_D', 8),
        base_channels=net_cfg.get('base_channels', 16),
        num_downsample=net_cfg.get('num_downsample', 3),
        num_residual=net_cfg.get('num_residual', 2),
        fixed_ds_size=net_cfg.get('fixed_ds_size', 256),
    )

    state_dict = torch.load(checkpoint_path, map_location=device)
    if 'params_ema' in state_dict:
        state_dict = state_dict['params_ema']
    elif 'params' in state_dict:
        state_dict = state_dict['params']

    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    if fp16:
        model = model.half()
        print('[INFO] FP16 inference enabled.')
    print(f'[INFO] Model loaded from: {checkpoint_path}')
    return model


# -----------------------------
# 推理
# -----------------------------
def sdr_path_to_tensor(path: str, max_size: Optional[int], fp16: bool) -> Tuple[torch.Tensor, Tuple[int, int]]:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f'Cannot read image: {path}')
    h, w = bgr.shape[:2]
    nh, nw = maybe_resize_hw(h, w, max_size)
    if (nh, nw) != (h, w):
        bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()  # C,H,W
    if fp16:
        t = t.half()
    return t, (nh, nw)


def infer_batch_from_paths(model, paths: List[str], device: str, max_size: Optional[int], fp16: bool) -> List[np.ndarray]:
    tensors = []
    shapes = []
    for p in paths:
        t, (h, w) = sdr_path_to_tensor(p, max_size=max_size, fp16=fp16)
        tensors.append(t)
        shapes.append((h, w))

    # pad 到同尺寸
    max_h = max(t.shape[1] for t in tensors)
    max_w = max(t.shape[2] for t in tensors)
    padded = []
    for t in tensors:
        ph = max_h - t.shape[1]
        pw = max_w - t.shape[2]
        if ph > 0 or pw > 0:
            t = torch.nn.functional.pad(t, (0, pw, 0, ph))
        padded.append(t)

    bt = torch.stack(padded).to(device, non_blocking=True)
    with torch.no_grad():
        out = model(bt)  # N,3,H,W

    out = out.float().cpu().numpy()  # N,3,H,W
    results = []
    for i, (h, w) in enumerate(shapes):
        rgb = np.transpose(out[i, :, :h, :w], (1, 2, 0))
        np.clip(rgb, 0.0, 1.0, out=rgb)
        u16 = (rgb * 65535.0 + 0.5).astype(np.uint16)
        results.append(u16)
    return results


def infer_batch_from_rgb_u8(model, rgbs: List[np.ndarray], device: str, fp16: bool) -> List[np.ndarray]:
    """
    rgbs: list of HxWx3 uint8, RGB
    """
    arr = np.stack(rgbs, axis=0).astype(np.float32) / 255.0  # N,H,W,C
    bt = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(device, non_blocking=True)
    if fp16:
        bt = bt.half()
    with torch.no_grad():
        out = model(bt)  # N,3,H,W

    out = out.float().cpu().numpy()
    results = []
    for i in range(out.shape[0]):
        rgb = np.transpose(out[i], (1, 2, 0))
        np.clip(rgb, 0.0, 1.0, out=rgb)
        u16 = (rgb * 65535.0 + 0.5).astype(np.uint16)
        results.append(u16)
    return results


# -----------------------------
# 编码 / 续传
# -----------------------------
def build_encoder_cmd(
    width: int,
    height: int,
    fps: float,
    output_path: str,
    master_display: str,
    max_cll: int,
    max_fall: int,
    crf: int,
    preset: str,
    lossless: bool
) -> List[str]:
    ensure_video_output_ext(output_path)
    ext = os.path.splitext(output_path)[1].lower()

    # x265 的 HDR10 参数写法：max-cll=MaxCLL,MaxFALL
    hdr_params = (
        f'hdr-opt=1:repeat-headers=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:'
        f'master-display={master_display}:max-cll={max_cll},{max_fall}:range=limited'
    )

    cmd = [
        'ffmpeg', '-loglevel', 'error', '-y',
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb48le',   # Python pipe 输入: 16bit RGB
        '-s', f'{width}x{height}',
        '-r', str(fps),
        '-i', 'pipe:0',
        '-an', '-sn',
        '-c:v', 'libx265',
        '-color_primaries', 'bt2020',
        '-color_trc', 'smpte2084',
        '-colorspace', 'bt2020nc',
    ]

    if lossless:
        # 极致质量（注意兼容性不如 4:2:0）
        cmd += [
            '-profile:v', 'main444-10',
            '-pix_fmt', 'yuv444p10le',
            '-preset', preset,
            '-x265-params', hdr_params + ':lossless=1'
        ]
    else:
        # 默认高质量近似“视觉无损”
        cmd += [
            '-profile:v', 'main10',
            '-pix_fmt', 'yuv420p10le',
            '-preset', preset,
            '-crf', str(crf),
            '-x265-params', hdr_params + ':aq-mode=3:aq-strength=0.9:qcomp=0.75'
        ]

    if ext == '.mp4':
        cmd += ['-tag:v', 'hvc1', '-movflags', '+faststart']

    cmd.append(output_path)
    return cmd


def _ffconcat_escape(path: str) -> str:
    # ffconcat: file '...'
    return path.replace("\\", "/").replace("'", "'\\''")


def concat_copy(file_a: str, file_b: str, out_file: str):
    """
    无重编码拼接（要求编码参数一致）
    """
    ext = os.path.splitext(out_file)[1].lower()
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False, encoding='utf-8') as f:
        list_txt = f.name
        f.write(f"file '{_ffconcat_escape(os.path.abspath(file_a))}'\n")
        f.write(f"file '{_ffconcat_escape(os.path.abspath(file_b))}'\n")

    tmp_out = out_file + '.concat_tmp' + ext
    cmd = ['ffmpeg', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', list_txt, '-c', 'copy']
    if ext == '.mp4':
        cmd += ['-movflags', '+faststart']
    cmd.append(tmp_out)
    run_cmd(cmd, 'Concat failed')

    os.replace(tmp_out, out_file)
    os.remove(list_txt)


def make_resume_part_path(output_path: str) -> str:
    base, ext = os.path.splitext(output_path)
    return f'{base}.resume_part{ext}'


# -----------------------------
# 三种处理入口
# -----------------------------
def process_single_image(args, model, device):
    t, _ = sdr_path_to_tensor(args.input_img, max_size=args.max_size, fp16=args.fp16)
    bt = t.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(bt)
    rgb = out.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
    np.clip(rgb, 0.0, 1.0, out=rgb)
    u16 = (rgb * 65535.0 + 0.5).astype(np.uint16)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_img)), exist_ok=True)
    cv2.imwrite(args.output_img, u16)
    print(f'[DONE] Saved: {args.output_img}')
    print(f'       shape={u16.shape}, dtype={u16.dtype}, min={u16.min()}, max={u16.max()}')


def process_image_dir(args, model, device):
    ensure_video_output_ext(args.output_video)

    # 收集图片
    img_paths = []
    for e in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff'):
        img_paths.extend(glob.glob(os.path.join(args.input_dir, e)))
    img_paths = sorted(img_paths)
    if not img_paths:
        raise RuntimeError(f'No images found in: {args.input_dir}')

    fps = args.fps if args.fps is not None else 60
    if args.fps is None:
        print('[INFO] FPS not provided for dir input, use default 60.')

    # duration 截断
    if args.duration is not None:
        limit = int(round(args.duration * fps))
        img_paths = img_paths[:limit]

    total = len(img_paths)
    if total == 0:
        raise RuntimeError('No frames to process after duration limit.')

    # 目标尺寸（编码器固定尺寸）
    first = cv2.imread(img_paths[0], cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f'Cannot read first image: {img_paths[0]}')
    h0, w0 = first.shape[:2]
    out_h, out_w = maybe_resize_hw(h0, w0, args.max_size)

    master_display, max_cll, max_fall = parse_hdr10_metadata(args.metadata)

    # 续传
    done = 0
    resume = not args.no_resume
    if resume:
        done = probe_frame_count(args.output_video, fps_hint=fps)
        if done > 0:
            print(f'[INFO] Resume detected: {done} frames already in output.')
    if done >= total:
        print(f'[DONE] Already complete ({done}/{total}), skip.')
        return

    start_idx = done
    out_segment = args.output_video if start_idx == 0 else make_resume_part_path(args.output_video)
    if os.path.exists(out_segment):
        os.remove(out_segment)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_video)), exist_ok=True)

    enc_cmd = build_encoder_cmd(
        width=out_w, height=out_h, fps=fps, output_path=out_segment,
        master_display=master_display, max_cll=max_cll, max_fall=max_fall,
        crf=args.crf, preset=args.preset, lossless=args.lossless
    )
    print(f'[INFO] Encoding -> {out_segment}')
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    batch_size = max(1, args.batch_size)
    pbar = tqdm(total=total - start_idx, desc='Processing(dir)', unit='frame')

    try:
        for i in range(start_idx, total, batch_size):
            paths = img_paths[i:i + batch_size]
            hdr_list = infer_batch_from_paths(model, paths, device=device, max_size=args.max_size, fp16=args.fp16)
            for hdr_u16 in hdr_list:
                if hdr_u16.shape[0] != out_h or hdr_u16.shape[1] != out_w:
                    hdr_u16 = cv2.resize(hdr_u16, (out_w, out_h), interpolation=cv2.INTER_AREA)
                enc.stdin.write(hdr_u16.tobytes())
            pbar.update(len(paths))
    finally:
        pbar.close()
        enc.stdin.close()
        err = enc.stderr.read().decode('utf-8', errors='ignore')
        ret = enc.wait()
        if ret != 0:
            raise RuntimeError(f'ffmpeg encode failed:\n{err[-3000:]}')

    # 续传拼接
    if start_idx > 0:
        print('[INFO] Concatenating old output + resumed segment ...')
        concat_copy(args.output_video, out_segment, args.output_video)
        os.remove(out_segment)

    print(f'[DONE] HDR10 saved: {args.output_video}')


def process_video_stream(args, model, device):
    ensure_video_output_ext(args.output_video)

    info = probe_video_info(args.input_video)
    src_w, src_h = info['width'], info['height']
    src_fps = info['fps'] if info['fps'] > 0 else 60.0
    src_duration = info.get('duration', None)

    fps = float(args.fps) if args.fps is not None else src_fps
    out_h, out_w = maybe_resize_hw(src_h, src_w, args.max_size)

    # 预计总帧（用于进度）
    if args.duration is not None:
        total_target = int(round(args.duration * fps))
    elif src_duration is not None and src_duration > 0:
        total_target = int(round(src_duration * fps))
    else:
        total_target = None

    master_display, max_cll, max_fall = parse_hdr10_metadata(args.metadata)

    # 续传
    resume = not args.no_resume
    done = 0
    if resume:
        done = probe_frame_count(args.output_video, fps_hint=fps)
        if done > 0:
            print(f'[INFO] Resume detected: {done} frames already encoded.')
    if total_target is not None and done >= total_target:
        print(f'[DONE] Already complete ({done}/{total_target}), skip.')
        return

    remaining = (total_target - done) if total_target is not None else None
    skip_sec = done / fps if done > 0 else 0.0

    out_segment = args.output_video if done == 0 else make_resume_part_path(args.output_video)
    if os.path.exists(out_segment):
        os.remove(out_segment)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_video)), exist_ok=True)

    # 解码命令（流式 raw rgb24）
    vf_chain = f'fps={fps}'
    if (out_w, out_h) != (src_w, src_h):
        vf_chain += f',scale={out_w}:{out_h}:flags=area'

    dec_cmd = ['ffmpeg', '-loglevel', 'error']
    if not args.cpu_decode:
        dec_cmd += ['-hwaccel', 'cuda']
    if skip_sec > 0:
        dec_cmd += ['-ss', f'{skip_sec:.6f}']
    dec_cmd += [
        '-i', args.input_video,
        '-vf', vf_chain,
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
        'pipe:1'
    ]

    enc_cmd = build_encoder_cmd(
        width=out_w, height=out_h, fps=fps, output_path=out_segment,
        master_display=master_display, max_cll=max_cll, max_fall=max_fall,
        crf=args.crf, preset=args.preset, lossless=args.lossless
    )

    print(f'[INFO] Source: {src_w}x{src_h} @{src_fps:.3f}fps')
    print(f'[INFO] Output: {out_w}x{out_h} @{fps:.3f}fps -> {out_segment}')
    if done > 0:
        print(f'[INFO] Resume from frame {done + 1} (skip ~{skip_sec:.3f}s)')

    dec = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    frame_bytes = out_w * out_h * 3
    batch_size = max(1, args.batch_size)
    batch_rgbs = []
    processed = 0

    pbar = tqdm(total=remaining, desc='Processing(video)', unit='frame')

    try:
        while True:
            if remaining is not None and processed >= remaining:
                break

            raw = dec.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break

            rgb = np.frombuffer(raw, dtype=np.uint8).reshape(out_h, out_w, 3)
            batch_rgbs.append(rgb.copy())  # copy，避免底层 buffer 复用问题

            if len(batch_rgbs) >= batch_size:
                hdr_list = infer_batch_from_rgb_u8(model, batch_rgbs, device=device, fp16=args.fp16)
                for hdr_u16 in hdr_list:
                    enc.stdin.write(hdr_u16.tobytes())
                processed += len(batch_rgbs)
                pbar.update(len(batch_rgbs))
                batch_rgbs = []

        if batch_rgbs:
            # 注意：剩余帧也计入 processed；不会重复加
            if remaining is not None:
                keep = max(0, remaining - processed)
                batch_rgbs = batch_rgbs[:keep]
            if batch_rgbs:
                hdr_list = infer_batch_from_rgb_u8(model, batch_rgbs, device=device, fp16=args.fp16)
                for hdr_u16 in hdr_list:
                    enc.stdin.write(hdr_u16.tobytes())
                processed += len(batch_rgbs)
                pbar.update(len(batch_rgbs))

    finally:
        pbar.close()

        # 关闭解码
        try:
            if dec.stdout:
                dec.stdout.close()
        except Exception:
            pass
        if dec.poll() is None:
            dec.terminate()
            try:
                dec.wait(timeout=3)
            except subprocess.TimeoutExpired:
                dec.kill()

        # 关闭编码
        enc.stdin.close()
        err = enc.stderr.read().decode('utf-8', errors='ignore')
        ret = enc.wait()
        if ret != 0:
            raise RuntimeError(f'ffmpeg encode failed:\n{err[-3000:]}')

    # 拼接续传段
    if done > 0:
        print('[INFO] Concatenating old output + resumed segment ...')
        concat_copy(args.output_video, out_segment, args.output_video)
        os.remove(out_segment)

    print(f'[DONE] HDR10 saved: {args.output_video}')
    print(f'[INFO] Frames this run: {processed}, total with resume: {done + processed}')


# -----------------------------
# 参数
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='SDR -> HDR10 inference (high quality, resume, mp4/mkv)')

    # 模型
    parser.add_argument('--config', type=str, default='options/train/Custom/train_hdrnet_sdr2hdr.yml',
                        help='训练配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型 checkpoint 路径(.pth)')

    # 输入（新接口 + 兼容旧接口）
    parser.add_argument('--input', type=str, default=None, help='输入路径: 视频/图片目录/单图')
    parser.add_argument('--input_video', type=str, default=None, help='兼容旧接口')
    parser.add_argument('--input_dir', type=str, default=None, help='兼容旧接口')
    parser.add_argument('--input_img', type=str, default=None, help='兼容旧接口')

    # 输出
    parser.add_argument('--output_video', type=str, default='./output_hdr10.mp4', help='视频输出路径(.mp4/.mkv)')
    parser.add_argument('--output_img', type=str, default='./output_hdr.png', help='单图输出路径')

    # 推理参数
    parser.add_argument('--fps', type=float, default=None, help='输出帧率: 视频默认跟随源fps，目录默认60')
    parser.add_argument('--duration', type=float, default=None, help='仅处理前N秒')
    parser.add_argument('--max_size', type=int, default=None, help='最大边长限制（如 2160）')
    parser.add_argument('--batch_size', type=int, default=4, help='推理 batch size')

    # 性能
    parser.add_argument('--fp16', action='store_true', help='FP16 推理')
    parser.add_argument('--cpu_decode', action='store_true', help='视频解码使用 CPU（默认尝试 GPU 解码）')

    # 编码质量
    parser.add_argument('--crf', type=int, default=8, help='x265 CRF（越小质量越高，默认8）')
    parser.add_argument('--preset', type=str, default='medium',
                        choices=['medium', 'slow', 'slower', 'veryslow'],
                        help='x265 preset')
    parser.add_argument('--lossless', action='store_true',
                        help='x265 lossless（体积极大，兼容性低于 yuv420）')

    # HDR 元数据
    parser.add_argument('--metadata', type=str, default=None, help='HDR metadata JSON 文件')

    # 续传
    parser.add_argument('--no_resume', action='store_true', help='禁用断点续传（默认开启）')

    return parser.parse_args()


def resolve_input(args) -> Tuple[str, str]:
    """
    返回: (input_type, input_path)
    """
    in_path = args.input
    if in_path is None:
        # 兼容旧参数
        in_path = args.input_video or args.input_dir or args.input_img
    if in_path is None:
        raise ValueError('Please provide --input (or one of --input_video/--input_dir/--input_img).')

    t = detect_input_type(in_path)
    return t, in_path


def main():
    args = parse_args()

    input_type, input_path = resolve_input(args)
    print(f'[INFO] Detected input: {input_type} ({input_path})')

    # 绑定兼容字段
    if input_type == 'video':
        args.input_video = input_path
    elif input_type == 'dir':
        args.input_dir = input_path
    else:
        args.input_img = input_path

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[INFO] Device: {device}')

    model = load_model(args.config, args.checkpoint, device=device, fp16=args.fp16)

    if input_type == 'img':
        process_single_image(args, model, device)
    elif input_type == 'dir':
        process_image_dir(args, model, device)
    else:
        process_video_stream(args, model, device)


if __name__ == '__main__':
    main()