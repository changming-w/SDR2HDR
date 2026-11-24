#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SDR -> HDR10 推理脚本（高质量 / 帧级精确断点续传 / mp4+mkv 输出）
============================================================================

设计要点
--------
1. 编码中间产物一律写 MKV。MP4 的 moov box 在编码结束时才落盘，进程被
   打断后文件完全不可解析，续传会静默从头开始。因此全程用 MKV 累积进度，
   最后一步用 `-c copy` remux 成目标容器（不重编码，SEI 元数据原样保留）。

2. 断点续传是帧级精确的：用 select='gte(n,N)' 按解码帧序号跳过，而不是 -ss 时间 seek。
   NVDEC 解 4K HEVC 有几百 fps，跳 2 万帧只要一两分钟，换来接缝零误差。
   赶时间可以用 --fast_resume 走 -ss，但接缝处可能 ±1 帧。

3. 色彩转换全部显式指定，不依赖 swscale 的隐式推断：
     解码端  YUV(BT.709, limited) -> RGB24(full)
     编码端  RGB48(full) -> YUV420P10(BT.2020nc, limited)
   swscale 在 RGB<->YUV 转换时如果拿不到明确的矩阵会 fallback 到 BT.601，
   那会给整条评测链引入系统性色偏。

4. decode / infer / encode 三线程 + 有界队列，内存封顶（约 600MB），三段真正并行。

5. 正式开跑前用 4 帧做一次编码参数预检，确认 x265 参数被接受、且 HDR10 SEI
   （Mastering Display + Content Light Level）真的写得进码流，避免跑几小时
   才发现元数据是空的。

用法
----
# SDR 视频 -> HDR10
python scripts/infer_sdr2hdr10.py \
    --input  /path/sdr.mp4 \
    --output_video /path/out_hdr10.mp4 \
    --config options/train/Custom/train_hdrnet_bt709_sdr2hdr.yml \
    --checkpoint experiments/.../net_g_175000.pth \
    --metadata /path/ref_hdr.json \
    --fp16 --batch_size 4

# 图片目录 -> HDR10
python scripts/infer_sdr2hdr10.py --input frames_dir --output_video out.mkv \
    --checkpoint ... --fps 60

# 单图测试
python scripts/infer_sdr2hdr10.py --input a.png --output_img a_hdr.png --checkpoint ...
"""

import os
import sys
import glob
import json
import time
import queue
import shutil
import hashlib
import argparse
import tempfile
import threading
import subprocess
import re
from typing import List, Optional, Tuple, Dict, Any, Callable

import cv2
import yaml
import torch
import numpy as np
from tqdm import tqdm

from basicsr.utils.registry import ARCH_REGISTRY
import basicsr.archs  # noqa: F401  触发 archs 自动扫描注册（否则 get(type) 找不到类）


VIDEO_EXTS = {'.mp4', '.mkv', '.mov', '.avi', '.flv', '.wmv', '.ts', '.m4v', '.webm'}
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
OUT_EXTS = {'.mp4', '.mkv'}

STATE_VERSION = 3


# =============================================================================
# 基础工具
# =============================================================================
def run_cmd(cmd: List[str], hint: str = 'Command failed') -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f'{hint}\nCMD: {" ".join(cmd)}\n{(r.stderr or "")[-4000:]}')
    return r


def parse_ratio(x) -> float:
    """解析 "30000/1001" / "0.6780" / 数值 三种形式。"""
    if x is None or x == 'N/A':
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if '/' in s:
        a, b = s.split('/', 1)
        try:
            b = float(b)
            return float(a) / b if b else 0.0
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def ffprobe_stream(path: str, count_packets: bool = False) -> Dict[str, Any]:
    """
    注意：用 -count_packets（只 demux）而不是 -count_frames（要完整解码一遍，
    4K HEVC 上要十几分钟）。对视频流 packet 数 == 帧数。
    """
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0']
    if count_packets:
        cmd += ['-count_packets']
    cmd += ['-show_entries',
            'stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,nb_read_packets,duration',
            '-of', 'json', path]
    data = json.loads(run_cmd(cmd, f'ffprobe failed on {path}').stdout)
    streams = data.get('streams') or []
    if not streams:
        raise RuntimeError(f'No video stream found in: {path}')
    return streams[0]


def probe_source(path: str) -> Dict[str, Any]:
    """探测源视频：分辨率、精确帧率字符串、总帧数。

    帧率优先取 avg_frame_rate（容器根据总时长/总帧数算出的平均帧率，
    决定播放时长）。不能用 r_frame_rate：它是"能表示所有时间戳的最小
    timebase"，对含 B 帧/VFR 的流经常被推成 2 倍甚至更高，按它编码会
    直接改变总时长，导致输出与原片播放进度对不齐。
    """
    st = ffprobe_stream(path, count_packets=True)

    rate_str = st.get('avg_frame_rate') or st.get('r_frame_rate') or '60/1'
    fps = parse_ratio(rate_str)
    if fps <= 0:
        rate_str = st.get('r_frame_rate') or '60/1'
        fps = parse_ratio(rate_str)
    if fps <= 0:
        rate_str, fps = '60/1', 60.0

    total = 0
    for key in ('nb_read_packets', 'nb_frames'):
        v = st.get(key)
        if v not in (None, 'N/A'):
            try:
                total = max(total, int(v))
            except ValueError:
                pass
    if total <= 0:
        dur = parse_ratio(st.get('duration'))
        total = int(round(dur * fps)) if dur > 0 else 0

    return {
        'width': int(st['width']),
        'height': int(st['height']),
        'fps': fps,
        'fps_str': rate_str,   # 保留有理数形式，避免 60000/1001 被浮点化
        'total': total,
    }


def probe_frame_count(path: str, fps_hint: float = 0.0) -> int:
    """帧数；文件不存在返回 0；文件存在但无法解析返回 -1（调用方当致命错误处理）"""
    if not os.path.exists(path) or os.path.getsize(path) <= 0:
        return 0

    for extra in ([], ['-err_detect', 'ignore_err'], ['-fflags', '+discardcorrupt']):
        cmd = (['ffprobe', '-v', 'error'] + extra +
               ['-select_streams', 'v:0', '-count_packets',
                '-show_entries', 'stream=nb_read_packets,nb_frames,duration,avg_frame_rate',
                '-of', 'json', path])
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            continue
        try:
            st = (json.loads(r.stdout).get('streams') or [{}])[0]
        except Exception:
            continue
        for key in ('nb_read_packets', 'nb_frames'):
            v = st.get(key)
            if v not in (None, 'N/A'):
                try:
                    n = int(v)
                    if n > 0:
                        return n
                except ValueError:
                    pass
        dur = parse_ratio(st.get('duration'))
        fps = fps_hint or parse_ratio(st.get('avg_frame_rate'))
        if dur > 0 and fps > 0:
            return int(round(dur * fps))
        return 0
    return -1


def probe_duration(path: str) -> float:
    """容器级时长（秒）；失败返回 0。注意 MP4 的 format duration 可能被更长的音频轨撑大。"""
    try:
        r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
                            'format=duration', '-of', 'json', path],
                           capture_output=True, text=True)
        return parse_ratio((json.loads(r.stdout).get('format') or {}).get('duration'))
    except Exception:
        return 0.0


def verify_sync(src: str, out: str):
    """对比源/输出的帧数与时长，量化"播放进度是否对得齐"。"""
    try:
        n_src, n_out = probe_frame_count(src), probe_frame_count(out)
        d_src, d_out = probe_duration(src), probe_duration(out)
        print(f'[CHECK] 帧数   源={n_src}  输出={n_out}')
        print(f'[CHECK] 时长   源={d_src:.3f}s  输出={d_out:.3f}s')
        if n_src > 0 and n_out > 0 and abs(n_src - n_out) > 1:
            print('[WARN] 帧数差超过 1 帧：电视上播放进度会与原片对不齐！'
                  '常见原因是源为 VFR（可变帧率）或含重复/丢弃帧。')
        elif d_src > 0 and d_out > 0 and abs(d_src - d_out) > 0.1:
            print('[WARN] 时长差超过 0.1s：若源的音频轨比视频轨长，'
                  '可把源音轨单独拷进输出来对齐总时长。')
    except Exception as e:
        print(f'[WARN] 同步校验失败: {e}')


def detect_input_type(path: str) -> str:
    if os.path.isdir(path):
        return 'dir'
    ext = os.path.splitext(path)[1].lower()
    if ext in VIDEO_EXTS:
        return 'video'
    if ext in IMAGE_EXTS:
        return 'img'
    if os.path.isfile(path):
        return 'img'
    raise FileNotFoundError(f'Cannot detect input type: {path}')


def ensure_out_ext(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext not in OUT_EXTS:
        raise ValueError(f'--output_video 只支持 .mp4 / .mkv，当前为: {path}')


def maybe_resize_hw(h: int, w: int, max_size: Optional[int]) -> Tuple[int, int]:
    """按最长边限制缩放；结果强制偶数（4:2:0 要求）。"""
    if max_size is None or max(h, w) <= max_size:
        return (h // 2) * 2, (w // 2) * 2
    s = max_size / max(h, w)
    return (int(round(h * s)) // 2) * 2, (int(round(w * s)) // 2) * 2


def human_size(n: float) -> str:
    for u in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024:
            return f'{n:.1f}{u}'
        n /= 1024
    return f'{n:.1f}PB'


# =============================================================================
# HDR10 元数据解析
# =============================================================================
def parse_hdr10_metadata(metadata_json: Optional[str]) -> Tuple[str, int, int]:
    """
    返回 (master_display_str, max_cll, max_fall)
    master_display 为 x265 `--master-display` 需要的格式：
        G(x,y)B(x,y)R(x,y)WP(x,y)L(max,min)
    色度坐标单位 1/50000，亮度单位 1/10000 nit。
    """
    # 兜底：BT.2020-D65 主显示器，4000 nits（本项目默认峰值与信号标记一致）
    # BT.2020 chromaticities (1/50000):
    #   R(0.708,0.292)->(35400,14600)  G(0.170,0.797)->(8500,39850)
    #   B(0.131,0.046)->(6550,2300)    WP D65(0.3127,0.3290)->(15635,16450)
    # Luminance unit is 1/10000 nit, so 4000 nit -> 40000000
    master = 'G(8500,39850)B(6550,2300)R(35400,14600)WP(15635,16450)L(40000000,1)'
    max_cll, max_fall = 1000, 400

    if not metadata_json or not os.path.exists(metadata_json):
        print('[WARN] 未提供 --metadata，使用默认 HDR10 元数据 (BT.2020-D65 / 4000nits)')
        print(f'[WARN]   MasteringDisplay: {master}')
        return master, max_cll, max_fall

    with open(metadata_json, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    to50k = lambda v: int(round(parse_ratio(v) * 50000))
    to10k = lambda v: int(round(parse_ratio(v) * 10000))

    streams = meta.get('streams') or []
    got_md = got_cll = False

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
                    max(1, to10k(sd.get('min_luminance'))),  # min_lum=0 在部分解析器上有坑
                )
                got_md = True
            elif tp == 'Content light level metadata':
                try:
                    max_cll = int(sd.get('max_content') or max_cll)
                    max_fall = int(sd.get('max_average') or max_fall)
                    got_cll = True
                except (TypeError, ValueError):
                    pass

    elif 'ext_meta_data' in meta:
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
        got_md = got_cll = True

    if not got_md:
        print('[WARN] JSON 中未找到 Mastering display metadata，使用默认值')
    if not got_cll:
        print('[WARN] JSON 中未找到 Content light level metadata，使用默认值')

    print(f'[INFO] MasteringDisplay = {master}')
    print(f'[INFO] MaxCLL = {max_cll}, MaxFALL = {max_fall}')
    return master, max_cll, max_fall


# =============================================================================
# 参考白锚点：可选显示级 (203nit) 重标定
# =============================================================================
def build_white_anchor_post_map(args) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
    """
    返回一个在 GPU 上执行的映射函数，输入 y 为模型输出 (N,C,H,W) float in [0,1]，
    表示 BT.2020/PQ 线性化前的编码域值。为了避免重复 PQ 往返，这里采用
    近似：先把 y 视作线性亮度比例，经全局比例/soft-knee 压缩，再回到 [0,1]。

    说明：--white_anchor=scene（默认）不改变像素；display 时，会把
    scene_white_nit（≈660）映射为 display_white_nit（≈203），并从
    knee_start_nit 起柔和压缩高光，尽量保留细节。
    """
    if args.white_anchor == 'scene':
        return None

    # 参数
    scene_w = max(1e-6, float(args.scene_white_nit))
    disp_w = max(1e-6, float(args.display_white_nit))
    knee = max(disp_w, float(args.knee_start_nit))
    peak = float(args.peak_nits or 4000.0)

    # 预计算比例
    scale = disp_w / scene_w

    def _map(y: torch.Tensor) -> torch.Tensor:
        # y in [0,1] (PQ coded domain). Map to nit approx by proportional to PQ code.
        # Lightweight monotonic remap: x_nit ≈ y * peak; then apply scale + knee.
        # This is an approximation to avoid PQ decode/encode overhead.
        x = y.clamp(0.0, 1.0)
        x_nit = x * peak
        # 线性缩放至让漫反射白对齐
        x_nit = x_nit * scale
        # soft-knee 压缩：从 knee 起使用平滑曲线渐进到 peak（不过界）
        if knee < peak:
            t = torch.clamp((x_nit - knee) / max(1e-6, (peak - knee)), 0.0, 1.0)
            # 曲线：y = knee + (peak-knee) * (1 - (1 - t)^2)  （二次缓入）
            x_nit = torch.where(x_nit <= knee, x_nit,
                                knee + (peak - knee) * (1.0 - (1.0 - t) * (1.0 - t)))
        # 重新归一化到 [0,1]
        y2 = torch.clamp(x_nit / max(1e-6, peak), 0.0, 1.0)
        return y2

    return _map


# =============================================================================
# ffmpeg 命令构造
# =============================================================================
def build_x265_params(md: Tuple[str, int, int], hdr_opt_key: str,
                      extra: Optional[str] = None) -> str:
    master, cll, fall = md
    parts = [
        f'{hdr_opt_key}=1',        # x265 3.x 叫 hdr10-opt，2.x 叫 hdr-opt，由预检自动选
        'repeat-headers=1',        # 每个 IDR 重复 VPS/SPS/PPS，容器无关地保住元数据
        'colorprim=bt2020',
        'transfer=smpte2084',
        'colormatrix=bt2020nc',
        f'master-display={master}',
        f'max-cll={cll},{fall}',   # x265 语法是 "MaxCLL,MaxFALL"，没有独立的 max-fall
        'range=limited',
    ]
    if extra:
        parts.append(extra.strip(':'))
    return ':'.join(parts)


def override_hdr10_md(md: Tuple[str, int, int], peak_nits: Optional[float],
                      max_cll: Optional[int], max_fall: Optional[int]) -> Tuple[str, int, int]:
    """Override MasteringDisplay peak and MaxCLL/FALL numerically.

    peak_nits: desired mastering max luminance in nit (e.g., 4000.0).
    """
    master, cll, fall = md
    if peak_nits is not None:
        # Replace L(max, min) first number in master string to 1/10000 nit units
        m = re.search(r'L\((\d+),(\d+)\)$', master)
        if m:
            min_l = m.group(2)
            max_l = int(round(float(peak_nits) * 10000.0))
            master = re.sub(r'L\(\d+,\d+\)$', f'L({max_l},{min_l})', master)
    if isinstance(max_cll, int):
        cll = max_cll
    if isinstance(max_fall, int):
        fall = max_fall
    return master, cll, fall


def build_encoder_cmd(width: int, height: int, fps_str: str, out_mkv: str,
                      md: Tuple[str, int, int], crf: float, preset: str,
                      hdr_opt_key: str, x265_extra: Optional[str]) -> List[str]:
    """
    输入：Python 管道送来的 rgb48le（16bit RGB，full range，已是 PQ / BT.2020 编码值）
    输出：HEVC Main10 / yuv420p10le / BT.2020nc limited，带完整 HDR10 SEI
    """
    return [
        'ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y',
        # ---------- 输入侧 ----------
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb48le',
        '-s', f'{width}x{height}',
        '-r', fps_str,
        '-color_primaries', 'bt2020',
        '-color_trc', 'smpte2084',
        '-colorspace', 'bt2020nc',
        '-color_range', 'pc',
        '-i', 'pipe:0',
        '-an', '-sn', '-dn',
        # ---------- 显式色彩转换：RGB(full) -> YUV420P10(BT.2020nc, limited) ----------
        # out_color_matrix=bt2020 会被 vf_scale 映射到 BT2020_NCL。
        # 不写这一句就完全依赖 swscale 的隐式推断，不同 ffmpeg 版本行为不一致。
        '-vf', 'scale=in_range=full:out_range=tv:out_color_matrix=bt2020:flags=accurate_rnd+full_chroma_int,format=yuv420p10le',
        # ---------- 编码 ----------
        '-c:v', 'libx265',
        '-profile:v', 'main10',
        '-preset', preset,
        '-crf', f'{crf:g}',
        '-x265-params', build_x265_params(md, hdr_opt_key, x265_extra),
        # ---------- 输出侧标签 ----------
        '-color_primaries', 'bt2020',
        '-color_trc', 'smpte2084',
        '-colorspace', 'bt2020nc',
        '-color_range', 'tv',
        '-chroma_sample_location', 'left',
        out_mkv,
    ]


def build_decoder_cmd(src: str, src_w: int, src_h: int, out_w: int, out_h: int,
                      fps_str: str, need_fps_filter: bool, skip: int,
                      hwaccel: str, fast_resume: bool, fps_float: float,
                      in_matrix: str, in_range: str, sws_flags: str) -> List[str]:
    """
    输出 rgb24 raw 到 stdout。
    滤镜顺序: [fps=] -> [select 跳帧] -> scale(显式矩阵/range) -> format=rgb24
    select 必须放在 fps 之后，这样 n 计的是重采样后的输出帧序号，与续传帧号一致。
    """
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'warning']
    if hwaccel and hwaccel != 'none':
        cmd += ['-hwaccel', hwaccel]
    if skip > 0 and fast_resume:
        cmd += ['-ss', f'{skip / max(fps_float, 1e-9):.6f}']
    cmd += ['-i', src]

    chain: List[str] = []
    if need_fps_filter:
        chain.append(f'fps={fps_str}')
    if skip > 0 and not fast_resume:
        chain.append(f"select='gte(n\\,{skip})'")

    sc = 'scale='
    if (out_w, out_h) != (src_w, src_h):
        sc += f'{out_w}:{out_h}:'
    sc += (f"flags='{sws_flags}':"
           f'in_color_matrix={in_matrix}:in_range={in_range}:out_range=full')
    chain.append(sc)
    chain.append('format=rgb24')

    cmd += [
        '-vf', ','.join(chain),
        '-an', '-sn', '-dn',
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
        'pipe:1',
    ]
    return cmd


# =============================================================================
# HDR10 校验 + 编码参数预检
# =============================================================================
def verify_hdr10(path: str) -> Tuple[bool, bool]:
    """返回 (有 MasteringDisplay, 有 ContentLightLevel)。"""
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


def preflight_encoder(md: Tuple[str, int, int], crf: float, preset: str,
                      x265_extra: Optional[str], fps_str: str) -> str:
    """
    用 4 帧 256x256 做一次真实编码，确认：
      1. x265 参数全部能被解析（写错一个字母 ffmpeg 就会直接失败）
      2. HDR10 SEI 真的进了码流
    同时自动在 hdr10-opt / hdr-opt 两个拼写间选择可用的那个。
    返回可用的 key，失败则抛异常。
    """
    tmp_dir = tempfile.mkdtemp(prefix='hdr10_preflight_')
    last_err = ''
    try:
        frame = np.full((256, 256, 3), 20000, dtype=np.uint16)
        for key in ('hdr10-opt', 'hdr-opt'):
            out = os.path.join(tmp_dir, f'probe_{key}.mkv')
            cmd = build_encoder_cmd(256, 256, fps_str, out, md, crf, preset, key, x265_extra)
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                for _ in range(4):
                    p.stdin.write(frame)
                p.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            err = p.stderr.read().decode('utf-8', 'ignore')
            rc = p.wait()

            if rc != 0:
                last_err = err[-1500:]
                print(f'[WARN] 预检: x265 不接受 "{key}"，尝试下一个拼写')
                continue

            has_md, has_cll = verify_hdr10(out)
            if has_md and has_cll:
                print(f'[INFO] 编码参数预检通过（HDR10 SEI 已确认写入，使用 {key}=1）')
                return key
            last_err = (f'编码成功但 SEI 缺失: MasteringDisplay={has_md}, '
                        f'ContentLightLevel={has_cll}\n{err[-1500:]}')
            print(f'[WARN] 预检: 使用 "{key}" 时 SEI 不完整')

        raise RuntimeError(
            '编码参数预检失败，拒绝开始长任务。\n'
            '常见原因: x265 版本过老 / master-display 字符串格式错误 / '
            'ffmpeg 未链接 libx265。\n'
            f'最后一次错误:\n{last_err}')
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# =============================================================================
# 容器 / 续传状态管理
# =============================================================================
def working_path(output_path: str) -> str:
    """
    编码中间产物强制 MKV。
    MP4 被 kill 后没有 moov，ffprobe 读不出来，续传直接失效。
    """
    base, ext = os.path.splitext(output_path)
    return output_path if ext.lower() == '.mkv' else base + '.work.mkv'


def next_part_path(work: str) -> str:
    base, ext = os.path.splitext(work)
    i = 0
    while True:
        p = f'{base}.part{i:03d}{ext}'
        if not os.path.exists(p):
            return p
        i += 1


def list_parts(work: str) -> List[str]:
    base, ext = os.path.splitext(work)
    return sorted(glob.glob(f'{base}.part[0-9][0-9][0-9]{ext}'))


def state_path(work_mkv: str) -> str:
    return work_mkv + '.state.json'


def encoder_fingerprint(md, crf, preset, hdr_opt_key, x265_extra,
                        w, h, fps_str) -> str:
    blob = json.dumps({
        'md': md, 'crf': crf, 'preset': preset, 'key': hdr_opt_key,
        'extra': x265_extra or '', 'w': w, 'h': h, 'fps': fps_str,
        'v': STATE_VERSION,
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode('utf-8')).hexdigest()[:16]


def load_state(work_mkv: str) -> Optional[dict]:
    sp = state_path(work_mkv)
    if not os.path.exists(sp):
        return None
    try:
        with open(sp, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def save_state(work_mkv: str, st: dict):
    sp = state_path(work_mkv)
    tmp = sp + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(st, f, ensure_ascii=False, indent=2)
    os.replace(tmp, sp)


def check_state_compatible(work_mkv: str, expect: dict, force: bool) -> bool:
    """
    续传前校验：源文件、总帧数、编码指纹必须一致，否则拼出来的是垃圾。
    返回 True 表示可以续传。
    """
    old = load_state(work_mkv)
    if old is None:
        return True   # 没有状态文件，按新任务处理（或旧版本产物，交给调用方决定）

    mismatch = []
    for k in ('source', 'total', 'fingerprint'):
        if old.get(k) != expect.get(k):
            mismatch.append(f'  {k}: 已有={old.get(k)!r}  当前={expect.get(k)!r}')

    if not mismatch:
        return True

    msg = '断点续传状态与当前参数不一致:\n' + '\n'.join(mismatch)
    if force:
        print(f'[WARN] {msg}\n[WARN] --force 已指定，将丢弃已有进度重新开始')
        return False
    raise RuntimeError(
        msg + '\n\n拼接不同参数编码的片段会产生损坏文件。\n'
              '请改用新的输出路径，或加 --force 丢弃已有进度重来。')


def concat_copy(file_a: str, file_b: str, out_file: str):
    """无重编码拼接两个 MKV（要求编码参数完全一致）。"""
    lst = None
    try:
        with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False,
                                         encoding='utf-8') as f:
            lst = f.name
            for p in (file_a, file_b):
                ap = os.path.abspath(p).replace('\\', '/').replace("'", "'\\''")
                f.write(f"file '{ap}'\n")

        tmp_out = out_file + '.concat_tmp.mkv'
        run_cmd(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                 '-f', 'concat', '-safe', '0', '-i', lst,
                 '-c', 'copy', '-map', '0:v:0', tmp_out],
                'Concat failed')
        os.replace(tmp_out, out_file)
    finally:
        if lst and os.path.exists(lst):
            os.remove(lst)


def reclaim_parts(work_mkv: str, fps_hint: float) -> int:
    """把所有遗留分片按顺序并回主文件。返回主文件帧数，探测失败返回 -1。"""
    parts = list_parts(work_mkv)
    for part in parts:
        n_part = probe_frame_count(part, fps_hint)
        if n_part <= 0:
            print(f'[WARN] 分片无内容，丢弃: {part}')
            os.remove(part)
            continue
        if not os.path.exists(work_mkv) or probe_frame_count(work_mkv, fps_hint) <= 0:
            os.replace(part, work_mkv)
            print(f'[INFO] 回收分片 {os.path.basename(part)}: {n_part} 帧')
            continue
        print(f'[INFO] 合并分片 {os.path.basename(part)}: +{n_part} 帧')
        concat_copy(work_mkv, part, work_mkv)
        os.remove(part)
    return probe_frame_count(work_mkv, fps_hint)


def finalize_output(work_mkv: str, output_path: str, keep_work: bool = False,
                    audio_src: Optional[str] = None):
    """
    work MKV -> 目标容器。`-c copy` 不动码流，HDR10 SEI 原样保留。
    audio_src 不为空时把源视频的音轨一并拷回：很多源片（尤其分片切割的）
    音频轨比视频轨长，容器时长由音频撑起；不拷音轨的话输出总时长变短，
    电视上播放进度会和原片对不齐。
    """
    same = os.path.abspath(work_mkv) == os.path.abspath(output_path)
    if same and not audio_src:
        return

    # 输出就是 work 文件本身（.mkv 直出）又要加音轨时，先写临时文件再替换，
    # 避免输入输出同路径互相踩踏。
    tmp_out = output_path + '.mux_tmp.mkv' if same else output_path

    print(f'[INFO] Remux -> {output_path}'
          + (f'（音轨拷自 {os.path.basename(audio_src)}）' if audio_src else ''))
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', work_mkv]
    if audio_src:
        cmd += ['-i', audio_src]
        # 1:a? ：源没有音轨时不报错；有几条拷几条（保留原始语言标签）
        # 不加 -shortest：音频比视频长时保留完整时长，和原片容器时长对齐
        cmd += ['-map', '0:v:0', '-map', '1:a?', '-c:v', 'copy', '-c:a', 'copy']
    else:
        cmd += ['-map', '0:v:0', '-c', 'copy']
    if os.path.splitext(output_path)[1].lower() == '.mp4':
        # hvc1 tag: 不加的话苹果系设备/QuickTime 不认这条 HEVC 轨
        cmd += ['-tag:v', 'hvc1', '-movflags', '+faststart']
    cmd.append(tmp_out)
    run_cmd(cmd, 'Remux failed')
    if same:
        os.replace(tmp_out, output_path)

    if not keep_work:
        os.remove(work_mkv)
        sp = state_path(work_mkv)
        if os.path.exists(sp):
            os.remove(sp)


def report_final(output_path: str):
    has_md, has_cll = verify_hdr10(output_path)
    size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
    print('')
    print('=' * 68)
    print(f'[DONE] {output_path}   ({human_size(size)})')
    print(f'       MasteringDisplay SEI : {"OK" if has_md else "*** MISSING ***"}')
    print(f'       ContentLightLevel SEI: {"OK" if has_cll else "*** MISSING ***"}')
    print('=' * 68)
    if not (has_md and has_cll):
        print('[ERROR] 输出缺少 HDR10 静态元数据，请检查 --metadata 与 x265 参数！')

# =============================================================================
# 优雅退出：收到 SIGINT/SIGTERM 时让编码器正常收尾，保证 MKV 可续传
# =============================================================================
import signal

GLOBAL_STOP = threading.Event()


class _Stopped(Exception):
    """内部信号：管线被主动中止"""
    pass


def _install_signal_handlers():
    def _handler(signum, frame):
        if not GLOBAL_STOP.is_set():
            print(f'\n[WARN] 收到信号 {signum}，正在收尾（编码器会写完 trailer，进度不会丢）...')
            GLOBAL_STOP.set()
        else:
            print('\n[WARN] 再次收到信号，强制退出')
            os._exit(130)

    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _handler)
        except (ValueError, OSError):
            pass


# =============================================================================
# 模型
# =============================================================================
def load_model(config_path: str, checkpoint_path: str, device: str = 'cuda',
               fp16: bool = False):
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
        global_channels=net_cfg.get('global_channels', 128),
    )

    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    except Exception:
        ckpt = torch.load(checkpoint_path, map_location='cpu')

    if isinstance(ckpt, dict):
        if 'params_ema' in ckpt:
            ckpt = ckpt['params_ema']
        elif 'params' in ckpt:
            ckpt = ckpt['params']
    model.load_state_dict(ckpt, strict=True)

    model.eval().to(device)
    if fp16:
        model = model.half()
        print('[INFO] FP16 半精度推理已启用')

    n_param = sum(p.numel() for p in model.parameters())
    print(f'[INFO] 模型已加载: {checkpoint_path}  ({n_param/1e6:.3f}M params)')
    return model


# =============================================================================
# GPU 侧推理 + 量化
# =============================================================================
_U16_OK: Optional[bool] = None


def _gpu_float_to_u16(y: torch.Tensor) -> np.ndarray:
    """
    y: (N,H,W,3) float，已 clamp 到 [0,65535] 并 round 过，位于 GPU 上，C 连续。
    优先在 GPU 上转 uint16 再回传（D2H 减半）；torch 版本不支持时退回 CPU 转换。
    """
    global _U16_OK
    if _U16_OK is None:
        try:
            _ = y[:1, :1, :1].to(torch.uint16).cpu().numpy()
            _U16_OK = True
            print('[INFO] 使用 GPU 侧 uint16 量化（D2H 带宽减半）')
        except Exception:
            _U16_OK = False
            print('[INFO] torch 不支持 uint16 张量，退回 CPU 侧量化')
    if _U16_OK:
        return y.to(torch.uint16).cpu().numpy()
    return y.cpu().numpy().astype(np.uint16)


def infer_rgb_u8_batch(model, batch_u8: np.ndarray, device: str,
                       fp16: bool,
                       post_map: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> np.ndarray:
    """
    batch_u8: (N,H,W,3) uint8, RGB
    return  : (N,H,W,3) uint16, RGB（模型输出已是 PQ 编码的 BT.2020 值）

    归一化、clamp、量化全部在 GPU 完成：
      上行只传 uint8（相对 float32 省 4 倍带宽）
      下行只传 uint16（相对 float32 省 2 倍带宽）
    """
    t = torch.from_numpy(batch_u8).to(device, non_blocking=True)
    x = t.permute(0, 3, 1, 2).contiguous().float().div_(255.0)
    if fp16:
        x = x.half()

    with torch.no_grad():
        y = model(x)
        if post_map is not None:
            y = post_map(y)

    y = y.float().clamp_(0.0, 1.0).mul_(65535.0).round_()
    y = y.permute(0, 2, 3, 1).contiguous()      # -> (N,H,W,3)
    return _gpu_float_to_u16(y)


# =============================================================================
# 三线程流水线
# =============================================================================
def _safe_put(q: queue.Queue, item, stop_evt: threading.Event):
    while True:
        if stop_evt.is_set() or GLOBAL_STOP.is_set():
            raise _Stopped()
        try:
            q.put(item, timeout=0.25)
            return
        except queue.Full:
            continue


def _safe_get(q: queue.Queue, stop_evt: threading.Event):
    while True:
        if stop_evt.is_set() or GLOBAL_STOP.is_set():
            raise _Stopped()
        try:
            return q.get(timeout=0.25)
        except queue.Empty:
            continue


def run_pipeline(model, produce_fn, enc_proc, batch_size: int, fp16: bool,
                 device: str, remaining: Optional[int], pbar,
                 queue_depth: int = 6,
                 post_map: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> int:
    """
    produce_fn(read_q, stop_evt, remaining) : 生产者，往 read_q 塞 (H,W,3) uint8 数组，
                                              结束时塞一个 None
    主线程负责推理，writer 线程负责写编码管道。

    内存上界 ≈ queue_depth * (24MB + 47MB) ≈ 430MB @ 4K
    """
    read_q: queue.Queue = queue.Queue(maxsize=queue_depth)
    write_q: queue.Queue = queue.Queue(maxsize=queue_depth)
    errors: Dict[str, BaseException] = {}
    stop_evt = threading.Event()

    def _producer():
        try:
            produce_fn(read_q, stop_evt, remaining)
        except _Stopped:
            pass
        except Exception as e:
            errors['producer'] = e
            stop_evt.set()
        finally:
            try:
                read_q.put(None, timeout=5)
            except queue.Full:
                pass

    def _writer():
        stdin = enc_proc.stdin
        try:
            while True:
                item = _safe_get(write_q, threading.Event())  # writer 自己不看 stop
                if item is None:
                    break
                stdin.write(item)          # ndarray 走 buffer 协议，零拷贝，不要 .tobytes()
        except _Stopped:
            pass
        except Exception as e:
            errors['writer'] = e
            stop_evt.set()

    t_prod = threading.Thread(target=_producer, name='producer', daemon=True)
    t_writ = threading.Thread(target=_writer, name='writer', daemon=True)
    t_prod.start()
    t_writ.start()

    processed = 0
    pending: List[np.ndarray] = []

    def _flush():
        nonlocal processed, pending
        if not pending:
            return
        batch = pending[0][None] if len(pending) == 1 else np.stack(pending, axis=0)
        out = infer_rgb_u8_batch(model, batch, device, fp16, post_map=post_map)
        for i in range(out.shape[0]):
            _safe_put(write_q, out[i], stop_evt)
        processed += len(pending)
        pbar.update(len(pending))
        pending = []

    interrupted = False
    try:
        while True:
            item = _safe_get(read_q, stop_evt)
            if item is None:
                break
            pending.append(item)
            if len(pending) >= batch_size:
                _flush()
        _flush()
    except _Stopped:
        interrupted = True
    finally:
        # 让 writer 把队列里剩下的写完再退出
        try:
            write_q.put(None, timeout=30)
        except queue.Full:
            pass
        t_writ.join(timeout=900)
        stop_evt.set()
        t_prod.join(timeout=10)

    if errors:
        raise RuntimeError('管线异常: ' + '; '.join(f'{k}={v!r}' for k, v in errors.items()))

    if interrupted:
        print(f'[WARN] 已中止，本次完成 {processed} 帧（进度已写入文件，可直接重跑续传）')

    return processed


# =============================================================================
# 生产者：视频解码管道
# =============================================================================
def make_video_producer(dec_proc, out_h: int, out_w: int):
    frame_bytes = out_w * out_h * 3

    def _produce(read_q: queue.Queue, stop_evt: threading.Event,
                 remaining: Optional[int]):
        stdout = dec_proc.stdout
        n = 0
        while True:
            if remaining is not None and n >= remaining:
                break
            if stop_evt.is_set() or GLOBAL_STOP.is_set():
                raise _Stopped()

            # 用 bytearray + readinto：得到可写数组，torch.from_numpy 不会告警
            buf = bytearray(frame_bytes)
            mv = memoryview(buf)
            got = 0
            while got < frame_bytes:
                k = stdout.readinto(mv[got:])
                if not k:
                    break
                got += k
            if got < frame_bytes:
                break

            _safe_put(read_q,
                      np.frombuffer(buf, dtype=np.uint8).reshape(out_h, out_w, 3),
                      stop_evt)
            n += 1

    return _produce


# =============================================================================
# 生产者：图片目录
# =============================================================================
def make_image_producer(paths: List[str], out_h: int, out_w: int):
    def _produce(read_q: queue.Queue, stop_evt: threading.Event,
                 remaining: Optional[int]):
        for i, p in enumerate(paths):
            if remaining is not None and i >= remaining:
                break
            if stop_evt.is_set() or GLOBAL_STOP.is_set():
                raise _Stopped()

            bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(f'无法读取图片: {p}')
            if bgr.shape[0] != out_h or bgr.shape[1] != out_w:
                interp = (cv2.INTER_AREA
                          if (bgr.shape[0] > out_h or bgr.shape[1] > out_w)
                          else cv2.INTER_LANCZOS4)
                bgr = cv2.resize(bgr, (out_w, out_h), interpolation=interp)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            _safe_put(read_q, np.ascontiguousarray(rgb), stop_evt)

    return _produce


# =============================================================================
# 编码器进程封装
# =============================================================================
def start_encoder(cmd: List[str], log_path: str):
    log = open(log_path, 'wb')
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=log,
                            start_new_session=True)   # Ctrl-C 只打 Python
    proc._log_fp = log          # type: ignore[attr-defined]
    return proc


def finish_encoder(proc, log_path: str):
    """关闭 stdin 让 ffmpeg 写完 trailer，然后检查返回码。"""
    try:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
    except (BrokenPipeError, OSError):
        pass
    rc = proc.wait()
    try:
        proc._log_fp.close()    # type: ignore[attr-defined]
    except Exception:
        pass
    # 255 / -2 / -15 是被信号打断，文件通常仍可用，不当作失败
    if rc not in (0, 255, -2, -15):
        tail = ''
        if os.path.exists(log_path):
            with open(log_path, 'rb') as f:
                tail = f.read()[-3000:].decode('utf-8', 'ignore')
        raise RuntimeError(f'ffmpeg 编码失败 (rc={rc}):\n{tail}')
    if rc != 0:
        print(f'[WARN] 编码器以 rc={rc} 退出（信号中断），已写入部分内容')


def stop_decoder(proc):
    try:
        if proc.stdout:
            proc.stdout.close()
    except Exception:
        pass
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    try:
        proc._log_fp.close()    # type: ignore[attr-defined]
    except Exception:
        pass


# =============================================================================
# 公共：准备续传状态
# =============================================================================
def prepare_resume(work: str, expect: dict, args, fps_float: float) -> int:
    if args.no_resume:
        # 清理所有分片
        for p in list_parts(work):
            os.remove(p)
        for p in (work, state_path(work)):
            if os.path.exists(p):
                os.remove(p)
        print('[INFO] --no_resume 已指定，从头开始')
        return 0

    if not check_state_compatible(work, expect, args.force):
        for p in list_parts(work):
            os.remove(p)
        for p in (work, state_path(work)):
            if os.path.exists(p):
                os.remove(p)
        return 0

    done = reclaim_parts(work, fps_float)
    if done < 0:
        raise RuntimeError(
            f'{work} 已存在但无法解析，拒绝覆盖以免丢数据。\n'
            f'请先人工检查该文件；确认可丢弃后用 --no_resume 重跑。')
    if done > 0:
        print(f'[INFO] 断点续传：已有 {done} 帧')
    return done


def run_and_merge(work: str, done: int, enc_cmd_builder, producer,
                  model, args, device: str, remaining: Optional[int],
                  total: Optional[int], desc: str,
                  post_map: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> int:
    """
    启动编码器 -> 跑管线 -> 收尾 -> 需要时把分片并回 work。
    返回本次处理帧数。
    """
    fresh = (done <= 0 and not os.path.exists(work))
    seg = work if fresh else next_part_path(work)
    enc_cmd = enc_cmd_builder(seg)          # 输出路径推迟到这里才确定

    enc_log = seg + '.enc.log'
    print(f'[INFO] 编码输出 -> {seg}')
    print(f'[INFO] 编码日志 -> {enc_log}')
    enc = start_encoder(enc_cmd, enc_log)

    pbar = tqdm(total=remaining, desc=desc, unit='frame', dynamic_ncols=True,
                smoothing=0.05)
    t0 = time.time()
    try:
        n = run_pipeline(model, producer, enc, max(1, args.batch_size),
                 args.fp16, device, remaining, pbar,
                 queue_depth=args.queue_depth, post_map=post_map)
    finally:
        pbar.close()
        finish_encoder(enc, enc_log)

    dt = time.time() - t0
    print(f'[INFO] 本次 {n} 帧 / {dt/60:.1f} min '
          f'({n/dt:.2f} fps)' if dt > 0 else f'[INFO] 本次 {n} 帧')

    if seg != work and os.path.exists(seg):
        n_seg = probe_frame_count(seg)
        if n_seg > 0:
            print(f'[INFO] 合并本次分片: +{n_seg} 帧')
            concat_copy(work, seg, work)
        os.remove(seg)

    return n


# =============================================================================
# 入口 1：视频
# =============================================================================
def process_video(args, model, device: str):
    ensure_out_ext(args.output_video)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_video)), exist_ok=True)

    src = probe_source(args.input_video)
    src_w, src_h = src['width'], src['height']
    src_fps, src_fps_str = src['fps'], src['fps_str']

    if args.fps is not None:
        out_fps = float(args.fps)
        out_fps_str = f'{out_fps:.10g}'
        need_fps_filter = abs(out_fps - src_fps) > 1e-6
    else:
        out_fps, out_fps_str, need_fps_filter = src_fps, src_fps_str, False

    out_h, out_w = maybe_resize_hw(src_h, src_w, args.max_size)

    total = src['total']
    if args.duration is not None:
        total = min(total, int(round(args.duration * out_fps))) if total > 0 \
            else int(round(args.duration * out_fps))
    total = total if total > 0 else None

    print(f'[INFO] 源: {src_w}x{src_h} @ {src_fps:.4f}fps ({src_fps_str}), '
          f'{total if total else "未知"} 帧')
    print(f'[INFO] 输出: {out_w}x{out_h} @ {out_fps:.4f}fps -> {args.output_video}')

    md = parse_hdr10_metadata(args.metadata)
    md = override_hdr10_md(md, peak_nits=args.peak_nits,
                           max_cll=args.max_cll, max_fall=args.max_fall)
    hdr_key = preflight_encoder(md, args.crf, args.preset, args.x265_extra, out_fps_str)

    work = working_path(args.output_video)
    if work != args.output_video and os.path.exists(args.output_video) \
            and not os.path.exists(work) and not args.force:
        print(f'[DONE] 目标文件已存在且无中间产物，视为已完成: {args.output_video}')
        report_final(args.output_video)
        return

    fp = encoder_fingerprint(md, args.crf, args.preset, hdr_key, args.x265_extra,
                             out_w, out_h, out_fps_str)
    expect = {'source': os.path.abspath(args.input_video), 'total': total,
              'fingerprint': fp}
    done = prepare_resume(work, expect, args, out_fps)

    if total is not None and done >= total:
        print(f'[INFO] 已完成 {done}/{total} 帧，直接封装')
        save_state(work, dict(expect, done=done, version=STATE_VERSION))
        finalize_output(work, args.output_video, keep_work=args.keep_work,
                       audio_src=None if args.no_audio else args.input_video)
        report_final(args.output_video)
        verify_sync(args.input_video, args.output_video)
        return

    remaining = (total - done) if total is not None else None

    dec_cmd = build_decoder_cmd(
        src=args.input_video, src_w=src_w, src_h=src_h, out_w=out_w, out_h=out_h,
        fps_str=out_fps_str, need_fps_filter=need_fps_filter, skip=done,
        hwaccel=args.hwaccel, fast_resume=args.fast_resume, fps_float=out_fps,
        in_matrix=args.in_matrix, in_range=args.in_range, sws_flags=args.sws_flags)

    enc_cmd_builder = lambda seg: build_encoder_cmd(
        out_w, out_h, out_fps_str, seg, md, args.crf, args.preset,
        hdr_key, args.x265_extra)

    if done > 0:
        mode = '-ss 时间 seek（可能 ±1 帧）' if args.fast_resume else 'select 帧级精确'
        print(f'[INFO] 从第 {done+1} 帧续传，跳帧方式: {mode}')
        if not args.fast_resume:
            print('[INFO] 正在解码并丢弃已完成部分，这一段没有进度条，请耐心等待 ...')

    dec_log = work + '.dec.log'
    dec = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE,
                           stderr=open(dec_log, 'wb'),
                           start_new_session=True)        # Ctrl-C 只打 Python
    dec._log_fp = dec.stderr    # type: ignore[attr-defined]

    # 提前写状态，确保崩溃后知道跑到了哪里
    save_state(work, dict(expect, done=done, version=STATE_VERSION))

    post_map = build_white_anchor_post_map(args)
    try:
        n = run_and_merge(work, done, enc_cmd_builder,
                          make_video_producer(dec, out_h, out_w),
                          model, args, device, remaining, total, 'SDR->HDR10',
                          post_map=post_map)
    finally:
        stop_decoder(dec)

    total_done = probe_frame_count(work, out_fps)
    save_state(work, dict(expect, done=total_done, version=STATE_VERSION))
    print(f'[INFO] 累计已编码 {total_done}'
          f'{"/" + str(total) if total else ""} 帧')

    if GLOBAL_STOP.is_set() or (total is not None and total_done < total):
        print('[INFO] 任务未完成。重跑同一条命令即可从断点继续。')
        return

    finalize_output(work, args.output_video, keep_work=args.keep_work,
                    audio_src=None if args.no_audio else args.input_video)
    report_final(args.output_video)
    verify_sync(args.input_video, args.output_video)


# =============================================================================
# 入口 2：图片目录
# =============================================================================
def process_image_dir(args, model, device: str):
    ensure_out_ext(args.output_video)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_video)), exist_ok=True)

    paths: List[str] = []
    for e in sorted(IMAGE_EXTS):
        paths += glob.glob(os.path.join(args.input_dir, '*' + e))
        paths += glob.glob(os.path.join(args.input_dir, '*' + e.upper()))
    paths = sorted(set(paths))
    if not paths:
        raise RuntimeError(f'目录中没有图片: {args.input_dir}')

    out_fps = float(args.fps) if args.fps is not None else 60.0
    out_fps_str = f'{out_fps:.10g}'
    if args.fps is None:
        print('[INFO] 目录输入未指定 --fps，使用默认 60')

    if args.duration is not None:
        paths = paths[:int(round(args.duration * out_fps))]
    total = len(paths)

    first = cv2.imread(paths[0], cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f'无法读取首帧: {paths[0]}')
    out_h, out_w = maybe_resize_hw(first.shape[0], first.shape[1], args.max_size)
    print(f'[INFO] {total} 张图片 -> {out_w}x{out_h} @ {out_fps:.4f}fps')

    md = parse_hdr10_metadata(args.metadata)
    md = override_hdr10_md(md, peak_nits=args.peak_nits,
                           max_cll=args.max_cll, max_fall=args.max_fall)
    hdr_key = preflight_encoder(md, args.crf, args.preset, args.x265_extra, out_fps_str)

    work = working_path(args.output_video)
    if work != args.output_video and os.path.exists(args.output_video) \
            and not os.path.exists(work) and not args.force:
        print(f'[DONE] 目标文件已存在: {args.output_video}')
        report_final(args.output_video)
        return

    fp = encoder_fingerprint(md, args.crf, args.preset, hdr_key, args.x265_extra,
                             out_w, out_h, out_fps_str)
    expect = {'source': os.path.abspath(args.input_dir), 'total': total,
              'fingerprint': fp}
    done = prepare_resume(work, expect, args, out_fps)

    if done >= total:
        print(f'[INFO] 已完成 {done}/{total} 帧，直接封装')
        finalize_output(work, args.output_video, keep_work=args.keep_work)
        report_final(args.output_video)
        return

    remaining = total - done
    enc_cmd_builder = lambda seg: build_encoder_cmd(
        out_w, out_h, out_fps_str, seg, md, args.crf, args.preset,
        hdr_key, args.x265_extra)

    post_map = build_white_anchor_post_map(args)
    run_and_merge(work, done, enc_cmd_builder,
                  make_image_producer(paths[done:], out_h, out_w),
                  model, args, device, remaining, total, 'IMG->HDR10',
                  post_map=post_map)

    total_done = probe_frame_count(work, out_fps)
    save_state(work, dict(expect, done=total_done, version=STATE_VERSION))

    if GLOBAL_STOP.is_set() or total_done < total:
        print(f'[INFO] 未完成 ({total_done}/{total})，重跑命令可续传')
        return

    finalize_output(work, args.output_video, keep_work=args.keep_work)
    report_final(args.output_video)


# =============================================================================
# 入口 3：单图
# =============================================================================
def process_single_image(args, model, device: str):
    bgr = cv2.imread(args.input_img, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f'无法读取: {args.input_img}')

    h, w = bgr.shape[:2]
    nh, nw = maybe_resize_hw(h, w, args.max_size)
    if (nh, nw) != (h, w):
        bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    post_map = build_white_anchor_post_map(args)
    out = infer_rgb_u8_batch(model, np.ascontiguousarray(rgb)[None],
                             device, args.fp16, post_map=post_map)[0]

    os.makedirs(os.path.dirname(os.path.abspath(args.output_img)), exist_ok=True)
    # cv2.imwrite 按 BGR 写盘，必须翻通道，否则 R/B 互换
    cv2.imwrite(args.output_img, out[:, :, ::-1])

    print(f'[DONE] 已保存: {args.output_img}')
    print(f'       shape={out.shape}, dtype={out.dtype}, '
          f'min={out.min()}, max={out.max()}')
    print('       注意: 16bit PNG 不含 HDR10 元数据，仅供数值检查，'
          '不要直接当 HDR 图看')


# =============================================================================
# 参数
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description='SDR -> HDR10 推理（高质量 / 断点续传 / mp4+mkv）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = p.add_argument_group('模型')
    g.add_argument('--config', type=str,
                   default='options/train/Custom/train_hdrnet_sdr2hdr.yml')
    g.add_argument('--checkpoint', type=str, required=True)

    g = p.add_argument_group('输入输出')
    g.add_argument('--input', type=str, default=None,
                   help='视频文件 / 图片目录 / 单张图片，自动识别')
    g.add_argument('--input_video', type=str, default=None, help='兼容旧接口')
    g.add_argument('--input_dir', type=str, default=None, help='兼容旧接口')
    g.add_argument('--input_img', type=str, default=None, help='兼容旧接口')
    g.add_argument('--output_video', type=str, default='./output_hdr10.mp4',
                   help='输出视频路径，仅支持 .mp4 / .mkv')
    g.add_argument('--output_img', type=str, default='./output_hdr.png')

    g = p.add_argument_group('推理')
    g.add_argument('--fps', type=float, default=None,
                   help='输出帧率；视频默认跟随源，目录默认 60')
    g.add_argument('--duration', type=float, default=None, help='只处理前 N 秒')
    g.add_argument('--max_size', type=int, default=None, help='最长边限制')
    g.add_argument('--batch_size', type=int, default=4)
    g.add_argument('--fp16', action='store_true')
    g.add_argument('--queue_depth', type=int, default=6,
                   help='流水线队列深度，4K 下每级约 71MB')

    g = p.add_argument_group('解码 / 色彩')
    g.add_argument('--hwaccel', type=str, default='cuda',
                   choices=['cuda', 'vaapi', 'qsv', 'none'],
                   help='解码硬件加速；精确续传要跳大量帧时强烈建议开')
    g.add_argument('--in_matrix', type=str, default='bt709',
                   choices=['bt709', 'bt601', 'smpte170m', 'bt2020'],
                   help='源视频 YUV->RGB 的矩阵，必须与训练数据一致')
    g.add_argument('--in_range', type=str, default='tv', choices=['tv', 'pc'],
                   help='源视频量化范围')
    g.add_argument('--sws_flags', type=str, default='bicubic+accurate_rnd',
                   help='swscale 标志；追求色度精度可加 +full_chroma_int')

    g = p.add_argument_group('编码质量')
    g.add_argument('--crf', type=float, default=12.0,
                   help='x265 CRF。10bit HDR 在 14 左右已到肉眼不可分辨门槛')
    g.add_argument('--preset', type=str, default='slow',
                   choices=['fast', 'medium', 'slow', 'slower', 'veryslow'])
    g.add_argument('--x265_extra', type=str, default=None,
                   help='追加 x265 参数，冒号分隔，如 "aq-mode=3:qcomp=0.75"')

    g = p.add_argument_group('HDR 元数据')
    g.add_argument('--metadata', type=str, default=None,
                   help='ffprobe 导出的参考 HDR JSON')
    g.add_argument('--peak_nits', type=float, default=4000.0,
                   help='覆盖 Mastering Display 最大亮度（nit）；仅影响元数据')
    g.add_argument('--max_cll', type=int, default=None,
                   help='覆盖 MaxCLL；默认保留解析/兜底值')
    g.add_argument('--max_fall', type=int, default=None,
                   help='覆盖 MaxFALL；默认保留解析/兜底值')

    g = p.add_argument_group('映射锚点（参考白）')
    g.add_argument('--white_anchor', type=str, default='scene',
                   choices=['scene', 'display'],
                   help='scene=按数据集漫反射白（≈660nit）不改变像素；display=把白锚到≈203nit并做高光软拐点压缩')
    g.add_argument('--scene_white_nit', type=float, default=660.0,
                   help='数据集漫反射白（nit），仅当 --white_anchor=display 时作为缩放参考')
    g.add_argument('--display_white_nit', type=float, default=203.0,
                   help='显示级参考白（nit），仅当 --white_anchor=display 时生效')
    g.add_argument('--knee_start_nit', type=float, default=1000.0,
                   help='高光软拐点起始亮度（nit），--white_anchor=display 时生效')

    g = p.add_argument_group('续传')
    g.add_argument('--no_resume', action='store_true', help='禁用续传，从头开始')
    g.add_argument('--force', action='store_true',
                   help='参数与已有进度不一致时，丢弃进度重来')
    g.add_argument('--fast_resume', action='store_true',
                   help='用 -ss 时间 seek 跳帧（快，但接缝可能 ±1 帧）')
    g.add_argument('--keep_work', action='store_true', help='保留中间 work.mkv')
    g.add_argument('--no_audio', action='store_true',
                   help='封装时不拷回源音轨；默认拷回，保证总时长与原片对齐')

    return p.parse_args()


def resolve_input(args) -> Tuple[str, str]:
    path = args.input or args.input_video or args.input_dir or args.input_img
    if path is None:
        raise ValueError('请提供 --input')
    return detect_input_type(path), path


def main():
    args = parse_args()
    _install_signal_handlers()

    kind, path = resolve_input(args)
    print(f'[INFO] 输入类型: {kind}  ({path})')
    if kind == 'video':
        args.input_video = path
    elif kind == 'dir':
        args.input_dir = path
    else:
        args.input_img = path

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        print(f'[INFO] 设备: {torch.cuda.get_device_name(0)}')
    else:
        print('[WARN] 未检测到 CUDA，使用 CPU（会非常慢）')

    model = load_model(args.config, args.checkpoint, device=device, fp16=args.fp16)

    if kind == 'img':
        process_single_image(args, model, device)
    elif kind == 'dir':
        process_image_dir(args, model, device)
    else:
        process_video(args, model, device)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n[WARN] 用户中断')
        sys.exit(130)
    except Exception as e:
        print(f'\n[ERROR] {e}', file=sys.stderr)
        sys.exit(1)