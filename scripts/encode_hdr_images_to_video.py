#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HDR 图片目录 -> HDR10 视频编码脚本（无推理版）
============================================================================

参考 scripts/infer_sdr2hdr.py 的编码链路，但输入图片本身已是
BT.2020 / PQ(SMPTE ST.2084) 编码的 16bit HDR 图，不需要任何模型推理，
直接把图片解码成 rgb48le 灌进 x265 即可。

设计要点（与 infer_sdr2hdr.py 保持一致）
----------------------------------------
1. 中间产物一律写 MKV（MP4 被杀后无 moov 不可解析），最后 `-c copy`
   remux 到目标容器，HDR10 SEI 原样保留。
2. 帧级断点续传：状态文件记录已完成帧数，重跑同一条命令自动续编。
3. 色彩转换显式指定：
     图片 -> RGB48(full, BT.2020 编码值，原样透传)
     编码端  RGB48(full) -> YUV420P10(BT.2020nc, limited)
4. 正式开跑前用 4 帧预检，确认 x265 参数可解析且 HDR10 SEI
   （Mastering Display + Content Light Level）真的写得进码流。

用法
----
python scripts/encode_hdr_images_to_video.py \
    --input_dir /mnt/data/wcm/data/hdr/train/train_hdr \
    --output_video /mnt/data/wcm/data/hdr/train/train_hdr_30fps.mp4 \
    --metadata /mnt/data/wcm/data/hdr/test/SDR/hdr10_metadata_default.json \
    --fps 30
"""

import os
import re
import sys
import glob
import json
import time
import queue
import shutil
import signal
import hashlib
import argparse
import tempfile
import threading
import subprocess
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
from tqdm import tqdm


IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
OUT_EXTS = {'.mp4', '.mkv'}

STATE_VERSION = 1

GLOBAL_STOP = threading.Event()


class _Stopped(Exception):
    """内部信号：管线被主动中止"""
    pass


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


def probe_frame_count(path: str, fps_hint: float = 0.0) -> int:
    """帧数；文件不存在返回 0；文件存在但无法解析返回 -1。"""
    if not os.path.exists(path) or os.path.getsize(path) <= 0:
        return 0
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_packets',
           '-show_entries', 'stream=nb_read_packets,nb_frames,duration,avg_frame_rate',
           '-of', 'json', path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return -1
    try:
        st = (json.loads(r.stdout).get('streams') or [{}])[0]
    except Exception:
        return -1
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


def human_size(n: float) -> str:
    for u in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024:
            return f'{n:.1f}{u}'
        n /= 1024
    return f'{n:.1f}PB'


def natural_sort_key(p: str):
    """black_2.png < black_10.png 的自然序，避免字典序把 10 排到 2 前面。"""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', os.path.basename(p))]


def collect_images(input_dir: str) -> List[str]:
    paths = []
    for root, _, files in os.walk(input_dir):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in IMAGE_EXTS:
                paths.append(os.path.join(root, fn))
    return sorted(paths, key=natural_sort_key)


# =============================================================================
# HDR10 元数据解析（与 infer_sdr2hdr.py 相同逻辑）
# =============================================================================
def parse_hdr10_metadata(metadata_json: Optional[str]) -> Tuple[str, int, int]:
    """
    返回 (master_display_str, max_cll, max_fall)
    master_display 为 x265 `--master-display` 需要的格式：
        G(x,y)B(x,y)R(x,y)WP(x,y)L(max,min)
    色度坐标单位 1/50000，亮度单位 1/10000 nit。
    """
    # 兜底：P3-D65 主显示器，1000 nits
    master = 'G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)'
    max_cll, max_fall = 1000, 400

    if not metadata_json or not os.path.exists(metadata_json):
        print('[WARN] 未提供 --metadata，使用默认 HDR10 元数据 (P3-D65 / 1000nits)')
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
                    max(1, to10k(sd.get('min_luminance'))),
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
        max_cll = int(e.get('max_content', e.get('max_luma', max_cll)))
        max_fall = int(e.get('max_average', max_fall))
        got_md = got_cll = True

    if not got_md:
        print('[WARN] JSON 中未找到 Mastering display metadata，使用默认值')
    if not got_cll:
        print('[WARN] JSON 中未找到 Content light level metadata，使用默认值')

    print(f'[INFO] MasteringDisplay = {master}')
    print(f'[INFO] MaxCLL = {max_cll}, MaxFALL = {max_fall}')
    return master, max_cll, max_fall


# =============================================================================
# ffmpeg 命令构造
# =============================================================================
def build_x265_params(md: Tuple[str, int, int], hdr_opt_key: str,
                      extra: Optional[str] = None) -> str:
    master, cll, fall = md
    parts = [
        f'{hdr_opt_key}=1',
        'repeat-headers=1',
        'colorprim=bt2020',
        'transfer=smpte2084',
        'colormatrix=bt2020nc',
        f'master-display={master}',
        f'max-cll={cll},{fall}',
        'range=limited',
    ]
    if extra:
        parts.append(extra.strip(':'))
    return ':'.join(parts)


def build_encoder_cmd(width: int, height: int, fps_str: str, out_mkv: str,
                      md: Tuple[str, int, int], crf: float, preset: str,
                      hdr_opt_key: str, x265_extra: Optional[str]) -> List[str]:
    """
    输入：rgb48le raw（16bit RGB，full range，已是 PQ / BT.2020 编码值）
    输出：HEVC Main10 / yuv420p10le / BT.2020nc limited，带完整 HDR10 SEI
    """
    return [
        'ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y',
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
        '-vf', 'scale=in_range=full:out_range=tv:out_color_matrix=bt2020:'
               'flags=accurate_rnd+full_chroma_int,format=yuv420p10le',
        '-c:v', 'libx265',
        '-profile:v', 'main10',
        '-preset', preset,
        '-crf', f'{crf:g}',
        '-x265-params', build_x265_params(md, hdr_opt_key, x265_extra),
        '-color_primaries', 'bt2020',
        '-color_trc', 'smpte2084',
        '-colorspace', 'bt2020nc',
        '-color_range', 'tv',
        '-chroma_sample_location', 'left',
        out_mkv,
    ]


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
    用 4 帧 256x256 做一次真实编码，确认 x265 参数可解析、HDR10 SEI 写得进码流，
    并自动在 hdr10-opt / hdr-opt 两个拼写间选择可用的那个。
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
            f'最后一次错误:\n{last_err}')
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# =============================================================================
# 容器 / 续传状态管理
# =============================================================================
def working_path(output_path: str) -> str:
    """编码中间产物强制 MKV，MP4 被杀后没有 moov 不可解析。"""
    base, ext = os.path.splitext(output_path)
    return output_path if ext.lower() == '.mkv' else base + '.work.mkv'


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


def prepare_resume(work: str, expect: dict, force: bool, no_resume: bool) -> int:
    """清理或校验续传状态，返回已完成帧数。"""
    def _wipe():
        for p in (work, state_path(work)):
            if os.path.exists(p):
                os.remove(p)

    if no_resume:
        _wipe()
        print('[INFO] --no_resume 已指定，从头开始')
        return 0

    old = load_state(work)
    if old is not None:
        mismatch = [f'  {k}: 已有={old.get(k)!r}  当前={expect.get(k)!r}'
                    for k in ('source', 'total', 'fingerprint')
                    if old.get(k) != expect.get(k)]
        if mismatch:
            msg = '断点续传状态与当前参数不一致:\n' + '\n'.join(mismatch)
            if not force:
                raise RuntimeError(
                    msg + '\n\n拼接不同参数编码的片段会产生损坏文件。\n'
                          '请改用新的输出路径，或加 --force 丢弃已有进度重来。')
            print(f'[WARN] {msg}\n[WARN] --force 已指定，将丢弃已有进度重新开始')
            _wipe()
            return 0

    if not os.path.exists(work):
        return 0
    done = probe_frame_count(work)
    if done < 0:
        raise RuntimeError(
            f'{work} 已存在但无法解析，拒绝覆盖以免丢数据。\n'
            f'请先人工检查该文件；确认可丢弃后用 --no_resume 重跑。')
    if done > 0:
        print(f'[INFO] 断点续传：已有 {done} 帧')
    return done


def finalize_output(work_mkv: str, output_path: str, keep_work: bool = False):
    """work MKV -> 目标容器，`-c copy` 不动码流，HDR10 SEI 原样保留。"""
    same = os.path.abspath(work_mkv) == os.path.abspath(output_path)
    if same:
        return

    print(f'[INFO] Remux -> {output_path}')
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', work_mkv,
           '-map', '0:v:0', '-c', 'copy']
    if os.path.splitext(output_path)[1].lower() == '.mp4':
        # hvc1 tag: 不加的话苹果系设备/QuickTime 不认这条 HEVC 轨
        cmd += ['-tag:v', 'hvc1', '-movflags', '+faststart']
    cmd.append(output_path)
    run_cmd(cmd, 'Remux failed')

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
# 优雅退出
# =============================================================================
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
# 图片读取（16bit 原样，8bit 放大到 16bit 保持满量程一致）
# =============================================================================
def read_image_rgb48(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    elif img.shape[2] == 4:
        img = img[:, :, :3]

    if img.dtype == np.uint8:
        # 8bit -> 16bit: v * 257（0xFF -> 0xFFFF），不用 v<<8 会损失满白
        img = img.astype(np.uint16) * 257
    elif img.dtype == np.uint16:
        pass
    else:
        img = cv2.normalize(img.astype(np.float32), None, 0, 65535,
                            cv2.NORM_MINMAX).astype(np.uint16)

    # cv2 按 BGR 读盘，喂 rgb48le 前必须翻通道
    return np.ascontiguousarray(img[:, :, ::-1])


# =============================================================================
# 管线：读图线程 + 写编码管道线程，主线程只做调度
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


def run_pipeline(paths: List[str], enc_proc, out_h: int, out_w: int,
                 queue_depth: int, pbar) -> int:
    q: queue.Queue = queue.Queue(maxsize=queue_depth)
    errors: Dict[str, BaseException] = {}
    stop_evt = threading.Event()

    def _reader():
        try:
            for p in paths:
                if stop_evt.is_set() or GLOBAL_STOP.is_set():
                    raise _Stopped()
                frame = read_image_rgb48(p)
                if frame is None:
                    raise FileNotFoundError(f'无法读取图片: {p}')
                if frame.shape[0] != out_h or frame.shape[1] != out_w:
                    raise ValueError(
                        f'图片尺寸 {frame.shape[1]}x{frame.shape[0]} 与首帧 '
                        f'{out_w}x{out_h} 不一致: {p}')
                _safe_put(q, frame, stop_evt)
        except _Stopped:
            pass
        except Exception as e:
            errors['reader'] = e
            stop_evt.set()
        finally:
            try:
                q.put(None, timeout=5)
            except queue.Full:
                pass

    def _writer():
        stdin = enc_proc.stdin
        try:
            while True:
                item = _safe_get(q, threading.Event())
                if item is None:
                    break
                stdin.write(item)          # ndarray 走 buffer 协议，零拷贝
                pbar.update(1)
        except _Stopped:
            pass
        except Exception as e:
            errors['writer'] = e
            stop_evt.set()

    t_read = threading.Thread(target=_reader, name='reader', daemon=True)
    t_writ = threading.Thread(target=_writer, name='writer', daemon=True)
    t_read.start()
    t_writ.start()
    t_read.join()
    t_writ.join()

    if errors:
        raise RuntimeError('管线异常: ' + '; '.join(f'{k}={v!r}' for k, v in errors.items()))
    return pbar.n


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
    if rc not in (0, 255, -2, -15):
        tail = ''
        if os.path.exists(log_path):
            with open(log_path, 'rb') as f:
                tail = f.read()[-3000:].decode('utf-8', 'ignore')
        raise RuntimeError(f'ffmpeg 编码失败 (rc={rc}):\n{tail}')
    if rc != 0:
        print(f'[WARN] 编码器以 rc={rc} 退出（信号中断），已写入部分内容')


# =============================================================================
# 主流程
# =============================================================================
def main():
    p = argparse.ArgumentParser(
        description='HDR 图片目录 -> 30Hz HDR10 视频（无推理，直接编码）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--input_dir', type=str, required=True,
                   help='HDR 图片目录（16bit PQ/BT.2020 PNG）')
    p.add_argument('--output_video', type=str, required=True,
                   help='输出视频路径，仅支持 .mp4 / .mkv')
    p.add_argument('--metadata', type=str, default=None,
                   help='ffprobe 导出的参考 HDR JSON')
    p.add_argument('--fps', type=float, default=30.0, help='输出帧率')
    p.add_argument('--crf', type=float, default=12.0,
                   help='x265 CRF。10bit HDR 在 14 左右已到肉眼不可分辨门槛')
    p.add_argument('--preset', type=str, default='slow',
                   choices=['fast', 'medium', 'slow', 'slower', 'veryslow'])
    p.add_argument('--x265_extra', type=str, default=None,
                   help='追加 x265 参数，冒号分隔，如 "aq-mode=3:qcomp=0.75"')
    p.add_argument('--queue_depth', type=int, default=6,
                   help='读图-编码队列深度，4K 下每级约 24MB')
    p.add_argument('--no_resume', action='store_true', help='禁用续传，从头开始')
    p.add_argument('--force', action='store_true',
                   help='参数与已有进度不一致时，丢弃进度重来')
    p.add_argument('--keep_work', action='store_true', help='保留中间 work.mkv')
    args = p.parse_args()

    _install_signal_handlers()

    if os.path.splitext(args.output_video)[1].lower() not in OUT_EXTS:
        raise ValueError(f'--output_video 只支持 .mp4 / .mkv，当前为: {args.output_video}')
    os.makedirs(os.path.dirname(os.path.abspath(args.output_video)), exist_ok=True)

    paths = collect_images(args.input_dir)
    if not paths:
        raise RuntimeError(f'目录中没有图片: {args.input_dir}')
    total = len(paths)

    first = read_image_rgb48(paths[0])
    if first is None:
        raise RuntimeError(f'无法读取首帧: {paths[0]}')
    out_h, out_w = first.shape[:2]
    if out_w % 2 or out_h % 2:
        raise ValueError(f'分辨率 {out_w}x{out_h} 非偶数，无法做 4:2:0 编码')

    out_fps_str = f'{float(args.fps):.10g}'
    print(f'[INFO] {total} 张图片 ({out_w}x{out_h}, uint16) '
          f'-> {out_fps_str}fps -> {args.output_video}')

    md = parse_hdr10_metadata(args.metadata)
    hdr_key = preflight_encoder(md, args.crf, args.preset, args.x265_extra, out_fps_str)

    work = working_path(args.output_video)
    if work != args.output_video and os.path.exists(args.output_video) \
            and not os.path.exists(work) and not args.force:
        print(f'[DONE] 目标文件已存在且无中间产物，视为已完成: {args.output_video}')
        report_final(args.output_video)
        return

    fp = encoder_fingerprint(md, args.crf, args.preset, hdr_key, args.x265_extra,
                             out_w, out_h, out_fps_str)
    expect = {'source': os.path.abspath(args.input_dir), 'total': total,
              'fingerprint': fp}
    done = prepare_resume(work, expect, args.force, args.no_resume)

    if done >= total:
        print(f'[INFO] 已完成 {done}/{total} 帧，直接封装')
        finalize_output(work, args.output_video, keep_work=args.keep_work)
        report_final(args.output_video)
        return

    # 续传时把分段输出到 part 文件，编完再无损拼接，避免破坏已有进度
    fresh = (done <= 0 and not os.path.exists(work))
    if fresh:
        seg = work
    else:
        base, ext = os.path.splitext(work)
        i = 0
        while os.path.exists(f'{base}.part{i:03d}{ext}'):
            i += 1
        seg = f'{base}.part{i:03d}{ext}'

    enc_cmd = build_encoder_cmd(out_w, out_h, out_fps_str, seg, md,
                                args.crf, args.preset, hdr_key, args.x265_extra)
    enc_log = seg + '.enc.log'
    print(f'[INFO] 编码输出 -> {seg}')
    print(f'[INFO] 编码日志 -> {enc_log}')
    enc = start_encoder(enc_cmd, enc_log)

    remaining = total - done
    pbar = tqdm(total=remaining, desc='IMG->HDR10', unit='frame',
                dynamic_ncols=True, smoothing=0.05)
    t0 = time.time()
    try:
        run_pipeline(paths[done:], enc, out_h, out_w, args.queue_depth, pbar)
    finally:
        pbar.close()
        finish_encoder(enc, enc_log)

    n = probe_frame_count(seg, float(args.fps))
    dt = time.time() - t0
    print(f'[INFO] 本次 {n} 帧 / {dt/60:.1f} min '
          f'({n/dt:.2f} fps)' if dt > 0 else f'[INFO] 本次 {n} 帧')

    if seg != work and os.path.exists(seg):
        if n > 0:
            print(f'[INFO] 合并本次分片: +{n} 帧')
            lst = None
            try:
                with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False,
                                                 encoding='utf-8') as f:
                    lst = f.name
                    for pp in (work, seg):
                        f.write(f"file '{os.path.abspath(pp)}'\n")
                tmp_out = work + '.concat_tmp.mkv'
                run_cmd(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                         '-f', 'concat', '-safe', '0', '-i', lst,
                         '-c', 'copy', '-map', '0:v:0', tmp_out],
                        'Concat failed')
                os.replace(tmp_out, work)
            finally:
                if lst and os.path.exists(lst):
                    os.remove(lst)
        os.remove(seg)

    total_done = probe_frame_count(work, float(args.fps))
    save_state(work, dict(expect, done=total_done, version=STATE_VERSION))
    print(f'[INFO] 累计已编码 {total_done}/{total} 帧')

    if GLOBAL_STOP.is_set() or total_done < total:
        print('[INFO] 任务未完成。重跑同一条命令即可从断点继续。')
        return

    finalize_output(work, args.output_video, keep_work=args.keep_work)
    report_final(args.output_video)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n[WARN] 用户中断')
        sys.exit(130)
    except Exception as e:
        print(f'\n[ERROR] {e}', file=sys.stderr)
        sys.exit(1)
