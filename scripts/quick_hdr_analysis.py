#!/usr/bin/env python3
"""快速计算 HDR 视频的 MaxCLL 和 MaxFALL"""
import subprocess, numpy as np, sys, json

video = sys.argv[1] if len(sys.argv) > 1 else None
sample_fps = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

if not video:
    print("用法: python quick_hdr_analysis.py <video> [sample_fps]")
    sys.exit(1)

# 获取视频信息
probe = subprocess.run([
    'ffprobe', '-v', 'error', '-select_streams', 'v:0',
    '-show_entries', 'stream=width,height,r_frame_rate,duration',
    '-of', 'json', video
], capture_output=True, text=True)
info = json.loads(probe.stdout)['streams'][0]
w, h = int(info['width']), int(info['height'])
dur = float(info.get('duration', 0))
fps_str = info.get('r_frame_rate', '60/1')
fps_parts = fps_str.split('/')
fps = int(fps_parts[0]) / int(fps_parts[1]) if '/' in fps_str else float(fps_str)
interval = max(1, int(fps / sample_fps))
total_est = int(dur * sample_fps)

print(f"视频: {w}x{h}, {dur:.0f}s, {fps:.1f}fps", flush=True)
print(f"采样: 每{interval}帧(={sample_fps}fps), 预计~{total_est}帧", flush=True)

# ffmpeg: 用 -r 控制输出帧率 + 缩放到320宽 + 转灰度16bit
cmd = [
    'ffmpeg', '-i', video,
    '-r', str(sample_fps),
    '-vf', 'scale=320:-1',
    '-pix_fmt', 'gray16le', '-f', 'rawvideo', '-'
]
print(f"命令: {' '.join(cmd[:6])}...", flush=True)

proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# PQ LUT (输入: 16-bit 值, 输出: nits)
pq_lut = np.zeros(65536, dtype=np.float32)
m1, m2, c1, c2, c3 = 0.1593017578125, 78.84375, 0.8359375, 18.8515625, 18.6875
for i in range(65536):
    pq = i / 65535.0  # 16-bit -> 0-1 PQ值
    num = max(pq ** (1/m2) - c1, 0)
    den = c2 - c3 * (pq ** (1/m2))
    if den > 0:
        pq_lut[i] = 10000 * (num / den) ** (1/m1)

# 缩放后的帧大小
sw = 320
sh = int(h * 320 / w)
frame_bytes = sw * sh * 2

maxcll = 0.0
maxfall = 0.0
n = 0

while True:
    data = proc.stdout.read(frame_bytes)
    if not data or len(data) < frame_bytes:
        break
    n += 1
    frame = np.frombuffer(data[:frame_bytes], dtype=np.uint16)
    nits = pq_lut[frame]
    fmax = float(np.max(nits))
    favg = float(np.mean(nits))
    if fmax > maxcll:
        maxcll = fmax
    if favg > maxfall:
        maxfall = favg
    if n % 100 == 0:
        print(f"  [{n}/{total_est}] MaxCLL={maxcll:.0f} MaxFALL={maxfall:.0f}", flush=True)

proc.terminate()
proc.wait()

print(f"\n===== 结果 =====", flush=True)
print(f"分析帧数: {n}", flush=True)
print(f"MaxCLL:  {maxcll:.0f} nits", flush=True)
print(f"MaxFALL: {maxfall:.0f} nits", flush=True)
