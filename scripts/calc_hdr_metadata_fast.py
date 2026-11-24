#!/usr/bin/env python3
"""快速计算 HDR 视频的 MaxCLL 和 MaxFALL (使用 ffmpeg hdr_metadata 滤镜)"""
import subprocess
import json
import sys
import re

def analyze_hdr_video_fast(video_path, sample_fps=1):
    """使用 ffmpeg 快速分析 HDR 视频元数据"""
    print(f"分析视频: {video_path}")
    print(f"采样帧率: {sample_fps} fps")

    # 先用 ffprobe 获取视频信息
    probe_cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate,nb_frames,duration',
        '-of', 'json', video_path
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    stream = info['streams'][0]

    width = int(stream['width'])
    height = int(stream['height'])
    duration = float(stream.get('duration', 0))
    fps_str = stream.get('r_frame_rate', '60/1')

    print(f"分辨率: {width}x{height}, 时长: {duration:.1f}s, 帧率: {fps_str}")

    # 计算采样间隔
    fps_parts = fps_str.split('/')
    fps = int(fps_parts[0]) / int(fps_parts[1]) if '/' in fps_str else float(fps_str)
    sample_interval = max(1, int(fps / sample_fps))

    # 使用 ffmpeg 的 hdr_metadata 滤镜提取元数据
    # 先用 select 滤镜降采样到每秒 1 帧
    cmd = [
        'ffmpeg', '-i', video_path,
        '-vf', f"select='not(mod(n\\,{sample_interval}))',hdr_metadata",
        '-f', 'null', '-'
    ]

    print("正在分析...")
    print(f"  采样间隔: 每 {sample_interval} 帧 (约 {sample_fps} fps)")
    print(f"  预计处理帧数: ~{int(duration * sample_fps)} 帧")
    sys.stdout.flush()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )

    maxcll_values = []
    maxfall_values = []

    # 解析 stderr 中的 hdr_metadata 输出
    for line in proc.stderr:
        if 'MaxCLL' in line or 'max_content' in line.lower():
            # 提取数值
            match = re.search(r'(?:MaxCLL|max_content)[=:\s]+(\d+)', line, re.IGNORECASE)
            if match:
                maxcll_values.append(int(match.group(1)))

        if 'MaxFALL' in line or 'max_average' in line.lower():
            match = re.search(r'(?:MaxFALL|max_average)[=:\s]+(\d+)', line, re.IGNORECASE)
            if match:
                maxfall_values.append(int(match.group(1)))

        # 进度提示
        if 'frame=' in line:
            match = re.search(r'frame=\s*(\d+)', line)
            if match:
                frame_num = int(match.group(1))
                if frame_num % 100 == 0:
                    print(f"  已处理 {frame_num} 帧...")

    proc.wait()

    # 如果没有从滤镜输出中提取到,使用备选方案
    if not maxcll_values or not maxfall_values:
        print("使用备选方案: 直接分析 PQ 值...")
        return analyze_pq_values(video_path, sample_fps)

    maxcll = max(maxcll_values) if maxcll_values else 0
    maxfall = max(maxfall_values) if maxfall_values else 0

    print(f"\n分析完成!")
    print(f"MaxCLL: {maxcll} nits")
    print(f"MaxFALL: {maxfall} nits")

    return maxcll, maxfall


def analyze_pq_values(video_path, sample_fps=1):
    """备选方案: 直接分析 PQ 编码值"""
    import numpy as np

    # 获取视频信息
    probe_cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate',
        '-of', 'json', video_path
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    stream = info['streams'][0]

    width = int(stream['width'])
    height = int(stream['height'])
    fps_parts = stream.get('r_frame_rate', '60/1').split('/')
    fps = int(fps_parts[0]) / int(fps_parts[1]) if '/' in fps_parts else float(fps_parts[0])

    # 计算采样间隔
    sample_interval = max(1, int(fps / sample_fps))

    # 使用 ffmpeg 解码,只取 Y 通道,降采样
    cmd = [
        'ffmpeg', '-i', video_path,
        '-vf', f"select='not(mod(n\\,{sample_interval}))',extractplanes=y",
        '-pix_fmt', 'gray16le',
        '-f', 'rawvideo', '-'
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    frame_bytes = width * height * 2  # 16-bit grayscale
    maxcll = 0
    maxfall = 0
    frame_count = 0

    # PQ 逆 EOTF 查找表 (加速)
    pq_lut = np.zeros(65536, dtype=np.float32)
    m1 = 0.1593017578125
    m2 = 78.84375
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875

    for i in range(65536):
        pq = i / 10000.0
        num = max(pq ** (1/m2) - c1, 0)
        den = c2 - c3 * (pq ** (1/m2))
        if den > 0:
            pq_lut[i] = 10000 * (num / den) ** (1/m1)

    try:
        while True:
            frame_data = proc.stdout.read(frame_bytes)
            if not frame_data or len(frame_data) < frame_bytes:
                break

            frame_count += 1
            frame = np.frombuffer(frame_data, dtype=np.uint16)

            # 使用查找表转换
            nits = pq_lut[frame]

            frame_max = np.max(nits)
            frame_avg = np.mean(nits)

            maxcll = max(maxcll, frame_max)
            maxfall = max(maxfall, frame_avg)

            if frame_count % 10 == 0:
                print(f"  已处理 {frame_count} 帧, 当前 MaxCLL={maxcll:.1f}, MaxFALL={maxfall:.1f}", flush=True)

    finally:
        proc.terminate()
        proc.wait()

    print(f"\n分析完成!")
    print(f"处理帧数: {frame_count}")
    print(f"MaxCLL: {maxcll:.1f} nits")
    print(f"MaxFALL: {maxfall:.1f} nits")

    return int(round(maxcll)), int(round(maxfall))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python calc_hdr_metadata_fast.py <video_path> [sample_fps]")
        print("示例: python calc_hdr_metadata_fast.py video.mp4 1  # 每秒1帧")
        sys.exit(1)

    video_path = sys.argv[1]
    sample_fps = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    maxcll, maxfall = analyze_hdr_video_fast(video_path, sample_fps)
    print(f"\n建议设置:")
    print(f"  MaxCLL (max_content): {maxcll}")
    print(f"  MaxFALL (max_average): {maxfall}")
