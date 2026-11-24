#!/usr/bin/env python
"""计算 HDR 视频的 MaxCLL 和 MaxFALL"""
import subprocess
import numpy as np
import sys

def pq_to_nits(pq_value):
    """PQ (SMPTE 2084) 转换为线性亮度 (nits)"""
    # PQ 逆 EOTF
    m1 = 0.1593017578125
    m2 = 78.84375
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875

    pq = pq_value / 10000.0  # 归一化到 [0, 1]

    # PQ 逆 EOTF
    num = np.maximum(pq ** (1/m2) - c1, 0)
    den = c2 - c3 * (pq ** (1/m2))

    if den <= 0:
        return 0

    linear = 10000 * (num / den) ** (1/m1)
    return linear

def analyze_hdr_video(video_path, sample_interval=1):
    """分析 HDR 视频,计算 MaxCLL 和 MaxFALL

    Args:
        video_path: 视频路径
        sample_interval: 采样间隔,1=每帧,10=每10帧取1帧
    """
    print(f"分析视频: {video_path}")
    print(f"采样间隔: 每 {sample_interval} 帧")

    # 使用 ffmpeg 解码为 rawvideo,保持 PQ 编码
    cmd = [
        'ffmpeg', '-i', video_path,
        '-pix_fmt', 'yuv420p10le',
        '-f', 'rawvideo', '-'
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # 获取视频信息
    probe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                 '-show_entries', 'stream=width,height,nb_frames,r_frame_rate',
                 '-of', 'json', video_path]
    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
    import json
    info = json.loads(probe_result.stdout)
    stream = info['streams'][0]
    width = int(stream['width'])
    height = int(stream['height'])
    total_frames = int(stream.get('nb_frames', 0))
    fps = stream['r_frame_rate']

    print(f"分辨率: {width}x{height}, 帧率: {fps}, 总帧数: {total_frames}")

    # 每帧字节数 (10-bit YUV420)
    frame_bytes = width * height * 3 // 2 * 2  # 2 bytes per pixel for 10-bit

    global_maxcll = 0
    global_maxfall = 0
    frame_count = 0
    processed_count = 0

    try:
        while True:
            # 读取一帧
            frame_data = proc.stdout.read(frame_bytes)
            if not frame_data or len(frame_data) < frame_bytes:
                break

            frame_count += 1

            # 采样:跳过不分析的帧
            if frame_count % sample_interval != 0:
                continue

            processed_count += 1

            # 解析为 numpy 数组
            frame = np.frombuffer(frame_data, dtype=np.uint16)
            frame = frame.reshape(height * 3 // 2, width)

            # 提取 Y 通道 (前 height 行)
            y_channel = frame[:height, :]

            # 转换为 nits
            y_nits = pq_to_nits(y_channel.astype(np.float32))

            # 计算该帧的 MaxCLL 和 MaxFALL
            frame_maxcll = np.max(y_nits)
            frame_maxfall = np.mean(y_nits)

            # 更新全局最大值
            global_maxcll = max(global_maxcll, frame_maxcll)
            global_maxfall = max(global_maxfall, frame_maxfall)

            if processed_count % 100 == 0:
                print(f"  已处理 {processed_count} 帧 (跳过 {frame_count - processed_count} 帧), "
                      f"当前 MaxCLL={global_maxcll:.1f}, MaxFALL={global_maxfall:.1f}")

    finally:
        proc.terminate()
        proc.wait()

    print(f"\n分析完成!")
    print(f"总帧数: {frame_count}, 分析帧数: {processed_count}")
    print(f"MaxCLL: {global_maxcll:.1f} nits")
    print(f"MaxFALL: {global_maxfall:.1f} nits")

    return int(round(global_maxcll)), int(round(global_maxfall))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python calc_hdr_metadata.py <video_path> [sample_interval]")
        print("示例: python calc_hdr_metadata.py video.mp4 10  # 每10帧取1帧")
        sys.exit(1)

    video_path = sys.argv[1]
    sample_interval = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    maxcll, maxfall = analyze_hdr_video(video_path, sample_interval)
    print(f"\n建议设置:")
    print(f"  --max_content {maxcll}")
    print(f"  --max_average {maxfall}")
