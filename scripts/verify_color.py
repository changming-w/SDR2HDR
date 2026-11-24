#!/usr/bin/env python
"""
验证色彩正确性：比较模型原始 uint16 输出 vs MKV 解码后的 rgb48

用法:
    python scripts/verify_color.py \
        --input /path/to/input_sdr.mp4 \
        --output /path/to/output_hdr.mkv \
        --config options/train/Custom/train_hdrnet_bt709_sdr2hdr.yml \
        --checkpoint experiments/train_hdrnet_bt709_sdr2hdr/models/net_g_175000.pth \
        --frame_idx 0 \
        --fp16
"""
import argparse
import os
import subprocess
import sys
import numpy as np
import cv2
import torch

# 加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basicsr.utils import img2tensor


def load_model(config_path, checkpoint_path, device, fp16=False):
    """加载模型"""
    from basicsr.utils.options import parse
    from basicsr.archs import build_network

    opt = parse(config_path, is_train=False)
    opt['checkpoint_path'] = checkpoint_path

    # 构建模型
    net = build_network(opt['network_g'])
    state = torch.load(checkpoint_path, map_location='cpu')
    if 'params_ema' in state:
        state = state['params_ema']
    elif 'params' in state:
        state = state['params']
    net.load_state_dict(state, strict=True)
    net.eval().to(device)
    if fp16:
        net = net.half()
    print(f'[INFO] 模型已加载: {checkpoint_path}')
    return net


def extract_frame_from_mkv(mkv_path, frame_idx, output_raw):
    """从 MKV 提取指定帧为 rgb48le raw"""
    cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error',
        '-i', mkv_path,
        '-vf', f"select='eq(n\\,{frame_idx})',scale=in_range=tv:in_color_matrix=bt2020:out_range=full",
        '-pix_fmt', 'rgb48le',
        '-frames:v', '1',
        '-f', 'rawvideo',
        output_raw
    ]
    subprocess.run(cmd, check=True)

    # 读取 raw 数据
    data = np.fromfile(output_raw, dtype=np.uint16)
    # 需要知道分辨率
    return data


def get_video_resolution(video_path):
    """获取视频分辨率"""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'csv=p=0:nk=1',
        video_path
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    w, h = r.stdout.strip().split(',')
    return int(w), int(h)


def run_model_on_frame(net, input_path, frame_idx, device, fp16=False):
    """对输入视频的单帧跑模型，返回 uint16 RGB"""
    # 读取帧
    cap = cv2.VideoCapture(input_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, bgr = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f'无法读取帧 {frame_idx}: {input_path}')

    # BGR -> RGB, HWC -> CHW, 归一化到 [0, 1]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_float = rgb.astype(np.float32) / 255.0

    # 转 tensor
    img_t = img2tensor(rgb_float, bgr2rgb=False, float32=True)
    img_t = img_t.unsqueeze(0).to(device)

    if fp16:
        img_t = img_t.half()

    # 推理
    with torch.no_grad():
        out = net(img_t)

    # 转回 uint16
    out = out.float().clamp_(0.0, 1.0).mul_(65535.0).round_()
    out_np = out[0].permute(1, 2, 0).cpu().numpy().astype(np.uint16)  # HWC
    return out_np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='输入 SDR 视频')
    parser.add_argument('--output', required=True, help='输出 HDR MKV')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--frame_idx', type=int, default=0, help='验证第几帧')
    parser.add_argument('--fp16', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. 获取分辨率
    w, h = get_video_resolution(args.output)
    print(f'[INFO] 分辨率: {w}x{h}')

    # 2. 从 MKV 解码一帧
    raw_path = '/tmp/verify_decoded.raw'
    print(f'[INFO] 从 MKV 解码第 {args.frame_idx} 帧 ...')
    decoded_data = extract_frame_from_mkv(args.output, args.frame_idx, raw_path)
    decoded = decoded_data.reshape(h, w, 3)
    print(f'[INFO] 解码完成: shape={decoded.shape}, dtype={decoded.dtype}')
    print(f'[INFO] 解码值范围: R[{decoded[:,:,0].min()}-{decoded[:,:,0].max()}] '
          f'G[{decoded[:,:,1].min()}-{decoded[:,:,1].max()}] '
          f'B[{decoded[:,:,2].min()}-{decoded[:,:,2].max()}]')

    # 3. 跑模型
    print(f'[INFO] 加载模型 ...')
    net = load_model(args.config, args.checkpoint, device, args.fp16)

    print(f'[INFO] 对第 {args.frame_idx} 帧跑模型 ...')
    model_out = run_model_on_frame(net, args.input, args.frame_idx, device, args.fp16)
    print(f'[INFO] 模型输出: shape={model_out.shape}, dtype={model_out.dtype}')
    print(f'[INFO] 模型值范围: R[{model_out[:,:,0].min()}-{model_out[:,:,0].max()}] '
          f'G[{model_out[:,:,1].min()}-{model_out[:,:,1].max()}] '
          f'B[{model_out[:,:,2].min()}-{model_out[:,:,2].max()}]')

    # 4. 比较
    diff = decoded.astype(np.int32) - model_out.astype(np.int32)
    abs_diff = np.abs(diff)

    print('\n' + '='*60)
    print('色彩正确性验证结果')
    print('='*60)
    print(f'平均差异: {abs_diff.mean():.2f} 码值')
    print(f'最大差异: {abs_diff.max()} 码值')
    print(f'差异分布:')
    for thresh in [10, 50, 100, 500, 1000]:
        pct = (abs_diff <= thresh).mean() * 100
        print(f'  ≤{thresh:4d}: {pct:.2f}%')

    print('\n通道平均差异:')
    for c, name in enumerate(['R', 'G', 'B']):
        print(f'  {name}: {abs_diff[:,:,c].mean():.2f}')

    # 5. 判断
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()

    if max_diff < 50 and mean_diff < 10:
        print('\n[OK] 色彩正确！差异在 4:2:0 + 有损编码的正常范围内 (±50 码)')
    elif max_diff < 200:
        print('\n[WARN] 差异偏大，但可能是 4:2:0 色度子采样导致的边缘差异')
        print('       建议检查多个帧确认')
    else:
        print('\n[ERROR] 差异过大！可能原因:')
        print('  1. 色彩矩阵接错（检查 scale 滤镜的 in_color_matrix/out_color_matrix）')
        print('  2. 范围转换错误（tv vs full）')
        print('  3. 模型输出不是 BT.2020 PQ')

    # 清理
    if os.path.exists(raw_path):
        os.remove(raw_path)


if __name__ == '__main__':
    main()
