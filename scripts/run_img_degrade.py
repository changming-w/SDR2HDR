'''
    批量图片退化: train_hdr (线性光 16bit) -> train_sdr_hable (8bit SDR)
    使用 dhr2sdr.py 的 img_hable 模式, 退化链路与视频退化完全一致
    (已用真实 HDR/SDR 视频对验证, 复现精度 46dB)
'''
import sys
sys.path.insert(0, '/mnt/data/wcm/HDRNet/scripts')

from dhr2sdr import main


if __name__ == "__main__":

    in_root = '/mnt/data/datasets/HDR/training_set/train_hdr'
    out_root = '/mnt/data/datasets/HDR/training_set/train_sdr_hable'

    gennerate_parmas = {
        'mode': 'img_hable',
        'ffmpeg_path': 'ffmpeg',  # Linux 系统 ffmpeg
        # 同时保存 PQ 编码的 HDR 图 (无需重跑时可设为 None)
        'pq_root': '/mnt/data/datasets/HDR/training_set/train_hdr_pq',
    }

    main(in_root, in_root, out_root, gennerate_parmas)

    print('over')
