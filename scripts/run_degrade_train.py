'''
    批量退化训练集: train_hdr (uint16 HDR) -> train_sdr_degrade (uint8 SDR)
    退化流程与参数与 hdr2sdr.py 保持一致, 仅在末尾增加 16bit->8bit 线性位深换算
    注意: 为保证映射可复现, 请勿修改本文件中的参数和 seed
'''
import sys
sys.path.insert(0, '/mnt/data/wcm/HDRNet/scripts')

from hdr2sdr import main


if __name__ == "__main__":

    in_root = '/mnt/data/datasets/HDR/training_set/train_hdr'
    out_root = '/mnt/data/datasets/HDR/training_set/train_sdr_degrade'

    gennerate_parmas = {
        # 高斯模糊
        'blur_ksize': 7,
        'blur_sigma': (0.5, 2.0),
        # 降采样后升回原尺寸
        'down_scale': (0.25, 0.75),
        'resize_back': True,
        'interp': 'bicubic',
        # 高斯噪声
        'noise_sigma': (1.0, 8.0),
        # jpeg 压缩 (转 8bit 后生效)
        'jpeg_quality': (30, 90),
        # HDR -> SDR: hable tonemap + bt709 gamma, peak_luma 控制高光压缩强度(对应 1000nits/100nits)
        'to_8bit': True,
        'peak_luma': 10.0,
        # 随机种子 (与之前保持一致)
        'seed': 42,
    }

    main(in_root, out_root, gennerate_parmas)

    print('over')
