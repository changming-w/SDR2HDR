'''
    批量 PQ 编码: train_hdr (线性光 16bit) -> train_hdr_pq (PQ 编码 16bit)
    纯 numpy 计算(标准 ST.2084 OETF, 1.0=10000nits), 与 img_hable 退化链路的前置步骤完全相同
    注意: 输出 PNG 无颜色标记, 后续使用时需显式指定:
          -color_trc smpte2084 -color_primaries bt2020 -colorspace bt2020nc
'''
import os
import sys
import glob
sys.path.insert(0, '/mnt/data/wcm/HDRNet/scripts')

import cv2
import numpy as np
from dhr2sdr import linear_to_pq16


if __name__ == "__main__":

    in_root = '/mnt/data/datasets/HDR/training_set/train_hdr'
    out_root = '/mnt/data/datasets/HDR/training_set/train_hdr_pq'

    in_paths = sorted(glob.glob(os.path.join(in_root, '*.png')))
    print('Debug : found img ==== ', len(in_paths))

    if not os.path.exists(out_root):
        os.makedirs(out_root)

    for iid, in_path in enumerate(in_paths):
        in_name = os.path.splitext(os.path.split(in_path)[-1])[0]
        out_path = os.path.join(out_root, in_name + '.png')
        if os.path.exists(out_path):
            continue
        print('Debug : process %d/%d == ' % (iid + 1, len(in_paths)), in_name)
        img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
        if img is None or img.dtype != np.uint16:
            print('Alarm : read failed or not 16bit == ', in_path)
            continue
        cv2.imwrite(out_path, linear_to_pq16(img))

    print('over')
