import os
import cv2
import lmdb
import pickle
from os import path as osp
from tqdm import tqdm

from basicsr.utils import scandir


def make_lmdb(folder_path, lmdb_path, suffix='.png', compress_level=3):
    img_path_list = sorted(list(scandir(folder_path, suffix=suffix, recursive=False, full_path=True)))
    if len(img_path_list) == 0:
        raise ValueError(f'No images found in {folder_path} with suffix {suffix}')

    print(f'Found {len(img_path_list)} images in {folder_path}')

    # lmdb key = 文件名去掉后缀，例如 000001
    keys = [osp.splitext(osp.basename(v))[0] for v in img_path_list]

    # 估算大小
    img = cv2.imread(img_path_list[0], cv2.IMREAD_UNCHANGED)
    H, W = img.shape[:2]
    C = 1 if img.ndim == 2 else img.shape[2]
    data_size_per_img = H * W * C * 100
    map_size = int(data_size_per_img * len(img_path_list))

    os.makedirs(lmdb_path, exist_ok=True) if lmdb_path.endswith('.lmdb') else os.makedirs(
        osp.dirname(lmdb_path), exist_ok=True)

    env = lmdb.open(lmdb_path, map_size=map_size)
    print(f'Creating {lmdb_path} ...')

    with env.begin(write=True) as txn:
        for path, key in tqdm(zip(img_path_list, keys), total=len(keys), desc=f'Write {osp.basename(lmdb_path)}'):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f'Warning: read fail: {path}')
                continue
            # 注意：这里存的是编码后的 PNG 字节流
            _, img_byte = cv2.imencode(suffix, img, [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
            txn.put(key.encode('ascii'), img_byte)

    env.sync()
    env.close()
    return keys, (H, W, C)


def write_meta_txt(lmdb_path, keys, res, compress_level=3, suffix='.png'):
    """写成 BasicSR 预期的 meta_info.txt 格式：
       000001.png (H,W,C) compress_level
    """
    h, w, c = res
    meta_path = osp.join(lmdb_path, 'meta_info.txt')
    print(f'Writing meta_info.txt to {meta_path}')

    with open(meta_path, 'w') as f:
        for k in keys:
            img_name = k + suffix  # 000001 + .png -> 000001.png
            f.write(f'{img_name} ({h},{w},{c}) {compress_level}\n')


if __name__ == '__main__':
    # 你的数据路径
    # lq_folder = '/mnt/data/datasets/HDR/train_set/train_sdr_lq'
    # gt_folder = '/mnt/data/datasets/HDR/train_set/train_sdr_gt'

    # lq_lmdb_path = '/mnt/data/datasets/HDR/train_set/train_sdr_lq/train_sdr_lq.lmdb'
    # gt_lmdb_path = '/mnt/data/datasets/HDR/train_set/train_sdr_gt/train_sdr_gt.lmdb'
    # meta_info_pkl_dir = '/mnt/data/datasets/HDR/train_set/meta_info'
    # meta_info_pkl = '/mnt/data/datasets/HDR/train_set/meta_info/meta_info.pkl'

    lq_folder = '/mnt/data/datasets/HDR/test_set/test_sdr_lq'
    gt_folder = '/mnt/data/datasets/HDR/test_set/test_sdr_gt'

    lq_lmdb_path = '/mnt/data/datasets/HDR/test_set/test_sdr_lq/test_sdr_lq.lmdb'
    gt_lmdb_path = '/mnt/data/datasets/HDR/test_set/test_sdr_gt/test_sdr_gt.lmdb'
    meta_info_pkl_dir = '/mnt/data/datasets/HDR/test_set/meta_info'
    meta_info_pkl = '/mnt/data/datasets/HDR/test_set/meta_info/meta_info.pkl'

    os.makedirs(osp.dirname(lq_lmdb_path), exist_ok=True)
    os.makedirs(osp.dirname(gt_lmdb_path), exist_ok=True)
    os.makedirs(osp.dirname(meta_info_pkl_dir), exist_ok=True)

    # 先给 LQ 建 lmdb
    keys_lq, res_lq = make_lmdb(lq_folder, lq_lmdb_path, suffix='.png', compress_level=3)
    # 再给 GT 建 lmdb
    keys_gt, res_gt = make_lmdb(gt_folder, gt_lmdb_path, suffix='.png', compress_level=3)

    # 检查 key 是否一致
    if keys_lq != keys_gt:
        print('Warning: LQ and GT keys not matched! You may need to check file names.')
    else:
        print('LQ and GT keys matched.')

    # 写 meta_info.txt，供 paired_paths_from_lmdb 使用
    write_meta_txt(lq_lmdb_path, keys_lq, res_lq, compress_level=3, suffix='.png')
    write_meta_txt(gt_lmdb_path, keys_gt, res_gt, compress_level=3, suffix='.png')

    # 另外保留一个 pkl 版本的 meta 信息（可选）
    meta_info = {
        'lq_folder': lq_folder,
        'gt_folder': gt_folder,
        'lq_lmdb_path': lq_lmdb_path,
        'gt_lmdb_path': gt_lmdb_path,
        'keys': keys_lq,
        'lq_resolution': '_'.join(map(str, res_lq)),  # H_W_C
        'gt_resolution': '_'.join(map(str, res_gt)),
        'compress_level': 3
    }
    with open(meta_info_pkl, 'wb') as f:
        pickle.dump(meta_info, f)
    print(f'Meta info (pkl) saved to {meta_info_pkl}')