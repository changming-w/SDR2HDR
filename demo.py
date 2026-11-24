# demo.py
import json
import logging
import os
import sys
from os import path as osp

import torch
from tqdm import tqdm

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import (
    get_env_info,
    get_root_logger,
    get_time_str,
    make_exp_dirs,
)
from basicsr.utils.options import dict2str, parse_options
from basicsr.utils.img_util import tensor2img, imwrite


def load_progress(progress_path):
    """从 progress.json 读取进度，没有则返回空 dict。"""
    if osp.isfile(progress_path):
        with open(progress_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_progress(progress_path, progress):
    """保存进度到 progress.json。"""
    tmp_path = progress_path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, progress_path)


def run_one_subfolder(opt_path, root_path, sub_input_dir, sub_output_dir,
                      batch_size, progress, progress_path):
    """
    对单个子文件夹跑一次流程，支持断点续跑，并统计成功/失败帧。
    """
    sub_input_dir = osp.abspath(sub_input_dir)
    sub_output_dir = osp.abspath(sub_output_dir)

    if not osp.isdir(sub_input_dir):
        raise FileNotFoundError(f'Sub input folder not found: {sub_input_dir}')
    os.makedirs(sub_output_dir, exist_ok=True)

    sub_name = osp.basename(sub_input_dir.rstrip('/'))

    # 1. 列出所有帧文件
    all_frames = sorted([
        f for f in os.listdir(sub_input_dir)
        if osp.isfile(osp.join(sub_input_dir, f))
        and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'))
    ])
    if not all_frames:
        print(f'[WARN] No frames found in {sub_input_dir}')
        return

    # 读取已有进度
    sub_prog = progress.get(sub_name, {})
    done_frames = set(sub_prog.get('done_frames', []))
    failed_frames = sub_prog.get('failed_frames', [])  # list[dict]
    failed_dict = {item['frame']: item['error'] for item in failed_frames}

    # 已存在输出文件也视为 done
    existing_outputs = {
        f for f in os.listdir(sub_output_dir)
        if osp.isfile(osp.join(sub_output_dir, f))
    }
    done_frames |= existing_outputs

    # 需要处理的帧：还没有成功输出的
    todo_frames = [f for f in all_frames if f not in done_frames]
    if not todo_frames:
        print(f'[INFO] Subfolder {sub_name} already done, skip.')
        progress[sub_name] = {
            'done': True,
            'done_frames': sorted(list(done_frames)),
            'failed_frames': [
                {'subfolder': sub_name, 'frame': f, 'error': failed_dict[f]}
                for f in sorted(failed_dict.keys())
            ],
        }
        save_progress(progress_path, progress)
        return

    print(f'[INFO] Subfolder {sub_name}: total {len(all_frames)} frames, '
          f'{len(done_frames)} done, {len(todo_frames)} to run '
          f'(historical failed: {len(failed_dict)}).')

    # 2. 用项目原始方式解析 opt
    #    模拟命令行参数：和 test_pipeline 里的用法保持一致
    sys.argv = [sys.argv[0], '-opt', opt_path]
    opt, _ = parse_options(root_path, is_train=False)

    # 这些字段 test_pipeline 默认会在 parse_options 中设置，这里只防御性覆盖一下
    opt.setdefault('dist', False)
    opt.setdefault('num_gpu', torch.cuda.device_count())
    opt.setdefault('manual_seed', 0)

    # 覆盖 dataset 的路径和 batch_size
    for phase, dataset_opt in opt['datasets'].items():
        # 注意：你的 hdrnet.yml 里应该是用 LQ 作为输入
        dataset_opt['dataroot_lq'] = sub_input_dir
        dataset_opt['phase'] = 'test'
        dataset_opt['name'] = ''  # 不要多一层 test_dataset 目录
        # BasicSR 新版默认用 batch_size_per_gpu / num_worker_per_gpu
        dataset_opt['batch_size_per_gpu'] = batch_size
        dataset_opt.setdefault('num_worker_per_gpu', 4)

    if 'path' not in opt:
        opt['path'] = {}
    opt['path']['visualization'] = sub_output_dir

    # 每个子文件夹单独一个实验名和 log
    opt['name'] = f"{opt['name']}_{sub_name}"
    torch.backends.cudnn.benchmark = True

    make_exp_dirs(opt)
    log_file = osp.join(
        opt['path']['log'],
        f"test_{opt['name']}_{get_time_str()}.log"
    )
    logger = get_root_logger(
        logger_name='basicsr',
        log_level=logging.INFO,
        log_file=log_file
    )
    logger.info(get_env_info())
    logger.info('Options after overriding paths for subfolder:')
    logger.info(dict2str(opt))

    # 3. 创建完整 dataset，再筛选出 todo_frames
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        if 'test' in phase:
            dataset_opt['phase'] = 'test'

        full_dataset = build_dataset(dataset_opt)
        logger.info(
            f"Full dataset frames in {dataset_opt['name'] or 'dataset'}: {len(full_dataset)}"
        )

        if hasattr(full_dataset, 'paths'):
            full_paths = full_dataset.paths
        elif hasattr(full_dataset, 'lq_paths'):
            full_paths = full_dataset.lq_paths
        else:
            raise AttributeError('Dataset has no "paths" or "lq_paths" attribute.')

        todo_set = set(todo_frames)
        selected_indices = [
            i for i, p in enumerate(full_paths)
            if osp.basename(p) in todo_set
        ]
        if not selected_indices:
            print(f'[INFO] All frames in {sub_name} already processed (by path), skip.')
            progress[sub_name] = {
                'done': True,
                'done_frames': sorted(list(done_frames)),
                'failed_frames': [
                    {'subfolder': sub_name, 'frame': f, 'error': failed_dict[f]}
                    for f in sorted(failed_dict.keys())
                ],
            }
            save_progress(progress_path, progress)
            return

        logger.info(
            f"Filtered dataset for subfolder {sub_name}: {len(selected_indices)} frames to run."
        )

        class SubsetDataset(torch.utils.data.Dataset):
            def __init__(self, base, indices):
                self.base = base
                self.indices = indices

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, idx):
                return self.base[self.indices[idx]]

        subset = SubsetDataset(full_dataset, selected_indices)

        # 直接用项目自己的 build_dataloader
        test_loader = build_dataloader(
            subset,
            dataset_opt,
            num_gpu=opt['num_gpu'],
            dist=opt['dist'],
            sampler=None,
            seed=opt['manual_seed']
        )
        break

    # 4. 创建模型，写推理循环 + 保存 + 断点 + 统计
    model = build_model(opt)
    logger.info(f'Start non-dist validation on subfolder: {sub_name}')

    model.net_g.eval()
    if hasattr(model, 'net_g_ema') and model.net_g_ema is not None:
        model.net_g_ema.eval()

    pbar = tqdm(total=len(subset), desc=sub_name, unit='img')

    with torch.no_grad():
        for data_idx, data in enumerate(test_loader):
            try:
                if 'lq_path' in data:
                    paths = data['lq_path']
                elif 'gt_path' in data:
                    paths = data['gt_path']
                else:
                    raise KeyError('Cannot find "lq_path" or "gt_path" in data batch.')

                frame_names = [osp.basename(p) for p in paths]

                model.feed_data(data)
                model.test()
                visuals = model.get_current_visuals()
                result = visuals['result']  # [B, C, H, W]

                for b_idx, frame_name in enumerate(frame_names):
                    save_img_path = osp.join(sub_output_dir, frame_name)

                    # 已存在：视作成功
                    if osp.isfile(save_img_path):
                        done_frames.add(frame_name)
                        failed_dict.pop(frame_name, None)
                        continue

                    img_tensor = result[b_idx:b_idx + 1, ...]
                    img = tensor2img(img_tensor)
                    imwrite(img, save_img_path)

                    done_frames.add(frame_name)
                    failed_dict.pop(frame_name, None)

            except Exception as e:
                err_msg = f'{type(e).__name__}: {e}'
                logger.error(f'Error processing batch in {sub_name}, paths={paths}: {err_msg}')
                for frame_name in frame_names:
                    failed_dict[frame_name] = err_msg

            # 更新 progress.json
            failed_frames = [
                {'subfolder': sub_name, 'frame': f, 'error': failed_dict[f]}
                for f in sorted(failed_dict.keys())
            ]
            progress[sub_name] = {
                'done': len(done_frames) + len(failed_dict) >= len(all_frames),
                'done_frames': sorted(list(done_frames)),
                'failed_frames': failed_frames,
            }
            save_progress(progress_path, progress)

            pbar.update(len(frame_names))

    pbar.close()

    # 当前子文件夹统计
    success_num = len(done_frames)
    fail_num = len(failed_dict)
    print(f'[RESULT] Subfolder {sub_name}: success={success_num}, failed={fail_num}')
    if fail_num > 0:
        print('[FAILED FRAMES in', sub_name, ']')
        for f, err in sorted(failed_dict.items()):
            print('  ', osp.join(sub_name, f), '->', err)

    logger.info(f'Non-dist validation done for subfolder: {sub_name}')
    logger.info(f'Sub input  : {sub_input_dir}')
    logger.info(f'Sub output : {sub_output_dir}')


def main(opt_path, input_root, output_root, batch_size):
    """按子文件夹顺序依次跑所有视频，支持断点续跑和总体统计。"""
    input_root = osp.abspath(input_root)
    output_root = osp.abspath(output_root)

    if not osp.isdir(input_root):
        raise FileNotFoundError(f'Input root not found: {input_root}')
    os.makedirs(output_root, exist_ok=True)

    progress_path = osp.join(output_root, 'progress.json')
    progress = load_progress(progress_path)

    subdirs = [
        d for d in os.listdir(input_root)
        if osp.isdir(osp.join(input_root, d))
    ]
    subdirs.sort()

    print('Found subfolders:')
    for d in subdirs:
        print('  ', d)
    print('Batch size:', batch_size)
    print('Progress file:', progress_path)

    # root_path 用和原 test_pipeline 相同的方式设置
    root_path = osp.abspath(osp.join(__file__, osp.pardir))

    total_success = 0
    total_failed = 0
    all_failed_list = []  # [(subfolder, frame, error), ...]

    for d in subdirs:
        sub_input_dir = osp.join(input_root, d)
        sub_output_dir = osp.join(output_root, d)

        sub_prog = progress.get(d, {})
        if sub_prog.get('done', False) and osp.isdir(sub_output_dir):
            print(f'[INFO] Skip already done subfolder: {d}')
            done_frames = sub_prog.get('done_frames', [])
            failed_frames = sub_prog.get('failed_frames', [])
            total_success += len(done_frames)
            total_failed += len(failed_frames)
            for item in failed_frames:
                all_failed_list.append((item['subfolder'], item['frame'], item['error']))
            continue

        print(f'\n=== Processing subfolder: {d} ===')
        run_one_subfolder(
            opt_path,
            root_path,
            sub_input_dir,
            sub_output_dir,
            batch_size,
            progress,
            progress_path
        )

        # 子目录结束后，更新总统计
        sub_prog = progress.get(d, {})
        done_frames = sub_prog.get('done_frames', [])
        failed_frames = sub_prog.get('failed_frames', [])
        total_success += len(done_frames)
        total_failed += len(failed_frames)
        for item in failed_frames:
            all_failed_list.append((item['subfolder'], item['frame'], item['error']))

    # 整体结果
    print('\n======= OVERALL RESULT =======')
    print('Total success frames:', total_success)
    print('Total failed frames :', total_failed)
    if total_failed > 0:
        print('\nFailed frame list (subfolder/frame -> error):')
        for subfolder, frame, err in all_failed_list:
            print('  ', osp.join(subfolder, frame), '->', err)
    print('===============================')


if __name__ == '__main__':
    opt_path = "/mnt/data/wcm/HDRNet/options/test/HDR/hdrnet.yml"
    input_root = "/mnt/data/datasets/deblur/sports_data/Demos/Deblur"
    output_root = "/mnt/data/datasets/deblur/sports_data/Demos/Deblur_HDR0.15"

    batch_size = 12  # 在这里改 batch 大小

    main(opt_path, input_root, output_root, batch_size)