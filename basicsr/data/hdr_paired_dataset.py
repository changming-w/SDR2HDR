import cv2
import numpy as np
from os import path as osp
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.transforms import augment, paired_pad_if_needed, paired_random_crop
from basicsr.utils import FileClient, bgr2ycbcr, img2tensor, scandir
from basicsr.utils.registry import DATASET_REGISTRY

IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.webp')


def imfrombytes_hdr(content, float32=True):
    """读取 16bit HDR PNG，归一化到 [0, 1]。

    与 imfrombytes 的区别：
    - 使用 IMREAD_UNCHANGED 保留 16bit 精度
    - 除以 65535 而非 255
    """
    img_np = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)
    if float32:
        img = img.astype(np.float32) / 65535.0
    return img


@DATASET_REGISTRY.register()
class HDRPairedImageDataset(data.Dataset):
    """SDR→HDR 配对数据集。

    GT (HDR) 为 16bit PNG，LQ (SDR) 为 8bit PNG。
    分别用不同的归一化方式读取：
      - HDR GT:  IMREAD_UNCHANGED → / 65535 → [0, 1]
      - SDR LQ:  正常读取       → / 255   → [0, 1]
    """

    def __init__(self, opt):
        super(HDRPairedImageDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        # 拷贝 io_backend 配置，避免 pop 修改原配置
        self.io_backend_opt = opt['io_backend'].copy()
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.paths = self._get_paired_paths()

        # 可选：将所有图片解码后加载到内存，消除 PNG 解码瓶颈
        self.load_to_memory = opt.get('load_to_memory', False)
        if self.load_to_memory:
            print(f'Decoding and loading {len(self.paths)} image pairs into memory...')
            self.gt_cache = []
            self.lq_cache = []
            for i, p in enumerate(self.paths):
                if (i + 1) % 50 == 0:
                    print(f'  [{i+1}/{len(self.paths)}]')
                # GT (HDR 16bit) -> uint16 numpy array
                gt_img = cv2.imread(p['gt_path'], cv2.IMREAD_UNCHANGED)
                self.gt_cache.append(gt_img)
                # LQ (SDR 8bit) -> uint8 numpy array
                lq_img = cv2.imread(p['lq_path'], cv2.IMREAD_COLOR)
                self.lq_cache.append(lq_img)
            print(f'Done! Loaded {len(self.paths)} pairs into memory (decoded).')

    def _get_paired_paths(self):
        """扫描文件夹，只匹配图像文件，生成配对路径列表。"""
        # 只收集图像文件
        lq_files = sorted([f for f in scandir(self.lq_folder) if f.lower().endswith(IMG_EXTENSIONS)])
        gt_files = sorted([f for f in scandir(self.gt_folder) if f.lower().endswith(IMG_EXTENSIONS)])
        assert len(lq_files) == len(gt_files), (
            f'lq and gt datasets have different number of images: '
            f'{len(lq_files)}, {len(gt_files)}.'
        )
        # 按文件名（不含扩展名）匹配
        lq_basenames = {osp.splitext(f)[0]: f for f in lq_files}
        gt_basenames = {osp.splitext(f)[0]: f for f in gt_files}
        assert lq_basenames.keys() == gt_basenames.keys(), (
            'lq and gt image filenames do not match.'
        )
        paths = []
        for name in sorted(lq_basenames.keys()):
            paths.append({
                'lq_path': osp.join(self.lq_folder, lq_basenames[name]),
                'gt_path': osp.join(self.gt_folder, gt_basenames[name]),
            })
        return paths

    def __getitem__(self, index):
        scale = self.opt['scale']
        gt_path = self.paths[index]['gt_path']
        lq_path = self.paths[index]['lq_path']

        # ---- 读取 GT (HDR, 16bit) ----
        if self.load_to_memory:
            img_gt = self.gt_cache[index]  # uint16, 不复制
        else:
            if self.file_client is None:
                self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
            img_bytes = self.file_client.get(gt_path, 'gt')
            # 保留 uint16，不转 float32
            img_np = np.frombuffer(img_bytes, np.uint8)
            img_gt = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)

        # ---- 读取 LQ (SDR, 8bit) ----
        if self.load_to_memory:
            img_lq = self.lq_cache[index]  # uint8, 不复制
        else:
            if self.file_client is None:
                self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
            img_bytes = self.file_client.get(lq_path, 'lq')
            # 保留 uint8，不转 float32
            img_np = np.frombuffer(img_bytes, np.uint8)
            img_lq = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        # ---- 数据增强 ----
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            img_gt, img_lq = paired_pad_if_needed(img_gt, img_lq, gt_size, scale)
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # copy 后再增强，避免污染缓存
            if self.load_to_memory:
                img_gt = img_gt.copy()
                img_lq = img_lq.copy()
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            # 需要先转 float32 才能做颜色空间转换
            img_gt = img_gt.astype(np.float32)
            img_lq = img_lq.astype(np.float32)
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        # validation 时裁剪不匹配的 GT
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # 统一转 float32 归一化
        if img_gt.dtype == np.uint16:
            img_gt = img_gt.astype(np.float32) / 65535.0
        elif img_gt.dtype == np.uint8:
            img_gt = img_gt.astype(np.float32) / 255.0
        elif img_gt.dtype != np.float32:
            img_gt = img_gt.astype(np.float32)

        if img_lq.dtype == np.uint16:
            img_lq = img_lq.astype(np.float32) / 65535.0
        elif img_lq.dtype == np.uint8:
            img_lq = img_lq.astype(np.float32) / 255.0
        elif img_lq.dtype != np.float32:
            img_lq = img_lq.astype(np.float32)

        # BGR→RGB, HWC→CHW, numpy→tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)


@DATASET_REGISTRY.register()
class HDRPairedMultiDataset(data.Dataset):
    """SDR→HDR 配对数据集（支持多组文件夹 + 按权重采样）。

    配置示例:
      dataroot_gt:
        - /path/to/hdr1
        - /path/to/hdr2
      dataroot_lq:
        - /path/to/sdr1
        - /path/to/sdr2
      dataset_weights: [0.5, 0.5]  # 可选，各组采样权重，默认均匀
    """

    def __init__(self, opt):
        super(HDRPairedMultiDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend'].copy()
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        # 支持多组文件夹
        gt_folders = opt['dataroot_gt']
        lq_folders = opt['dataroot_lq']
        if isinstance(gt_folders, str):
            gt_folders = [gt_folders]
        if isinstance(lq_folders, str):
            lq_folders = [lq_folders]
        assert len(gt_folders) == len(lq_folders), (
            f'dataroot_gt 和 dataroot_lq 数量不一致: {len(gt_folders)} vs {len(lq_folders)}'
        )

        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        # 收集所有组的配对路径，合并为一个列表
        self.paths = []
        for i, (gt_folder, lq_folder) in enumerate(zip(gt_folders, lq_folders)):
            paths = self._get_paired_paths(gt_folder, lq_folder)
            print(f'[Dataset] 组{i+1}: 加载 {len(paths)} 对图片 from {lq_folder} <-> {gt_folder}')
            self.paths.extend(paths)
        print(f'[Dataset] 总计 {len(self.paths)} 对图片，均匀采样')

        # 可选：加载到内存
        self.load_to_memory = opt.get('load_to_memory', False)
        if self.load_to_memory:
            print(f'Decoding and loading {len(self.paths)} image pairs into memory...')
            self.gt_cache = []
            self.lq_cache = []
            for i, p in enumerate(self.paths):
                if (i + 1) % 50 == 0:
                    print(f'  [{i+1}/{len(self.paths)}]')
                gt_img = cv2.imread(p['gt_path'], cv2.IMREAD_UNCHANGED)
                self.gt_cache.append(gt_img)
                lq_img = cv2.imread(p['lq_path'], cv2.IMREAD_COLOR)
                self.lq_cache.append(lq_img)
            print(f'Done! Loaded {len(self.paths)} pairs into memory (decoded).')

    def _get_paired_paths(self, gt_folder, lq_folder):
        """扫描文件夹，生成配对路径列表。"""
        lq_files = sorted([f for f in scandir(lq_folder) if f.lower().endswith(IMG_EXTENSIONS)])
        gt_files = sorted([f for f in scandir(gt_folder) if f.lower().endswith(IMG_EXTENSIONS)])
        if len(lq_files) != len(gt_files):
            print(f'[WARN] {lq_folder} ({len(lq_files)}) 与 {gt_folder} ({len(gt_files)}) 数量不同')
        # 按文件名匹配
        lq_basenames = {osp.splitext(f)[0]: f for f in lq_files}
        gt_basenames = {osp.splitext(f)[0]: f for f in gt_files}
        common = sorted(lq_basenames.keys() & gt_basenames.keys())
        paths = []
        for name in common:
            paths.append({
                'lq_path': osp.join(lq_folder, lq_basenames[name]),
                'gt_path': osp.join(gt_folder, gt_basenames[name]),
            })
        return paths

    def __getitem__(self, index):
        scale = self.opt['scale']

        # 直接从合并的列表中按索引取
        gt_path = self.paths[index]['gt_path']
        lq_path = self.paths[index]['lq_path']

        # ---- 读取 GT (HDR, 16bit) ----
        if self.load_to_memory:
            img_gt = self.gt_cache[index]
        else:
            if self.file_client is None:
                self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_np = np.frombuffer(img_bytes, np.uint8)
            img_gt = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)

        # ---- 读取 LQ (SDR, 8bit) ----
        if self.load_to_memory:
            img_lq = self.lq_cache[index]
        else:
            if self.file_client is None:
                self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
            img_bytes = self.file_client.get(lq_path, 'lq')
            img_np = np.frombuffer(img_bytes, np.uint8)
            img_lq = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        # ---- 数据增强 ----
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            img_gt, img_lq = paired_pad_if_needed(img_gt, img_lq, gt_size, scale)
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            if self.load_to_memory:
                img_gt = img_gt.copy()
                img_lq = img_lq.copy()
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = img_gt.astype(np.float32)
            img_lq = img_lq.astype(np.float32)
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # 转 float32 归一化
        if img_gt.dtype == np.uint16:
            img_gt = img_gt.astype(np.float32) / 65535.0
        elif img_gt.dtype == np.uint8:
            img_gt = img_gt.astype(np.float32) / 255.0
        elif img_gt.dtype != np.float32:
            img_gt = img_gt.astype(np.float32)

        if img_lq.dtype == np.uint16:
            img_lq = img_lq.astype(np.float32) / 65535.0
        elif img_lq.dtype == np.uint8:
            img_lq = img_lq.astype(np.float32) / 255.0
        elif img_lq.dtype != np.float32:
            img_lq = img_lq.astype(np.float32)

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)


@DATASET_REGISTRY.register()
class HDRPairedMultiDataset10bitGT(HDRPairedMultiDataset):
    """SDR→HDR 配对数据集（多组文件夹），GT 为 10bit 数据左移 6bit 存成的 16bit PNG。

    与 HDRPairedMultiDataset 的唯一区别：
      - GT (uint16): 为 10bit 数据左移 6bit 存成（10bit<<6），满量程 1023*64=65472
        归一化直接 / 65472（等价于 (v>>6)/1023 的连续形式，无中间量化误差），
        保留插值/缩放产生的亚 LSB 精度，量程端点严格对齐（1023*64 -> 1.0）
      - LQ (SDR 8bit): 正常读取 -> / 255 -> [0, 1]

    配置示例:
      type: HDRPairedMultiDataset10bitGT
      dataroot_gt:
        - /path/to/sony_light/hdr
      dataroot_lq:
        - /path/to/sony_light/sdr
    """

    def __getitem__(self, index):
        scale = self.opt['scale']

        gt_path = self.paths[index]['gt_path']
        lq_path = self.paths[index]['lq_path']

        # ---- 读取 GT (10bit<<6 的 16bit PNG) ----
        if self.load_to_memory:
            img_gt = self.gt_cache[index]
        else:
            if self.file_client is None:
                self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_np = np.frombuffer(img_bytes, np.uint8)
            img_gt = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)

        # ---- 读取 LQ (SDR, 8bit) ----
        if self.load_to_memory:
            img_lq = self.lq_cache[index]
        else:
            if self.file_client is None:
                self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
            img_bytes = self.file_client.get(lq_path, 'lq')
            img_np = np.frombuffer(img_bytes, np.uint8)
            img_lq = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        # ---- 数据增强 ----
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            img_gt, img_lq = paired_pad_if_needed(img_gt, img_lq, gt_size, scale)
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            if self.load_to_memory:
                img_gt = img_gt.copy()
                img_lq = img_lq.copy()
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = img_gt.astype(np.float32)
            img_lq = img_lq.astype(np.float32)
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # 转 float32 归一化（GT: 10bit<<6, 满量程 65472 = 1023*64, 直接除避免中间量化误差）
        if img_gt.dtype == np.uint16:
            img_gt = img_gt.astype(np.float32) / 65472.0
        elif img_gt.dtype == np.uint8:
            img_gt = img_gt.astype(np.float32) / 255.0
        elif img_gt.dtype != np.float32:
            img_gt = img_gt.astype(np.float32)

        if img_lq.dtype == np.uint16:
            img_lq = img_lq.astype(np.float32) / 65535.0
        elif img_lq.dtype == np.uint8:
            img_lq = img_lq.astype(np.float32) / 255.0
        elif img_lq.dtype != np.float32:
            img_lq = img_lq.astype(np.float32)

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)
