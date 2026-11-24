"""标准SDR2HDR训练数据 light_scene/hdrframes: PQ -> BT.2390 tone mapping (10000->4000nit)
-> 按 1000nit 满量程归一化 -> gamma 2.2 编码 -> 16bit PNG

流程:
  1. PQ (ST 2084) 解码: 16bit 码值 E -> 线性亮度 Y (相对 10000nit)
  2. BT.2390 EETF (libplacebo pl_tone_map_bt2390 移植): PQ 域 hermite 滚降,
     10000nit -> 4000nit。knee ≈ 2717nit, 本批数据 max ~740nit,
     全部低于 knee, EETF 为纯恒等 (中间调不受任何压缩, 仅 >2717nit 才滚降)
  3. 归一化: 1000nit = 1.0 (用户指定的显示满量程基准)
  4. gamma 2.2 编码: V = Y_norm^(1/2.2), V∈[0,1] -> uint16 (×65535)

输出: hdrframes_g22/ (与源同目录, 同名 16bit gamma2.2 PNG, uint16 直存)
- 用 HDRPairedMultiDataset (uint16 /65535) 即可正确读取, 无需 10bit<<6 专用 dataset
- 幂等可重跑 (已存在则跳过), 16 进程并行
"""
import os
from multiprocessing import Pool

import cv2
import numpy as np

ROOT = '/mnt/data/wcm/data/hdr/train/标准SDR2HDR训练数据/light_scene'
SRC_DIR = os.path.join(ROOT, 'hdrframes')
DST_DIR = os.path.join(ROOT, 'hdrframes_g22')

# ---- PQ (ST 2084) 常数 ----
M1, M2 = 0.1593017578125, 78.84375
C1, C2, C3 = 0.8359375, 18.8515625, 18.6875

# ---- BT.2390 EETF 参数 ----
L_IN_MAX = 1.0                  # 输入满量程: 10000 nit (PQ 满量程, 相对 1.0)
L_OUT_MAX = 4000.0 / 10000.0    # tone mapping 目标满量程: 4000 nit
# knee 偏移: 0.5 为 ITU-R BT.2390 规范值 (libplacebo 实现默认 1.0)
KNEE_OFFSET = 0.5

# ---- 输出归一化基准: 1000 nit = 1.0 ----
NORM_NIT = 1000.0

GAMMA = 2.2


def pq_decode(E):
    """PQ 码值 [0,1] -> 线性亮度 Y (相对 10000nit, 即 [0,1] 对应 0~10000nit)"""
    E = np.clip(E, 0.0, 1.0)
    Eip = np.power(E, 1.0 / M2)
    num = np.maximum(Eip - C1, 0.0)
    den = C2 - C3 * Eip
    return np.power(num / den, 1.0 / M1)


def pq_encode(Y):
    """线性亮度 Y (相对 10000nit) -> PQ 码值 E [0,1]"""
    Y = np.clip(Y, 0.0, 1.0)
    Ym = np.power(Y, M1)
    return np.power((C1 + C2 * Ym) / (1.0 + C3 * Ym), M2)


def bt2390_eetf(Y_lin):
    """BT.2390 EETF (libplacebo src/tone_mapping.c bt2390() 的 Python 移植)。

    归一化域: 1.0 = 源满量程 (PQ 域), 与 libplacebo 完全一致:
        maxLum = PQ(L_out_max) / PQ(L_in_max)   目标满量程 (同域)
        ks     = (1 + KNEE_OFFSET) * maxLum - KNEE_OFFSET   knee 起点
        x < ks: 恒等; 否则 hermite 样条 (ks,ks) -> (1, maxLum)

    10000->4000nit 时 ks≈0.8611 (≈2717nit): 低于 knee 的内容 (含全部
    中间调) 完全不变, 源满量程 10000nit 映射到目标满量程 4000nit。
    本批数据 max ~740nit, EETF 为纯恒等。
    黑点适配项 (libplacebo 的 bp/gain) 在输出黑点=0 时退化为恒等, 故省略。
    """
    E_in = pq_encode(np.clip(Y_lin, 0.0, 1.0))
    maxLum = pq_encode(np.array([L_OUT_MAX]))[0] / pq_encode(np.array([L_IN_MAX]))[0]
    ks = (1.0 + KNEE_OFFSET) * maxLum - KNEE_OFFSET

    if ks >= 1.0:  # 目标量程不小于源量程, 无需压缩
        return Y_lin

    tb = np.clip((E_in - ks) / (1.0 - ks), 0.0, 1.0)
    tb2 = tb * tb
    tb3 = tb2 * tb
    E_out = np.where(
        E_in < ks,
        E_in,
        (2.0 * tb3 - 3.0 * tb2 + 1.0) * ks
        + (tb3 - 2.0 * tb2 + tb) * (1.0 - ks)
        + (-2.0 * tb3 + 3.0 * tb2) * maxLum,
    )
    return pq_decode(np.clip(E_out, 0.0, 1.0))


def process(fname):
    src = os.path.join(SRC_DIR, fname)
    dst = os.path.join(DST_DIR, fname)
    if os.path.exists(dst):
        return True, 'skip'
    img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    if img is None:
        return False, 'read failed'

    E = img.astype(np.float64) / 65535.0        # PQ 码值 (BGR 顺序, 逐通道处理与顺序无关)
    Y = pq_decode(E)                            # 线性亮度 (相对 10000nit)
    Y_tm = bt2390_eetf(Y)                       # BT.2390 tone mapping (10000 -> 4000 nit)
    Y_norm = Y_tm * (10000.0 / NORM_NIT)        # 1000nit = 1.0 归一化
    Y_norm = np.clip(Y_norm, 0.0, 1.0)          # 超过 1000nit 的内容 clip 到满量程
    V = np.power(Y_norm, 1.0 / GAMMA)           # gamma 2.2 编码
    out = np.rint(V * 65535.0).astype(np.uint16)

    ok = cv2.imwrite(dst, out)
    return bool(ok), 'ok' if ok else 'write failed'


if __name__ == '__main__':
    os.makedirs(DST_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(SRC_DIR) if f.lower().endswith('.png'))
    print(f'total {len(files)} files')

    # 单文件自检: 打印转换链路关键数值
    maxLum = pq_encode(np.array([L_OUT_MAX]))[0]
    ks = (1.0 + KNEE_OFFSET) * maxLum - KNEE_OFFSET
    print(f'EETF: 10000->4000nit, knee_offset={KNEE_OFFSET}, '
          f'knee=PQ {ks:.4f} (~{pq_decode(ks)*10000:.0f} nit, 低于此恒等)')
    E_test = np.array([0.0, 0.25, 0.5, 0.5806, 0.75, 0.9026, 1.0])
    Y_test = pq_decode(E_test)
    Y_tm = bt2390_eetf(Y_test)
    V_test = np.power(np.clip(Y_tm * 10.0, 0, 1), 1/2.2)   # 1000nit=1.0
    print('E      :', ' '.join(f'{e:.4f}' for e in E_test))
    print('Y(nit) :', ' '.join(f'{y*10000:7.1f}' for y in Y_test))
    print('tm(nit):', ' '.join(f'{y*10000:7.1f}' for y in Y_tm))
    print('V(1000nit=1.0):', ' '.join(f'{v:.4f}' for v in V_test))

    fails = 0
    with Pool(16) as pool:
        for i, (ok, msg) in enumerate(pool.imap_unordered(process, files)):
            if not ok:
                fails += 1
                print(f'[FAIL] {msg}')
            if (i + 1) % 200 == 0:
                print(f'{i+1}/{len(files)}')
    print(f'finished, {fails} failed')
