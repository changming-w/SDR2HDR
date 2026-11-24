#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BT.709 8bit PNG -> BT.2020 8bit PNG (SDR, gamma 2.2)

数据来源: BT.709 视频在电视上播放, TCON 前截帧, 再按比例缩放得到的 8bit PNG
          (TCON 前原色为 BT.709, 电视 gamma 设置为 2.2)
流程:     码值 -> 归一化 -> 线性光 L = V^2.2 -> 709→2020 矩阵 -> V = L^(1/2.2) -> 量化
输出标记: cICP = 9/4/0/1 (BT.2020 原色 / gamma 2.2 / RGB / 全范围)
"""
import argparse
import struct
import zlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np

GAMMA = 2.2

# BT.709 -> BT.2020 线性 RGB 转换矩阵 (ITU-R BT.2087, 同为 D65 白点, 无需色适应)
M_709_TO_2020 = np.array([
    [0.627403895934699, 0.329283038377884, 0.043313065687417],
    [0.069097289358232, 0.919540395075459, 0.011362315566309],
    [0.016391438875150, 0.088013307877226, 0.895595253247624],
], dtype=np.float32)

# H.273: primaries 9 = BT.2020, transfer 4 = gamma 2.2, matrix 0 = RGB, full range = 1
CICP = (9, 4, 0, 1)


# ---------------- PNG cICP ----------------
def insert_cicp(png_bytes: bytes, cicp=CICP) -> bytes:
    """在 IHDR 之后、IDAT 之前插入 cICP chunk"""
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG stream"
    ihdr_len = struct.unpack(">I", png_bytes[8:12])[0]
    ihdr_end = 8 + 4 + 4 + ihdr_len + 4
    data = bytes(cicp)
    chunk = (struct.pack(">I", len(data)) + b"cICP" + data +
             struct.pack(">I", zlib.crc32(b"cICP" + data) & 0xFFFFFFFF))
    return png_bytes[:ihdr_end] + chunk + png_bytes[ihdr_end:]


# ---------------- 核心转换 ----------------
def build_decode_lut(in_limited: bool) -> np.ndarray:
    """256 项查找表: 8bit 码值 -> 线性光"""
    code = np.arange(256, dtype=np.float64)
    if in_limited:
        norm = np.clip((code - 16.0) / 219.0, 0.0, 1.0)
    else:
        norm = code / 255.0
    return np.power(norm, GAMMA).astype(np.float32)


def convert_rgb8(rgb8: np.ndarray, dither: bool = False,
                 in_limited: bool = False, seed: int = 0) -> np.ndarray:
    """rgb8: HxWx3 uint8 (RGB 顺序) -> HxWx3 uint8 BT.2020 RGB (gamma 2.2)"""
    # 1. 线性化 (查表)
    lin = build_decode_lut(in_limited)[rgb8]

    # 2. 线性域原色转换
    lin2020 = lin @ M_709_TO_2020.T
    np.clip(lin2020, 0.0, 1.0, out=lin2020)    # 709 在 2020 内, 只清理浮点误差

    # 3. gamma 2.2 编码 + 量化
    v = np.power(lin2020, np.float32(1.0 / GAMMA)) * np.float32(255.0)
    if dither:                                  # TPDF 抖动, 减轻 8bit 色带
        rng = np.random.default_rng(seed)
        v += (rng.random(v.shape, dtype=np.float32) -
              rng.random(v.shape, dtype=np.float32))
    return np.clip(np.rint(v), 0, 255).astype(np.uint8)


def self_test():
    px = np.array([[[255, 0, 0], [255, 255, 255], [128, 128, 128], [0, 0, 0]]], np.uint8)
    out = convert_rgb8(px)[0]
    assert tuple(out[0]) == (206, 76, 39), f"纯红异常: {tuple(out[0])}"
    assert tuple(out[1]) == (255, 255, 255), f"白色异常: {tuple(out[1])}"
    assert tuple(out[2]) == (128, 128, 128), f"灰阶异常: {tuple(out[2])}"
    assert tuple(out[3]) == (0, 0, 0), f"黑色异常: {tuple(out[3])}"
    print("[self-test] gamma 2.2, 纯红 (255,0,0) -> (206,76,39)  OK")


# ---------------- 单文件处理 ----------------
def process_one(src: str, dst: str, dither: bool, in_limited: bool, cicp: bool):
    img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    if img is None:
        return src, "read failed"
    if img.dtype != np.uint8:
        return src, f"unsupported dtype {img.dtype} (仅支持 8bit)"
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    alpha = img[..., 3:4] if img.shape[2] == 4 else None
    rgb = np.ascontiguousarray(img[..., 2::-1])            # BGR(A) -> RGB

    seed = zlib.crc32(Path(src).name.encode())             # 抖动可复现
    out = convert_rgb8(rgb, dither, in_limited, seed)[..., ::-1]   # RGB -> BGR
    if alpha is not None:
        out = np.concatenate([out, alpha], axis=2)

    ok, buf = cv2.imencode(".png", np.ascontiguousarray(out),
                           [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        return src, "encode failed"
    data = buf.tobytes()
    if cicp:
        data = insert_cicp(data)

    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_bytes(data)
    return src, None


# ---------------- 主程序 ----------------
def main():
    ap = argparse.ArgumentParser(
        description="BT.709 8bit PNG -> BT.2020 8bit PNG (gamma 2.2, TCON 前截帧)")
    ap.add_argument("input", help="输入 PNG 文件或目录 (递归)")
    ap.add_argument("output", help="输出 PNG 文件或目录")
    ap.add_argument("--dither", action="store_true", help="量化前加 TPDF 抖动")
    ap.add_argument("--in-limited", action="store_true", help="输入 RGB 为 16-235 窄范围")
    ap.add_argument("--no-cicp", action="store_true", help="不写入 cICP 色彩标记")
    ap.add_argument("--workers", type=int, default=8, help="进程数")
    args = ap.parse_args()

    self_test()

    src, dst = Path(args.input), Path(args.output)
    if src.is_file():
        jobs = [(src, dst if dst.suffix.lower() == ".png" else dst / src.name)]
    else:
        files = sorted(p for p in src.rglob("*") if p.suffix.lower() == ".png")
        jobs = [(p, dst / p.relative_to(src)) for p in files]
    if not jobs:
        print("没有找到 PNG 文件")
        return

    cv2.setNumThreads(1)
    fails = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_one, str(s), str(d), args.dither,
                          args.in_limited, not args.no_cicp) for s, d in jobs]
        for i, fu in enumerate(as_completed(futs), 1):
            s, err = fu.result()
            if err:
                fails += 1
                print(f"[FAIL] {s}: {err}")
            if i % 200 == 0 or i == len(futs):
                print(f"{i}/{len(futs)} done")
    print(f"finished: {len(jobs) - fails} ok, {fails} failed")


if __name__ == "__main__":
    main()