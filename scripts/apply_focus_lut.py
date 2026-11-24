#!/usr/bin/env python3
"""将 .cube 3D LUT 批量应用到图片目录。

用法：
  python3 scripts/apply_focus_lut.py focus_brighten_65.cube 输入目录 输出目录
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from focus_map import read_image, write_image
from generate_focus_lut import apply_lut

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def load_cube(filepath: str) -> np.ndarray:
    """解析 .cube 文件，返回 (size, size, size, 3) [0,1]，索引顺序 [b, g, r]。"""
    size = None
    rows = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('LUT_3D_SIZE'):
                size = int(line.split()[1])
                continue
            parts = line.split()
            try:
                values = [float(x) for x in parts]
            except ValueError:
                continue  # TITLE / DOMAIN_* 等头信息
            if len(values) == 3:
                rows.append(values)

    if size is None:
        raise ValueError(f'未找到 LUT_3D_SIZE: {filepath}')
    data = np.asarray(rows, dtype=np.float32)
    if data.shape[0] != size ** 3:
        raise ValueError(f'数据行数 {data.shape[0]} != {size}^3')
    # 写入约定（generate_focus_lut.write_cube）：行序 b0 最外层、r0 最内层，
    # 每行值为 lut[r0, g0, b0]；reshape 后得 data[i,j,k] = lut[k,j,i]，
    # 交换 0↔2 轴即还原为 lut 的 [r, g, b] 索引，再转为 apply_lut 需要的
    # [b, g, r] 索引（二者恰好等价，直接返回转置结果）。
    return data.reshape(size, size, size, 3).transpose(2, 1, 0, 3)


def main():
    ap = argparse.ArgumentParser(description='批量应用 3D LUT (.cube)')
    ap.add_argument('cube', help='.cube LUT 文件路径')
    ap.add_argument('input', help='输入图片目录')
    ap.add_argument('output', help='输出目录')
    a = ap.parse_args()

    lut = load_cube(a.cube)
    print(f'已加载 LUT: {a.cube} ({lut.shape[0]}^3)')

    input_dir = Path(a.input)
    output_dir = Path(a.output)
    files = sorted(p for p in input_dir.rglob('*')
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    print(f'共 {len(files)} 张图片')

    for i, f in enumerate(files):
        try:
            img = read_image(f)
            out = apply_lut(img, lut)
            write_image(output_dir / f.relative_to(input_dir), out)
            if (i + 1) % 20 == 0 or len(files) <= 20:
                print(f'[{i+1}/{len(files)}] {f.name}')
        except Exception as e:
            print(f'错误 {f.name}: {e}')

    print(f'完成！输出: {output_dir}')


if __name__ == '__main__':
    main()
