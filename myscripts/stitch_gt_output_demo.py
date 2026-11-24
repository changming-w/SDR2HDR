import os
from multiprocessing import Pool

from PIL import Image

# gt 与 output 各取宽度 1/4~3/4 的中间条带，左右拼接（左 GT、右 output）
# OUT_DIR 可指向 output（BT.709）或 output_bt2020（已转 BT.2020）
BASE = '/mnt/data/wcm/data/hdr/test/demo'
GT_DIR = os.path.join(BASE, 'gt')
OUT_DIR = os.path.join(BASE, 'output_bt2020')
DST_DIR = os.path.join(BASE, 'demo_bt2020')


def process(fname):
    gt = Image.open(os.path.join(GT_DIR, fname))
    out = Image.open(os.path.join(OUT_DIR, fname))
    if gt.size != out.size:
        return fname, False, f'size mismatch: {gt.size} vs {out.size}'
    w, h = gt.size
    x0, x1 = w // 4, w * 3 // 4
    band_w = x1 - x0
    canvas = Image.new(gt.mode, (band_w * 2, h))
    canvas.paste(gt.crop((x0, 0, x1, h)), (0, 0))
    canvas.paste(out.crop((x0, 0, x1, h)), (band_w, 0))
    canvas.save(os.path.join(DST_DIR, fname))
    return fname, True, 'ok'


def main():
    os.makedirs(DST_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(GT_DIR) if f.lower().endswith('.png'))
    print(f'total: {len(files)}')
    n_fail = 0
    with Pool(16) as pool:
        for i, (fname, ok, msg) in enumerate(pool.imap_unordered(process, files)):
            if not ok:
                n_fail += 1
                print(f'FAIL: {fname} ({msg})')
            if (i + 1) % 200 == 0:
                print(f'progress: {i + 1}/{len(files)}')
    print(f'done, failed: {n_fail}')


if __name__ == '__main__':
    main()
