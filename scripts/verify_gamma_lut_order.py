#!/usr/bin/env python3
"""
验证: 等比缩放模型 (out = k * RGB) 放在屏幕 gamma 校正 LUT 之前 / 之后的差异

面板模型（假设）:
  - 三通道原生响应 P_c(V') 各不相同（非 2.2，且互不相同）
  - 三通道满幅亮度 a_c 不同 => 原生白点偏离 D65
  - 原色按 BT.709 计算 XYZ

gamma 校正 LUT（三条）:
  C_c(V) = P_c^{-1}( s * V^2.2 / a_c )，使 a_c * P_c(C_c(V)) = s * V^2.2
  即校正后屏幕严格为 2.2 且灰阶为 D65（同时承担白平衡）

两条流程:
  A 模型在校正前: V -> k*V -> LUT -> 面板    （理想: 发光 = s*(kV)^2.2，白点不变）
  B 模型在校正后: V -> LUT -> k*V' -> 面板   （当前做法）

用法:
  python3 verify_gamma_lut_order.py
  python3 verify_gamma_lut_order.py --in_dir model_input --out_dir out_bt709 --save_dir verify_out
"""
import argparse
import os

import cv2
import numpy as np

# ---------------- 面板假设 ----------------
NATIVE_GAMMA = {'R': 2.45, 'G': 2.25, 'B': 2.05}   # 原生幂指数
S_CURVE = {'R': 0.10, 'G': 0.05, 'B': 0.15}         # 叠加 S 形扭曲强度
AMP = np.array([0.90, 1.00, 1.10])                   # 原生满幅亮度 => 偏蓝原生白点
TARGET_GAMMA = 2.2
LUT_SIZE = 1024        # 校正 LUT 输入点数（10bit）
LUT_OUT_BITS = 10      # 校正 LUT 输出位宽

M709 = np.array([[0.4124564, 0.3575761, 0.1804375],
                 [0.2126729, 0.7151522, 0.0721750],
                 [0.0193339, 0.1191920, 0.9503041]])
WHITE_XYZ = M709 @ np.ones(3)


def panel_native(v, ch):
    """面板原生响应: 码值 -> 归一化线性亮度（单调）"""
    v = np.clip(v, 0.0, 1.0)
    base = np.power(v, NATIVE_GAMMA[ch])
    s = S_CURVE[ch]
    return np.clip(base + s * np.sin(np.pi * v) * base * (1 - v) * 2, 0.0, 1.0)


def build_luts():
    """生成三条校正 LUT（数值求逆 + 量化）"""
    s = AMP.min()                                    # 满足所有通道可达的最大目标亮度
    dense = np.linspace(0, 1, 65536)
    vin = np.linspace(0, 1, LUT_SIZE)
    luts = []
    for i, ch in enumerate('RGB'):
        resp = panel_native(dense, ch)
        resp = np.maximum.accumulate(resp)           # 保证单调便于求逆
        target = s * np.power(vin, TARGET_GAMMA) / AMP[i]
        code = np.interp(target, resp, dense)
        q = 2 ** LUT_OUT_BITS - 1
        luts.append(np.round(code * q) / q)
    return np.stack(luts), s


def apply_lut(v, luts):
    """v: (..., 3) in [0,1] -> 校正后码值"""
    x = np.linspace(0, 1, LUT_SIZE)
    return np.stack([np.interp(v[..., i], x, luts[i]) for i in range(3)], axis=-1)


def panel_emit(code, s):
    """校正后码值 -> 三通道线性亮度（相对 D65 白 = 1）"""
    lin = np.stack([AMP[i] * panel_native(code[..., i], ch) for i, ch in enumerate('RGB')], axis=-1)
    return lin / s


def to_xyz(lin):
    return lin @ M709.T


def xyz_to_uv(xyz):
    X, Y, Z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    d = np.maximum(X + 15 * Y + 3 * Z, 1e-12)
    return np.stack([4 * X / d, 9 * Y / d], axis=-1)


def xyz_to_lab(xyz):
    t = xyz / WHITE_XYZ
    f = np.where(t > (6 / 29) ** 3, np.cbrt(t), t / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], axis=-1)


def path_a(v, k, luts, s):
    return panel_emit(apply_lut(np.clip(v * k, 0, 1), luts), s)


def path_b(v, k, luts, s):
    return panel_emit(np.clip(apply_lut(v, luts) * k, 0, 1), s)


def gray_test(luts, s):
    uv_d65 = xyz_to_uv(WHITE_XYZ)
    grays = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    ks = [0.6, 0.8, 1.0, 1.2]

    print("\n[校验] 无模型时校正后灰阶 Δu'v' (应≈0):",
          f"{max(np.abs(xyz_to_uv(to_xyz(panel_emit(apply_lut(np.full(3, g), luts), s))) - uv_d65).max() for g in grays):.5f}")

    print("\n=== 灰阶测试: 白点偏移 Δu'v'（B 相对 D65；>0.004 人眼可见）/ ΔE76(B vs A) / 亮度增益 ===")
    print(f"{'V':>5} {'k':>4} | {'A Δuv':>8} {'B Δuv':>8} | {'ΔE(A,B)':>8} | {'增益A':>7} {'增益B':>7} {'k^2.2':>7}")
    for g in grays:
        y_in = to_xyz(path_a(np.full(3, g), 1.0, luts, s))[1]
        for k in ks:
            la, lb = path_a(np.full(3, g), k, luts, s), path_b(np.full(3, g), k, luts, s)
            xa, xb = to_xyz(la), to_xyz(lb)
            dua = np.linalg.norm(xyz_to_uv(xa) - uv_d65)
            dub = np.linalg.norm(xyz_to_uv(xb) - uv_d65)
            de = np.linalg.norm(xyz_to_lab(xa) - xyz_to_lab(xb))
            print(f"{g:5.2f} {k:4.1f} | {dua:8.5f} {dub:8.5f} | {de:8.3f} | "
                  f"{xa[1] / y_in:7.3f} {xb[1] / y_in:7.3f} {min(k * g, 1) ** 2.2 / g ** 2.2:7.3f}")


def image_test(in_dir, out_dir, save_dir, luts, s):
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n=== 真实图像测试: k = sum(out)/sum(in)，B 相对 A 的 ΔE76 ===")
    for n in sorted(os.listdir(in_dir)):
        if not n.lower().endswith('.png'):
            continue
        i = cv2.imread(os.path.join(in_dir, n))[..., ::-1].astype(np.float32) / 255
        o = cv2.imread(os.path.join(out_dir, n))[..., ::-1].astype(np.float32) / 255
        i, o = i[::4, ::4], o[::4, ::4]              # 降采样加速
        si = i.sum(-1)
        k = np.where(si > 1e-3, o.sum(-1) / np.maximum(si, 1e-6), 1.0)[..., None]
        la, lb = path_a(i, k, luts, s), path_b(i, k, luts, s)
        de = np.linalg.norm(xyz_to_lab(to_xyz(la)) - xyz_to_lab(to_xyz(lb)), axis=-1)
        gray = np.ptp(i, axis=-1) < 2 / 255
        uv_d65 = xyz_to_uv(WHITE_XYZ)
        duv_b = np.linalg.norm(xyz_to_uv(to_xyz(lb)) - uv_d65, axis=-1)[gray & (si > 0.15)]
        print(f"{n}: ΔE 均值={de.mean():.3f} p99={np.percentile(de, 99):.3f} 最大={de.max():.3f} "
              f"ΔE>2占比={(de > 2).mean() * 100:.2f}% | 灰色像素B路径 Δu'v' 均值="
              f"{duv_b.mean() if duv_b.size else float('nan'):.5f}")
        cv2.imwrite(os.path.join(save_dir, n), np.clip(de / 5 * 255, 0, 255).astype(np.uint8))
    print(f"[save] ΔE 热力图（0-5 映射到 0-255）: {save_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_dir', default=None, help='模型输入目录')
    ap.add_argument('--out_dir', default=None, help='模型输出目录')
    ap.add_argument('--save_dir', default='verify_gamma_lut_order')
    args = ap.parse_args()

    luts, s = build_luts()
    print("面板假设: 原生 gamma", NATIVE_GAMMA, "S 扭曲", S_CURVE, "满幅", AMP.tolist())
    print(f"校正目标: 2.2 + D65，峰值利用率 s={s:.3f}")
    for v in [0.2, 0.5, 0.8]:
        print(f"  灰 {v:.1f} 经 LUT -> RGB 码值 {np.round(apply_lut(np.full(3, v), luts), 4)}")

    gray_test(luts, s)
    if args.in_dir and args.out_dir:
        image_test(args.in_dir, args.out_dir, args.save_dir, luts, s)


if __name__ == '__main__':
    main()
