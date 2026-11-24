#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绿色色块提取脚本
=================
在 SDR 图像上按亮度从暗到亮选取 N 个绿色像素，以该像素为中心绘制 SxS 纯色块。
EDR/GT 在完全相同的坐标绘制同尺寸色块，色值取各自该点的像素值。

亮度分档策略:
    统计绿色像素的实际亮度区间（用百分位抗噪），在该区间内线性均分 N 档，
    再吸附到真实存在的亮度值。某档无候选时自动上下平移搜索。

用法:
    # 单图模式
    python extract_green_blocks.py --sdr sdr.png --edr edr.png --out ./out

    # 仅 SDR 模式（不需要 EDR）
    python extract_green_blocks.py --sdr sdr.png --out ./out --sdr-only

    # 目录模式（批量处理）
    python extract_green_blocks.py --sdr ./sdr_dir --edr ./edr_dir --out ./out

    # 顶部排列色块
    python extract_green_blocks.py --sdr sdr.png --edr edr.png --out ./out --top-patches
"""

import os
import csv
import json
import copy
import re
import shutil
import argparse

import numpy as np
import cv2


# ==================== 基础工具函数 ====================

def box_range(cy, cx, size, anchor="floor"):
    """
    以 (cy, cx) 为中心计算 size x size 区域的坐标范围。

    Args:
        cy, cx: 中心点坐标
        size: 色块边长
        anchor: 偶数尺寸时的对齐方式
            - "floor": 上/左多一行（像素位于块内 size//2 位置）
            - "ceil":  下/右多一行（像素位于块内 size//2-1 位置）

    Returns:
        (y0, y1, x0, x1, up, left): 半开区间 [y0, y1), [x0, x1)
    """
    if size % 2 == 1:
        up = left = size // 2
    else:
        up = left = size // 2 if anchor == "floor" else size // 2 - 1
    return cy - up, cy - up + size, cx - left, cx - left + size, up, left


def load_img(path):
    """
    加载图片为 float32 BGR 格式，值域 [0, 1]。
    支持 uint8、uint16、float 格式。
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = img[:, :, :3]
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        denom = 1023.0 if img.max() <= 1023 else 65535.0
        return img.astype(np.float32) / denom
    return np.clip(img.astype(np.float32), 0.0, 1.0)


def luma(bgr, coef="709"):
    """
    计算 BT.709 或 BT.601 亮度。

    Args:
        bgr: BGR 格式图像
        coef: "709" 或 "601"

    Returns:
        亮度图，值域 [0, 1]
    """
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    if coef == "601":
        return 0.299 * r + 0.587 * g + 0.114 * b
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def green_mask(bgr, s_min, v_min, dom, h_lo=35, h_hi=90):
    """
    绿色像素检测：HSV 色相区间 + G 通道主导，双判据。

    Args:
        bgr: BGR 格式图像，值域 [0, 1]
        s_min: 最低饱和度
        v_min: 最低明度
        dom: G 通道相对于 R/B 的最小差值
        h_lo, h_hi: 绿色色相范围（OpenCV 0-179）

    Returns:
        (mask, saturation): 绿色掩码和饱和度图
    """
    u8 = np.clip(bgr * 255.0, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(u8, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0]
    s = hsv[..., 1] / 255.0
    v = hsv[..., 2] / 255.0

    # 判据1: HSV 色相区间
    m_hsv = (h >= h_lo) & (h <= h_hi) & (s >= s_min) & (v >= v_min)
    # 判据2: G 通道主导
    g = bgr[..., 1]
    m_dom = (g - np.maximum(bgr[..., 0], bgr[..., 2])) > dom

    return m_hsv & m_dom, s


# ==================== 亮度分档算法 ====================

def plan_levels(Yv, n, lo_pct, hi_pct, min_count, min_gap_warn=25.0):
    """
    在实际亮度区间内线性均分 n 档，并吸附到真实存在的亮度值。

    Args:
        Yv: 绿色像素的亮度值数组（已排序）
        n: 要分的档数
        lo_pct, hi_pct: 亮度区间的百分位边界（抗噪）
        min_count: 某亮度值至少需有多少像素才算可用
        min_gap_warn: 最小间隔告警阈值

    Returns:
        (targets, tols, info):
            - targets: 每档的目标亮度值
            - tols: 每档的容忍范围（与邻档间距的一半）
            - info: 统计信息
    """
    # 统计每个亮度值的像素数量
    cnt = np.bincount(np.clip(np.round(Yv), 0, 255).astype(int), minlength=256)
    lo = float(np.percentile(Yv, lo_pct))
    hi = float(np.percentile(Yv, hi_pct))

    # 找出可用的亮度值
    vals = np.nonzero(cnt >= min_count)[0].astype(float)
    vals = vals[(vals >= lo - 0.5) & (vals <= hi + 0.5)]

    if len(vals) < n:
        raise RuntimeError(
            f"区间 [{lo:.0f},{hi:.0f}] 内可用亮度值仅 {len(vals)} 个(<{n}), "
            f"请降低 --min-count / 放宽绿色判据 / 减少 --levels")

    # 在可用亮度区间内均分 n 档
    ideal = np.linspace(vals[0], vals[-1], n)
    targets, prev = [], -1e9
    for k, t in enumerate(ideal):
        pool = vals[vals > prev]  # 保证严格递增
        rest = n - k - 1  # 给后续档位留够空间
        if len(pool) > rest:
            pool = pool[:len(pool) - rest]
        targets.append(float(pool[np.argmin(np.abs(pool - t))]))
        prev = targets[-1]
    targets = np.array(targets, dtype=np.float64)

    # 计算每档的容忍范围（与最近邻档间距的一半，留 5% 余量）
    d = np.diff(targets)
    tols = np.empty(n)
    tols[0], tols[-1] = d[0] / 2.0, d[-1] / 2.0
    for k in range(1, n - 1):
        tols[k] = min(d[k - 1], d[k]) / 2.0
    tols = np.maximum(tols * 0.95, 0.5)

    info = dict(
        range=[float(vals[0]), float(vals[-1])],
        gap_mean=float(d.mean()),
        gap_min=float(d.min()),
        n_support=int(len(vals))
    )

    if d.min() < min_gap_warn:
        print(f"[warn] 实际亮度跨度仅 {vals[-1] - vals[0]:.0f}, "
              f"均分后最小间隔 {d.min():.1f} < {min_gap_warn:.0f}, "
              f"相邻色块差异较小(非错误)")

    return targets, tols, info


# ==================== 空间约束求解 ====================

def nms(cands, topk, gap):
    """
    档内 NMS（非极大值抑制）：保留互不重叠的前 topk 个候选。

    Args:
        cands: 候选列表 [(cost, y, x), ...]，已按 cost 升序
        topk: 保留数量
        gap: 不重叠所需的最小 Chebyshev 距离
    """
    kept = []
    for c, y, x in cands:
        if all(max(abs(y - yy), abs(x - xx)) >= gap for _, yy, xx in kept):
            kept.append((c, y, x))
            if len(kept) >= topk:
                break
    return kept


def solve(bins, gap):
    """
    回溯搜索：每档选一个中心点，两两 Chebyshev 距离 >= gap（矩形严格不相交）。
    稀缺档优先搜索，提高效率。

    Args:
        bins: 每档的候选列表
        gap: 不重叠所需的最小距离

    Returns:
        dict: {档号: (y, x)} 或 None（无解）
    """
    order = sorted(range(len(bins)), key=lambda i: len(bins[i]))  # 稀缺档优先
    chosen = {}

    def dfs(i):
        if i == len(order):
            return True
        bi = order[i]
        for _, y, x in bins[bi]:
            if all(max(abs(y - yy), abs(x - xx)) >= gap for yy, xx in chosen.values()):
                chosen[bi] = (y, x)
                if dfs(i + 1):
                    return True
                del chosen[bi]
        return False

    return chosen if dfs(0) else None


# ==================== 主流程 ====================

def run(a):
    """处理单张图片的主流程。"""
    os.makedirs(a.out, exist_ok=True)
    S = a.size  # 色块尺寸

    # 1. 加载图像
    sdr = load_img(a.sdr)
    if getattr(a, 'sdr_only', False):
        edr = np.zeros_like(sdr)
    else:
        edr = load_img(a.edr)

    # 检查分辨率一致性
    if sdr.shape[:2] != edr.shape[:2]:
        raise SystemExit(
            f"[error] SDR {sdr.shape[1]}x{sdr.shape[0]} 与 EDR "
            f"{edr.shape[1]}x{edr.shape[0]} 分辨率不一致")

    # 2. Resize 到 4K (3840x2160)
    target_h, target_w = 2160, 3840
    orig_h, orig_w = sdr.shape[:2]
    if (orig_h, orig_w) != (target_h, target_w):
        print(f"[info] Resize {orig_w}x{orig_h} -> {target_w}x{target_h}")
        sdr = cv2.resize(sdr, (target_w, target_h), interpolation=cv2.INTER_AREA)
        if getattr(a, 'sdr_only', False):
            edr = np.zeros_like(sdr)
        else:
            edr = cv2.resize(edr, (target_w, target_h), interpolation=cv2.INTER_AREA)

    H, W = edr.shape[:2]
    N = a.levels

    # 3. 计算亮度图和绿色掩码
    Y_edr = luma(edr, a.coef) * 255.0

    # 中心点合法范围（保证整块落在图内，且色块四周与图像边界至少留 safe 像素安全间隔）
    _, _, _, _, up, left = box_range(0, 0, S, a.anchor)
    down, right = S - 1 - up, S - 1 - left
    safe = max(0, int(getattr(a, 'safe_margin', 50)))
    inner = np.zeros((H, W), dtype=bool)
    inner[max(up + safe, 1):H - max(down + safe, 1),
          max(left + safe, 1):W - max(right + safe, 1)] = True

    Y = luma(sdr, a.coef) * 255.0
    gm, sat = green_mask(sdr, a.s_min, a.v_min, a.dom, a.h_lo, a.h_hi)
    gm &= inner

    ys, xs = np.nonzero(gm)
    if len(ys) == 0:
        raise SystemExit("[error] SDR 中未检出符合条件的绿色像素, 放宽 --s-min / --dom")

    # 按亮度升序排序
    Yv, Sv = Y[ys, xs], sat[ys, xs]
    o = np.argsort(Yv)
    ys, xs, Yv, Sv = ys[o], xs[o], Yv[o], Sv[o]

    print(f"[info] 图像 {W}x{H}, 块尺寸 {S}x{S}, "
          f"中心像素位于块内 ({up},{left})"
          f"{' [严格居中]' if S % 2 else ' [偶数尺寸, 半像素偏心]'}")
    print(f"[info] 基于 SDR 选择绿色像素位置")
    print(f"[info] 候选绿色像素 {len(ys)}, 亮度 [{Yv[0]:.1f}, {Yv[-1]:.1f}], "
          f"P{a.lo_pct}-P{a.hi_pct} [{np.percentile(Yv, a.lo_pct):.1f}, "
          f"{np.percentile(Yv, a.hi_pct):.1f}]")

    # 4. 计算分档目标
    targets, tols, info = plan_levels(Yv, N, a.lo_pct, a.hi_pct,
                                      a.min_count, a.min_gap)
    print(f"[plan] 区间 [{info['range'][0]:.0f}, {info['range'][1]:.0f}] "
          f"均分 {N} 档, 平均间隔 {info['gap_mean']:.1f} (最小 {info['gap_min']:.1f})")
    print(f"[plan] 档位: {', '.join(f'{t:.0f}' for t in targets)}")

    # 5. 求解空间约束
    cnt_all = np.bincount(np.clip(np.round(Yv), 0, 255).astype(int), minlength=256)
    vals_ok = np.nonzero(cnt_all >= a.min_count)[0].astype(float)
    sgap = S + a.margin  # 不重叠所需的最小 Chebyshev 距离
    tg = targets.copy()
    tl = tols.copy()
    topk = a.topk

    def build_bins(tg, tl, topk):
        """为每档构建候选列表。"""
        bins = []
        for k, t in enumerate(tg):
            i0 = int(np.searchsorted(Yv, t - tl[k]))
            i1 = int(np.searchsorted(Yv, t + tl[k]))
            if i1 <= i0:
                bins.append([])
                continue
            # 代价 = 亮度贴近度 + 饱和度（绿得纯）
            c = np.abs(Yv[i0:i1] - t) / max(tl[k], 1e-6) + 0.4 * (1.0 - Sv[i0:i1])
            idx = np.argsort(c)[:a.pool]
            cands = [(float(c[i]), int(ys[i0 + i]), int(xs[i0 + i])) for i in idx]
            bins.append(nms(cands, topk, sgap))
        return bins

    sel = None
    for it in range(a.max_iter):
        bins = build_bins(tg, tl, topk)
        empty = [k for k in range(N) if not bins[k]]

        if empty:
            # 空档：在不与邻档窗口重叠的前提下，平移到最近的可用亮度
            moved = []
            for k in empty:
                lo_b = (tg[k - 1] + tl[k - 1] + tl[k]) if k > 0 else float(Yv[0])
                hi_b = (tg[k + 1] - tl[k + 1] - tl[k]) if k < N - 1 else float(Yv[-1])
                cand = [v for v in vals_ok if lo_b <= v <= hi_b and
                        np.searchsorted(Yv, v + tl[k]) -
                        np.searchsorted(Yv, v - tl[k]) > 0]
                if not cand:
                    continue
                nv = float(min(cand, key=lambda v: abs(v - tg[k])))
                if abs(nv - tg[k]) < 1e-6:
                    continue
                moved.append(f"L{k}: {tg[k]:.0f}->{nv:.0f}")
                tg[k] = nv
            if moved:
                print(f"[fix#{it}] 空档平移 " + "; ".join(moved))
                continue
            print(f"[fix#{it}] 档位 {empty} 仍无候选, tol 放宽 1.3x")
            tl = tl * 1.3
            continue

        sel = solve(bins, sgap)
        if sel is not None and len(sel) == N:
            print(f"[ok] 第 {it} 轮求解成功")
            print(f"[ok] 最终档位: {', '.join(f'{t:.0f}' for t in tg)}")
            break
        k = int(np.argmin([len(b) for b in bins]))
        print(f"[fix#{it}] 空间冲突(最紧 L{k}, 候选 {len(bins[k])}), 扩 topk 与 tol")
        topk += 8
        tl = tl * 1.15
        sel = None

    if sel is None:
        raise RuntimeError("多轮调整后仍无解: 试试减少 --levels / 减小 --size / "
                           "--margin 0 / 放宽 --s-min --dom")

    # 6. 采样（从原图读取，不从已绘制的画布读）
    picks = []
    for k in sorted(sel, key=lambda i: tg[i]):
        cy, cx = sel[k]
        if a.stabilize:  # 可选 3x3 中值抗噪
            pe = np.median(edr[cy - 1:cy + 2, cx - 1:cx + 2].reshape(-1, 3), axis=0)
            ps = np.median(sdr[cy - 1:cy + 2, cx - 1:cx + 2].reshape(-1, 3), axis=0)
        else:
            pe, ps = edr[cy, cx], sdr[cy, cx]
        ce = tuple(int(round(v * 255)) for v in np.clip(pe, 0, 1))  # EDR 像素值
        cs = tuple(int(round(v * 255)) for v in np.clip(ps, 0, 1))  # SDR 像素值
        picks.append((k, cy, cx, ce, cs))

    # 7. 绘制色块
    edr_o = np.clip(edr * 255, 0, 255).astype(np.uint8).copy()
    sdr_o = np.clip(sdr * 255, 0, 255).astype(np.uint8).copy()
    ce_list, cs_list, rows = [], [], []

    # 顶部参考块：白块 + 中灰块（位置默认对齐给定 dataset.json：gray=3600, white=3700, y=140）
    margin_edge = 100
    gx, gy, wx, wy = a.gray_x, a.gray_y, a.white_x, a.white_y
    for im in (edr_o, sdr_o):
        cv2.rectangle(im, (wx, wy), (wx + S - 1, wy + S - 1), (255, 255, 255), -1)   # 白块
        cv2.rectangle(im, (gx, gy), (gx + S - 1, gy + S - 1), (128, 128, 128), -1)   # 中灰块

    # 可选：在顶部排列所有绿色色块（从暗到亮）
    if a.top_patches:
        patch_colors_sdr = []
        patch_colors_edr = []
        for k, cy, cx, ce, cs in sorted(picks, key=lambda x: x[4][1]):  # 按 G 通道排序
            patch_colors_sdr.append(cs)
            patch_colors_edr.append(ce)
        # 不添加白块，只要 10 个绿色

        margin_gap = 20
        total_width = len(patch_colors_sdr) * S + (len(patch_colors_sdr) - 1) * margin_gap

        # 获取所有原始色块的 x 范围
        original_patch_x_ranges = []
        for k, cy, cx, ce, cs in picks:
            y0, y1, x0, x1, _, _ = box_range(cy, cx, S, a.anchor)
            original_patch_x_ranges.append((x0, x1))

        top_row_y0 = margin_edge
        top_row_y1 = margin_edge + S

        def check_overlap(start_x):
            """检查顶部色块行是否与原始色块重叠。"""
            for i in range(len(patch_colors_sdr)):
                tx0 = start_x + i * (S + margin_gap)
                tx1 = tx0 + S
                for ox0, ox1 in original_patch_x_ranges:
                    for k, cy, cx, ce, cs in picks:
                        oy0, oy1, _, _, _, _ = box_range(cy, cx, S, a.anchor)
                        if not (tx1 < ox0 or tx0 > ox1 or top_row_y1 < oy0 or top_row_y0 > oy1):
                            return True
            return False

        # 搜索不重叠的位置
        start_x = (W - total_width) // 2
        min_x = margin_edge
        max_x = W - total_width - margin_edge
        start_x = max(min_x, min(start_x, max_x))

        if check_overlap(start_x):
            found = False
            for offset in range(10, W, 10):  # 10px 步进搜索
                candidate = start_x - offset
                if candidate >= min_x and not check_overlap(candidate):
                    start_x = candidate
                    found = True
                    break
                candidate = start_x + offset
                if candidate <= max_x and not check_overlap(candidate):
                    start_x = candidate
                    found = True
                    break
            if not found:
                print(f"[warn] 无法找到不重叠的顶部色块位置，使用居中位置")

        for i, (cs, ce) in enumerate(zip(patch_colors_sdr, patch_colors_edr)):
            x0 = start_x + i * (S + margin_gap)
            y0 = margin_edge
            edr_o[y0:y0 + S, x0:x0 + S] = ce
            sdr_o[y0:y0 + S, x0:x0 + S] = cs

    # 绘制选中的色块
    for k, cy, cx, ce, cs in picks:
        y0, y1, x0, x1, _, _ = box_range(cy, cx, S, a.anchor)
        edr_o[y0:y1, x0:x1] = ce
        sdr_o[y0:y1, x0:x1] = cs

        # 自检
        assert (y1 - y0, x1 - x0) == (S, S), "块尺寸不完整"
        assert (y0 + up, x0 + left) == (cy, cx), "块未以该像素为中心"
        if not a.stabilize:
            assert tuple(int(round(v * 255)) for v in np.clip(edr[cy, cx], 0, 1)) == ce
            assert tuple(int(round(v * 255)) for v in np.clip(sdr[cy, cx], 0, 1)) == cs

        # 可选：画红框
        if a.outline:
            for im in (edr_o, sdr_o):
                cv2.rectangle(im, (x0, y0), (x0 + S - 1, y0 + S - 1), (0, 0, 255), 1)

        # 可选：在色块左侧写档位数字编号
        if a.label:
            text_x = max(5, x0 - 25)
            text_y = y0 + S // 2 + 8
            for im in (edr_o, sdr_o):
                cv2.putText(im, str(k), (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(im, str(k), (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        # 可选：中心画十字
        if a.mark_center:
            for im in (edr_o, sdr_o):
                cv2.drawMarker(im, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 9, 1)

        ce_list.append(np.full((S, S, 3), ce, np.uint8))
        cs_list.append(np.full((S, S, 3), cs, np.uint8))
        rows.append(dict(
            level=k, target_Y=round(float(tg[k]), 2),
            cy=int(cy), cx=int(cx),
            box_y0=int(y0), box_x0=int(x0), box_y1=int(y1 - 1), box_x1=int(x1 - 1),
            edr_Y=round(float(Y_edr[cy, cx]), 2),
            edr_R=ce[2], edr_G=ce[1], edr_B=ce[0],
            sdr_Y=round(float(Y[cy, cx]), 2),
            sdr_R=cs[2], sdr_G=cs[1], sdr_B=cs[0]))

    # 8. 保存结果
    cv2.imwrite(os.path.join(a.out, "sdr_with_patch.png"), sdr_o)
    if not getattr(a, 'sdr_only', False):
        cv2.imwrite(os.path.join(a.out, "edr_with_patch.png"), edr_o)

    sep = np.full((8, S * len(rows), 3), 255, np.uint8)
    if not getattr(a, 'sdr_only', False):
        cv2.imwrite(os.path.join(a.out, "colorchart.png"),
                    np.vstack([np.hstack(ce_list), sep, np.hstack(cs_list)]))
    else:
        cv2.imwrite(os.path.join(a.out, "colorchart.png"), np.hstack(cs_list))

    with open(os.path.join(a.out, "patches.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with open(os.path.join(a.out, "patches.json"), "w", encoding="utf-8") as f:
        json.dump(dict(plan=dict(targets=tg.tolist(), **info),
                       size=S, anchor=a.anchor, levels=rows),
                  f, indent=2, ensure_ascii=False)

    # 打印结果表格
    print("\n L   target  edr_Y  Δprev  EDR(R,G,B)      sdr_Y  SDR(R,G,B)     center")
    prev = None
    for r in rows:
        d = "  -  " if prev is None else f"{r['edr_Y'] - prev:>5.1f}"
        prev = r['edr_Y']
        print(f" L{r['level']:<2d} {r['target_Y']:>6.1f} {r['edr_Y']:>6.1f} {d} "
              f"({r['edr_R']:>3},{r['edr_G']:>3},{r['edr_B']:>3})  "
              f"{r['sdr_Y']:>6.1f} ({r['sdr_R']:>3},{r['sdr_G']:>3},{r['sdr_B']:>3})  "
              f"({r['cx']:>5},{r['cy']:>5})")
    print(f"\n[done] -> {a.out}")
    return rows


# ==================== 批量处理 ====================

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def make_stem(idx, filename):
    """规范化图片名：{序号}_{原名开头数字}，无数字则仅用序号。无中文、无短横线。"""
    base = os.path.splitext(os.path.basename(filename))[0]
    m = re.match(r"\D*(\d+)", base)
    num = m.group(1) if m else ""
    return f"{idx:03d}_{num}" if num else f"{idx:03d}"


def items_from_rows(rows, idx, image, a):
    """由某图分档结果生成 dataset.json items：original(green) + gray + white。
    顶部不含绿色块(green-top)，只保留白/灰参考块；x,y 为色块【中心】坐标(与软件参考一致)，rgb 取 SDR 值。
    绘制时 gray/white 用左上角 a.gray_x/a.white_x，故 json 中心 = 左上角 + size//2，物理位置不变。"""
    pid = f"{idx:03d}"
    h = a.size // 2
    items = []
    for r in sorted(rows, key=lambda r: r["level"]):        # original：原图采样中心像素
        items.append({
            "image": image,
            "id": f"{pid}-original-L{r['level']}",
            "type": "green",
            "x": int(r["cx"]), "y": int(r["cy"]),
            "rgb": [int(r["sdr_R"]), int(r["sdr_G"]), int(r["sdr_B"])],
        })
    items.append({"image": image, "id": f"{pid}-gray", "type": "gray",
                  "x": int(a.gray_x + h), "y": int(a.gray_y + h), "rgb": [128, 128, 128]})
    items.append({"image": image, "id": f"{pid}-white", "type": "white",
                  "x": int(a.white_x + h), "y": int(a.white_y + h), "rgb": [255, 255, 255]})
    return items


def write_dataset(out_dir, items, a):
    """写出与给定模板同结构的 dataset.json。"""
    dataset = {
        "version": 1,
        "name": a.dataset_name,
        "width": 3840,
        "height": 2160,
        "patch_size": a.size,
        "items": items,
    }
    path = os.path.join(out_dir, a.dataset_json)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    return path


def main():
    p = argparse.ArgumentParser(
        description="EDR/SDR 配对绿色色块生成（以选中像素为中心，纯色填充）")

    # 输入输出
    p.add_argument("--sdr", required=True, help="SDR 图片路径或目录")
    p.add_argument("--edr", required=False,
                   help="EDR GT 图片路径或目录（须与 SDR 同分辨率），--sdr-only 时可省略")
    p.add_argument("--out", default="./out", help="输出根目录")

    # 色块参数
    p.add_argument("--levels", type=int, default=10, help="色块数量/亮度档数")
    p.add_argument("--size", type=int, default=80, help="色块边长")
    p.add_argument("--anchor", default="floor", choices=["floor", "ceil"],
                   help="偶数尺寸时中心像素偏左上(floor)或右下(ceil)")
    p.add_argument("--margin", type=int, default=0, help="色块间额外间距，0=紧贴不重叠")

    # 亮度分档参数
    p.add_argument("--min-gap", type=float, default=25.0,
                   help="亮度间隔参考值，低于它只告警不报错")
    p.add_argument("--lo-pct", type=float, default=0.5, help="亮度下界百分位（抗噪）")
    p.add_argument("--hi-pct", type=float, default=99.5, help="亮度上界百分位（抗噪）")
    p.add_argument("--min-count", type=int, default=8,
                   help="某亮度值至少需有多少像素才算可用档位")

    # 绿色检测参数
    p.add_argument("--coef", default="709", choices=["709", "601"], help="亮度权重")
    p.add_argument("--s-min", type=float, default=0.20, help="绿色最低饱和度")
    p.add_argument("--v-min", type=float, default=0.02, help="绿色最低明度")
    p.add_argument("--dom", type=float, default=0.03, help="G 通道主导阈值")
    p.add_argument("--h-lo", type=int, default=35, help="绿色色相下界（OpenCV 0-179）")
    p.add_argument("--h-hi", type=int, default=90, help="绿色色相上界")

    # 求解参数
    p.add_argument("--pool", type=int, default=20000, help="每档参与排序的候选上限")
    p.add_argument("--topk", type=int, default=15, help="每档 NMS 后保留候选数")
    p.add_argument("--max-iter", type=int, default=12, help="自动修正最大轮数")

    # 可视化选项
    p.add_argument("--outline", action="store_true", help="画红框与档位编号")
    p.add_argument("--label", action="store_true", help="在色块左侧写档位数字编号")
    p.add_argument("--mark-center", action="store_true", help="中心画十字便于核对")
    p.add_argument("--stabilize", action="store_true", help="用 3x3 中值代替单像素")
    p.add_argument("--top-patches", action="store_true", help="在顶部排列所有色块（从暗到亮）")
    p.add_argument("--sdr-only", action="store_true", help="只给 SDR 加色块，不处理 EDR/GT")

    # dataset.json 输出（格式对齐给定模板：original + gray + white，顶部不含绿色块）
    p.add_argument("--dataset-name", default="green-scenes-sdr", help="dataset.json 的 name 字段")
    p.add_argument("--dataset-json", default="dataset.json", help="dataset.json 文件名")
    p.add_argument("--gray-x", type=int, default=3600, help="灰块左上 x（默认对齐模板 3600）")
    p.add_argument("--gray-y", type=int, default=140, help="灰块左上 y（默认 140）")
    p.add_argument("--white-x", type=int, default=3700, help="白块左上 x（默认对齐模板 3700）")
    p.add_argument("--white-y", type=int, default=140, help="白块左上 y（默认 140）")
    p.add_argument("--safe-margin", type=int, default=50,
                   help="所有色块与图像边界的最小安全间隔像素（默认 50）")

    args = p.parse_args()

    # 参数检查
    if not getattr(args, 'sdr_only', False) and args.edr is None:
        p.error("--edr 是必需的，除非使用 --sdr-only")

    # 判断单图还是目录模式
    sdr_is_dir = os.path.isdir(args.sdr)
    edr_is_dir = os.path.isdir(args.edr) if args.edr else False

    if not getattr(args, 'sdr_only', False) and sdr_is_dir != edr_is_dir:
        raise SystemExit("[error] --sdr 和 --edr 必须同时为文件或同时为目录")

    if not sdr_is_dir:
        # 单图模式
        rows = run(args)
        stem = make_stem(1, args.sdr)
        img_dir = os.path.join(args.out, "images")
        os.makedirs(img_dir, exist_ok=True)
        src = os.path.join(args.out, "sdr_with_patch.png")
        if os.path.exists(src):
            shutil.copy(src, os.path.join(img_dir, stem + ".png"))
        items = items_from_rows(rows, 1, f"images/{stem}.png", args)
        path = write_dataset(args.out, items, args)
        print(f"[out]  dataset.json -> {path}  "
              f"({len(items)} 条: {len(rows)} green + gray + white)")
    else:
        # 目录模式
        sdr_files = {f for f in os.listdir(args.sdr)
                     if os.path.splitext(f)[1].lower() in IMAGE_EXTS
                     and '_depth' not in f and '_weight' not in f}

        if getattr(args, 'sdr_only', False):
            common = sorted(sdr_files)
        else:
            edr_files = {f for f in os.listdir(args.edr)
                         if os.path.splitext(f)[1].lower() in IMAGE_EXTS}
            common = sorted(sdr_files & edr_files)

        if not common:
            if getattr(args, 'sdr_only', False):
                raise SystemExit(f"[error] SDR 目录中没有图片:\n  SDR: {args.sdr}")
            else:
                raise SystemExit(f"[error] 目录中没有共同图片:\n  SDR: {args.sdr}\n  EDR: {args.edr}")

        print(f"[info] 找到 {len(common)} 对图片")

        # 创建输出目录
        sdr_out_dir = os.path.join(args.out, "SDR")
        os.makedirs(sdr_out_dir, exist_ok=True)
        if not getattr(args, 'sdr_only', False):
            gt_out_dir = os.path.join(args.out, "GT")
            os.makedirs(gt_out_dir, exist_ok=True)
        img_dir = os.path.join(args.out, "images")
        os.makedirs(img_dir, exist_ok=True)

        success, failed, all_items = 0, [], []
        for i, name in enumerate(common, 1):
            print(f"\n{'='*60}")
            print(f"[{i}/{len(common)}] 处理: {name}")
            print(f"{'='*60}")

            # 复制 args 并修改路径
            sub_args = copy.copy(args)
            sub_args.sdr = os.path.join(args.sdr, name)
            if args.edr:
                sub_args.edr = os.path.join(args.edr, name)
            sub_args.out = os.path.join(args.out, "_tmp_" + os.path.splitext(name)[0])

            try:
                rows = run(sub_args)

                # 收集 dataset.json items，并按规范名复制 SDR 图到 images/
                stem = make_stem(i, name)
                tmp_sdr = os.path.join(sub_args.out, "sdr_with_patch.png")
                if os.path.exists(tmp_sdr):
                    shutil.copy(tmp_sdr, os.path.join(img_dir, stem + ".png"))
                    shutil.move(tmp_sdr, os.path.join(sdr_out_dir, name))
                all_items += items_from_rows(rows, i, f"images/{stem}.png", args)

                if not getattr(args, 'sdr_only', False):
                    tmp_gt = os.path.join(sub_args.out, "edr_with_patch.png")
                    if os.path.exists(tmp_gt):
                        shutil.move(tmp_gt, os.path.join(gt_out_dir, name))

                # 清理临时目录
                shutil.rmtree(sub_args.out, ignore_errors=True)
                success += 1
            except Exception as e:
                print(f"[error] {name}: {e}")
                failed.append((name, str(e)))

        if all_items:
            ds_path = write_dataset(args.out, all_items, args)
        print(f"\n{'='*60}")
        print(f"[done] 完成 {success}/{len(common)} 张")
        print(f"[out]  SDR -> {sdr_out_dir}")
        if not getattr(args, 'sdr_only', False):
            print(f"[out]  GT  -> {gt_out_dir}")
        print(f"[out]  images -> {img_dir}")
        if all_items:
            print(f"[out]  dataset.json -> {ds_path}  ({len(all_items)} 条)")
        if failed:
            print(f"[fail] {len(failed)} 张失败:")
            for name, err in failed:
                print(f"  {name}: {err}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
