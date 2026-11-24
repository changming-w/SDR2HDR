#!/usr/bin/env python
"""Clean and select 1200 most typical dark-scene HDR/SDR pairs.

Strategy:
1. Match HDR and SDR files by name
2. Sample frames and compute luminance statistics (darkness score)
3. Filter out: too-dark (black), too-bright, corrupted frames
4. Select 1200 diverse, representative dark-scene pairs
5. Copy to /data/wcm/datasets/dark_scene_1200/

Selection criteria for "typical dark scenes":
- Mean luminance in a good dark range (not pure black, not bright)
- Sufficient content (not blank/uniform)
- Proportional representation from all video sources
- Temporal diversity (avoid consecutive frames)
"""
import os
import shutil
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2

# Paths
HDR_DIR = "/mnt/data/wcm/data/hdr/train/hdrframes1"
SDR_DIR = "/mnt/data/wcm/data/hdr/train/sdrframes_hable"
OUT_DIR = "/mnt/data/wcm/data/hdr/train/dark_scene_1200"

TARGET_COUNT = 1200
SAMPLE_SIZE = 2500  # Number of frames to analyze in detail

# Luminance thresholds (in PQ code value space, 0-1)
# These are for 16-bit PQ encoded HDR
MIN_MEAN_LUMA = 0.02    # Too dark if below this (nearly black)
MAX_MEAN_LUMA = 0.45    # Too bright if above this (not a dark scene)
MIN_PEAK_LUMA = 0.05    # Must have some highlights
MIN_STD_LUMA = 0.01     # Must have contrast/texture


def get_source_name(filename: str) -> str:
    """Extract video source name from filename."""
    # Remove _pXXX_XXXXXXXX.png suffix
    import re
    match = re.match(r'(.+?)_p\d+_\d+\.png', filename)
    if match:
        return match.group(1)
    return "unknown"


def compute_frame_stats(hdr_path: str) -> dict:
    """Compute luminance statistics for an HDR frame (16-bit PQ PNG).

    Returns dict with: mean_luma, peak_luma, std_luma, content_ratio
    """
    # Read at reduced size for speed
    img = cv2.imread(hdr_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    # Convert to grayscale luminance (simple average of channels for PQ)
    if img.ndim == 3:
        # Convert BGR to float [0, 1]
        img_float = img.astype(np.float64) / 65535.0
        # Use green channel as luminance proxy (fastest)
        gray = img_float[:, :, 1]  # Green channel
    else:
        gray = img.astype(np.float64) / 65535.0

    # Downsample for faster stats
    h, w = gray.shape
    if h > 270:  # Downsample to ~270p
        scale = 270 / h
        gray = cv2.resize(gray, (int(w * scale), 270), interpolation=cv2.INTER_AREA)

    stats = {
        'mean_luma': float(np.mean(gray)),
        'peak_luma': float(np.percentile(gray, 99.5)),  # 99.5th percentile as peak
        'std_luma': float(np.std(gray)),
        'content_ratio': float(np.mean(gray > 0.01)),  # Fraction of non-black pixels
    }
    return stats


def score_dark_frame(stats: dict) -> float:
    """Score a frame for being a 'typical dark scene'. Higher = better.

    Criteria:
    - Mean luminance in sweet spot for dark scenes
    - Has some highlights (peak)
    - Has texture/contrast (std)
    - Has meaningful content
    """
    if stats is None:
        return -1

    mean_l = stats['mean_luma']
    peak_l = stats['peak_luma']
    std_l = stats['std_luma']
    content = stats['content_ratio']

    # Filter out bad frames
    if mean_l < MIN_MEAN_LUMA or mean_l > MAX_MEAN_LUMA:
        return -1
    if peak_l < MIN_PEAK_LUMA:
        return -1
    if std_l < MIN_STD_LUMA:
        return -1
    if content < 0.3:  # Less than 30% content
        return -1

    score = 0.0

    # Prefer mean luminance in typical dark range (0.05-0.25 in PQ space)
    # Bell curve centered at 0.12
    ideal_mean = 0.12
    mean_score = np.exp(-((mean_l - ideal_mean) ** 2) / (2 * 0.08 ** 2))
    score += 40 * mean_score

    # Prefer having some highlights (HDR detail in dark scenes)
    # Peak around 0.3-0.7 is good for dark HDR
    peak_score = min(peak_l / 0.5, 1.0)  # Saturate at 0.5
    score += 25 * peak_score

    # Prefer good contrast
    std_score = min(std_l / 0.10, 1.0)  # Saturate at 0.10
    score += 20 * std_score

    # Prefer high content ratio
    score += 15 * content

    return score


def main():
    print("=" * 60)
    print("Dark Scene Data Cleaning & Selection")
    print("=" * 60)

    # 1. Get all HDR and SDR files
    print("\n[1/5] Scanning directories...")
    hdr_files = sorted([f for f in os.listdir(HDR_DIR) if f.endswith('.png')])
    sdr_files = set(f for f in os.listdir(SDR_DIR) if f.endswith('.png'))

    # Find matching pairs
    paired_files = [f for f in hdr_files if f in sdr_files]
    print(f"  HDR files: {len(hdr_files)}")
    print(f"  SDR files: {len(sdr_files)}")
    print(f"  Matched pairs: {len(paired_files)}")

    # 2. Group by source
    print("\n[2/5] Grouping by video source...")
    source_groups = defaultdict(list)
    for f in paired_files:
        source = get_source_name(f)
        source_groups[source].append(f)

    for source, files in sorted(source_groups.items()):
        print(f"  {source[:50]}...: {len(files)} frames")

    # 3. Sample and compute statistics
    print(f"\n[3/5] Analyzing frame statistics (sampling {SAMPLE_SIZE} frames)...")

    # Stratified sampling: sample proportionally from each source
    sample_files = []
    for source, files in source_groups.items():
        n_sample = max(100, int(SAMPLE_SIZE * len(files) / len(paired_files)))
        n_sample = min(n_sample, len(files))
        # Uniform sampling across the sequence
        indices = np.linspace(0, len(files) - 1, n_sample, dtype=int)
        sample_files.extend([files[i] for i in indices])

    print(f"  Sampling {len(sample_files)} frames for analysis...")

    # Compute stats for sampled frames
    frame_scores = {}
    frame_stats = {}
    failed_count = 0

    for i, fname in enumerate(sample_files):
        hdr_path = os.path.join(HDR_DIR, fname)
        stats = compute_frame_stats(hdr_path)

        if stats is None:
            failed_count += 1
            continue

        score = score_dark_frame(stats)
        frame_scores[fname] = score
        frame_stats[fname] = stats

        if (i + 1) % 200 == 0:
            valid = sum(1 for s in frame_scores.values() if s > 0)
            print(f"  {i+1}/{len(sample_files)} analyzed, {valid} valid dark frames")

    print(f"  Analysis complete. Failed reads: {failed_count}")

    # 4. Select best frames
    print(f"\n[4/5] Selecting top {TARGET_COUNT} frames...")

    # Filter valid frames (score > 0)
    valid_frames = [(f, s) for f, s in frame_scores.items() if s > 0]
    valid_frames.sort(key=lambda x: x[1], reverse=True)

    print(f"  Valid dark frames: {len(valid_frames)}")

    if len(valid_frames) < TARGET_COUNT:
        print(f"  WARNING: Only {len(valid_frames)} valid frames, less than target {TARGET_COUNT}")
        TARGET_ACTUAL = len(valid_frames)
    else:
        TARGET_ACTUAL = TARGET_COUNT

    # Select with diversity constraint:
    # Ensure proportional representation from each source
    selected = []
    source_selected = defaultdict(int)

    # Calculate target per source (proportional)
    source_targets = {}
    for source, files in source_groups.items():
        source_in_valid = sum(1 for f, s in valid_frames if get_source_name(f) == source)
        source_targets[source] = int(TARGET_ACTUAL * source_in_valid / len(valid_frames))

    print(f"  Target per source: {source_targets}")

    # First pass: fill each source to its target
    for source, target in source_targets.items():
        source_frames = [(f, s) for f, s in valid_frames if get_source_name(f) == source]
        # Add temporal diversity: skip every N frames
        step = max(1, len(source_frames) // target) if target > 0 else 1
        count = 0
        for i in range(0, len(source_frames), step):
            if count >= target:
                break
            selected.append(source_frames[i][0])
            source_selected[source] += 1
            count += 1

    # Fill remaining slots with highest-scored frames not yet selected
    selected_set = set(selected)
    for fname, score in valid_frames:
        if len(selected) >= TARGET_ACTUAL:
            break
        if fname not in selected_set:
            selected.append(fname)
            selected_set.add(fname)
            source_selected[get_source_name(fname)] += 1

    print(f"  Selected {len(selected)} frames")
    for source, count in sorted(source_selected.items()):
        print(f"    {source[:40]}...: {count}")

    # 5. Copy files
    print(f"\n[5/5] Copying files to {OUT_DIR}...")

    hdr_out = os.path.join(OUT_DIR, "hdr")
    sdr_out = os.path.join(OUT_DIR, "sdr")
    os.makedirs(hdr_out, exist_ok=True)
    os.makedirs(sdr_out, exist_ok=True)

    # Create list file
    list_path = os.path.join(OUT_DIR, "list.txt")

    copied = 0
    with open(list_path, 'w') as list_f:
        for fname in sorted(selected):
            # Copy HDR
            src_hdr = os.path.join(HDR_DIR, fname)
            dst_hdr = os.path.join(hdr_out, fname)
            if not os.path.exists(dst_hdr):
                shutil.copy2(src_hdr, dst_hdr)

            # Copy SDR
            src_sdr = os.path.join(SDR_DIR, fname)
            dst_sdr = os.path.join(sdr_out, fname)
            if not os.path.exists(dst_sdr):
                shutil.copy2(src_sdr, dst_sdr)

            # Write to list (without extension)
            name_no_ext = os.path.splitext(fname)[0]
            list_f.write(f"{name_no_ext}\n")

            copied += 1
            if copied % 100 == 0:
                print(f"  Copied {copied}/{len(selected)} pairs...")

    print(f"\n{'=' * 60}")
    print(f"DONE! {copied} pairs saved to {OUT_DIR}")
    print(f"  HDR: {hdr_out}")
    print(f"  SDR: {sdr_out}")
    print(f"  List: {list_path}")
    print(f"{'=' * 60}")

    # Print some statistics about selected frames
    print("\nSelected frame statistics:")
    selected_means = [frame_stats[f]['mean_luma'] for f in selected if f in frame_stats]
    selected_peaks = [frame_stats[f]['peak_luma'] for f in selected if f in frame_stats]
    if selected_means:
        print(f"  Mean luminance: {np.mean(selected_means):.4f} ± {np.std(selected_means):.4f}")
        print(f"  Peak luminance: {np.mean(selected_peaks):.4f} ± {np.std(selected_peaks):.4f}")


if __name__ == "__main__":
    main()
