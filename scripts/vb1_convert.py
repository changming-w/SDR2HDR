#!/usr/bin/env python3
"""Batch-convert VB1 PNG frames to normal display order."""

from __future__ import annotations

import argparse
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from PIL import Image


def convert_vb1(image: np.ndarray, horizontal_flip: bool = True) -> np.ndarray:
    """Apply half-frame swap, reverse each 8-pixel lane, then optionally mirror."""
    if image.ndim not in (2, 3):
        raise ValueError(f"Unsupported image shape: {image.shape}")

    height, width = image.shape[:2]
    if width % 8 != 0:
        raise ValueError(f"Image width must be divisible by 8, got {width}")

    converted = np.roll(image, -(width // 2), axis=1)
    lane_shape = (height, width // 8, 8) + image.shape[2:]
    converted = converted.reshape(lane_shape)[:, :, ::-1, ...].reshape(image.shape)

    if horizontal_flip:
        converted = converted[:, ::-1, ...]

    return np.ascontiguousarray(converted)


def process_single(args_tuple: tuple) -> tuple[str, str, int, int]:
    """Process a single image file. Returns (status, path, width, height)."""
    src_path, dst_path, horizontal_flip, overwrite = args_tuple

    if dst_path.exists() and not overwrite:
        return ("SKIP", str(dst_path), 0, 0)

    with Image.open(src_path) as pil_image:
        image = np.asarray(pil_image)
        converted = convert_vb1(image, horizontal_flip=horizontal_flip)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(converted, mode=pil_image.mode).save(dst_path)

    return ("OK", str(dst_path), image.shape[1], image.shape[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-convert VB1 PNG frames: swap half-frames, reverse 8-pixel lanes, and horizontally flip."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing input PNG files")
    parser.add_argument("output_dir", type=Path, help="Directory for converted PNG files")
    parser.add_argument(
        "--no-horizontal-flip",
        action="store_true",
        help="Do not horizontally flip the image after VB1 lane conversion",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process PNG files recursively and preserve subdirectories",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files that already exist in the output directory",
    )
    parser.add_argument(
        "-j", "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    if input_dir == output_dir:
        raise SystemExit("Input and output directories must be different")

    pattern = "**/*.png" if args.recursive else "*.png"
    sources = sorted(input_dir.glob(pattern))
    if not sources:
        raise SystemExit(f"No PNG files found in: {input_dir}")

    horizontal_flip = not args.no_horizontal_flip

    # Build task list
    tasks = []
    for source in sources:
        relative_path = source.relative_to(input_dir)
        destination = output_dir / relative_path
        tasks.append((source, destination, horizontal_flip, args.overwrite))

    print(f"Processing {len(tasks)} files with {args.workers} workers...")

    converted_count = 0
    skipped_count = 0

    with Pool(processes=args.workers) as pool:
        for i, (status, path, w, h) in enumerate(
            pool.imap_unordered(process_single, tasks), 1
        ):
            if status == "SKIP":
                skipped_count += 1
            else:
                converted_count += 1
                print(f"[{i}/{len(tasks)}] OK    {Path(path).name} ({w}x{h})")

    print(
        f"Done: converted={converted_count}, skipped={skipped_count}, "
        f"output={output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
