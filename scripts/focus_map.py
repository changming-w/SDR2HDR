"""Batch-generate vegetation focus maps in HSV color space.

The grayscale output is a uint8 focus map: 0 means non-vegetation and larger
values mean a stronger vegetation response.  A color heatmap is saved beside
it for visualization.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def read_image(path: Path) -> np.ndarray:
    """Read an image, including paths containing Chinese characters."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    """Write an image, including paths containing Chinese characters."""
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix or ".png"
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise ValueError(f"无法编码图片: {path}")
    encoded.tofile(str(path))


def vegetation_focus_map(
    image_bgr: np.ndarray,
    hue_min: int = 25,
    hue_max: int = 95,
    saturation_min: int = 35,
    value_min: int = 35,
    blur_sigma: float = 5.0,
    value_gamma: float = 2.0,
) -> np.ndarray:
    """Return a soft vegetation response map in the range [0, 255].

    OpenCV represents HSV hue in [0, 179]. The default hue interval covers
    yellow-green through blue-green. Hue is only a binary vegetation mask;
    response strength inside that mask is driven by the V channel, with the
    original square-root saturation modulation retained.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue, saturation, value = cv2.split(hsv)

    # H only selects the vegetation area and does not weight its intensity.
    hue_mask = ((hue >= hue_min) & (hue <= hue_max)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    hue_mask = cv2.morphologyEx(hue_mask, cv2.MORPH_OPEN, kernel)
    hue_mask = cv2.morphologyEx(hue_mask, cv2.MORPH_CLOSE, kernel)

    sat_score = np.clip(
        (saturation - saturation_min) / max(255 - saturation_min, 1),
        0.0,
        1.0,
    )
    val_score = np.clip((value - value_min) / max(255 - value_min, 1), 0.0, 1.0)
    # Non-linear V mapping. Gamma > 1 expands the difference between darker
    # and brighter vegetation (for example, 0.5 -> 0.25 with gamma=2).
    val_score = np.power(val_score, value_gamma)

    focus = val_score * np.sqrt(sat_score) * (hue_mask.astype(np.float32) / 255.0)
    # focus = val_score  * (hue_mask.astype(np.float32) / 255.0)

    if blur_sigma > 0:
        focus = cv2.GaussianBlur(focus, (0, 0), blur_sigma)
        # Keep H as a strict mask after smoothing.
        focus *= hue_mask.astype(np.float32) / 255.0

    return np.clip(focus * 255.0, 0, 255).astype(np.uint8)


def make_comparison(
    image_bgr: np.ndarray, focus: np.ndarray, heatmap: np.ndarray
) -> np.ndarray:
    """Place input, grayscale focus map, and heatmap in one horizontal image."""
    focus_bgr = cv2.cvtColor(focus, cv2.COLOR_GRAY2BGR)
    panels = [image_bgr, focus_bgr, heatmap]
    labels = ["Input", "Focus map", "Heatmap"]
    title_height = 42
    labelled_panels = []

    for panel, label in zip(panels, labels):
        canvas = np.full(
            (panel.shape[0] + title_height, panel.shape[1], 3), 255, dtype=np.uint8
        )
        canvas[title_height:] = panel
        cv2.putText(
            canvas, label, (12, 29), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 0, 0), 2, cv2.LINE_AA
        )
        labelled_panels.append(canvas)

    return np.hstack(labelled_panels)


def process_directory(
    input_dir: Path,
    output_dir: Path,
    hue_min: int,
    hue_max: int,
    saturation_min: int,
    value_min: int,
    blur_sigma: float,
    value_gamma: float,
) -> tuple[int, int]:
    """Process all supported images recursively. Return (successes, failures)."""
    if not input_dir.is_dir():
        raise NotADirectoryError(f"输入目录不存在: {input_dir}")

    image_paths = sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        print(f"输入目录中未找到图片: {input_dir}")
        return 0, 0

    success_count = 0
    failure_count = 0
    for path in image_paths:
        relative = path.relative_to(input_dir)
        save_relative = relative.with_suffix(".png")
        try:
            image = read_image(path)
            focus = vegetation_focus_map(
                image, hue_min, hue_max, saturation_min, value_min,
                blur_sigma, value_gamma
            )
            heatmap = cv2.applyColorMap(focus, cv2.COLORMAP_JET)
            comparison = make_comparison(image, focus, heatmap)

            # Inputs retain their source encoding; derived images use lossless PNG.
            write_image(output_dir / "input" / relative, image)
            write_image(output_dir / "output" / save_relative, focus)
            write_image(output_dir / "heatmap" / save_relative, heatmap)
            write_image(output_dir / "comparison" / save_relative, comparison)
            success_count += 1
            print(f"[{success_count}/{len(image_paths)}] 已处理: {relative}")
        except Exception as error:  # Continue the batch if one file is damaged.
            failure_count += 1
            print(f"处理失败 {path}: {error}")

    return success_count, failure_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量生成基于 HSV 绿植区域的 focus map")
    parser.add_argument("input_dir", nargs="?", type=Path, default=Path("G:/test_images/231_sun_green"), help="输入图片目录")
    parser.add_argument("output_dir", nargs="?", type=Path, default=Path("focus_results"), help="结果根目录")
    parser.add_argument("--hue-min", type=int, default=25, help="绿色最小色相，OpenCV 范围 0-179")
    parser.add_argument("--hue-max", type=int, default=95, help="绿色最大色相，OpenCV 范围 0-179")
    parser.add_argument("--saturation-min", type=int, default=35, help="最小饱和度，范围 0-255")
    parser.add_argument("--value-min", type=int, default=35, help="最小明度，范围 0-255")
    parser.add_argument("--blur-sigma", type=float, default=5.0, help="focus map 高斯平滑强度，0 表示不平滑")
    parser.add_argument("--value-gamma", type=float, default=0.5, help="V 通道非线性指数；越大明暗差距越明显")
    args = parser.parse_args()

    if not 0 <= args.hue_min < args.hue_max <= 179:
        parser.error("必须满足 0 <= hue-min < hue-max <= 179")
    if not 0 <= args.saturation_min <= 255 or not 0 <= args.value_min <= 255:
        parser.error("saturation-min 和 value-min 必须位于 0-255")
    if args.blur_sigma < 0:
        parser.error("blur-sigma 不能小于 0")
    if args.value_gamma <= 0:
        parser.error("value-gamma 必须大于 0")
    return args


def main() -> None:
    args = parse_args()
    succeeded, failed = process_directory(
        args.input_dir,
        args.output_dir,
        args.hue_min,
        args.hue_max,
        args.saturation_min,
        args.value_min,
        args.blur_sigma,
        args.value_gamma,
    )
    print(f"完成：成功 {succeeded} 张，失败 {failed} 张；结果目录: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
