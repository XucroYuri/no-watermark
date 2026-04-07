from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from no_watermar.provider_workers import restore_with_simple_lama


def main() -> int:
    parser = argparse.ArgumentParser(description="LaMa sidecar restore provider.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output-image", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--options-json", default=None)
    args = parser.parse_args()

    image = _read_color_image(args.image)
    mask = _read_gray_image(args.mask)
    options = json.loads(args.options_json) if args.options_json else {}
    result = restore_with_simple_lama(
        image,
        mask,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        options=options,
    )
    _write_image(args.output_image, result["restored"])
    payload = {
        "latency_ms": result["latency_ms"],
        "meta": result["meta"],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def _read_color_image(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to decode color image: {path}")
    return image


def _read_gray_image(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to decode grayscale image: {path}")
    return image


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix.lower() or ".png", image)
    if not ok:
        raise ValueError(f"Unable to encode image for: {path}")
    encoded.tofile(str(path))


if __name__ == "__main__":
    raise SystemExit(main())
