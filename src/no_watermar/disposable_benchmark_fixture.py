from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def create_disposable_benchmark_fixture(input_root: Path) -> dict[str, Any]:
    input_root = input_root.resolve()
    items = [
        _write_regular_fixture(input_root / "regular" / "fixture-regular-a.jpg", accent=(34, 83, 201), label="fixture-a"),
        _write_regular_fixture(input_root / "regular" / "fixture-regular-b.jpg", accent=(79, 168, 61), label="fixture-b"),
        _write_cover_fixture(input_root / "cover" / "fixture-cover-0.jpg"),
    ]
    return {
        "input_root": str(input_root),
        "image_count": len(items),
        "items": items,
    }


def _write_regular_fixture(path: Path, *, accent: tuple[int, int, int], label: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = 220, 144
    image = np.full((height, width, 3), 244, dtype=np.uint8)

    # Add simple structure so Telea/noop/corner-crop comparisons produce stable artifacts.
    cv2.rectangle(image, (0, 0), (width - 1, int(height * 0.36)), accent, thickness=-1)
    cv2.circle(image, (int(width * 0.5), int(height * 0.48)), 28, (255, 223, 186), thickness=-1)
    cv2.rectangle(image, (int(width * 0.22), int(height * 0.58)), (int(width * 0.78), height - 18), (220, 220, 220), thickness=-1)
    cv2.line(image, (12, height - 56), (width - 14, height - 56), (210, 210, 210), thickness=2)
    cv2.putText(image, label, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Seed-mask heuristics inspect both bottom corners, so keep light watermark text there.
    cv2.putText(image, "example.com", (4, height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(
        image,
        "@mark",
        (width - 54, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.33,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    _write_encoded_image(path, image)
    return {
        "relative_path": str(path.name if path.parent.name == "" else path.relative_to(path.parents[1])),
        "kind": "regular",
    }


def _write_cover_fixture(path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = 240, 156
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (28, 28, 28)
    cv2.rectangle(image, (0, 0), (width - 1, int(height * 0.28)), (76, 96, 136), thickness=-1)
    cv2.rectangle(image, (0, int(height * 0.72)), (int(width * 0.24), height - 1), (248, 248, 248), thickness=-1)
    cv2.putText(image, "COVER", (18, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(image, "fixture", (24, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(image, "sample", (42, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (18, 18, 18), 1, cv2.LINE_AA)

    _write_encoded_image(path, image)
    return {
        "relative_path": str(path.name if path.parent.name == "" else path.relative_to(path.parents[1])),
        "kind": "cover",
    }


def _write_encoded_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"Failed to encode disposable benchmark fixture: {path}")
    encoded.tofile(str(path))
