from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import cv2
import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from no_watermar.scanner import build_scan_items, scan_image_paths


def _write_image(path: Path, shape: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full(shape, 255, dtype=np.uint8)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError("encode failed")
    encoded.tofile(str(path))


class ScannerTests(unittest.TestCase):
    def test_recursive_scan_includes_nested_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_image(root / "a.jpg", (100, 80, 3))
            _write_image(root / "nested" / "b.jpg", (100, 80, 3))
            _write_image(root / "docs" / "ignored.jpg", (100, 80, 3))

            paths = scan_image_paths(root, recursive=True)

            self.assertEqual(len(paths), 2)
            self.assertEqual({path.name for path in paths}, {"a.jpg", "b.jpg"})

    def test_build_scan_items_classifies_landscape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_image(root / "landscape.jpg", (120, 220, 3))
            items = build_scan_items(root, recursive=False)
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].category, "landscape_regular")


if __name__ == "__main__":
    unittest.main()
