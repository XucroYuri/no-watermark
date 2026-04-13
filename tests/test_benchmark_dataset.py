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

from no_watermar.benchmark_dataset import DATASET_COVER, DATASET_REGULAR, prepare_benchmark_dataset


def _write_image(path: Path, shape: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full(shape, 255, dtype=np.uint8)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError("encode failed")
    encoded.tofile(str(path))


class BenchmarkDatasetTests(unittest.TestCase):
    def test_prepare_benchmark_dataset_writes_manifests_and_seed_masks(self) -> None:
        with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as bench_dir:
            input_root = Path(input_dir)
            benchmark_root = Path(bench_dir)
            _write_image(input_root / "regular.jpg", (180, 120, 3))
            _write_image(input_root / "nested" / "0001-cover.jpg", (180, 120, 3))

            summary = prepare_benchmark_dataset(input_root, benchmark_root, recursive=True)

            self.assertEqual(summary["image_count"], 2)
            self.assertEqual(summary["dataset_counts"][DATASET_REGULAR], 1)
            self.assertEqual(summary["dataset_counts"][DATASET_COVER], 1)
            self.assertTrue((benchmark_root / "datasets" / DATASET_REGULAR / "manifest.json").exists())
            self.assertTrue((benchmark_root / "datasets" / DATASET_COVER / "manifest.json").exists())
            self.assertTrue((benchmark_root / "datasets" / DATASET_REGULAR / "masks_seed" / "regular.png").exists())
            self.assertTrue((benchmark_root / "datasets" / DATASET_REGULAR / "prompts" / "regular.txt").exists())
            self.assertTrue((benchmark_root / "datasets" / DATASET_COVER / "inputs" / "nested" / "0001-cover.jpg").exists())


if __name__ == "__main__":
    unittest.main()
