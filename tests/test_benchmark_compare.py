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

from no_watermar.benchmark_compare import compare_benchmark_reports
from no_watermar.benchmark_dataset import DATASET_REGULAR, prepare_benchmark_dataset
from no_watermar.benchmark_runner import run_benchmark


def _write_image(path: Path, shape: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full(shape, 255, dtype=np.uint8)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError("encode failed")
    encoded.tofile(str(path))


class BenchmarkCompareTests(unittest.TestCase):
    def test_compare_benchmark_reports_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as bench_dir:
            input_root = Path(input_dir)
            benchmark_root = Path(bench_dir)
            _write_image(input_root / "regular.jpg", (180, 120, 3))

            prepare_benchmark_dataset(input_root, benchmark_root, recursive=False)
            summary = run_benchmark(
                benchmark_root,
                dataset_id=DATASET_REGULAR,
                mask_provider_name="seed_manifest",
                restore_provider_name="telea",
            )

            report_json = Path(summary["dataset_summaries"][0]["report_json"])
            comparison = compare_benchmark_reports(
                baseline_report=report_json,
                candidate_report=report_json,
                output_dir=benchmark_root / "comparisons",
            )

            self.assertEqual(comparison["common_item_count"], 1)
            self.assertEqual(comparison["status_pair_counts"]["restored__restored"], 1)
            self.assertEqual(comparison["mask_overlap_summary"]["mean_iou"], 1.0)
            self.assertEqual(comparison["delta_summary"]["mask_ratio"]["mean"], 0.0)
            self.assertIn("mask_latency_ms", comparison["delta_summary"])
            self.assertEqual(comparison["delta_summary"]["mask_latency_ms"]["mean"], 0.0)
            self.assertIn("ocr_residual_latency_ms", comparison["delta_summary"])
            self.assertTrue(Path(comparison["output_json"]).exists())
            self.assertTrue(Path(comparison["output_csv"]).exists())


if __name__ == "__main__":
    unittest.main()
