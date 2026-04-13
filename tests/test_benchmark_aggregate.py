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

from no_watermar.benchmark_aggregate import aggregate_benchmark_reports
from no_watermar.benchmark_dataset import DATASET_REGULAR, prepare_benchmark_dataset
from no_watermar.benchmark_runner import run_benchmark


def _write_image(path: Path, shape: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full(shape, 255, dtype=np.uint8)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError("encode failed")
    encoded.tofile(str(path))


class BenchmarkAggregateTests(unittest.TestCase):
    def test_aggregate_benchmark_reports_groups_runs(self) -> None:
        with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as bench_dir:
            input_root = Path(input_dir)
            benchmark_root = Path(bench_dir)
            _write_image(input_root / "regular.jpg", (180, 120, 3))

            prepare_benchmark_dataset(input_root, benchmark_root, recursive=False)
            first = run_benchmark(
                benchmark_root,
                dataset_id=DATASET_REGULAR,
                mask_provider_name="seed_manifest",
                restore_provider_name="telea",
            )
            second = run_benchmark(
                benchmark_root,
                dataset_id=DATASET_REGULAR,
                mask_provider_name="seed_manifest",
                restore_provider_name="telea",
            )
            third = run_benchmark(
                benchmark_root,
                dataset_id=DATASET_REGULAR,
                mask_provider_name="seed_manifest",
                restore_provider_name="noop",
            )

            summary = aggregate_benchmark_reports(
                reports_root=benchmark_root / "runs",
                dataset_id=DATASET_REGULAR,
                output_dir=benchmark_root / "aggregations",
            )

            self.assertEqual(summary["group_count"], 2)
            self.assertEqual(summary["report_count"], 3)
            self.assertEqual(summary["filters"]["dataset_id"], DATASET_REGULAR)

            telea_group = next(group for group in summary["groups"] if group["restore_provider"] == "telea")
            self.assertEqual(telea_group["run_count"], 2)
            self.assertEqual(telea_group["item_count"], 2)
            self.assertEqual(telea_group["run_ids"], [first["run_id"], second["run_id"]])
            self.assertIn("mask_latency_ms", telea_group["mean_metrics"])
            self.assertTrue(Path(summary["output_json"]).exists())
            self.assertTrue(Path(summary["output_csv"]).exists())

            filtered = aggregate_benchmark_reports(
                reports_root=benchmark_root / "runs",
                dataset_id=DATASET_REGULAR,
                mask_provider="seed_manifest",
                restore_provider="telea",
                run_after=second["run_id"],
                run_before=second["run_id"],
            )

            self.assertEqual(filtered["group_count"], 1)
            self.assertEqual(filtered["report_count"], 1)
            filtered_group = filtered["groups"][0]
            self.assertEqual(filtered_group["restore_provider"], "telea")
            self.assertEqual(filtered_group["run_count"], 1)
            self.assertEqual(filtered_group["run_ids"], [second["run_id"]])
            self.assertNotIn(third["run_id"], filtered_group["run_ids"])


if __name__ == "__main__":
    unittest.main()
