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
from no_watermar.benchmark_compare import compare_benchmark_reports
from no_watermar.benchmark_dataset import DATASET_REGULAR, prepare_benchmark_dataset
from no_watermar.benchmark_runner import run_benchmark
from no_watermar.benchmark_trends import build_benchmark_trend_snapshot


def _write_image(path: Path, shape: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full(shape, 255, dtype=np.uint8)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError("encode failed")
    encoded.tofile(str(path))


class BenchmarkTrendTests(unittest.TestCase):
    def test_build_benchmark_trend_snapshot_merges_latest_compare_and_aggregate(self) -> None:
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
            candidate = run_benchmark(
                benchmark_root,
                dataset_id=DATASET_REGULAR,
                mask_provider_name="seed_manifest",
                restore_provider_name="noop",
            )

            second_report = Path(second["dataset_summaries"][0]["report_json"])
            candidate_report = Path(candidate["dataset_summaries"][0]["report_json"])
            comparison = compare_benchmark_reports(
                baseline_report=second_report,
                candidate_report=candidate_report,
                output_dir=benchmark_root / "comparisons",
            )
            aggregation = aggregate_benchmark_reports(
                reports_root=benchmark_root / "runs",
                dataset_id=DATASET_REGULAR,
                mask_provider="seed_manifest",
                restore_provider="telea",
                output_dir=benchmark_root / "aggregations",
            )

            summary = build_benchmark_trend_snapshot(
                comparisons_root=benchmark_root / "comparisons",
                aggregations_root=benchmark_root / "aggregations",
                output_dir=benchmark_root / "trends",
            )

            self.assertEqual(summary["dataset_id"], DATASET_REGULAR)
            self.assertEqual(summary["comparison"]["path"], comparison["output_json"])
            self.assertEqual(summary["baseline_latest"]["run_id"], second["run_id"])
            self.assertEqual(summary["candidate_latest"]["run_id"], candidate["run_id"])
            self.assertEqual(summary["baseline_aggregate"]["path"], aggregation["output_json"])
            self.assertEqual(summary["baseline_aggregate"]["group"]["run_count"], 2)
            self.assertEqual(summary["candidate_latest"]["restore_provider"], "noop")
            self.assertIsNone(summary["candidate_aggregate"])
            self.assertTrue(Path(summary["output_json"]).exists())
            self.assertTrue(Path(summary["output_markdown"]).exists())
            self.assertTrue(Path(summary["latest_json"]).exists())
            markdown = Path(summary["output_markdown"]).read_text(encoding="utf-8")
            self.assertIn("# Benchmark Trend Snapshot", markdown)
            self.assertIn("Candidate Vs Baseline Aggregate", markdown)
            self.assertEqual(first["mask_provider"], "seed_manifest")


if __name__ == "__main__":
    unittest.main()
