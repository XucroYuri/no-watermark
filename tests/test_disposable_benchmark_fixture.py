from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from no_watermar.benchmark_dataset import DATASET_COVER, DATASET_REGULAR, prepare_benchmark_dataset
from no_watermar.disposable_benchmark_fixture import create_disposable_benchmark_fixture


class DisposableBenchmarkFixtureTests(unittest.TestCase):
    def test_create_disposable_benchmark_fixture_writes_redistributable_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_root = Path(tmp_dir) / "inputs"

            summary = create_disposable_benchmark_fixture(input_root)

            self.assertEqual(summary["image_count"], 3)
            self.assertTrue((input_root / "regular" / "fixture-regular-a.jpg").exists())
            self.assertTrue((input_root / "regular" / "fixture-regular-b.jpg").exists())
            self.assertTrue((input_root / "cover" / "fixture-cover-0.jpg").exists())

    def test_create_disposable_benchmark_fixture_prepares_regular_and_cover_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_root = root / "inputs"
            benchmark_root = root / "benchmarks"

            create_disposable_benchmark_fixture(input_root)
            summary = prepare_benchmark_dataset(input_root, benchmark_root, recursive=True)

            self.assertEqual(summary["dataset_counts"][DATASET_REGULAR], 2)
            self.assertEqual(summary["dataset_counts"][DATASET_COVER], 1)
            self.assertTrue((benchmark_root / "datasets" / DATASET_REGULAR / "manifest.json").exists())
            self.assertTrue((benchmark_root / "datasets" / DATASET_COVER / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
