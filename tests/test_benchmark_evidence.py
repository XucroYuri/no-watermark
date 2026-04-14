from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
import sys

import cv2
import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from no_watermar.benchmark_dataset import DATASET_REGULAR, prepare_benchmark_dataset
from no_watermar.benchmark_evidence import build_stable_baseline_evidence
from no_watermar.benchmark_runner import run_benchmark


def _write_image(path: Path, shape: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full(shape, 255, dtype=np.uint8)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError("encode failed")
    encoded.tofile(str(path))


class BenchmarkEvidenceTests(unittest.TestCase):
    def test_capture_disposable_evidence_script_exists(self) -> None:
        script_path = Path(__file__).resolve().parents[1] / "tools" / "benchmark" / "capture-disposable-evidence.py"
        self.assertTrue(script_path.exists(), "disposable evidence helper should exist")
        content = script_path.read_text(encoding="utf-8")

        self.assertIn("create_disposable_benchmark_fixture", content)
        self.assertIn("build_stable_baseline_evidence", content)

    def test_capture_stable_baseline_script_exists(self) -> None:
        script_path = Path(__file__).resolve().parents[1] / "tools" / "benchmark" / "capture-stable-baseline.ps1"
        self.assertTrue(script_path.exists(), "stable baseline capture helper should exist")
        content = script_path.read_text(encoding="utf-8")

        self.assertIn("run-release-smoke.ps1", content)
        self.assertIn("benchmark\", \"evidence", content)
        self.assertIn("--candidate-mask-provider", content)

    def test_build_stable_baseline_evidence_writes_ready_summary(self) -> None:
        with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as bench_dir:
            input_root = Path(input_dir)
            benchmark_root = Path(bench_dir)
            _write_image(input_root / "regular.jpg", (180, 120, 3))

            prepare_benchmark_dataset(input_root, benchmark_root, recursive=False)
            for _ in range(2):
                run_benchmark(
                    benchmark_root,
                    dataset_id=DATASET_REGULAR,
                    mask_provider_name="seed_manifest",
                    restore_provider_name="telea",
                )
                run_benchmark(
                    benchmark_root,
                    dataset_id=DATASET_REGULAR,
                    mask_provider_name="seed_manifest",
                    restore_provider_name="noop",
                )
                run_benchmark(
                    benchmark_root,
                    dataset_id=DATASET_REGULAR,
                    mask_provider_name="seed_manifest",
                    restore_provider_name="corner_crop",
                )

            summary = build_stable_baseline_evidence(
                benchmark_root=benchmark_root,
                dataset_id=DATASET_REGULAR,
                baseline_mask_provider="seed_manifest",
                baseline_restore_provider="telea",
                candidate_mask_provider="seed_manifest",
                candidate_restore_provider="noop",
                optional_mask_provider="seed_manifest",
                optional_restore_provider="corner_crop",
                minimum_run_count=2,
                output_dir=benchmark_root / "evidence",
            )

            self.assertEqual(summary["status"], "ready")
            self.assertTrue(summary["release_blocking"]["ready"])
            self.assertEqual(summary["release_blocking"]["baseline"]["run_count"], 2)
            self.assertEqual(summary["release_blocking"]["candidate"]["run_count"], 2)
            self.assertTrue(summary["optional_stable"]["ready"])
            self.assertEqual(summary["optional_stable"]["candidate"]["run_count"], 2)
            self.assertTrue(Path(summary["output_json"]).exists())
            markdown = Path(summary["output_markdown"]).read_text(encoding="utf-8")
            self.assertIn("# Stable Baseline Evidence", markdown)
            self.assertIn("seed_manifest + telea", markdown)
            self.assertIn("seed_manifest + noop", markdown)

    def test_build_stable_baseline_evidence_reports_missing_repeated_runs(self) -> None:
        with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as bench_dir:
            input_root = Path(input_dir)
            benchmark_root = Path(bench_dir)
            _write_image(input_root / "regular.jpg", (180, 120, 3))

            prepare_benchmark_dataset(input_root, benchmark_root, recursive=False)
            run_benchmark(
                benchmark_root,
                dataset_id=DATASET_REGULAR,
                mask_provider_name="seed_manifest",
                restore_provider_name="telea",
            )
            run_benchmark(
                benchmark_root,
                dataset_id=DATASET_REGULAR,
                mask_provider_name="seed_manifest",
                restore_provider_name="noop",
            )

            summary = build_stable_baseline_evidence(
                benchmark_root=benchmark_root,
                dataset_id=DATASET_REGULAR,
                baseline_mask_provider="seed_manifest",
                baseline_restore_provider="telea",
                candidate_mask_provider="seed_manifest",
                candidate_restore_provider="noop",
                include_optional=False,
                minimum_run_count=2,
            )

            self.assertEqual(summary["status"], "action_required")
            self.assertFalse(summary["release_blocking"]["ready"])
            issue_codes = {issue["issue_code"] for issue in summary["release_blocking"]["issues"]}
            self.assertIn("run_count_below_threshold", issue_codes)
            self.assertEqual(summary["optional_stable"]["status"], "skipped")

    def test_capture_stable_baseline_script_runs_in_evidence_only_mode(self) -> None:
        shell = shutil.which("powershell") or shutil.which("pwsh")
        if shell is None:
            self.skipTest("PowerShell is not available")

        with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as bench_dir:
            input_root = Path(input_dir)
            benchmark_root = Path(bench_dir)
            _write_image(input_root / "regular.jpg", (180, 120, 3))

            prepare_benchmark_dataset(input_root, benchmark_root, recursive=False)
            for _ in range(2):
                run_benchmark(
                    benchmark_root,
                    dataset_id=DATASET_REGULAR,
                    mask_provider_name="seed_manifest",
                    restore_provider_name="telea",
                )
                run_benchmark(
                    benchmark_root,
                    dataset_id=DATASET_REGULAR,
                    mask_provider_name="seed_manifest",
                    restore_provider_name="noop",
                )

            script_path = Path(__file__).resolve().parents[1] / "tools" / "benchmark" / "capture-stable-baseline.ps1"
            completed = subprocess.run(
                [
                    shell,
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(script_path),
                    "-BenchmarkRoot",
                    str(benchmark_root),
                    "-Dataset",
                    DATASET_REGULAR,
                    "-Repetitions",
                    "2",
                    "-MinimumRunCount",
                    "2",
                    "-SkipSmoke",
                    "-SkipOptionalEvidence",
                    "-CandidateMaskProvider",
                    "seed_manifest",
                    "-CandidateRestoreProvider",
                    "noop",
                ],
                cwd=str(Path(__file__).resolve().parents[1]),
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, msg=completed.stderr or completed.stdout)
            self.assertIn("EVIDENCE status=ready", completed.stdout)

            output_json_line = next(
                line for line in completed.stdout.splitlines() if line.startswith("OUTPUT json=")
            )
            output_json = Path(output_json_line.split("=", 1)[1].strip())
            self.assertTrue(output_json.exists())

            latest_json = benchmark_root / "evidence" / "latest.json"
            summary = json.loads(latest_json.read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "ready")
            self.assertEqual(summary["release_blocking"]["candidate"]["restore_provider"], "noop")

    def test_capture_disposable_evidence_script_runs_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            script_path = Path(__file__).resolve().parents[1] / "tools" / "benchmark" / "capture-disposable-evidence.py"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--workspace-root",
                    str(workspace_root),
                    "--repetitions",
                    "2",
                    "--minimum-run-count",
                    "2",
                    "--clean",
                ],
                cwd=str(Path(__file__).resolve().parents[1]),
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, msg=completed.stderr or completed.stdout)
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["evidence"]["status"], "ready")
            self.assertEqual(payload["fixture"]["image_count"], 3)
            self.assertTrue((workspace_root / "benchmarks" / "evidence" / "latest.json").exists())


if __name__ == "__main__":
    unittest.main()
