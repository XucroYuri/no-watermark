from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

import cv2
import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from no_watermar.benchmark_dataset import DATASET_REGULAR, prepare_benchmark_dataset
from no_watermar.benchmark_paddleocr_session import PaddleOCRExecutionContext
from no_watermar.benchmark_restore_session import PowerPaintExecutionContext
from no_watermar.benchmark_providers import MASK_PROVIDER_REGISTRY, RESTORE_PROVIDER_REGISTRY, ProviderUnavailableError
from no_watermar.benchmark_models import MaskResult, RestoreResult
from no_watermar.benchmark_runner import run_benchmark


def _write_image(path: Path, shape: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full(shape, 255, dtype=np.uint8)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError("encode failed")
    encoded.tofile(str(path))


def _read_image_shape(path: Path) -> tuple[int, int]:
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("decode failed")
    height, width = image.shape[:2]
    return height, width


class FakeOCRSession:
    def __init__(self) -> None:
        self.closed = False
        self.calls: list[Path] = []

    def detect(self, *, image_path: Path, source_category: str, score_threshold: float = 0.45):
        self.calls.append(image_path)
        height, width = _read_image_shape(image_path)
        mask = np.zeros((height, width), dtype=np.uint8)
        payload = {
            "confidence": 0.0,
            "boxes": [],
            "latency_ms": 12.5,
            "meta": {
                "matched_texts": [],
                "source_category": source_category,
                "score_threshold": score_threshold,
                "session_mode": "persistent-sidecar",
            },
        }
        return payload, mask

    def close(self) -> None:
        self.closed = True


def _make_fake_context() -> tuple[PaddleOCRExecutionContext, FakeOCRSession]:
    fake_session = FakeOCRSession()
    context = PaddleOCRExecutionContext(
        requested_mode="auto",
        resolved_mode="persistent-sidecar",
        python_executable="C:\\mock\\python.exe",
        note="mock persistent OCR session",
        session=fake_session,
    )
    return context, fake_session


class FakePowerPaintSession:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _make_fake_powerpaint_context() -> tuple[PowerPaintExecutionContext, FakePowerPaintSession]:
    fake_session = FakePowerPaintSession()
    context = PowerPaintExecutionContext(
        requested_mode="auto",
        resolved_mode="persistent-sidecar",
        python_executable="C:\\mock\\powerpaint\\python.exe",
        note="mock persistent PowerPaint session",
        session=fake_session,
    )
    return context, fake_session


class BenchmarkRunnerTests(unittest.TestCase):
    def test_run_benchmark_rejects_empty_dataset_before_starting_ocr_context(self) -> None:
        with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as bench_dir:
            input_root = Path(input_dir)
            benchmark_root = Path(bench_dir)
            prepare_benchmark_dataset(input_root, benchmark_root, recursive=False)

            with patch(
                "no_watermar.benchmark_runner.create_paddleocr_execution_context",
                side_effect=AssertionError("ocr context should not be created for an empty dataset"),
            ):
                with self.assertRaisesRegex(ValueError, "contains no benchmark items"):
                    run_benchmark(
                        benchmark_root,
                        dataset_id=DATASET_REGULAR,
                        mask_provider_name="seed_manifest",
                        restore_provider_name="telea",
                    )

    def test_run_benchmark_records_latency_and_session_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as bench_dir:
            input_root = Path(input_dir)
            benchmark_root = Path(bench_dir)
            _write_image(input_root / "regular.jpg", (180, 120, 3))

            prepare_benchmark_dataset(input_root, benchmark_root, recursive=False)
            fake_context, fake_session = _make_fake_context()
            with patch("no_watermar.benchmark_runner.create_paddleocr_execution_context", return_value=fake_context):
                summary = run_benchmark(
                    benchmark_root,
                    dataset_id=DATASET_REGULAR,
                    mask_provider_name="seed_manifest",
                    restore_provider_name="telea",
                    restore_prompt="preserve texture detail",
                    restore_negative_prompt="blur",
                    restore_options={"steps": 22},
                )

            self.assertEqual(summary["status_counts"]["restored"], 1)
            self.assertEqual(summary["ocr_session_mode_requested"], "auto")
            self.assertEqual(summary["ocr_session_mode_resolved"], "persistent-sidecar")
            self.assertEqual(summary["restore_prompt"], "preserve texture detail")
            self.assertEqual(summary["restore_negative_prompt"], "blur")
            self.assertEqual(summary["restore_options"]["steps"], 22)
            self.assertTrue(fake_session.closed)
            self.assertGreaterEqual(len(fake_session.calls), 1)

            report_json = Path(summary["dataset_summaries"][0]["report_json"])
            report = json.loads(report_json.read_text(encoding="utf-8"))
            result = report["results"][0]

            self.assertIsNotNone(result["metrics"]["mask_latency_ms"])
            self.assertIsNotNone(result["metrics"]["restore_latency_ms"])
            self.assertEqual(result["metrics"]["ocr_residual_latency_ms"], 12.5)
            self.assertEqual(result["metrics"]["ocr_residual_hits"], 0)
            self.assertEqual(result["restore_result"]["meta"]["restore_options"]["steps"], 22)
            self.assertEqual(result["restore_result"]["meta"]["ocr_residual"]["session_mode"], "persistent-sidecar")

    def test_run_benchmark_records_provider_unavailable_status(self) -> None:
        class FailingMaskProvider:
            name = "failing_mask"

            def detect(self, item, image, *, hints=None):
                raise ProviderUnavailableError("mocked missing env")

        class FailingRestoreProvider:
            name = "failing_restore"

            def restore(self, item, image, mask, *, prompt=None, negative_prompt=None, meta=None):
                raise ProviderUnavailableError("mocked missing restore env")

        with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as bench_dir:
            input_root = Path(input_dir)
            benchmark_root = Path(bench_dir)
            _write_image(input_root / "regular.jpg", (180, 120, 3))
            prepare_benchmark_dataset(input_root, benchmark_root, recursive=False)

            fake_context, _ = _make_fake_context()
            with patch("no_watermar.benchmark_runner.create_paddleocr_execution_context", return_value=fake_context):
                with patch.dict(MASK_PROVIDER_REGISTRY, {"failing_mask": FailingMaskProvider}, clear=False):
                    summary = run_benchmark(
                        benchmark_root,
                        dataset_id=DATASET_REGULAR,
                        mask_provider_name="failing_mask",
                        restore_provider_name="telea",
                    )
                    self.assertEqual(summary["status_counts"]["mask_provider_unavailable"], 1)

            fake_context, _ = _make_fake_context()
            with patch("no_watermar.benchmark_runner.create_paddleocr_execution_context", return_value=fake_context):
                with patch.dict(RESTORE_PROVIDER_REGISTRY, {"failing_restore": FailingRestoreProvider}, clear=False):
                    summary = run_benchmark(
                        benchmark_root,
                        dataset_id=DATASET_REGULAR,
                        mask_provider_name="seed_manifest",
                        restore_provider_name="failing_restore",
                    )
                    self.assertEqual(summary["status_counts"]["restore_provider_unavailable"], 1)

    def test_run_benchmark_supports_corner_crop_restore_provider(self) -> None:
        class BottomLeftMaskProvider:
            name = "bottom-left-mask"

            def detect(self, item, image, *, hints=None):
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                mask[120:180, 0:64] = 255
                return MaskResult(
                    provider_name=self.name,
                    mask=mask,
                    confidence=0.88,
                    boxes=[{"region": "bottom_left", "x1": 0, "y1": 120, "x2": 64, "y2": 180}],
                    latency_ms=8.0,
                    meta={"source_category": item.source_category},
                )

        with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as bench_dir:
            input_root = Path(input_dir)
            benchmark_root = Path(bench_dir)
            _write_image(input_root / "regular.jpg", (180, 120, 3))
            prepare_benchmark_dataset(input_root, benchmark_root, recursive=False)

            fake_context, fake_session = _make_fake_context()
            with patch("no_watermar.benchmark_runner.create_paddleocr_execution_context", return_value=fake_context):
                with patch.dict(MASK_PROVIDER_REGISTRY, {"bottom-left-mask": BottomLeftMaskProvider}, clear=False):
                    summary = run_benchmark(
                        benchmark_root,
                        dataset_id=DATASET_REGULAR,
                        mask_provider_name="bottom-left-mask",
                        restore_provider_name="corner_crop",
                    )

            self.assertEqual(summary["status_counts"]["cropped"], 1)
            self.assertTrue(fake_session.closed)

            report_json = Path(summary["dataset_summaries"][0]["report_json"])
            report = json.loads(report_json.read_text(encoding="utf-8"))
            result = report["results"][0]
            self.assertEqual(result["status"], "cropped")
            self.assertIsNone(result["metrics"]["changed_nonzero"])
            self.assertIsNone(result["metrics"]["mean_abs_diff"])
            self.assertIsNone(result["metrics"]["edge_delta"])
            self.assertEqual(result["restore_result"]["meta"]["crop_edge"], "bottom")
            self.assertEqual(result["restore_result"]["meta"]["output_shape"], [120, 120])

    def test_run_benchmark_records_restore_session_metadata_for_powerpaint(self) -> None:
        class FakePowerPaintRestoreProvider:
            name = "powerpaint_v2_1"

            def __init__(self) -> None:
                self.received_context = None

            def restore(self, item, image, mask, *, prompt=None, negative_prompt=None, meta=None):
                self.received_context = (meta or {}).get("powerpaint_context")
                return RestoreResult(
                    provider_name=self.name,
                    restored=image.copy(),
                    latency_ms=987.0,
                    meta={
                        "session_mode": "persistent-sidecar",
                        "prompt_used": prompt,
                        "negative_prompt_used": negative_prompt,
                        "restore_options": dict((meta or {}).get("restore_options") or {}),
                    },
                )

        with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as bench_dir:
            input_root = Path(input_dir)
            benchmark_root = Path(bench_dir)
            _write_image(input_root / "regular.jpg", (180, 120, 3))

            prepare_benchmark_dataset(input_root, benchmark_root, recursive=False)
            fake_ocr_context, fake_ocr_session = _make_fake_context()
            fake_restore_context, fake_restore_session = _make_fake_powerpaint_context()
            fake_restore_provider = FakePowerPaintRestoreProvider()

            with patch("no_watermar.benchmark_runner.create_paddleocr_execution_context", return_value=fake_ocr_context):
                with patch("no_watermar.benchmark_runner.create_powerpaint_execution_context", return_value=fake_restore_context):
                    with patch("no_watermar.benchmark_runner.create_restore_provider", return_value=fake_restore_provider):
                        summary = run_benchmark(
                            benchmark_root,
                            dataset_id=DATASET_REGULAR,
                            mask_provider_name="seed_manifest",
                            restore_provider_name="powerpaint_v2_1",
                            restore_prompt="preserve fabric detail",
                            restore_negative_prompt="text, logo",
                            restore_options={"checkpoint_dir": "C:\\mock\\ppt-v2-1", "session_mode": "auto"},
                        )

            self.assertEqual(summary["status_counts"]["restored"], 1)
            self.assertEqual(summary["restore_session_mode_requested"], "auto")
            self.assertEqual(summary["restore_session_mode_resolved"], "persistent-sidecar")
            self.assertEqual(summary["restore_session_note"], "mock persistent PowerPaint session")
            self.assertIs(fake_restore_provider.received_context, fake_restore_context)
            self.assertTrue(fake_ocr_session.closed)
            self.assertTrue(fake_restore_session.closed)

            report_json = Path(summary["dataset_summaries"][0]["report_json"])
            report = json.loads(report_json.read_text(encoding="utf-8"))
            result = report["results"][0]
            self.assertEqual(report["restore_session_mode_resolved"], "persistent-sidecar")
            self.assertEqual(result["restore_result"]["meta"]["session_mode"], "persistent-sidecar")
            self.assertEqual(result["restore_result"]["meta"]["restore_options"]["checkpoint_dir"], "C:\\mock\\ppt-v2-1")


if __name__ == "__main__":
    unittest.main()
