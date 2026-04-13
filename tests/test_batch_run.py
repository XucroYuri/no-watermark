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

from no_watermar.benchmark_models import MaskResult
from no_watermar.benchmark_models import RestoreResult
from no_watermar.benchmark_restore_session import PowerPaintExecutionContext
from no_watermar.benchmark_providers import RuleBasedMaskProvider
from no_watermar.batch_run import load_batch_run_summary, resolve_batch_run_summary
from no_watermar.pipeline import resume_pipeline, run_pipeline


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


class BatchRunTests(unittest.TestCase):
    def test_run_pipeline_writes_summary_manifest_and_results_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_root = root / "inputs"
            runs_root = root / "runtime" / "runs"
            _write_image(input_root / "sample.jpg", (120, 220, 3))

            summary = run_pipeline(
                input_root=input_root,
                output_root=runs_root,
                recursive=False,
                restore_prompt="restore fine texture",
                restore_negative_prompt="smear",
                restore_options={"steps": 20},
            )

            summary_json = Path(summary["summary_json"])
            manifest_json = Path(summary["manifest_json"])
            results_jsonl = Path(summary["results_jsonl"])
            latest_summary_json = Path(summary["latest_summary_json"])
            self.assertTrue(summary_json.exists())
            self.assertTrue(manifest_json.exists())
            self.assertTrue(results_jsonl.exists())
            self.assertTrue(latest_summary_json.exists())
            self.assertEqual(summary["run_status"], "completed")
            self.assertEqual(summary["completed_item_count"], 1)
            self.assertEqual(summary["pending_item_count"], 0)
            self.assertEqual(summary["mask_provider"], "rule_based_roi")
            self.assertEqual(summary["restore_provider"], "telea")
            self.assertEqual(summary["restore_prompt"], "restore fine texture")
            self.assertEqual(summary["restore_negative_prompt"], "smear")
            self.assertEqual(summary["restore_options"]["steps"], 20)
            self.assertEqual(len(results_jsonl.read_text(encoding="utf-8").splitlines()), 1)

            manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
            self.assertEqual(manifest["item_count"], 1)
            self.assertEqual(manifest["items"][0]["relative_path"], "sample.jpg")

    def test_resolve_batch_run_summary_defaults_to_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_root = root / "inputs"
            runs_root = root / "runtime" / "runs"
            _write_image(input_root / "sample.jpg", (120, 220, 3))

            summary = run_pipeline(input_root=input_root, output_root=runs_root, recursive=False)
            loaded, lookup_mode, resolved_source = resolve_batch_run_summary(runs_root=runs_root, latest=True)

            self.assertEqual(lookup_mode, "latest")
            self.assertEqual(loaded["run_id"], summary["run_id"])
            self.assertTrue(str(resolved_source).endswith("latest.json"))

    def test_load_batch_run_summary_supports_legacy_report_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = Path(tmp_dir) / "reports" / "report.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(
                    {
                        "command": "batch apply",
                        "run_id": "legacy-run",
                        "output_root": str(report_path.parent.parent),
                        "image_count": 1,
                        "results": [{"status": "restored"}],
                    }
                ),
                encoding="utf-8",
            )

            summary = load_batch_run_summary(report_path)

            self.assertEqual(summary["run_id"], "legacy-run")
            self.assertEqual(summary["run_status"], "completed")
            self.assertEqual(summary["completed_item_count"], 1)

    def test_resume_pipeline_continues_only_pending_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_root = root / "inputs"
            runs_root = root / "runtime" / "runs"
            _write_image(input_root / "first.jpg", (120, 220, 3))
            _write_image(input_root / "second.jpg", (120, 220, 3))

            call_count = {"value": 0}

            real_provider = RuleBasedMaskProvider()

            class FlakyMaskProvider:
                name = real_provider.name

                def detect(self, item, image, *, hints=None) -> MaskResult:
                    call_count["value"] += 1
                    if call_count["value"] == 2:
                        raise RuntimeError("simulated detector failure")
                    return real_provider.detect(item, image, hints=hints)

            def create_flaky_mask_provider(name: str):
                self.assertEqual(name, "rule_based_roi")
                return FlakyMaskProvider()

            with patch("no_watermar.pipeline.create_mask_provider", side_effect=create_flaky_mask_provider):
                with self.assertRaisesRegex(RuntimeError, "simulated detector failure"):
                    run_pipeline(input_root=input_root, output_root=runs_root, recursive=False)

            interrupted_summary, _, _ = resolve_batch_run_summary(runs_root=runs_root, latest=True)
            self.assertEqual(interrupted_summary["run_status"], "interrupted")
            self.assertEqual(interrupted_summary["completed_item_count"], 1)
            self.assertEqual(interrupted_summary["pending_item_count"], 1)

            resumed = resume_pipeline(
                interrupted_summary,
                summary_context={
                    "command": "batch resume",
                    "mode": "resume",
                },
            )

            self.assertEqual(resumed["run_status"], "completed")
            self.assertEqual(resumed["completed_item_count"], 2)
            self.assertEqual(resumed["pending_item_count"], 0)
            self.assertEqual(resumed["resumed_item_count"], 1)
            self.assertEqual(resumed["resume_count"], 1)
            self.assertEqual({result["item"]["relative_path"] for result in resumed["results"]}, {"first.jpg", "second.jpg"})

    def test_resume_pipeline_falls_back_when_summary_json_is_truncated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_root = root / "inputs"
            runs_root = root / "runtime" / "runs"
            _write_image(input_root / "first.jpg", (120, 220, 3))
            _write_image(input_root / "second.jpg", (120, 220, 3))

            call_count = {"value": 0}
            real_provider = RuleBasedMaskProvider()

            class FlakyMaskProvider:
                name = real_provider.name

                def detect(self, item, image, *, hints=None) -> MaskResult:
                    call_count["value"] += 1
                    if call_count["value"] == 2:
                        raise RuntimeError("simulated detector failure")
                    return real_provider.detect(item, image, hints=hints)

            with patch("no_watermar.pipeline.create_mask_provider", return_value=FlakyMaskProvider()):
                with self.assertRaisesRegex(RuntimeError, "simulated detector failure"):
                    run_pipeline(input_root=input_root, output_root=runs_root, recursive=False)

            interrupted_summary, _, _ = resolve_batch_run_summary(runs_root=runs_root, latest=True)
            Path(interrupted_summary["summary_json"]).write_text("", encoding="utf-8")

            reloaded, lookup_mode, _ = resolve_batch_run_summary(runs_root=runs_root, run_id=interrupted_summary["run_id"])
            self.assertEqual(lookup_mode, "run_id")
            self.assertEqual(reloaded["run_status"], "interrupted")
            self.assertEqual(reloaded["completed_item_count"], 1)

            resumed = resume_pipeline(reloaded, summary_context={"command": "batch resume", "mode": "resume"})

            self.assertEqual(resumed["run_status"], "completed")
            self.assertEqual(resumed["completed_item_count"], 2)
            self.assertEqual(resumed["pending_item_count"], 0)

    def test_run_pipeline_serializes_provider_metadata_with_numpy_scalars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_root = root / "inputs"
            runs_root = root / "runtime" / "runs"
            _write_image(input_root / "sample.jpg", (120, 220, 3))

            class JsonSafeMaskProvider:
                name = "jsonsafe-mask"

                def detect(self, item, image, *, hints=None) -> MaskResult:
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    mask[0:10, 0:10] = 255
                    return MaskResult(
                        provider_name=self.name,
                        mask=mask,
                        confidence=0.75,
                        boxes=[{"x1": np.int32(1), "y1": np.int32(2), "x2": np.int32(3), "y2": np.int32(4)}],
                        meta={"raw_box_count": np.int32(1)},
                    )

            class JsonSafeRestoreProvider:
                name = "jsonsafe-restore"

                def restore(self, item, image, mask, *, prompt=None, negative_prompt=None, meta=None) -> RestoreResult:
                    return RestoreResult(
                        provider_name=self.name,
                        restored=image.copy(),
                        meta={
                            "applied": np.bool_(True),
                            "prompt": prompt,
                            "negative_prompt": negative_prompt,
                            "restore_options": dict((meta or {}).get("restore_options") or {}),
                        },
                    )

            with patch("no_watermar.pipeline.create_mask_provider", return_value=JsonSafeMaskProvider()):
                with patch("no_watermar.pipeline.create_restore_provider", return_value=JsonSafeRestoreProvider()):
                    summary = run_pipeline(
                        input_root=input_root,
                        output_root=runs_root,
                        recursive=False,
                        mask_provider_name="jsonsafe-mask",
                        restore_provider_name="jsonsafe-restore",
                        restore_prompt="preserve fabric texture",
                        restore_negative_prompt="melted text",
                        restore_options={"steps": 18},
                    )

            self.assertEqual(summary["status_counts"]["restored"], 1)
            loaded = load_batch_run_summary(Path(summary["summary_json"]))
            result = loaded["results"][0]
            self.assertEqual(result["mask_meta"]["raw_box_count"], 1)
            self.assertTrue(result["restore_meta"]["applied"])
            self.assertEqual(result["restore_meta"]["prompt"], "preserve fabric texture")
            self.assertEqual(result["restore_meta"]["negative_prompt"], "melted text")
            self.assertEqual(result["restore_meta"]["restore_options"]["steps"], 18)

    def test_run_pipeline_supports_corner_crop_restore_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_root = root / "inputs"
            runs_root = root / "runtime" / "runs"
            _write_image(input_root / "sample.jpg", (120, 220, 3))

            class BottomLeftMaskProvider:
                name = "bottom-left-mask"

                def detect(self, item, image, *, hints=None) -> MaskResult:
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    mask[90:120, 0:60] = 255
                    return MaskResult(
                        provider_name=self.name,
                        mask=mask,
                        confidence=0.92,
                        boxes=[{"region": "bottom_left", "x1": 0, "y1": 90, "x2": 60, "y2": 120}],
                    )

            with patch("no_watermar.pipeline.create_mask_provider", return_value=BottomLeftMaskProvider()):
                summary = run_pipeline(
                    input_root=input_root,
                    output_root=runs_root,
                    recursive=False,
                    mask_provider_name="bottom-left-mask",
                    restore_provider_name="corner_crop",
                    restore_options={"edge_tolerance": 24},
                )

            self.assertEqual(summary["status_counts"]["cropped"], 1)
            loaded = load_batch_run_summary(Path(summary["summary_json"]))
            result = loaded["results"][0]
            self.assertEqual(result["status"], "cropped")
            self.assertEqual(result["restore_provider"], "corner_crop")
            self.assertEqual(result["restore_meta"]["crop_edge"], "bottom")
            self.assertEqual(result["restore_meta"]["output_shape"], [90, 220])

            restored_height, restored_width = _read_image_shape(Path(result["restored_path"]))
            self.assertEqual((restored_height, restored_width), (90, 220))

    def test_run_pipeline_records_powerpaint_restore_session_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_root = root / "inputs"
            runs_root = root / "runtime" / "runs"
            _write_image(input_root / "sample.jpg", (120, 220, 3))

            class FakePowerPaintSession:
                def __init__(self) -> None:
                    self.closed = False

                def close(self) -> None:
                    self.closed = True

            fake_session = FakePowerPaintSession()
            fake_context = PowerPaintExecutionContext(
                requested_mode="auto",
                resolved_mode="persistent-sidecar",
                python_executable="C:\\mock\\powerpaint\\python.exe",
                note="mock persistent PowerPaint session",
                session=fake_session,
            )

            class JsonSafeMaskProvider:
                name = "jsonsafe-mask"

                def detect(self, item, image, *, hints=None) -> MaskResult:
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    mask[0:10, 0:10] = 255
                    return MaskResult(provider_name=self.name, mask=mask, confidence=0.9)

            class FakePowerPaintRestoreProvider:
                name = "powerpaint_v2_1"

                def __init__(self) -> None:
                    self.received_context = None

                def restore(self, item, image, mask, *, prompt=None, negative_prompt=None, meta=None) -> RestoreResult:
                    self.received_context = (meta or {}).get("powerpaint_context")
                    return RestoreResult(
                        provider_name=self.name,
                        restored=image.copy(),
                        latency_ms=456.0,
                        meta={
                            "session_mode": "persistent-sidecar",
                            "restore_options": dict((meta or {}).get("restore_options") or {}),
                        },
                    )

            fake_restore_provider = FakePowerPaintRestoreProvider()

            with patch("no_watermar.pipeline.create_mask_provider", return_value=JsonSafeMaskProvider()):
                with patch("no_watermar.pipeline.create_restore_provider", return_value=fake_restore_provider):
                    with patch("no_watermar.pipeline.create_powerpaint_execution_context", return_value=fake_context):
                        summary = run_pipeline(
                            input_root=input_root,
                            output_root=runs_root,
                            recursive=False,
                            mask_provider_name="jsonsafe-mask",
                            restore_provider_name="powerpaint_v2_1",
                            restore_options={"checkpoint_dir": "C:\\mock\\ppt-v2-1", "session_mode": "auto"},
                        )

            self.assertEqual(summary["restore_session_mode_requested"], "auto")
            self.assertEqual(summary["restore_session_mode_resolved"], "persistent-sidecar")
            self.assertEqual(summary["restore_session_note"], "mock persistent PowerPaint session")
            self.assertIs(fake_restore_provider.received_context, fake_context)
            self.assertTrue(fake_session.closed)

            loaded = load_batch_run_summary(Path(summary["summary_json"]))
            result = loaded["results"][0]
            self.assertEqual(result["restore_meta"]["session_mode"], "persistent-sidecar")
            self.assertEqual(result["restore_meta"]["restore_options"]["checkpoint_dir"], "C:\\mock\\ppt-v2-1")


if __name__ == "__main__":
    unittest.main()
