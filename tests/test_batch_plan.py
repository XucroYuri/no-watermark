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

from no_watermar.batch_plan import apply_batch_plan, create_batch_plan, validate_batch_paths
from no_watermar.scan_manifest import create_scan_manifest


def _write_image(path: Path, shape: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full(shape, 255, dtype=np.uint8)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError("encode failed")
    encoded.tofile(str(path))


class BatchPlanTests(unittest.TestCase):
    def test_create_batch_plan_writes_plan_and_latest_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_root = root / "inputs"
            output_root = root / "runtime" / "runs"
            _write_image(input_root / "sample.jpg", (160, 100, 3))

            summary = create_batch_plan(input_root=input_root, output_root=output_root, recursive=False)

            plan_path = Path(summary["plan_path"])
            latest_path = Path(summary["latest_plan_path"])
            self.assertEqual(summary["command"], "batch plan")
            self.assertEqual(summary["status"], "ok")
            self.assertEqual(summary["item_count"], 1)
            self.assertTrue(plan_path.exists())
            self.assertTrue(latest_path.exists())

    def test_apply_batch_plan_runs_pipeline_with_plan_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_root = root / "inputs"
            output_root = root / "runtime" / "runs"
            _write_image(input_root / "sample.jpg", (180, 120, 3))

            plan = create_batch_plan(input_root=input_root, output_root=output_root, recursive=False)
            summary = apply_batch_plan(Path(plan["plan_path"]))

            self.assertEqual(summary["command"], "batch apply")
            self.assertEqual(summary["mode"], "planned")
            self.assertEqual(summary["plan_id"], plan["plan_id"])
            self.assertEqual(summary["status_counts"]["restored"], 1)
            self.assertEqual(summary["warnings"], [])
            self.assertTrue(Path(summary["report_json"]).exists())
            self.assertTrue(Path(summary["report_csv"]).exists())

    def test_apply_batch_plan_reuses_provider_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_root = root / "inputs"
            output_root = root / "runtime" / "runs"
            _write_image(input_root / "sample.jpg", (180, 120, 3))

            plan = create_batch_plan(
                input_root=input_root,
                output_root=output_root,
                recursive=False,
                mask_provider="rule_based_roi",
                restore_provider="telea",
                ocr_session_mode="auto",
                restore_prompt="remove corner watermark and keep texture",
                restore_negative_prompt="blur",
                restore_options={"steps": 24},
            )
            summary = apply_batch_plan(Path(plan["plan_path"]))

            self.assertEqual(summary["mask_provider"], "rule_based_roi")
            self.assertEqual(summary["restore_provider"], "telea")
            self.assertEqual(summary["ocr_session_mode_requested"], "auto")
            self.assertEqual(summary["restore_prompt"], "remove corner watermark and keep texture")
            self.assertEqual(summary["restore_negative_prompt"], "blur")
            self.assertEqual(summary["restore_options"]["steps"], 24)

    def test_create_batch_plan_can_reuse_scan_manifest_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_root = root / "inputs"
            output_root = root / "runtime" / "runs"
            scans_root = root / "runtime" / "scans"
            _write_image(input_root / "sample.jpg", (180, 120, 3))

            scan_manifest = create_scan_manifest(input_root, scans_root=scans_root, recursive=False)
            summary = create_batch_plan(
                input_root=None,
                output_root=output_root,
                scan_manifest_path=Path(scan_manifest["manifest_path"]),
            )

            self.assertEqual(summary["input_mode"], "scan_manifest")
            self.assertEqual(summary["scan_manifest_path"], scan_manifest["manifest_path"])
            self.assertEqual(summary["source_scan_id"], scan_manifest["scan_id"])
            self.assertEqual(summary["item_count"], 1)

    def test_apply_batch_plan_uses_planned_items_without_rescanning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_root = root / "inputs"
            output_root = root / "runtime" / "runs"
            scans_root = root / "runtime" / "scans"
            _write_image(input_root / "first.jpg", (180, 120, 3))

            scan_manifest = create_scan_manifest(input_root, scans_root=scans_root, recursive=False)
            plan = create_batch_plan(
                input_root=None,
                output_root=output_root,
                scan_manifest_path=Path(scan_manifest["manifest_path"]),
            )
            _write_image(input_root / "second.jpg", (180, 120, 3))

            summary = apply_batch_plan(Path(plan["plan_path"]))

            self.assertEqual(summary["image_count"], 1)
            self.assertEqual(summary["input_mode"], "scan_manifest")
            self.assertEqual(summary["scan_manifest_path"], scan_manifest["manifest_path"])
            self.assertEqual(summary["source_scan_id"], scan_manifest["scan_id"])
            self.assertEqual(summary["status_counts"]["restored"], 1)
            self.assertEqual(summary["results"][0]["item"]["relative_path"], "first.jpg")

    def test_validate_batch_paths_rejects_output_inside_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_root = root / "inputs"
            input_root.mkdir(parents=True)
            output_root = input_root / "generated"
            plans_root = input_root / "plans"

            with self.assertRaisesRegex(ValueError, "Output root must not be inside input root"):
                validate_batch_paths(input_root=input_root, output_root=output_root, plans_root=plans_root)


if __name__ == "__main__":
    unittest.main()
