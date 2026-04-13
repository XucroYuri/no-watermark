from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
import sys
from unittest.mock import patch

import cv2
import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from no_watermar.benchmark_models import MaskResult
from no_watermar.benchmark_providers import RuleBasedMaskProvider
from no_watermar.benchmark_cli import main as benchmark_main
from no_watermar.batch_plan import create_batch_plan
from no_watermar.cli import main
from no_watermar.scan_manifest import create_scan_manifest
from run import main as run_main


def _write_image(path: Path, shape: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full(shape, 255, dtype=np.uint8)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError("encode failed")
    encoded.tofile(str(path))


class CliTests(unittest.TestCase):
    def test_batch_apply_dispatches_to_pipeline(self) -> None:
        buffer = io.StringIO()
        summary = {"status": "ok", "items": 0}

        with patch("no_watermar.cli.commands.batch.run_pipeline", return_value=summary) as run_pipeline:
            with patch.dict(os.environ, {}, clear=True), redirect_stdout(buffer):
                exit_code = main(["batch", "apply", "--scan-only"])

        payload = json.loads(buffer.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(run_pipeline.called)
        self.assertEqual(run_pipeline.call_args.kwargs["mask_provider_name"], "rule_based_roi")
        self.assertEqual(run_pipeline.call_args.kwargs["restore_provider_name"], "telea")
        self.assertEqual(run_pipeline.call_args.kwargs["ocr_session_mode"], "auto")

    def test_batch_plan_creates_plan_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_root = root / "inputs"
            output_root = root / "runtime" / "runs"
            _write_image(input_root / "sample.jpg", (160, 100, 3))
            buffer = io.StringIO()

            with patch.dict(os.environ, {}, clear=True), redirect_stdout(buffer):
                exit_code = main(["batch", "plan", "--input", str(input_root), "--output", str(output_root), "--no-recursive"])

            payload = json.loads(buffer.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["command"], "batch plan")
            self.assertEqual(payload["item_count"], 1)
            self.assertTrue(Path(payload["plan_path"]).exists())

    def test_batch_plan_accepts_scan_manifest_input_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_root = root / "inputs"
            output_root = root / "runtime" / "runs"
            scans_root = root / "runtime" / "scans"
            _write_image(input_root / "sample.jpg", (160, 100, 3))
            scan_manifest = create_scan_manifest(input_root, scans_root=scans_root, recursive=False)
            buffer = io.StringIO()

            with patch.dict(os.environ, {}, clear=True), redirect_stdout(buffer):
                exit_code = main(
                    [
                        "batch",
                        "plan",
                        "--scan-manifest",
                        scan_manifest["manifest_path"],
                        "--output",
                        str(output_root),
                    ]
                )

            payload = json.loads(buffer.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["input_mode"], "scan_manifest")
            self.assertEqual(payload["scan_manifest_path"], scan_manifest["manifest_path"])
            self.assertEqual(payload["source_scan_id"], scan_manifest["scan_id"])
            self.assertEqual(payload["item_count"], 1)

    def test_batch_plan_rejects_mixed_scan_manifest_and_direct_flags(self) -> None:
        buffer = io.StringIO()

        with patch.dict(os.environ, {}, clear=True), redirect_stdout(buffer):
            exit_code = main(["batch", "plan", "--scan-manifest", "missing.json", "--limit", "1"])

        payload = json.loads(buffer.getvalue())
        self.assertEqual(exit_code, 2)
        self.assertEqual(payload["status"], "error")
        self.assertIn("do not also pass --dataset-profile, --input, --no-recursive, or --limit", payload["error"])

    def test_batch_apply_rejects_mixed_plan_and_direct_flags(self) -> None:
        buffer = io.StringIO()

        with patch.dict(os.environ, {}, clear=True), redirect_stdout(buffer):
            exit_code = main(["batch", "apply", "--plan", "missing.json", "--scan-only"])

        payload = json.loads(buffer.getvalue())
        self.assertEqual(exit_code, 2)
        self.assertEqual(payload["status"], "error")
        self.assertIn("do not also pass direct input/output or execution flags", payload["error"])

    def test_batch_apply_plan_requires_yes_with_no_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_root = root / "inputs"
            output_root = root / "runtime" / "runs"
            _write_image(input_root / "sample.jpg", (160, 100, 3))
            plan = create_batch_plan(input_root=input_root, output_root=output_root, recursive=False)
            buffer = io.StringIO()

            with patch.dict(os.environ, {}, clear=True), redirect_stdout(buffer):
                exit_code = main(["batch", "apply", "--plan", plan["plan_path"], "--no-input"])

            payload = json.loads(buffer.getvalue())
            self.assertEqual(exit_code, 2)
            self.assertEqual(payload["status"], "error")
            self.assertIn("requires --yes", payload["error"])

    def test_batch_apply_plan_yes_skips_prompt_and_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_root = root / "inputs"
            output_root = root / "runtime" / "runs"
            _write_image(input_root / "sample.jpg", (160, 100, 3))
            plan = create_batch_plan(input_root=input_root, output_root=output_root, recursive=False)
            buffer = io.StringIO()

            with patch("builtins.input", side_effect=AssertionError("input should not be called")):
                with patch.dict(os.environ, {}, clear=True), redirect_stdout(buffer):
                    exit_code = main(["batch", "apply", "--plan", plan["plan_path"], "--yes"])

            payload = json.loads(buffer.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["confirmation_mode"], "yes_flag")
            self.assertTrue(payload["confirmation_required"])
            self.assertEqual(payload["status_counts"]["restored"], 1)

    def test_batch_apply_plan_interactive_confirm_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_root = root / "inputs"
            output_root = root / "runtime" / "runs"
            _write_image(input_root / "sample.jpg", (160, 100, 3))
            plan = create_batch_plan(input_root=input_root, output_root=output_root, recursive=False)
            buffer = io.StringIO()

            with patch("builtins.input", return_value="yes"):
                with patch.dict(os.environ, {}, clear=True), redirect_stdout(buffer):
                    exit_code = main(["batch", "apply", "--plan", plan["plan_path"]])

            payload = json.loads(buffer.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["confirmation_mode"], "interactive")
            self.assertEqual(payload["status_counts"]["restored"], 1)

    def test_batch_apply_plan_interactive_reject_returns_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_root = root / "inputs"
            output_root = root / "runtime" / "runs"
            _write_image(input_root / "sample.jpg", (160, 100, 3))
            plan = create_batch_plan(input_root=input_root, output_root=output_root, recursive=False)
            buffer = io.StringIO()

            with patch("builtins.input", return_value="n"):
                with patch.dict(os.environ, {}, clear=True), redirect_stdout(buffer):
                    exit_code = main(["batch", "apply", "--plan", plan["plan_path"]])

            payload = json.loads(buffer.getvalue())
            self.assertEqual(exit_code, 2)
            self.assertEqual(payload["status"], "error")
            self.assertIn("aborted by user", payload["error"])

    def test_batch_apply_direct_rejects_yes_without_plan(self) -> None:
        buffer = io.StringIO()

        with patch.dict(os.environ, {}, clear=True), redirect_stdout(buffer):
            exit_code = main(["batch", "apply", "--yes"])

        payload = json.loads(buffer.getvalue())
        self.assertEqual(exit_code, 2)
        self.assertEqual(payload["status"], "error")
        self.assertIn("only supported with --plan", payload["error"])

    def test_batch_report_loads_latest_run_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_root = root / "inputs"
            runs_root = root / "runtime" / "runs"
            _write_image(input_root / "sample.jpg", (160, 100, 3))
            run_summary = create_batch_plan(input_root=input_root, output_root=runs_root, recursive=False)
            buffer = io.StringIO()

            with patch("builtins.input", side_effect=AssertionError("input should not be called")):
                with patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()):
                    main(["batch", "apply", "--plan", run_summary["plan_path"], "--yes"])

            with patch.dict(os.environ, {}, clear=True), redirect_stdout(buffer):
                exit_code = main(["batch", "report", "--runs-root", str(runs_root)])

            payload = json.loads(buffer.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["command"], "batch report")
            self.assertEqual(payload["lookup_mode"], "latest")
            self.assertEqual(payload["run_status"], "completed")
            self.assertEqual(payload["completed_item_count"], 1)
            self.assertTrue(Path(payload["summary_json"]).exists())
            self.assertTrue(Path(payload["manifest_json"]).exists())

    def test_batch_report_loads_run_by_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_root = root / "inputs"
            runs_root = root / "runtime" / "runs"
            _write_image(input_root / "sample.jpg", (160, 100, 3))
            plan = create_batch_plan(input_root=input_root, output_root=runs_root, recursive=False)
            run_buffer = io.StringIO()
            report_buffer = io.StringIO()

            with patch("builtins.input", side_effect=AssertionError("input should not be called")):
                with patch.dict(os.environ, {}, clear=True), redirect_stdout(run_buffer):
                    main(["batch", "apply", "--plan", plan["plan_path"], "--yes"])

            run_payload = json.loads(run_buffer.getvalue())
            with patch.dict(os.environ, {}, clear=True), redirect_stdout(report_buffer):
                exit_code = main(["batch", "report", "--runs-root", str(runs_root), "--run-id", run_payload["run_id"]])

            payload = json.loads(report_buffer.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["run_id"], run_payload["run_id"])
            self.assertEqual(payload["lookup_mode"], "run_id")

    def test_batch_report_returns_error_for_missing_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            buffer = io.StringIO()

            with patch.dict(os.environ, {}, clear=True), redirect_stdout(buffer):
                exit_code = main(["batch", "report", "--runs-root", tmp_dir, "--run-id", "missing-run"])

            payload = json.loads(buffer.getvalue())
            self.assertEqual(exit_code, 2)
            self.assertEqual(payload["status"], "error")
            self.assertIn("does not exist", payload["error"])

    def test_batch_resume_continues_interrupted_run(self) -> None:
        from no_watermar.pipeline import run_pipeline

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_root = root / "inputs"
            runs_root = root / "runtime" / "runs"
            _write_image(input_root / "first.jpg", (160, 100, 3))
            _write_image(input_root / "second.jpg", (160, 100, 3))

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

            buffer = io.StringIO()
            with patch.dict(os.environ, {}, clear=True), redirect_stdout(buffer):
                exit_code = main(["batch", "resume", "--runs-root", str(runs_root)])

            payload = json.loads(buffer.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["command"], "batch resume")
            self.assertEqual(payload["lookup_mode"], "latest")
            self.assertEqual(payload["run_status"], "completed")
            self.assertEqual(payload["resumed_item_count"], 1)
            self.assertEqual(payload["resume_count"], 1)

    def test_batch_resume_noops_when_run_is_already_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_root = root / "inputs"
            runs_root = root / "runtime" / "runs"
            _write_image(input_root / "sample.jpg", (160, 100, 3))
            plan = create_batch_plan(input_root=input_root, output_root=runs_root, recursive=False)

            with patch("builtins.input", side_effect=AssertionError("input should not be called")):
                with patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()):
                    main(["batch", "apply", "--plan", plan["plan_path"], "--yes"])

            buffer = io.StringIO()
            with patch.dict(os.environ, {}, clear=True), redirect_stdout(buffer):
                exit_code = main(["batch", "resume", "--runs-root", str(runs_root)])

            payload = json.loads(buffer.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["run_status"], "completed")
            self.assertEqual(payload["resumed_item_count"], 0)
            self.assertTrue(any("No pending items remain" in warning for warning in payload["warnings"]))

    def test_config_show_reports_existing_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = root / "generated.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[watermark_keywords]",
                        'active_presets = ["brand"]',
                        "",
                        "[watermark_keywords.presets]",
                        'brand = ["Acme Studio"]',
                    ]
                ),
                encoding="utf-8",
            )

            buffer = io.StringIO()
            with patch.dict(os.environ, {}, clear=True), redirect_stdout(buffer):
                exit_code = main(["config", "show", "--config", str(config_path)])

            payload = json.loads(buffer.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["command"], "config show")
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["config"]["path"], str(config_path))
            self.assertEqual(payload["watermark_keywords"]["active_presets"], ["brand"])

    def test_config_validate_reports_defaults_when_no_config_file_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            buffer = io.StringIO()
            with patch.dict(os.environ, {}, clear=True), redirect_stdout(buffer):
                exit_code = main(["config", "validate", "--start-dir", tmp_dir])

            payload = json.loads(buffer.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["config"]["path"], None)
            self.assertEqual(payload["config"]["resolution_mode"], "defaults-only")
            self.assertTrue(any("No local no-watermar.toml file was found" in warning for warning in payload["warnings"]))

    def test_config_init_writes_selected_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            buffer = io.StringIO()
            target_path = root / "generated.toml"

            with patch.dict(os.environ, {}, clear=True), redirect_stdout(buffer):
                exit_code = main(["config", "init", "--config", str(target_path), "--template", "brand-social"])

            payload = json.loads(buffer.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["template"], "brand-social")
            self.assertEqual(payload["config_path"], str(target_path))
            self.assertTrue(target_path.exists())
            self.assertIn('brand_social = [', target_path.read_text(encoding="utf-8"))

    def test_config_init_returns_error_when_target_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "no-watermar.toml").write_text("existing\n", encoding="utf-8")
            buffer = io.StringIO()

            with patch.dict(os.environ, {}, clear=True), redirect_stdout(buffer):
                exit_code = main(["config", "init", "--start-dir", str(root)])

            payload = json.loads(buffer.getvalue())
            self.assertEqual(exit_code, 2)
            self.assertEqual(payload["status"], "error")
            self.assertIn("already exists", payload["error"])

    def test_config_validate_returns_error_for_unknown_active_preset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = root / "no-watermar.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[watermark_keywords]",
                        'active_presets = ["missing"]',
                    ]
                ),
                encoding="utf-8",
            )

            buffer = io.StringIO()
            with patch.dict(os.environ, {}, clear=True), redirect_stdout(buffer):
                exit_code = main(["config", "validate", "--config", str(config_path)])

            payload = json.loads(buffer.getvalue())
            self.assertEqual(exit_code, 2)
            self.assertEqual(payload["status"], "error")
            self.assertIn("missing", payload["error"])

    def test_scan_show_accepts_dataset_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_root = root / "inputs" / "smoke"
            config_path = root / "no-watermar.toml"
            _write_image(input_root / "sample.jpg", (160, 100, 3))
            config_path.write_text(
                "\n".join(
                    [
                        "[watermark_keywords]",
                        "",
                        "[profiles.datasets.smoke]",
                        'input = "./inputs/smoke"',
                        "recursive = false",
                        "limit = 1",
                    ]
                ),
                encoding="utf-8",
            )
            buffer = io.StringIO()

            with patch.dict(os.environ, {}, clear=True), redirect_stdout(buffer):
                with patch("pathlib.Path.cwd", return_value=root):
                    exit_code = main(["scan", "show", "--dataset-profile", "smoke"])

            payload = json.loads(buffer.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["item_count"], 1)
            self.assertEqual(payload["dataset_profile"], "smoke")
            self.assertEqual(payload["dataset_profile_config"]["input_root"], str(input_root.resolve()))

    def test_scan_show_reports_discovery_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_root = root / "inputs"
            _write_image(input_root / "sample.jpg", (160, 100, 3))
            buffer = io.StringIO()

            with patch.dict(os.environ, {}, clear=True), redirect_stdout(buffer):
                exit_code = main(["scan", "show", "--input", str(input_root), "--no-recursive"])

            payload = json.loads(buffer.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["command"], "scan show")
            self.assertEqual(payload["mode"], "show")
            self.assertEqual(payload["item_count"], 1)
            self.assertEqual(payload["category_counts"], {"portrait_regular": 1})

    def test_scan_run_persists_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_root = root / "inputs"
            scans_root = root / "runtime" / "scans"
            _write_image(input_root / "sample.jpg", (120, 220, 3))
            buffer = io.StringIO()

            with patch.dict(os.environ, {}, clear=True), redirect_stdout(buffer):
                exit_code = main(
                    ["scan", "run", "--input", str(input_root), "--scans-root", str(scans_root), "--no-recursive"]
                )

            payload = json.loads(buffer.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["command"], "scan run")
            self.assertEqual(payload["mode"], "manifest")
            self.assertTrue(Path(payload["manifest_path"]).exists())
            self.assertTrue(Path(payload["latest_manifest_path"]).exists())

    def test_batch_plan_accepts_dataset_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_root = root / "inputs" / "quick"
            output_root = root / "runtime" / "runs"
            _write_image(input_root / "sample.jpg", (160, 100, 3))
            (root / "no-watermar.toml").write_text(
                "\n".join(
                    [
                        "[watermark_keywords]",
                        "",
                        "[profiles.datasets.quick]",
                        'input = "./inputs/quick"',
                        "recursive = false",
                        "limit = 1",
                    ]
                ),
                encoding="utf-8",
            )
            buffer = io.StringIO()

            with patch.dict(os.environ, {}, clear=True), redirect_stdout(buffer):
                with patch("pathlib.Path.cwd", return_value=root):
                    exit_code = main(["batch", "plan", "--dataset-profile", "quick", "--output", str(output_root)])

            payload = json.loads(buffer.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["item_count"], 1)
            self.assertEqual(payload["dataset_profile"], "quick")
            self.assertEqual(payload["dataset_profile_config"]["input_root"], str(input_root.resolve()))

    def test_batch_plan_accepts_dataset_and_provider_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_root = root / "inputs" / "quick"
            output_root = root / "runtime" / "runs"
            _write_image(input_root / "sample.jpg", (160, 100, 3))
            (root / "no-watermar.toml").write_text(
                "\n".join(
                    [
                        "[watermark_keywords]",
                        "",
                        "[profiles.datasets.quick]",
                        'input = "./inputs/quick"',
                        "recursive = false",
                        "",
                        "[profiles.providers.ocr_telea]",
                        'mask_provider = "paddleocr"',
                        'restore_provider = "telea"',
                        'ocr_session_mode = "persistent"',
                        'restore_prompt = "remove watermark text only"',
                        "",
                        "[profiles.providers.ocr_telea.restore_options]",
                        "steps = 28",
                    ]
                ),
                encoding="utf-8",
            )
            buffer = io.StringIO()

            with patch.dict(os.environ, {}, clear=True), redirect_stdout(buffer):
                with patch("pathlib.Path.cwd", return_value=root):
                    exit_code = main(
                        [
                            "batch",
                            "plan",
                            "--dataset-profile",
                            "quick",
                            "--provider-profile",
                            "ocr_telea",
                            "--output",
                            str(output_root),
                        ]
                    )

            payload = json.loads(buffer.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["dataset_profile"], "quick")
            self.assertEqual(payload["provider_profile"], "ocr_telea")
            self.assertEqual(payload["mask_provider"], "paddleocr")
            self.assertEqual(payload["restore_provider"], "telea")
            self.assertEqual(payload["ocr_session_mode"], "persistent")
            self.assertEqual(payload["restore_prompt"], "remove watermark text only")
            self.assertEqual(payload["restore_options"]["steps"], 28)

    def test_batch_apply_accepts_corner_crop_provider_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_root = root / "inputs" / "quick"
            output_root = root / "runtime" / "runs"
            _write_image(input_root / "sample.jpg", (160, 100, 3))
            (root / "no-watermar.toml").write_text(
                "\n".join(
                    [
                        "[watermark_keywords]",
                        "",
                        "[profiles.datasets.quick]",
                        'input = "./inputs/quick"',
                        "recursive = false",
                        "",
                        "[profiles.providers.ocr_corner_crop]",
                        'mask_provider = "paddleocr"',
                        'restore_provider = "corner_crop"',
                        'ocr_session_mode = "persistent"',
                        "",
                        "[profiles.providers.ocr_corner_crop.restore_options]",
                        "edge_tolerance = 24",
                    ]
                ),
                encoding="utf-8",
            )
            buffer = io.StringIO()
            summary = {"status": "ok", "run_id": "20260406-000000-000003"}

            with patch("no_watermar.cli.commands.batch.run_pipeline", return_value=summary) as run_pipeline:
                with patch.dict(os.environ, {}, clear=True), redirect_stdout(buffer):
                    with patch("pathlib.Path.cwd", return_value=root):
                        exit_code = main(
                            [
                                "batch",
                                "apply",
                                "--dataset-profile",
                                "quick",
                                "--provider-profile",
                                "ocr_corner_crop",
                                "--output",
                                str(output_root),
                            ]
                        )

            payload = json.loads(buffer.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(run_pipeline.call_args.kwargs["mask_provider_name"], "paddleocr")
            self.assertEqual(run_pipeline.call_args.kwargs["restore_provider_name"], "corner_crop")
            self.assertEqual(run_pipeline.call_args.kwargs["ocr_session_mode"], "persistent")
            self.assertEqual(run_pipeline.call_args.kwargs["restore_options"]["edge_tolerance"], 24)
            self.assertEqual(payload["dataset_profile"], "quick")
            self.assertEqual(payload["provider_profile"], "ocr_corner_crop")

    def test_batch_apply_accepts_dataset_and_provider_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_root = root / "inputs" / "quick"
            output_root = root / "runtime" / "runs"
            _write_image(input_root / "sample.jpg", (160, 100, 3))
            (root / "no-watermar.toml").write_text(
                "\n".join(
                    [
                        "[watermark_keywords]",
                        "",
                        "[profiles.datasets.quick]",
                        'input = "./inputs/quick"',
                        "recursive = false",
                        "",
                        "[profiles.providers.ocr_telea]",
                        'mask_provider = "paddleocr"',
                        'restore_provider = "telea"',
                        'ocr_session_mode = "persistent"',
                        'restore_prompt = "restore background detail"',
                        'restore_negative_prompt = "smudge"',
                        "",
                        "[profiles.providers.ocr_telea.restore_options]",
                        "steps = 32",
                    ]
                ),
                encoding="utf-8",
            )
            buffer = io.StringIO()
            summary = {"status": "ok", "run_id": "20260405-000000-000002"}

            with patch("no_watermar.cli.commands.batch.run_pipeline", return_value=summary) as run_pipeline:
                with patch.dict(os.environ, {}, clear=True), redirect_stdout(buffer):
                    with patch("pathlib.Path.cwd", return_value=root):
                        exit_code = main(
                            [
                                "batch",
                                "apply",
                                "--dataset-profile",
                                "quick",
                                "--provider-profile",
                                "ocr_telea",
                                "--output",
                                str(output_root),
                            ]
                        )

            payload = json.loads(buffer.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(run_pipeline.call_args.kwargs["mask_provider_name"], "paddleocr")
            self.assertEqual(run_pipeline.call_args.kwargs["restore_provider_name"], "telea")
            self.assertEqual(run_pipeline.call_args.kwargs["ocr_session_mode"], "persistent")
            self.assertEqual(run_pipeline.call_args.kwargs["restore_prompt"], "restore background detail")
            self.assertEqual(run_pipeline.call_args.kwargs["restore_negative_prompt"], "smudge")
            self.assertEqual(run_pipeline.call_args.kwargs["restore_options"]["steps"], 32)
            self.assertEqual(payload["dataset_profile"], "quick")
            self.assertEqual(payload["provider_profile"], "ocr_telea")

    def test_scan_run_rejects_scans_root_inside_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_root = root / "inputs"
            _write_image(input_root / "sample.jpg", (120, 220, 3))
            buffer = io.StringIO()

            with patch.dict(os.environ, {}, clear=True), redirect_stdout(buffer):
                exit_code = main(
                    [
                        "scan",
                        "run",
                        "--input",
                        str(input_root),
                        "--scans-root",
                        str(input_root / "runtime" / "scans"),
                        "--no-recursive",
                    ]
                )

            payload = json.loads(buffer.getvalue())
            self.assertEqual(exit_code, 2)
            self.assertEqual(payload["status"], "error")
            self.assertIn("must not be inside input root", payload["error"])

    def test_providers_list_reports_descriptors(self) -> None:
        buffer = io.StringIO()

        with patch("no_watermar.cli.commands.providers.list_provider_descriptors", return_value=[{"name": "mock"}]):
            with patch.dict(os.environ, {}, clear=True), redirect_stdout(buffer):
                exit_code = main(["providers", "list"])

        payload = json.loads(buffer.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload, [{"name": "mock"}])

    def test_providers_doctor_reports_diagnostics(self) -> None:
        buffer = io.StringIO()
        summary = {"status": "ok", "summary": {"implemented_total": 1}}

        with patch("no_watermar.cli.commands.providers.build_provider_doctor_report", return_value=summary):
            with patch.dict(os.environ, {}, clear=True), redirect_stdout(buffer):
                exit_code = main(["providers", "doctor"])

        payload = json.loads(buffer.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["summary"]["implemented_total"], 1)

    def test_benchmark_trends_reports_snapshot(self) -> None:
        buffer = io.StringIO()
        summary = {"snapshot_id": "20260405-010203-000001", "dataset_id": "regular_corner_text"}

        with patch("no_watermar.cli.commands.benchmark.build_benchmark_trend_snapshot", return_value=summary):
            with patch.dict(os.environ, {}, clear=True), redirect_stdout(buffer):
                exit_code = main(["benchmark", "trends"])

        payload = json.loads(buffer.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["snapshot_id"], "20260405-010203-000001")
        self.assertEqual(payload["dataset_id"], "regular_corner_text")

    def test_benchmark_run_accepts_dataset_and_provider_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "no-watermar.toml").write_text(
                "\n".join(
                    [
                        "[watermark_keywords]",
                        "",
                        "[profiles.datasets.local_smoke]",
                        "limit = 2",
                        'benchmark_dataset = "regular_corner_text"',
                        "",
                        "[profiles.providers.ocr_telea]",
                        'mask_provider = "paddleocr"',
                        'restore_provider = "telea"',
                        'ocr_session_mode = "persistent"',
                        'restore_prompt = "repair texture with a generative model"',
                        "",
                        "[profiles.providers.ocr_telea.restore_options]",
                        "steps = 40",
                    ]
                ),
                encoding="utf-8",
            )
            buffer = io.StringIO()
            summary = {"run_id": "20260405-000000-000001"}

            with patch("no_watermar.cli.commands.benchmark.run_benchmark", return_value=summary) as run_benchmark:
                with patch.dict(os.environ, {}, clear=True), redirect_stdout(buffer):
                    with patch("pathlib.Path.cwd", return_value=root):
                        exit_code = main(
                            [
                                "benchmark",
                                "run",
                                "--dataset-profile",
                                "local_smoke",
                                "--provider-profile",
                                "ocr_telea",
                            ]
                        )

            payload = json.loads(buffer.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(run_benchmark.call_args.kwargs["dataset_id"], "regular_corner_text")
            self.assertEqual(run_benchmark.call_args.kwargs["mask_provider_name"], "paddleocr")
            self.assertEqual(run_benchmark.call_args.kwargs["restore_provider_name"], "telea")
            self.assertEqual(run_benchmark.call_args.kwargs["ocr_session_mode"], "persistent")
            self.assertEqual(run_benchmark.call_args.kwargs["restore_prompt"], "repair texture with a generative model")
            self.assertEqual(run_benchmark.call_args.kwargs["restore_options"]["steps"], 40)
            self.assertEqual(run_benchmark.call_args.kwargs["limit"], 2)
            self.assertEqual(payload["dataset_profile"], "local_smoke")
            self.assertEqual(payload["provider_profile"], "ocr_telea")

    def test_run_py_compatibility_wrapper_passes_through_scan_group(self) -> None:
        with patch("no_watermar.cli.main", return_value=0) as root_main:
            with patch.object(sys, "argv", ["run.py", "scan", "show"]):
                exit_code = run_main()

        self.assertEqual(exit_code, 0)
        root_main.assert_called_once_with(["scan", "show"])

    def test_run_py_compatibility_wrapper_passes_through_benchmark_group(self) -> None:
        with patch("no_watermar.cli.main", return_value=0) as root_main:
            with patch.object(sys, "argv", ["run.py", "benchmark", "trends"]):
                exit_code = run_main()

        self.assertEqual(exit_code, 0)
        root_main.assert_called_once_with(["benchmark", "trends"])

    def test_benchmark_cli_compatibility_wrapper_prefixes_root_group(self) -> None:
        with patch("no_watermar.benchmark_cli.root_main", return_value=0) as root_main:
            exit_code = benchmark_main(["probe-providers"])

        self.assertEqual(exit_code, 0)
        root_main.assert_called_once_with(["benchmark", "probe-providers"])


if __name__ == "__main__":
    unittest.main()
