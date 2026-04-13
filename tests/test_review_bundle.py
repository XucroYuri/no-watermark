from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from no_watermar.io_utils import write_json
from no_watermar.review_bundle import build_review_bundle


class ReviewBundleTests(unittest.TestCase):
    def test_build_review_bundle_copies_common_inputs_and_provider_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            inputs = root / "dataset" / "inputs"
            seed_masks = root / "dataset" / "seed_masks"
            seed_overlays = root / "dataset" / "seed_overlays"
            provider_root = root / "runs"
            artifact_root = root / "artifacts"

            input_path = inputs / "sample.jpg"
            seed_mask_path = seed_masks / "sample.png"
            seed_overlay_path = seed_overlays / "sample.jpg"
            prompt_path = root / "dataset" / "prompts" / "sample.txt"
            self._write_bytes(input_path, b"input")
            self._write_bytes(seed_mask_path, b"seed-mask")
            self._write_bytes(seed_overlay_path, b"seed-overlay")
            self._write_bytes(prompt_path, b"prompt")

            telea_result = self._build_result(
                item_id="regular-0001",
                input_path=input_path,
                seed_mask_path=seed_mask_path,
                seed_overlay_path=seed_overlay_path,
                prompt_path=prompt_path,
                relative_path="sample.jpg",
                provider_root=provider_root / "telea",
                confidence=0.91,
                mean_abs_diff=0.41,
                edge_delta=0.52,
                restore_latency_ms=121.3,
                mask_latency_ms=45.0,
            )
            brushnet_result = self._build_result(
                item_id="regular-0001",
                input_path=input_path,
                seed_mask_path=seed_mask_path,
                seed_overlay_path=seed_overlay_path,
                prompt_path=prompt_path,
                relative_path="sample.jpg",
                provider_root=provider_root / "brushnet",
                confidence=0.91,
                mean_abs_diff=0.73,
                edge_delta=0.88,
                restore_latency_ms=26032.1,
                mask_latency_ms=55036.8,
            )

            telea_report = artifact_root / "telea.json"
            brushnet_report = artifact_root / "brushnet.json"
            compare_path = artifact_root / "compare.json"
            trend_path = artifact_root / "latest.md"

            write_json(
                telea_report,
                {
                    "dataset_id": "regular_corner_text",
                    "mask_provider": "paddleocr",
                    "restore_provider": "telea",
                    "results": [telea_result],
                    "status_counts": {"restored": 1},
                },
            )
            write_json(
                brushnet_report,
                {
                    "dataset_id": "regular_corner_text",
                    "mask_provider": "paddleocr",
                    "restore_provider": "brushnet",
                    "results": [brushnet_result],
                    "status_counts": {"restored": 1},
                },
            )
            write_json(compare_path, {"delta_summary": {"mean_abs_diff": {"mean": 0.32}}})
            trend_path.parent.mkdir(parents=True, exist_ok=True)
            trend_path.write_text("# trend\n", encoding="utf-8")

            bundle = build_review_bundle(
                report_paths=[telea_report, brushnet_report],
                compare_paths=[compare_path],
                trend_paths=[trend_path],
                output_dir=root / "review",
            )

            review_root = root / "review"
            self.assertEqual(bundle["provider_labels"], ["telea", "brushnet"])
            self.assertTrue((review_root / "review.json").exists())
            self.assertTrue((review_root / "README.md").exists())
            self.assertTrue((review_root / "artifacts" / "reports" / "telea.json").exists())
            self.assertTrue((review_root / "artifacts" / "comparisons" / "compare.json").exists())
            self.assertTrue((review_root / "artifacts" / "trends" / "latest.md").exists())
            self.assertTrue((review_root / "items" / "regular-0001" / "source" / "input.jpg").exists())
            self.assertTrue((review_root / "items" / "regular-0001" / "telea" / "restored.jpg").exists())
            self.assertTrue((review_root / "items" / "regular-0001" / "brushnet" / "restored.jpg").exists())

            review_payload = json.loads((review_root / "review.json").read_text(encoding="utf-8"))
            self.assertEqual(review_payload["item_count"], 1)
            self.assertEqual(review_payload["providers"]["telea"]["mean_metrics"]["mean_abs_diff"], 0.41)
            self.assertEqual(review_payload["providers"]["brushnet"]["mean_metrics"]["restore_latency_ms"], 26032.1)
            self.assertIn("artifacts/comparisons/compare.json", review_payload["comparison_artifacts"])
            self.assertIn("artifacts/trends/latest.md", review_payload["trend_artifacts"])
            self.assertIn("telea", (review_root / "README.md").read_text(encoding="utf-8"))
            self.assertIn("brushnet", (review_root / "README.md").read_text(encoding="utf-8"))

    def test_build_review_bundle_dedupes_duplicate_provider_names_and_accepts_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_path = root / "input.jpg"
            seed_mask_path = root / "seed_mask.png"
            seed_overlay_path = root / "seed_overlay.jpg"
            prompt_path = root / "prompt.txt"
            self._write_bytes(input_path, b"input")
            self._write_bytes(seed_mask_path, b"seed-mask")
            self._write_bytes(seed_overlay_path, b"seed-overlay")
            self._write_bytes(prompt_path, b"prompt")

            base_report = root / "base.json"
            tuned_report = root / "tuned.json"
            write_json(
                base_report,
                {
                    "run_id": "run-base",
                    "dataset_id": "regular_corner_text",
                    "restore_provider": "brushnet",
                    "results": [
                        self._build_result(
                            item_id="regular-0001",
                            input_path=input_path,
                            seed_mask_path=seed_mask_path,
                            seed_overlay_path=seed_overlay_path,
                            prompt_path=prompt_path,
                            relative_path="sample.jpg",
                            provider_root=root / "base",
                            confidence=0.9,
                            mean_abs_diff=0.8,
                            edge_delta=1.0,
                            restore_latency_ms=10.0,
                            mask_latency_ms=20.0,
                        )
                    ],
                },
            )
            write_json(
                tuned_report,
                {
                    "run_id": "run-tuned",
                    "dataset_id": "regular_corner_text",
                    "restore_provider": "brushnet",
                    "results": [
                        self._build_result(
                            item_id="regular-0001",
                            input_path=input_path,
                            seed_mask_path=seed_mask_path,
                            seed_overlay_path=seed_overlay_path,
                            prompt_path=prompt_path,
                            relative_path="sample.jpg",
                            provider_root=root / "tuned",
                            confidence=0.9,
                            mean_abs_diff=1.1,
                            edge_delta=1.2,
                            restore_latency_ms=30.0,
                            mask_latency_ms=20.0,
                        )
                    ],
                },
            )

            auto_bundle = build_review_bundle(
                report_paths=[base_report, tuned_report],
                output_dir=root / "review-auto",
            )
            self.assertEqual(auto_bundle["provider_labels"], ["brushnet@run-base", "brushnet@run-tuned"])

            labeled_bundle = build_review_bundle(
                report_paths=[base_report, tuned_report],
                provider_labels=["brushnet_base", "brushnet_tuned"],
                output_dir=root / "review-labeled",
            )
            self.assertEqual(labeled_bundle["provider_labels"], ["brushnet_base", "brushnet_tuned"])

    def test_build_review_bundle_keeps_duplicate_artifact_filenames_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_path = root / "input.jpg"
            seed_mask_path = root / "seed_mask.png"
            seed_overlay_path = root / "seed_overlay.jpg"
            prompt_path = root / "prompt.txt"
            self._write_bytes(input_path, b"input")
            self._write_bytes(seed_mask_path, b"seed-mask")
            self._write_bytes(seed_overlay_path, b"seed-overlay")
            self._write_bytes(prompt_path, b"prompt")

            base_report = root / "runs" / "run-base" / "reports" / "regular_corner_text_paddleocr__brushnet.json"
            tuned_report = root / "runs" / "run-tuned" / "reports" / "regular_corner_text_paddleocr__brushnet.json"
            compare_a = root / "comparisons" / "base-vs-a" / "regular_corner_text_paddleocr__brushnet.json"
            compare_b = root / "comparisons" / "base-vs-b" / "regular_corner_text_paddleocr__brushnet.json"

            write_json(
                base_report,
                {
                    "run_id": "run-base",
                    "dataset_id": "regular_corner_text",
                    "restore_provider": "brushnet",
                    "results": [
                        self._build_result(
                            item_id="regular-0001",
                            input_path=input_path,
                            seed_mask_path=seed_mask_path,
                            seed_overlay_path=seed_overlay_path,
                            prompt_path=prompt_path,
                            relative_path="sample.jpg",
                            provider_root=root / "base",
                            confidence=0.9,
                            mean_abs_diff=0.8,
                            edge_delta=1.0,
                            restore_latency_ms=10.0,
                            mask_latency_ms=20.0,
                        )
                    ],
                },
            )
            write_json(
                tuned_report,
                {
                    "run_id": "run-tuned",
                    "dataset_id": "regular_corner_text",
                    "restore_provider": "brushnet",
                    "results": [
                        self._build_result(
                            item_id="regular-0001",
                            input_path=input_path,
                            seed_mask_path=seed_mask_path,
                            seed_overlay_path=seed_overlay_path,
                            prompt_path=prompt_path,
                            relative_path="sample.jpg",
                            provider_root=root / "tuned",
                            confidence=0.9,
                            mean_abs_diff=1.1,
                            edge_delta=1.2,
                            restore_latency_ms=30.0,
                            mask_latency_ms=20.0,
                        )
                    ],
                },
            )
            write_json(compare_a, {"name": "a"})
            write_json(compare_b, {"name": "b"})

            bundle = build_review_bundle(
                report_paths=[base_report, tuned_report],
                provider_labels=["brushnet_base", "brushnet_tuned"],
                compare_paths=[compare_a, compare_b],
                output_dir=root / "review-distinct-artifacts",
            )

            self.assertIn(
                "artifacts/reports/run-base/reports/regular_corner_text_paddleocr__brushnet.json",
                bundle["report_artifacts"],
            )
            self.assertIn(
                "artifacts/reports/run-tuned/reports/regular_corner_text_paddleocr__brushnet.json",
                bundle["report_artifacts"],
            )
            self.assertIn(
                "artifacts/comparisons/base-vs-a/regular_corner_text_paddleocr__brushnet.json",
                bundle["comparison_artifacts"],
            )
            self.assertIn(
                "artifacts/comparisons/base-vs-b/regular_corner_text_paddleocr__brushnet.json",
                bundle["comparison_artifacts"],
            )

    def _build_result(
        self,
        *,
        item_id: str,
        input_path: Path,
        seed_mask_path: Path,
        seed_overlay_path: Path,
        prompt_path: Path,
        relative_path: str,
        provider_root: Path,
        confidence: float,
        mean_abs_diff: float,
        edge_delta: float,
        restore_latency_ms: float,
        mask_latency_ms: float,
    ) -> dict[str, object]:
        mask_path = provider_root / "masks" / "sample.png"
        overlay_path = provider_root / "overlays" / "sample.jpg"
        restored_path = provider_root / "restored" / "sample.jpg"
        self._write_bytes(mask_path, b"mask")
        self._write_bytes(overlay_path, b"overlay")
        self._write_bytes(restored_path, b"restored")
        return {
            "item": {
                "item_id": item_id,
                "dataset_id": "regular_corner_text",
                "input_path": str(input_path),
                "relative_path": relative_path,
                "source_category": "portrait_regular",
                "benchmark_category": "regular_corner_text",
                "prompt": "remove text",
                "prompt_path": str(prompt_path),
                "seed_mask_path": str(seed_mask_path),
                "seed_overlay_path": str(seed_overlay_path),
            },
            "status": "restored",
            "note": "",
            "mask_path": str(mask_path),
            "overlay_path": str(overlay_path),
            "restored_path": str(restored_path),
            "metrics": {
                "mean_abs_diff": mean_abs_diff,
                "edge_delta": edge_delta,
                "restore_latency_ms": restore_latency_ms,
                "mask_latency_ms": mask_latency_ms,
            },
            "mask_result": {
                "confidence": confidence,
            },
        }

    def _write_bytes(self, path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
