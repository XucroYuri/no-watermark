from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from no_watermar.benchmark_models import BenchmarkDatasetItem
from no_watermar.benchmark_restore_session import PowerPaintExecutionContext
from no_watermar.benchmark_providers import (
    BrushNetRestoreProvider,
    DiffusersInpaintRestoreProvider,
    PowerPaintV21RestoreProvider,
    list_provider_descriptors,
)


class DiffusersProviderTests(unittest.TestCase):
    def test_diffusers_restore_provider_uses_sidecar_and_preserves_prompt_contract(self) -> None:
        provider = DiffusersInpaintRestoreProvider()
        image = np.full((32, 24, 3), 255, dtype=np.uint8)
        mask = np.zeros((32, 24), dtype=np.uint8)
        item = BenchmarkDatasetItem(
            item_id="sample",
            dataset_id="regular_corner_text",
            source_path=Path("source.jpg"),
            input_path=Path("input.jpg"),
            relative_path=Path("input.jpg"),
            width=24,
            height=32,
            source_category="portrait_regular",
            benchmark_category="regular_corner_text",
            prompt="restore background",
            prompt_path=Path("prompt.txt"),
        )

        with patch("no_watermar.benchmark_providers._diffusers_module_available", return_value=False):
            with patch.dict(os.environ, {"NO_WATERMAR_DIFFUSERS_PYTHON": "C:\\mock\\diffusers\\python.exe"}, clear=True):
                with patch(
                    "no_watermar.benchmark_providers._run_restore_sidecar",
                    return_value=(
                        {
                            "latency_ms": 321.5,
                            "meta": {
                                "engine": "diffusers_autopipeline",
                                "model_id": "mock/model",
                                "prompt_used": "restore background",
                                "restore_options": {"model_id": "mock/model", "steps": 28},
                            },
                        },
                        image.copy(),
                    ),
                ) as run_sidecar:
                    result = provider.restore(
                        item,
                        image,
                        mask,
                        prompt="restore background",
                        negative_prompt="blur",
                        meta={"restore_options": {"model_id": "mock/model", "steps": 28}},
                    )

        self.assertEqual(result.provider_name, "diffusers_inpaint")
        self.assertEqual(result.latency_ms, 321.5)
        self.assertEqual(result.meta["model_id"], "mock/model")
        self.assertEqual(result.meta["restore_options"]["steps"], 28)
        self.assertTrue(run_sidecar.called)
        self.assertEqual(run_sidecar.call_args.kwargs["prompt"], "restore background")
        self.assertEqual(run_sidecar.call_args.kwargs["negative_prompt"], "blur")
        self.assertEqual(run_sidecar.call_args.kwargs["options"]["model_id"], "mock/model")

    def test_list_provider_descriptors_includes_diffusers_restore_provider(self) -> None:
        with patch("no_watermar.benchmark_providers._describe_paddleocr_runtime", return_value=(True, "ok", {"ok": True})):
            with patch("no_watermar.benchmark_providers._describe_lama_runtime", return_value=(True, "ok", {"ok": True})):
                with patch(
                    "no_watermar.benchmark_providers._describe_diffusers_runtime",
                    return_value=(False, "NO_WATERMAR_DIFFUSERS_PYTHON is not set.", {"ok": False}),
                ):
                    with patch(
                        "no_watermar.benchmark_providers._describe_powerpaint_runtime",
                        return_value=(False, "NO_WATERMAR_POWERPAINT_PYTHON is not set.", {"ok": False}),
                    ):
                        with patch(
                            "no_watermar.benchmark_providers._describe_brushnet_runtime",
                            return_value=(False, "NO_WATERMAR_BRUSHNET_PYTHON is not set.", {"ok": False}),
                        ):
                            descriptors = list_provider_descriptors()

        diffusers_descriptor = next(
            descriptor for descriptor in descriptors["restore_providers"] if descriptor["name"] == "diffusers_inpaint"
        )
        self.assertTrue(diffusers_descriptor["implemented"])
        self.assertEqual(diffusers_descriptor["default_mode"], "oneshot-sidecar")
        self.assertIn("NO_WATERMAR_DIFFUSERS_PYTHON", diffusers_descriptor["required_env_vars"])

    def test_powerpaint_restore_provider_uses_sidecar_and_preserves_prompt_contract(self) -> None:
        provider = PowerPaintV21RestoreProvider()
        image = np.full((32, 24, 3), 255, dtype=np.uint8)
        mask = np.zeros((32, 24), dtype=np.uint8)
        item = BenchmarkDatasetItem(
            item_id="sample",
            dataset_id="regular_corner_text",
            source_path=Path("source.jpg"),
            input_path=Path("input.jpg"),
            relative_path=Path("input.jpg"),
            width=24,
            height=32,
            source_category="portrait_regular",
            benchmark_category="regular_corner_text",
            prompt="remove watermark",
            prompt_path=Path("prompt.txt"),
        )

        with patch("no_watermar.benchmark_providers._powerpaint_module_available", return_value=False):
            with patch.dict(os.environ, {"NO_WATERMAR_POWERPAINT_PYTHON": "C:\\mock\\powerpaint\\python.exe"}, clear=True):
                with patch(
                    "no_watermar.benchmark_providers._run_restore_sidecar",
                    return_value=(
                        {
                            "latency_ms": 654.0,
                            "meta": {
                                "engine": "powerpaint_v2_1",
                                "checkpoint_dir": "C:\\mock\\checkpoints\\ppt-v2-1",
                                "prompt_used": "remove watermark",
                                "restore_options": {
                                    "checkpoint_dir": "C:\\mock\\checkpoints\\ppt-v2-1",
                                    "steps": 36,
                                },
                            },
                        },
                        image.copy(),
                    ),
                ) as run_sidecar:
                    result = provider.restore(
                        item,
                        image,
                        mask,
                        prompt="remove watermark",
                        negative_prompt="text, logo",
                        meta={"restore_options": {"checkpoint_dir": "C:\\mock\\checkpoints\\ppt-v2-1", "steps": 36}},
                    )

        self.assertEqual(result.provider_name, "powerpaint_v2_1")
        self.assertEqual(result.latency_ms, 654.0)
        self.assertEqual(result.meta["checkpoint_dir"], "C:\\mock\\checkpoints\\ppt-v2-1")
        self.assertEqual(result.meta["restore_options"]["steps"], 36)
        self.assertTrue(run_sidecar.called)
        self.assertEqual(run_sidecar.call_args.kwargs["prompt"], "remove watermark")
        self.assertEqual(run_sidecar.call_args.kwargs["negative_prompt"], "text, logo")
        self.assertEqual(run_sidecar.call_args.kwargs["options"]["checkpoint_dir"], "C:\\mock\\checkpoints\\ppt-v2-1")

    def test_powerpaint_restore_provider_uses_persistent_context_when_provided(self) -> None:
        provider = PowerPaintV21RestoreProvider()
        image = np.full((32, 24, 3), 255, dtype=np.uint8)
        mask = np.zeros((32, 24), dtype=np.uint8)
        item = BenchmarkDatasetItem(
            item_id="sample",
            dataset_id="regular_corner_text",
            source_path=Path("source.jpg"),
            input_path=Path("input.jpg"),
            relative_path=Path("input.jpg"),
            width=24,
            height=32,
            source_category="portrait_regular",
            benchmark_category="regular_corner_text",
            prompt="remove watermark",
            prompt_path=Path("prompt.txt"),
        )

        class FakePowerPaintSession:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def restore(self, **kwargs):
                self.calls.append(kwargs)
                return (
                    {
                        "latency_ms": 4321.0,
                        "meta": {
                            "engine": "powerpaint_v2_1",
                            "session_mode": "persistent-sidecar",
                            "prompt_used": kwargs["prompt"],
                            "restore_options": dict(kwargs.get("options") or {}),
                        },
                    },
                    image.copy(),
                )

        fake_session = FakePowerPaintSession()
        fake_context = PowerPaintExecutionContext(
            requested_mode="auto",
            resolved_mode="persistent-sidecar",
            python_executable="C:\\mock\\powerpaint\\python.exe",
            note="mock persistent PowerPaint session",
            session=fake_session,
        )

        with patch("no_watermar.benchmark_providers._powerpaint_module_available", return_value=False):
            result = provider.restore(
                item,
                image,
                mask,
                prompt="remove watermark",
                negative_prompt="text, logo",
                meta={
                    "restore_options": {"checkpoint_dir": "C:\\mock\\checkpoints\\ppt-v2-1", "steps": 36},
                    "powerpaint_context": fake_context,
                },
            )

        self.assertEqual(result.provider_name, "powerpaint_v2_1")
        self.assertEqual(result.latency_ms, 4321.0)
        self.assertEqual(result.meta["session_mode"], "persistent-sidecar")
        self.assertEqual(result.meta["restore_options"]["steps"], 36)
        self.assertEqual(fake_session.calls[0]["prompt"], "remove watermark")
        self.assertEqual(fake_session.calls[0]["negative_prompt"], "text, logo")

    def test_list_provider_descriptors_includes_powerpaint_restore_provider(self) -> None:
        with patch("no_watermar.benchmark_providers._describe_paddleocr_runtime", return_value=(True, "ok", {"ok": True})):
            with patch("no_watermar.benchmark_providers._describe_lama_runtime", return_value=(True, "ok", {"ok": True})):
                with patch("no_watermar.benchmark_providers._describe_diffusers_runtime", return_value=(True, "ok", {"ok": True})):
                    with patch(
                        "no_watermar.benchmark_providers._describe_powerpaint_runtime",
                        return_value=(False, "NO_WATERMAR_POWERPAINT_PYTHON is not set.", {"ok": False}),
                    ):
                        with patch(
                            "no_watermar.benchmark_providers._describe_brushnet_runtime",
                            return_value=(False, "NO_WATERMAR_BRUSHNET_PYTHON is not set.", {"ok": False}),
                        ):
                            descriptors = list_provider_descriptors()

        powerpaint_descriptor = next(
            descriptor for descriptor in descriptors["restore_providers"] if descriptor["name"] == "powerpaint_v2_1"
        )
        self.assertTrue(powerpaint_descriptor["implemented"])
        self.assertEqual(powerpaint_descriptor["default_mode"], "persistent-sidecar")
        self.assertIn("persistent-sidecar", powerpaint_descriptor["execution_modes"])
        self.assertIn("NO_WATERMAR_POWERPAINT_PYTHON", powerpaint_descriptor["required_env_vars"])
        self.assertIn("NO_WATERMAR_POWERPAINT_CHECKPOINT_DIR", powerpaint_descriptor["required_env_vars"])

    def test_brushnet_restore_provider_uses_sidecar_and_preserves_prompt_contract(self) -> None:
        provider = BrushNetRestoreProvider()
        image = np.full((32, 24, 3), 255, dtype=np.uint8)
        mask = np.zeros((32, 24), dtype=np.uint8)
        item = BenchmarkDatasetItem(
            item_id="sample",
            dataset_id="regular_corner_text",
            source_path=Path("source.jpg"),
            input_path=Path("input.jpg"),
            relative_path=Path("input.jpg"),
            width=24,
            height=32,
            source_category="portrait_regular",
            benchmark_category="regular_corner_text",
            prompt="remove watermark",
            prompt_path=Path("prompt.txt"),
        )

        with patch("no_watermar.benchmark_providers._brushnet_module_available", return_value=False):
            with patch.dict(os.environ, {"NO_WATERMAR_BRUSHNET_PYTHON": "C:\\mock\\brushnet\\python.exe"}, clear=True):
                with patch(
                    "no_watermar.benchmark_providers._run_restore_sidecar",
                    return_value=(
                        {
                            "latency_ms": 777.0,
                            "meta": {
                                "engine": "brushnet",
                                "brushnet_model_path": "C:\\mock\\brushnet\\segmentation_mask_brushnet_ckpt",
                                "prompt_used": "remove watermark",
                                "restore_options": {
                                    "brushnet_model_path": "C:\\mock\\brushnet\\segmentation_mask_brushnet_ckpt",
                                    "steps": 40,
                                },
                            },
                        },
                        image.copy(),
                    ),
                ) as run_sidecar:
                    result = provider.restore(
                        item,
                        image,
                        mask,
                        prompt="remove watermark",
                        negative_prompt="text, logo",
                        meta={
                            "restore_options": {
                                "brushnet_model_path": "C:\\mock\\brushnet\\segmentation_mask_brushnet_ckpt",
                                "steps": 40,
                            }
                        },
                    )

        self.assertEqual(result.provider_name, "brushnet")
        self.assertEqual(result.latency_ms, 777.0)
        self.assertEqual(result.meta["brushnet_model_path"], "C:\\mock\\brushnet\\segmentation_mask_brushnet_ckpt")
        self.assertEqual(result.meta["restore_options"]["steps"], 40)
        self.assertTrue(run_sidecar.called)
        self.assertEqual(run_sidecar.call_args.kwargs["prompt"], "remove watermark")
        self.assertEqual(run_sidecar.call_args.kwargs["negative_prompt"], "text, logo")
        self.assertEqual(
            run_sidecar.call_args.kwargs["options"]["brushnet_model_path"],
            "C:\\mock\\brushnet\\segmentation_mask_brushnet_ckpt",
        )

    def test_list_provider_descriptors_includes_brushnet_restore_provider(self) -> None:
        with patch("no_watermar.benchmark_providers._describe_paddleocr_runtime", return_value=(True, "ok", {"ok": True})):
            with patch("no_watermar.benchmark_providers._describe_lama_runtime", return_value=(True, "ok", {"ok": True})):
                with patch("no_watermar.benchmark_providers._describe_diffusers_runtime", return_value=(True, "ok", {"ok": True})):
                    with patch("no_watermar.benchmark_providers._describe_powerpaint_runtime", return_value=(True, "ok", {"ok": True})):
                        with patch(
                            "no_watermar.benchmark_providers._describe_brushnet_runtime",
                            return_value=(False, "NO_WATERMAR_BRUSHNET_PYTHON is not set.", {"ok": False}),
                        ):
                            descriptors = list_provider_descriptors()

        brushnet_descriptor = next(
            descriptor for descriptor in descriptors["restore_providers"] if descriptor["name"] == "brushnet"
        )
        self.assertTrue(brushnet_descriptor["implemented"])
        self.assertEqual(brushnet_descriptor["default_mode"], "oneshot-sidecar")
        self.assertIn("inprocess", brushnet_descriptor["execution_modes"])
        self.assertIn("NO_WATERMAR_BRUSHNET_PYTHON", brushnet_descriptor["required_env_vars"])
        self.assertIn("NO_WATERMAR_BRUSHNET_MODEL", brushnet_descriptor["required_env_vars"])


if __name__ == "__main__":
    unittest.main()
