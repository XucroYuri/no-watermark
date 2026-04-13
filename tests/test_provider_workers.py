from __future__ import annotations

import os
import tempfile
import types
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
from PIL import Image

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from no_watermar.provider_workers import (
    _BRUSHNET_PIPELINE_CACHE,
    _POWERPAINT_PIPELINE_CACHE,
    _fit_restored_to_original_size,
    _looks_like_watermark_text,
    detect_mask_with_paddleocr,
    restore_with_brushnet,
    restore_with_diffusers_inpaint,
    restore_with_powerpaint_v2_1,
)
from no_watermar.models import DetectionResult


class ProviderWorkersTests(unittest.TestCase):
    def test_fit_restored_to_original_size_crops_padding(self) -> None:
        restored = np.zeros((1600, 1072, 3), dtype=np.uint8)
        fitted = _fit_restored_to_original_size(restored, 1600, 1066)

        self.assertEqual(fitted.shape, (1600, 1066, 3))

    def test_fit_restored_to_original_size_keeps_matching_shape(self) -> None:
        restored = np.zeros((512, 512, 3), dtype=np.uint8)
        fitted = _fit_restored_to_original_size(restored, 512, 512)

        self.assertIs(fitted, restored)

    def test_looks_like_watermark_text_uses_configured_presets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = root / "no-watermar.toml"
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

            with patch.dict(os.environ, {"NO_WATERMAR_CONFIG": str(config_path)}, clear=True):
                self.assertTrue(_looks_like_watermark_text("Acme Studio", "bottom_left"))

    def test_looks_like_watermark_text_accepts_domain_like_bottom_text(self) -> None:
        self.assertTrue(_looks_like_watermark_text("BestGiriSexy.Com", "bottom_left", ()))

    def test_detect_mask_with_paddleocr_uses_enhanced_bottom_strip_when_primary_is_empty(self) -> None:
        image = np.full((120, 80, 3), 240, dtype=np.uint8)
        image[100:118, 48:79] = 220

        primary_result: list[object] = []
        enhanced_result = [
            {
                "dt_polys": [
                    np.array(
                        [
                            [150, 14],
                            [195, 14],
                            [195, 28],
                            [150, 28],
                        ]
                    )
                ],
                "rec_texts": ["BestGiriSexy.Com"],
                "rec_scores": [0.62],
            }
        ]

        with patch(
            "no_watermar.provider_workers._run_paddleocr_inference",
            side_effect=[primary_result, enhanced_result],
        ):
            payload = detect_mask_with_paddleocr(
                image,
                "portrait_regular",
                score_threshold=0.45,
                ocr_engine=object(),
            )

        self.assertGreater(int(np.count_nonzero(payload["mask"])), 0)
        self.assertEqual(payload["meta"]["detection_strategy"], "enhanced_bottom_strip_ocr")
        self.assertEqual(payload["meta"]["fallback_provider"], None)
        self.assertTrue(any(box["region"] == "bottom_right" for box in payload["boxes"]))

    def test_detect_mask_with_paddleocr_uses_rule_based_fallback_when_ocr_still_fails(self) -> None:
        image = np.full((120, 80, 3), 240, dtype=np.uint8)
        fallback_mask = np.zeros((120, 80), dtype=np.uint8)
        fallback_mask[108:120, 0:54] = 255
        detection = DetectionResult(
            category="portrait_regular",
            mask_nonzero=int(np.count_nonzero(fallback_mask)),
            confidence=0.73,
            boxes=[
                {"region": "bottom_left", "x1": 0, "y1": 114, "x2": 54, "y2": 120},
                {"region": "bottom_right", "x1": 61, "y1": 110, "x2": 80, "y2": 120},
            ],
        )

        with patch("no_watermar.provider_workers._run_paddleocr_inference", side_effect=[[], []]):
            with patch("no_watermar.provider_workers.detect_watermarks", return_value=(fallback_mask, detection)):
                payload = detect_mask_with_paddleocr(
                    image,
                    "portrait_regular",
                    score_threshold=0.45,
                    ocr_engine=object(),
                )

        self.assertEqual(payload["meta"]["detection_strategy"], "rule_based_roi_fallback")
        self.assertEqual(payload["meta"]["fallback_provider"], "rule_based_roi")
        self.assertEqual(payload["meta"]["fallback_compacted_regions"], ["bottom_left", "bottom_right"])
        self.assertEqual(payload["boxes"][0]["y1"], 115)
        self.assertEqual(payload["boxes"][0]["x2"], 47)
        self.assertEqual(payload["boxes"][1]["x1"], 49)
        self.assertEqual(payload["boxes"][1]["y1"], 113)
        self.assertLess(int(np.count_nonzero(payload["mask"])), detection.mask_nonzero)

    def test_restore_with_diffusers_inpaint_supports_single_file_mode(self) -> None:
        image = np.full((32, 24, 3), 128, dtype=np.uint8)
        mask = np.zeros((32, 24), dtype=np.uint8)
        mask[8:24, 6:18] = 255

        class FakeGenerator:
            def __init__(self, device: str) -> None:
                self.device = device
                self.seed = None

            def manual_seed(self, seed: int) -> "FakeGenerator":
                self.seed = seed
                return self

        class FakeCuda:
            @staticmethod
            def is_available() -> bool:
                return False

            @staticmethod
            def empty_cache() -> None:
                return None

        class FakePipeline:
            def __init__(self) -> None:
                self.device = None
                self.attention_slicing = False
                self.called_with = None

            def enable_attention_slicing(self) -> None:
                self.attention_slicing = True

            def to(self, device: str) -> "FakePipeline":
                self.device = device
                return self

            def __call__(self, **kwargs):
                self.called_with = kwargs
                pil_image = kwargs["image"].convert("RGB")
                return types.SimpleNamespace(images=[pil_image])

        class FakeSingleFilePipelineClass:
            last_source = None
            last_kwargs = None
            last_pipeline = None

            @classmethod
            def from_single_file(cls, source: str, **kwargs):
                cls.last_source = source
                cls.last_kwargs = kwargs
                cls.last_pipeline = FakePipeline()
                return cls.last_pipeline

        fake_torch = types.ModuleType("torch")
        fake_torch.float16 = "float16"
        fake_torch.float32 = "float32"
        fake_torch.bfloat16 = "bfloat16"
        fake_torch.cuda = FakeCuda()
        fake_torch.Generator = FakeGenerator

        fake_diffusers = types.ModuleType("diffusers")
        fake_diffusers.AutoPipelineForInpainting = object()
        fake_diffusers.StableDiffusionInpaintPipeline = FakeSingleFilePipelineClass
        fake_diffusers.StableDiffusionXLInpaintPipeline = FakeSingleFilePipelineClass
        fake_diffusers.FluxFillPipeline = object()

        with patch.dict(sys.modules, {"torch": fake_torch, "diffusers": fake_diffusers}):
            payload = restore_with_diffusers_inpaint(
                image,
                mask,
                prompt="restore texture",
                negative_prompt="blur",
                options={
                    "model_id": "https://hf-mirror.com/runwayml/stable-diffusion-inpainting/resolve/main/sd-v1-5-inpainting.ckpt",
                    "load_mode": "single_file",
                    "pipeline_class": "StableDiffusionInpaintPipeline",
                    "device": "cpu",
                    "torch_dtype": "float32",
                    "steps": 12,
                    "guidance_scale": 6.0,
                    "strength": 0.95,
                    "seed": 123,
                },
            )

        self.assertEqual(payload["meta"]["engine"], "diffusers_single_file")
        self.assertEqual(payload["meta"]["load_mode"], "single_file")
        self.assertEqual(payload["meta"]["pipeline_class"], "StableDiffusionInpaintPipeline")
        self.assertEqual(
            payload["meta"]["model_source"],
            "https://hf-mirror.com/runwayml/stable-diffusion-inpainting/resolve/main/sd-v1-5-inpainting.ckpt",
        )
        self.assertEqual(FakeSingleFilePipelineClass.last_source, payload["meta"]["model_source"])
        self.assertEqual(FakeSingleFilePipelineClass.last_kwargs["torch_dtype"], "float32")
        self.assertEqual(FakeSingleFilePipelineClass.last_pipeline.device, "cpu")
        self.assertEqual(FakeSingleFilePipelineClass.last_pipeline.called_with["prompt"], "restore texture")
        self.assertEqual(FakeSingleFilePipelineClass.last_pipeline.called_with["negative_prompt"], "blur")
        self.assertIsInstance(FakeSingleFilePipelineClass.last_pipeline.called_with["image"], Image.Image)
        self.assertIsInstance(FakeSingleFilePipelineClass.last_pipeline.called_with["mask_image"], Image.Image)

    def test_restore_with_diffusers_inpaint_supports_flux_fill_with_layerwise_casting(self) -> None:
        image = np.full((48, 40, 3), 128, dtype=np.uint8)
        mask = np.zeros((48, 40), dtype=np.uint8)
        mask[8:24, 6:18] = 255

        class FakeGenerator:
            def __init__(self, device: str) -> None:
                self.device = device
                self.seed = None

            def manual_seed(self, seed: int) -> "FakeGenerator":
                self.seed = seed
                return self

        class FakeCuda:
            @staticmethod
            def is_available() -> bool:
                return False

            @staticmethod
            def empty_cache() -> None:
                return None

        class FakeTransformer:
            def __init__(self) -> None:
                self.layerwise_kwargs = None

            def enable_layerwise_casting(self, **kwargs) -> None:
                self.layerwise_kwargs = kwargs

        class FakeFluxPipeline:
            def __init__(self) -> None:
                self.device = None
                self.attention_slicing = False
                self.called_with = None
                self.transformer = FakeTransformer()

            def enable_attention_slicing(self) -> None:
                self.attention_slicing = True

            def to(self, device: str) -> "FakeFluxPipeline":
                self.device = device
                return self

            def __call__(self, **kwargs):
                self.called_with = kwargs
                pil_image = kwargs["image"].convert("RGB")
                return types.SimpleNamespace(images=[pil_image])

        class FakeFluxFillPipelineClass:
            last_source = None
            last_kwargs = None
            last_pipeline = None

            @classmethod
            def from_pretrained(cls, source: str, **kwargs):
                cls.last_source = source
                cls.last_kwargs = kwargs
                cls.last_pipeline = FakeFluxPipeline()
                return cls.last_pipeline

        fake_torch = types.ModuleType("torch")
        fake_torch.float16 = "float16"
        fake_torch.float32 = "float32"
        fake_torch.bfloat16 = "bfloat16"
        fake_torch.float8_e4m3fn = "float8_e4m3fn"
        fake_torch.float8_e5m2 = "float8_e5m2"
        fake_torch.cuda = FakeCuda()
        fake_torch.Generator = FakeGenerator

        fake_diffusers = types.ModuleType("diffusers")
        fake_diffusers.AutoPipelineForInpainting = object()
        fake_diffusers.StableDiffusionInpaintPipeline = object()
        fake_diffusers.StableDiffusionXLInpaintPipeline = object()
        fake_diffusers.FluxFillPipeline = FakeFluxFillPipelineClass

        with patch.dict(sys.modules, {"torch": fake_torch, "diffusers": fake_diffusers}):
            payload = restore_with_diffusers_inpaint(
                image,
                mask,
                prompt="restore texture",
                negative_prompt="blur",
                options={
                    "model_id": "black-forest-labs/FLUX.1-Fill-dev",
                    "pipeline_class": "FluxFillPipeline",
                    "device": "cpu",
                    "torch_dtype": "bfloat16",
                    "steps": 16,
                    "guidance_scale": 24.0,
                    "max_sequence_length": 256,
                    "enable_layerwise_casting": True,
                    "layerwise_storage_dtype": "float8_e4m3fn",
                    "layerwise_compute_dtype": "bfloat16",
                    "seed": 123,
                },
            )

        self.assertEqual(payload["meta"]["engine"], "diffusers_autopipeline")
        self.assertEqual(payload["meta"]["pipeline_class"], "FluxFillPipeline")
        self.assertEqual(payload["meta"]["torch_dtype"], "bfloat16")
        self.assertEqual(payload["meta"]["layerwise_casting"]["storage_dtype"], "float8_e4m3fn")
        self.assertEqual(payload["meta"]["layerwise_casting"]["compute_dtype"], "bfloat16")
        self.assertEqual(FakeFluxFillPipelineClass.last_source, "black-forest-labs/FLUX.1-Fill-dev")
        self.assertEqual(FakeFluxFillPipelineClass.last_kwargs["torch_dtype"], "bfloat16")
        self.assertEqual(FakeFluxFillPipelineClass.last_pipeline.transformer.layerwise_kwargs["storage_dtype"], "float8_e4m3fn")
        self.assertEqual(FakeFluxFillPipelineClass.last_pipeline.transformer.layerwise_kwargs["compute_dtype"], "bfloat16")
        self.assertEqual(FakeFluxFillPipelineClass.last_pipeline.called_with["height"], 48)
        self.assertEqual(FakeFluxFillPipelineClass.last_pipeline.called_with["width"], 40)
        self.assertEqual(FakeFluxFillPipelineClass.last_pipeline.called_with["max_sequence_length"], 256)
        self.assertNotIn("strength", FakeFluxFillPipelineClass.last_pipeline.called_with)

    def test_restore_with_diffusers_inpaint_supports_flux_fill_single_file_mode(self) -> None:
        image = np.full((48, 40, 3), 128, dtype=np.uint8)
        mask = np.zeros((48, 40), dtype=np.uint8)
        mask[8:24, 6:18] = 255

        class FakeGenerator:
            def __init__(self, device: str) -> None:
                self.device = device
                self.seed = None

            def manual_seed(self, seed: int) -> "FakeGenerator":
                self.seed = seed
                return self

        class FakeCuda:
            @staticmethod
            def is_available() -> bool:
                return False

            @staticmethod
            def empty_cache() -> None:
                return None

        class FakeFluxPipeline:
            def __init__(self) -> None:
                self.device = None
                self.called_with = None

            def to(self, device: str) -> "FakeFluxPipeline":
                self.device = device
                return self

            def __call__(self, **kwargs):
                self.called_with = kwargs
                pil_image = kwargs["image"].convert("RGB")
                return types.SimpleNamespace(images=[pil_image])

        class FakeFluxFillPipelineClass:
            last_source = None
            last_kwargs = None
            last_pipeline = None

            @classmethod
            def from_single_file(cls, source: str, **kwargs):
                cls.last_source = source
                cls.last_kwargs = kwargs
                cls.last_pipeline = FakeFluxPipeline()
                return cls.last_pipeline

        fake_torch = types.ModuleType("torch")
        fake_torch.float16 = "float16"
        fake_torch.float32 = "float32"
        fake_torch.bfloat16 = "bfloat16"
        fake_torch.cuda = FakeCuda()
        fake_torch.Generator = FakeGenerator

        fake_diffusers = types.ModuleType("diffusers")
        fake_diffusers.AutoPipelineForInpainting = object()
        fake_diffusers.StableDiffusionInpaintPipeline = object()
        fake_diffusers.StableDiffusionXLInpaintPipeline = object()
        fake_diffusers.FluxFillPipeline = FakeFluxFillPipelineClass

        with patch.dict(sys.modules, {"torch": fake_torch, "diffusers": fake_diffusers}):
            payload = restore_with_diffusers_inpaint(
                image,
                mask,
                prompt="restore texture",
                negative_prompt="blur",
                options={
                    "model_id": "camenduru/FLUX.1-Fill-dev-ungated",
                    "load_mode": "single_file",
                    "single_file_url": "https://hf-mirror.com/camenduru/FLUX.1-Fill-dev-ungated/resolve/main/flux1-fill-dev.safetensors",
                    "pipeline_class": "FluxFillPipeline",
                    "config": "camenduru/FLUX.1-Fill-dev-ungated",
                    "device": "cpu",
                    "torch_dtype": "bfloat16",
                    "steps": 12,
                    "guidance_scale": 24.0,
                    "max_sequence_length": 256,
                    "seed": 123,
                },
            )

        self.assertEqual(payload["meta"]["engine"], "diffusers_single_file")
        self.assertEqual(payload["meta"]["pipeline_class"], "FluxFillPipeline")
        self.assertEqual(
            payload["meta"]["model_source"],
            "https://hf-mirror.com/camenduru/FLUX.1-Fill-dev-ungated/resolve/main/flux1-fill-dev.safetensors",
        )
        self.assertEqual(FakeFluxFillPipelineClass.last_source, payload["meta"]["model_source"])
        self.assertEqual(FakeFluxFillPipelineClass.last_kwargs["config"], "camenduru/FLUX.1-Fill-dev-ungated")
        self.assertEqual(FakeFluxFillPipelineClass.last_kwargs["torch_dtype"], "bfloat16")
        self.assertEqual(FakeFluxFillPipelineClass.last_pipeline.device, "cpu")
        self.assertEqual(FakeFluxFillPipelineClass.last_pipeline.called_with["height"], 48)
        self.assertEqual(FakeFluxFillPipelineClass.last_pipeline.called_with["width"], 40)
        self.assertEqual(FakeFluxFillPipelineClass.last_pipeline.called_with["max_sequence_length"], 256)
        self.assertNotIn("strength", FakeFluxFillPipelineClass.last_pipeline.called_with)

    def test_restore_with_brushnet_supports_local_checkpoint_contract(self) -> None:
        image = np.full((32, 24, 3), 128, dtype=np.uint8)
        mask = np.zeros((32, 24), dtype=np.uint8)
        mask[8:24, 6:18] = 255

        class FakeGenerator:
            def __init__(self, device: str) -> None:
                self.device = device
                self.seed = None

            def manual_seed(self, seed: int) -> "FakeGenerator":
                self.seed = seed
                return self

        class FakeCuda:
            @staticmethod
            def is_available() -> bool:
                return False

            @staticmethod
            def empty_cache() -> None:
                return None

        class FakeScheduler:
            @classmethod
            def from_config(cls, config):
                return {"copied_from": config}

        class FakePipeline:
            def __init__(self) -> None:
                self.device = None
                self.attention_slicing = False
                self.called_with = None
                self.scheduler = types.SimpleNamespace(config={"name": "orig"})

            def enable_attention_slicing(self) -> None:
                self.attention_slicing = True

            def enable_model_cpu_offload(self) -> None:
                return None

            def to(self, device: str) -> "FakePipeline":
                self.device = device
                return self

            def __call__(self, **kwargs):
                self.called_with = kwargs
                return types.SimpleNamespace(images=[kwargs["image"].convert("RGB")])

        class FakeBrushNetPipelineType:
            last_source = None
            last_kwargs = None
            last_pipeline = None

            @classmethod
            def from_pretrained(cls, source: str, **kwargs):
                cls.last_source = source
                cls.last_kwargs = kwargs
                cls.last_pipeline = FakePipeline()
                return cls.last_pipeline

        class FakeBrushNetType:
            last_source = None
            last_kwargs = None

            @classmethod
            def from_pretrained(cls, source: str, **kwargs):
                cls.last_source = source
                cls.last_kwargs = kwargs
                return types.SimpleNamespace(source=source, kwargs=kwargs)

        fake_torch = types.ModuleType("torch")
        fake_torch.float16 = "float16"
        fake_torch.float32 = "float32"
        fake_torch.bfloat16 = "bfloat16"
        fake_torch.cuda = FakeCuda()
        fake_torch.Generator = FakeGenerator

        fake_diffusers = types.ModuleType("diffusers")
        fake_diffusers.UniPCMultistepScheduler = FakeScheduler
        fake_diffusers.BrushNetModel = FakeBrushNetType
        fake_diffusers.StableDiffusionBrushNetPipeline = FakeBrushNetPipelineType

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_root = Path(tmp_dir)
            (model_root / "segmentation_mask_brushnet_ckpt").mkdir(parents=True, exist_ok=True)
            (model_root / "realisticVisionV60B1_v51VAE").mkdir(parents=True, exist_ok=True)
            _BRUSHNET_PIPELINE_CACHE.clear()

            with patch.dict(os.environ, {}, clear=True):
                with patch.dict(sys.modules, {"torch": fake_torch, "diffusers": fake_diffusers}):
                    payload = restore_with_brushnet(
                        image,
                        mask,
                        prompt="remove watermark",
                        negative_prompt="text, logo",
                        options={
                            "brushnet_model_path": str(model_root / "segmentation_mask_brushnet_ckpt"),
                            "device": "cpu",
                            "torch_dtype": "float32",
                            "steps": 28,
                            "guidance_scale": 8.0,
                            "local_files_only": True,
                            "seed": 123,
                        },
                    )

        self.assertEqual(payload["meta"]["engine"], "brushnet")
        self.assertEqual(payload["meta"]["session_mode"], "inprocess")
        self.assertEqual(payload["meta"]["torch_dtype"], "float32")
        self.assertEqual(
            payload["meta"]["brushnet_model_path"],
            str((model_root / "segmentation_mask_brushnet_ckpt").resolve()),
        )
        self.assertEqual(
            payload["meta"]["base_model_path"],
            str((model_root / "realisticVisionV60B1_v51VAE").resolve()),
        )
        self.assertTrue(payload["meta"]["composite_back"])
        self.assertEqual(
            FakeBrushNetType.last_source,
            str((model_root / "segmentation_mask_brushnet_ckpt").resolve()),
        )
        self.assertEqual(
            FakeBrushNetPipelineType.last_source,
            str((model_root / "realisticVisionV60B1_v51VAE").resolve()),
        )
        self.assertEqual(FakeBrushNetPipelineType.last_pipeline.device, "cpu")
        self.assertEqual(FakeBrushNetPipelineType.last_pipeline.called_with["prompt"], "remove watermark")
        self.assertEqual(FakeBrushNetPipelineType.last_pipeline.called_with["negative_prompt"], "text, logo")
        self.assertEqual(FakeBrushNetPipelineType.last_pipeline.called_with["guidance_scale"], 8.0)
        self.assertEqual(FakeBrushNetPipelineType.last_pipeline.called_with["num_inference_steps"], 28)
        self.assertEqual(FakeBrushNetPipelineType.last_pipeline.called_with["brushnet_conditioning_scale"], 1.0)

    def test_restore_with_powerpaint_v21_supports_checkpoint_dir_contract(self) -> None:
        image = np.full((32, 24, 3), 128, dtype=np.uint8)
        mask = np.zeros((32, 24), dtype=np.uint8)
        mask[8:24, 6:18] = 255

        class FakeGenerator:
            def __init__(self, device: str) -> None:
                self.device = device
                self.seed = None

            def manual_seed(self, seed: int) -> "FakeGenerator":
                self.seed = seed
                return self

        class FakeCuda:
            @staticmethod
            def is_available() -> bool:
                return False

            @staticmethod
            def empty_cache() -> None:
                return None

        class FakeScheduler:
            @classmethod
            def from_config(cls, config):
                return {"copied_from": config}

        class FakePipeline:
            def __init__(self) -> None:
                self.device = None
                self.attention_slicing = False
                self.called_with = None
                self.scheduler = types.SimpleNamespace(config={"name": "orig"})
                self.text_encoder_brushnet = types.SimpleNamespace(load_state_dict=lambda state, strict=False: None)
                self.brushnet = object()
                self.unet = None
                self.tokenizer = None

            def enable_attention_slicing(self) -> None:
                self.attention_slicing = True

            def enable_model_cpu_offload(self) -> None:
                return None

            def to(self, device: str) -> "FakePipeline":
                self.device = device
                return self

            def __call__(self, **kwargs):
                self.called_with = kwargs
                return types.SimpleNamespace(images=[kwargs["image"].convert("RGB")])

        class FakePipelineType:
            last_source = None
            last_kwargs = None
            last_pipeline = None

            @classmethod
            def from_pretrained(cls, source: str, **kwargs):
                cls.last_source = source
                cls.last_kwargs = kwargs
                cls.last_pipeline = FakePipeline()
                return cls.last_pipeline

        class FakeUNetType:
            calls = []

            @classmethod
            def from_pretrained(cls, source: str, **kwargs):
                cls.calls.append((source, kwargs))
                return types.SimpleNamespace(source=source, kwargs=kwargs)

        class FakeBrushNetType:
            last_unet = None

            @classmethod
            def from_unet(cls, unet):
                cls.last_unet = unet
                return types.SimpleNamespace(unet=unet)

        class FakeTextEncoderType:
            calls = []

            @classmethod
            def from_pretrained(cls, source: str, **kwargs):
                cls.calls.append((source, kwargs))
                return types.SimpleNamespace(source=source, kwargs=kwargs, load_state_dict=lambda state, strict=False: None)

        class FakeTokenizerWrapper:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

        fake_add_tokens_calls = []

        def fake_add_tokens(**kwargs):
            fake_add_tokens_calls.append(kwargs)

        fake_torch = types.ModuleType("torch")
        fake_torch.float16 = "float16"
        fake_torch.float32 = "float32"
        fake_torch.bfloat16 = "bfloat16"
        fake_torch.cuda = FakeCuda()
        fake_torch.Generator = FakeGenerator
        fake_torch.load = lambda path, map_location=None: {"path": path, "map_location": map_location}

        fake_diffusers = types.ModuleType("diffusers")
        fake_diffusers.UniPCMultistepScheduler = FakeScheduler

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.CLIPTextModel = FakeTextEncoderType

        fake_safetensors = types.ModuleType("safetensors")
        fake_safetensors_torch = types.ModuleType("safetensors.torch")
        fake_safetensors_torch.load_model = lambda model, path: None

        fake_powerpaint = types.ModuleType("powerpaint")
        fake_powerpaint_models = types.ModuleType("powerpaint.models")
        fake_powerpaint_brushnet = types.ModuleType("powerpaint.models.BrushNet_CA")
        fake_powerpaint_brushnet.BrushNetModel = FakeBrushNetType
        fake_powerpaint_unet = types.ModuleType("powerpaint.models.unet_2d_condition")
        fake_powerpaint_unet.UNet2DConditionModel = FakeUNetType
        fake_powerpaint_pipelines = types.ModuleType("powerpaint.pipelines")
        fake_powerpaint_pipeline_brushnet = types.ModuleType("powerpaint.pipelines.pipeline_PowerPaint_Brushnet_CA")
        fake_powerpaint_pipeline_brushnet.StableDiffusionPowerPaintBrushNetPipeline = FakePipelineType
        fake_powerpaint_utils = types.ModuleType("powerpaint.utils")
        fake_powerpaint_utils_utils = types.ModuleType("powerpaint.utils.utils")
        fake_powerpaint_utils_utils.TokenizerWrapper = FakeTokenizerWrapper
        fake_powerpaint_utils_utils.add_tokens = fake_add_tokens

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = Path(tmp_dir) / "ppt-v2-1"
            (checkpoint_dir / "PowerPaint_Brushnet").mkdir(parents=True, exist_ok=True)
            (checkpoint_dir / "realisticVisionV60B1_v51VAE" / "unet").mkdir(parents=True, exist_ok=True)
            (checkpoint_dir / "realisticVisionV60B1_v51VAE" / "tokenizer").mkdir(parents=True, exist_ok=True)
            (checkpoint_dir / "PowerPaint_Brushnet" / "diffusion_pytorch_model.safetensors").write_text("", encoding="utf-8")
            (checkpoint_dir / "PowerPaint_Brushnet" / "pytorch_model.bin").write_text("", encoding="utf-8")
            _POWERPAINT_PIPELINE_CACHE.clear()

            with patch.dict(
                sys.modules,
                {
                    "torch": fake_torch,
                    "diffusers": fake_diffusers,
                    "transformers": fake_transformers,
                    "safetensors": fake_safetensors,
                    "safetensors.torch": fake_safetensors_torch,
                    "powerpaint": fake_powerpaint,
                    "powerpaint.models": fake_powerpaint_models,
                    "powerpaint.models.BrushNet_CA": fake_powerpaint_brushnet,
                    "powerpaint.models.unet_2d_condition": fake_powerpaint_unet,
                    "powerpaint.pipelines": fake_powerpaint_pipelines,
                    "powerpaint.pipelines.pipeline_PowerPaint_Brushnet_CA": fake_powerpaint_pipeline_brushnet,
                    "powerpaint.utils": fake_powerpaint_utils,
                    "powerpaint.utils.utils": fake_powerpaint_utils_utils,
                },
            ):
                payload = restore_with_powerpaint_v2_1(
                    image,
                    mask,
                    prompt="remove watermark",
                    negative_prompt="text, logo",
                    options={
                        "checkpoint_dir": str(checkpoint_dir),
                        "device": "cpu",
                        "torch_dtype": "float32",
                        "steps": 18,
                        "guidance_scale": 9.5,
                        "local_files_only": True,
                        "seed": 123,
                    },
                )

        self.assertEqual(payload["meta"]["engine"], "powerpaint_v2_1")
        self.assertEqual(payload["meta"]["task"], "object-removal")
        self.assertEqual(payload["meta"]["session_mode"], "inprocess")
        self.assertEqual(payload["meta"]["torch_dtype"], "float32")
        self.assertEqual(payload["meta"]["checkpoint_dir"], str(checkpoint_dir.resolve()))
        self.assertEqual(payload["meta"]["base_model_path"], str((checkpoint_dir / "realisticVisionV60B1_v51VAE").resolve()))
        self.assertEqual(payload["meta"]["backbone_source"], str((checkpoint_dir / "realisticVisionV60B1_v51VAE").resolve()))
        self.assertTrue(payload["meta"]["composite_back"])
        self.assertEqual(FakePipelineType.last_source, str((checkpoint_dir / "realisticVisionV60B1_v51VAE").resolve()))
        self.assertEqual(FakePipelineType.last_pipeline.device, "cpu")
        self.assertEqual(FakePipelineType.last_pipeline.called_with["promptU"], "remove watermark")
        self.assertEqual(FakePipelineType.last_pipeline.called_with["negative_promptU"], "text, logo")
        self.assertEqual(FakePipelineType.last_pipeline.called_with["guidance_scale"], 9.5)
        self.assertEqual(FakePipelineType.last_pipeline.called_with["num_inference_steps"], 18)
        self.assertTrue(fake_add_tokens_calls)


if __name__ == "__main__":
    unittest.main()
