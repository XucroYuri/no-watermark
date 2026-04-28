from __future__ import annotations

import contextlib
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageFilter

from .config import get_watermark_keyword_settings
from .detector import detect_watermarks
from .model_store import (
    download_hf_repo_snapshot,
    resolve_diffusers_model_store_dir,
    using_hf_project_store,
)
from .models import ScanItem

_BRUSHNET_PIPELINE_CACHE: dict[tuple[str, ...], dict[str, Any]] = {}
_POWERPAINT_PIPELINE_CACHE: dict[tuple[str, ...], dict[str, Any]] = {}


def detect_mask_with_paddleocr(
    image: np.ndarray,
    source_category: str,
    *,
    score_threshold: float = 0.45,
    ocr_engine: Any | None = None,
    session_mode: str | None = None,
) -> dict[str, Any]:
    start = perf_counter()
    rgb_image = image[:, :, ::-1]
    if ocr_engine is None:
        ocr_engine = create_paddleocr_engine()
        if session_mode is None:
            session_mode = "inprocess"
    elif session_mode is None:
        session_mode = "persistent-sidecar"

    raw_result = _run_paddleocr_inference(ocr_engine, rgb_image)
    lines = _flatten_paddleocr_result(raw_result)
    keyword_settings = get_watermark_keyword_settings()

    height, width = image.shape[:2]
    primary_detection = _build_paddleocr_mask_from_lines(
        lines,
        width=width,
        height=height,
        source_category=source_category,
        score_threshold=score_threshold,
        watermark_tokens=keyword_settings.tokens,
    )
    detection_strategy = "primary_ocr"
    detection_attempts = ["primary_ocr"]
    enhanced_score_threshold = max(0.25, min(score_threshold, 0.35))
    resolved_detection = primary_detection

    if resolved_detection["mask_nonzero"] == 0:
        enhanced_detection = _detect_mask_with_enhanced_bottom_strip_ocr(
            image,
            source_category=source_category,
            score_threshold=enhanced_score_threshold,
            ocr_engine=ocr_engine,
            watermark_tokens=keyword_settings.tokens,
        )
        detection_attempts.append("enhanced_bottom_strip_ocr")
        if enhanced_detection["mask_nonzero"] > 0:
            resolved_detection = enhanced_detection
            detection_strategy = "enhanced_bottom_strip_ocr"
        else:
            detection_attempts.append("rule_based_roi_fallback")
            fallback_detection = _detect_mask_with_rule_based_fallback(
                image,
                source_category=source_category,
            )
            if fallback_detection["mask_nonzero"] > 0:
                resolved_detection = fallback_detection
                detection_strategy = "rule_based_roi_fallback"

    latency_ms = (perf_counter() - start) * 1000.0
    return {
        "mask": resolved_detection["mask"],
        "boxes": resolved_detection["boxes"],
        "confidence": resolved_detection["confidence"],
        "latency_ms": latency_ms,
        "meta": {
            "matched_texts": resolved_detection["matched_texts"],
            "score_threshold": score_threshold,
            "enhanced_score_threshold": enhanced_score_threshold if len(detection_attempts) > 1 else None,
            "source_category": source_category,
            "session_mode": session_mode,
            "watermark_keyword_presets": list(keyword_settings.active_presets),
            "watermark_keyword_config": str(keyword_settings.config_path) if keyword_settings.config_path else None,
            "detection_strategy": detection_strategy,
            "detection_attempts": detection_attempts,
            "fallback_provider": resolved_detection.get("fallback_provider"),
            "fallback_compacted_regions": list(resolved_detection.get("fallback_compacted_regions") or []),
        },
    }


def create_paddleocr_engine() -> Any:
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    from paddleocr import PaddleOCR

    try:
        return PaddleOCR(
            lang="en",
            device="cpu",
            enable_hpi=False,
            enable_mkldnn=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    except (TypeError, ValueError):
        # Fall back to the older constructor style used by legacy PaddleOCR releases.
        try:
            return PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
        except (TypeError, ValueError) as exc:
            if "show_log" not in str(exc):
                raise
            return PaddleOCR(use_angle_cls=False, lang="en")


def _run_paddleocr_inference(ocr_engine: Any, rgb_image: np.ndarray) -> Any:
    try:
        return ocr_engine.predict(rgb_image)
    except TypeError as exc:
        if "unexpected keyword" not in str(exc) and "predict" not in str(exc):
            raise
        return ocr_engine.ocr(rgb_image, cls=False)


def _build_paddleocr_mask_from_lines(
    lines: list[dict[str, Any]],
    *,
    width: int,
    height: int,
    source_category: str,
    score_threshold: float,
    watermark_tokens: tuple[str, ...],
    offset_x: int = 0,
    offset_y: int = 0,
) -> dict[str, Any]:
    mask = np.zeros((height, width), dtype=np.uint8)
    boxes: list[dict[str, Any]] = []
    matched_texts: list[dict[str, Any]] = []
    accepted_scores: list[float] = []

    for line in lines:
        polygon = [
            (int(point[0]) + offset_x, int(point[1]) + offset_y)
            for point in list(line.get("polygon") or [])
        ]
        text = str(line.get("text") or "")
        score = float(line.get("score") or 0.0)
        if score < score_threshold:
            continue
        if not polygon:
            continue

        region = _classify_text_region(polygon, width, height, source_category)
        if region is None:
            continue

        if source_category != "cover_heavy" and not _looks_like_watermark_text(text, region, watermark_tokens):
            continue

        x1 = max(0, min(point[0] for point in polygon))
        y1 = max(0, min(point[1] for point in polygon))
        x2 = min(width, max(point[0] for point in polygon))
        y2 = min(height, max(point[1] for point in polygon))
        x1, y1, x2, y2 = _expand_box(x1, y1, x2, y2, width, height, region)
        if x2 <= x1 or y2 <= y1:
            continue
        mask[y1:y2, x1:x2] = 255
        accepted_scores.append(score)
        matched_texts.append({"text": text, "score": round(score, 4), "region": region})
        boxes.append({"region": region, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "text": text, "score": round(score, 4)})

    confidence = float(sum(accepted_scores) / len(accepted_scores)) if accepted_scores else 0.0
    return {
        "mask": mask,
        "mask_nonzero": int(np.count_nonzero(mask)),
        "boxes": boxes,
        "confidence": confidence,
        "matched_texts": matched_texts,
        "fallback_provider": None,
    }


def _detect_mask_with_enhanced_bottom_strip_ocr(
    image: np.ndarray,
    *,
    source_category: str,
    score_threshold: float,
    ocr_engine: Any,
    watermark_tokens: tuple[str, ...],
) -> dict[str, Any]:
    height, width = image.shape[:2]
    if source_category not in {"portrait_regular", "landscape_regular"}:
        return _build_paddleocr_mask_from_lines(
            [],
            width=width,
            height=height,
            source_category=source_category,
            score_threshold=score_threshold,
            watermark_tokens=watermark_tokens,
        )

    strip_top_ratio = 0.82 if source_category == "portrait_regular" else 0.78
    strip_y1 = max(0, min(height - 1, int(height * strip_top_ratio)))
    strip = image[strip_y1:, :, :]
    if strip.size == 0:
        return _build_paddleocr_mask_from_lines(
            [],
            width=width,
            height=height,
            source_category=source_category,
            score_threshold=score_threshold,
            watermark_tokens=watermark_tokens,
        )

    upscaled = cv2.resize(strip, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    raw_result = _run_paddleocr_inference(ocr_engine, np.ascontiguousarray(enhanced_rgb))
    lines = _flatten_paddleocr_result(raw_result)
    scale_x = strip.shape[1] / float(max(1, upscaled.shape[1]))
    scale_y = strip.shape[0] / float(max(1, upscaled.shape[0]))
    normalized_lines = []
    for line in lines:
        scaled_polygon = [
            (
                int(round(point[0] * scale_x)),
                int(round(point[1] * scale_y)),
            )
            for point in list(line.get("polygon") or [])
        ]
        normalized_lines.append(
            {
                "polygon": scaled_polygon,
                "text": line.get("text"),
                "score": line.get("score"),
            }
        )

    return _build_paddleocr_mask_from_lines(
        normalized_lines,
        width=width,
        height=height,
        source_category=source_category,
        score_threshold=score_threshold,
        watermark_tokens=watermark_tokens,
        offset_y=strip_y1,
    )


def _detect_mask_with_rule_based_fallback(
    image: np.ndarray,
    *,
    source_category: str,
) -> dict[str, Any]:
    height, width = image.shape[:2]
    if source_category not in {"portrait_regular", "landscape_regular"}:
        return {
            "mask": np.zeros((height, width), dtype=np.uint8),
            "mask_nonzero": 0,
            "boxes": [],
            "confidence": 0.0,
            "matched_texts": [],
            "fallback_provider": None,
        }

    item = ScanItem(
        source_path=Path("<paddleocr-fallback>"),
        relative_path=Path("<paddleocr-fallback>"),
        width=width,
        height=height,
        category=source_category,
    )
    mask, detection = detect_watermarks(item, image)
    compacted_regions: list[str] = []
    normalized_boxes = [
        _normalize_rule_based_fallback_box(
            {
                "region": str(box.get("region") or ""),
                "x1": int(box.get("x1", 0)),
                "y1": int(box.get("y1", 0)),
                "x2": int(box.get("x2", 0)),
                "y2": int(box.get("y2", 0)),
            },
            width=width,
            height=height,
            source_category=source_category,
            compacted_regions=compacted_regions,
        )
        for box in list(detection.boxes)
    ]
    compact_mask = np.zeros((height, width), dtype=np.uint8)
    for box in normalized_boxes:
        compact_mask[box["y1"] : box["y2"], box["x1"] : box["x2"]] = 255
    mask_nonzero = int(np.count_nonzero(compact_mask))
    confidence = min(1.0, mask_nonzero / float(max(1, width * height * 0.08)))
    return {
        "mask": compact_mask,
        "mask_nonzero": mask_nonzero,
        "boxes": normalized_boxes,
        "confidence": float(confidence),
        "matched_texts": [],
        "fallback_provider": "rule_based_roi",
        "fallback_compacted_regions": compacted_regions,
    }


def _normalize_rule_based_fallback_box(
    box: dict[str, Any],
    *,
    width: int,
    height: int,
    source_category: str,
    compacted_regions: list[str],
) -> dict[str, Any]:
    region = str(box.get("region") or "")
    legacy_box = _legacy_rule_based_fallback_box(width, height, source_category, region)
    compact_box = _compact_rule_based_fallback_box(width, height, source_category, region)
    normalized = {
        "region": region,
        "x1": int(box.get("x1", 0)),
        "y1": int(box.get("y1", 0)),
        "x2": int(box.get("x2", 0)),
        "y2": int(box.get("y2", 0)),
    }
    if legacy_box is not None and compact_box is not None:
        current = (normalized["x1"], normalized["y1"], normalized["x2"], normalized["y2"])
        if current == legacy_box:
            normalized["x1"], normalized["y1"], normalized["x2"], normalized["y2"] = compact_box
            if region and region not in compacted_regions:
                compacted_regions.append(region)
    return normalized


def _legacy_rule_based_fallback_box(
    width: int,
    height: int,
    source_category: str,
    region: str,
) -> tuple[int, int, int, int] | None:
    fallback_ratios: dict[str, tuple[float, float, float, float]]
    if source_category == "portrait_regular":
        fallback_ratios = {
            "bottom_left": (0.00, 0.952, 0.68, 1.00),
            "bottom_right": (0.77, 0.92, 1.00, 1.00),
        }
    elif source_category == "landscape_regular":
        fallback_ratios = {
            "bottom_left": (0.00, 0.915, 0.52, 1.00),
            "bottom_right": (0.76, 0.89, 1.00, 1.00),
        }
    else:
        return None
    ratios = fallback_ratios.get(region)
    if ratios is None:
        return None
    return (
        int(width * ratios[0]),
        int(height * ratios[1]),
        int(width * ratios[2]),
        int(height * ratios[3]),
    )


def _compact_rule_based_fallback_box(
    width: int,
    height: int,
    source_category: str,
    region: str,
) -> tuple[int, int, int, int] | None:
    compact_ratios: dict[str, tuple[float, float, float, float]]
    if source_category == "portrait_regular":
        compact_ratios = {
            "bottom_left": (0.00, 0.965, 0.595, 0.999),
            "bottom_right": (0.617, 0.944, 0.991, 0.99),
        }
    elif source_category == "landscape_regular":
        compact_ratios = {
            "bottom_left": (0.00, 0.932, 0.56, 0.995),
            "bottom_right": (0.617, 0.883, 0.991, 0.973),
        }
    else:
        return None
    ratios = compact_ratios.get(region)
    if ratios is None:
        return None
    return (
        int(width * ratios[0]),
        int(height * ratios[1]),
        int(width * ratios[2]),
        int(height * ratios[3]),
    )


def restore_with_simple_lama(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    prompt: str | None = None,
    negative_prompt: str | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        from simple_lama_inpainting import SimpleLama
        engine_name = "simple_lama_inpainting"
    except ImportError:
        from simple_lama import SimpleLama
        engine_name = "simple_lama"

    start = perf_counter()
    lama = SimpleLama()
    pil_image = Image.fromarray(image[:, :, ::-1])
    pil_mask = Image.fromarray(mask.astype(np.uint8)).convert("L")
    result = lama(pil_image, pil_mask)
    restored = np.array(result)[:, :, ::-1]
    original_height, original_width = image.shape[:2]
    padded_height, padded_width = restored.shape[:2]
    restored = _fit_restored_to_original_size(restored, original_height, original_width)
    latency_ms = (perf_counter() - start) * 1000.0
    return {
        "restored": restored,
        "latency_ms": latency_ms,
        "meta": {
            "engine": engine_name,
            "original_shape": [int(original_height), int(original_width)],
            "output_shape": [int(padded_height), int(padded_width)],
            "cropped_to_original": bool(padded_height != original_height or padded_width != original_width),
            "prompt_used": bool(prompt),
            "negative_prompt_used": bool(negative_prompt),
            "restore_options": dict(options or {}),
        },
    }


def restore_with_diffusers_inpaint(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    prompt: str | None = None,
    negative_prompt: str | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import torch

    resolved_options = dict(options or {})
    model_id = str(resolved_options.get("model_id") or os.getenv("NO_WATERMAR_DIFFUSERS_MODEL") or "").strip()
    if not model_id:
        raise ValueError(
            "Diffusers inpaint requires restore_options.model_id or the NO_WATERMAR_DIFFUSERS_MODEL environment variable."
        )

    start = perf_counter()
    device = _resolve_diffusers_device(torch, resolved_options)
    torch_dtype, torch_dtype_name = _resolve_diffusers_dtype(torch, device, resolved_options)
    pipeline, engine_name, load_mode, pipeline_class_name, model_source = _load_diffusers_inpaint_pipeline(
        model_id=model_id,
        resolved_options=resolved_options,
        torch_dtype=torch_dtype,
    )
    layerwise_casting_meta = _maybe_enable_diffusers_layerwise_casting(
        pipeline,
        torch_module=torch,
        options=resolved_options,
    )

    if bool(resolved_options.get("enable_attention_slicing", True)) and hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()

    execution_device = device
    use_cpu_offload = bool(resolved_options.get("enable_model_cpu_offload", False))
    if use_cpu_offload and device == "cuda" and hasattr(pipeline, "enable_model_cpu_offload"):
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to(device)

    prompt_text = (
        str(prompt).strip()
        if prompt and str(prompt).strip()
        else str(
            resolved_options.get("default_prompt")
            or "remove the watermark and reconstruct the covered background naturally"
        ).strip()
    )
    negative_prompt_text = (
        str(negative_prompt).strip()
        if negative_prompt and str(negative_prompt).strip()
        else _read_optional_string_option(resolved_options, "default_negative_prompt")
    )

    original_height, original_width = image.shape[:2]
    align_multiple = max(1, int(resolved_options.get("align_to_multiple_of", 8)))
    working_width, working_height = _align_diffusers_size(original_width, original_height, align_multiple)
    resized_for_model = working_width != original_width or working_height != original_height

    pil_image = Image.fromarray(image[:, :, ::-1])
    pil_mask = Image.fromarray(mask.astype(np.uint8)).convert("L")
    if resized_for_model:
        resampling = getattr(Image, "Resampling", Image)
        pil_image = pil_image.resize((working_width, working_height), resampling.LANCZOS)
        pil_mask = pil_mask.resize((working_width, working_height), resampling.NEAREST)

    mask_blur_radius = float(resolved_options.get("mask_blur_radius", 0.0))
    if mask_blur_radius > 0:
        pil_mask = pil_mask.filter(ImageFilter.GaussianBlur(radius=mask_blur_radius))

    generator = None
    seed = resolved_options.get("seed")
    if seed is not None:
        generator = torch.Generator(device=execution_device).manual_seed(int(seed))

    if pipeline_class_name == "FluxFillPipeline":
        default_steps = 50
        default_guidance_scale = 30.0
    else:
        default_steps = 30
        default_guidance_scale = 7.5

    inference_kwargs: dict[str, Any] = {
        "prompt": prompt_text,
        "image": pil_image,
        "mask_image": pil_mask,
        "num_inference_steps": int(resolved_options.get("steps", resolved_options.get("num_inference_steps", default_steps))),
        "guidance_scale": float(resolved_options.get("guidance_scale", default_guidance_scale)),
    }
    if negative_prompt_text:
        inference_kwargs["negative_prompt"] = negative_prompt_text
    if generator is not None:
        inference_kwargs["generator"] = generator
    if pipeline_class_name == "FluxFillPipeline":
        inference_kwargs["height"] = working_height
        inference_kwargs["width"] = working_width
        if "max_sequence_length" in resolved_options:
            inference_kwargs["max_sequence_length"] = int(resolved_options["max_sequence_length"])
        prompt_2 = _read_optional_string_option(resolved_options, "prompt_2")
        if prompt_2:
            inference_kwargs["prompt_2"] = prompt_2
    elif "strength" in resolved_options:
        inference_kwargs["strength"] = float(resolved_options["strength"])
    if "eta" in resolved_options:
        inference_kwargs["eta"] = float(resolved_options["eta"])

    output = pipeline(**inference_kwargs).images[0].convert("RGB")
    if resized_for_model:
        resampling = getattr(Image, "Resampling", Image)
        output = output.resize((original_width, original_height), resampling.LANCZOS)
    restored = np.array(output)[:, :, ::-1]
    latency_ms = (perf_counter() - start) * 1000.0

    if device == "cuda" and bool(resolved_options.get("clear_cuda_cache", True)) and hasattr(torch.cuda, "empty_cache"):
        torch.cuda.empty_cache()

    return {
        "restored": restored,
        "latency_ms": latency_ms,
        "meta": {
            "engine": engine_name,
            "model_id": model_id,
            "model_source": model_source,
            "requested_model_id": resolved_options.get("requested_model_id", model_id),
            "load_mode": load_mode,
            "pipeline_class": pipeline_class_name,
            "device": device,
            "torch_dtype": torch_dtype_name,
            "original_shape": [int(original_height), int(original_width)],
            "working_shape": [int(working_height), int(working_width)],
            "resized_for_model": resized_for_model,
            "mask_blur_radius": mask_blur_radius,
            "prompt_used": prompt_text,
            "negative_prompt_used": negative_prompt_text,
            "layerwise_casting": layerwise_casting_meta,
            "restore_options": dict(resolved_options),
        },
    }


def restore_with_powerpaint_v2_1(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    prompt: str | None = None,
    negative_prompt: str | None = None,
    options: dict[str, Any] | None = None,
    session_mode: str | None = None,
) -> dict[str, Any]:
    resolved_options = dict(options or {})
    source_dir = _resolve_powerpaint_source_dir(resolved_options)

    with _powerpaint_source_context(source_dir):
        import torch
        from diffusers import UniPCMultistepScheduler
        from safetensors.torch import load_model
        from transformers import CLIPTextModel

        from powerpaint.models.BrushNet_CA import BrushNetModel
        from powerpaint.models.unet_2d_condition import UNet2DConditionModel
        from powerpaint.pipelines.pipeline_PowerPaint_Brushnet_CA import StableDiffusionPowerPaintBrushNetPipeline
        from powerpaint.utils.utils import TokenizerWrapper, add_tokens

        start = perf_counter()
        device = _resolve_powerpaint_device(torch, resolved_options)
        torch_dtype, torch_dtype_name = _resolve_powerpaint_dtype(torch, device, resolved_options)
        checkpoint_dir = _resolve_powerpaint_checkpoint_dir(resolved_options)
        base_model_path = _resolve_powerpaint_base_model_path(resolved_options, checkpoint_dir)
        local_files_only = bool(resolved_options.get("local_files_only", False))
        backbone_source = _resolve_powerpaint_backbone_source(
            resolved_options,
            base_model_path=base_model_path,
            local_files_only=local_files_only,
        )

        pipeline_meta = _load_powerpaint_v21_pipeline(
            checkpoint_dir=checkpoint_dir,
            base_model_path=base_model_path,
            backbone_source=backbone_source,
            source_dir=source_dir,
            device=device,
            torch_dtype=torch_dtype,
            torch_dtype_name=torch_dtype_name,
            local_files_only=local_files_only,
            enable_model_cpu_offload=bool(resolved_options.get("enable_model_cpu_offload", device == "cuda")),
            torch_module=torch,
            scheduler_type=UniPCMultistepScheduler,
            text_encoder_type=CLIPTextModel,
            brushnet_type=BrushNetModel,
            unet_type=UNet2DConditionModel,
            pipeline_type=StableDiffusionPowerPaintBrushNetPipeline,
            tokenizer_type=TokenizerWrapper,
            add_tokens_fn=add_tokens,
            load_model_fn=load_model,
        )
        pipeline = pipeline_meta["pipeline"]

        if bool(resolved_options.get("enable_attention_slicing", True)) and hasattr(pipeline, "enable_attention_slicing"):
            pipeline.enable_attention_slicing()

        task = str(resolved_options.get("task") or "object-removal").strip().lower() or "object-removal"
        prompt_text = _resolve_powerpaint_prompt(prompt, resolved_options, task)
        negative_prompt_text = _resolve_powerpaint_negative_prompt(negative_prompt, resolved_options)
        prompt_a, prompt_b, negative_prompt_a, negative_prompt_b = _prepare_powerpaint_task_prompts(
            prompt_text=prompt_text,
            negative_prompt_text=negative_prompt_text,
            task=task,
        )

        original_height, original_width = image.shape[:2]
        pil_image = Image.fromarray(image[:, :, ::-1]).convert("RGB")
        pil_mask = Image.fromarray(mask.astype(np.uint8)).convert("L")

        target_longest_side = int(resolved_options.get("resize_longest_side", 640))
        allow_upscale = bool(resolved_options.get("allow_upscale", False))
        resized_width, resized_height = _resize_to_longest_side(
            original_width,
            original_height,
            target_longest_side,
            allow_upscale=allow_upscale,
        )
        working_width, working_height = _floor_align_size(
            resized_width,
            resized_height,
            max(1, int(resolved_options.get("align_to_multiple_of", 8))),
        )
        resized_for_model = working_width != original_width or working_height != original_height

        if resized_for_model:
            resampling = getattr(Image, "Resampling", Image)
            pil_image = pil_image.resize((working_width, working_height), resampling.LANCZOS)
            pil_mask = pil_mask.resize((working_width, working_height), resampling.NEAREST)

        input_mask_blur_radius = float(resolved_options.get("input_mask_blur_radius", 0.0))
        if input_mask_blur_radius > 0:
            pil_mask = pil_mask.filter(ImageFilter.GaussianBlur(radius=input_mask_blur_radius))

        masked_rgb = np.array(pil_image, dtype=np.float32)
        mask_array = np.array(pil_mask, dtype=np.float32) / 255.0
        masked_rgb *= 1.0 - mask_array[:, :, None]
        masked_image = Image.fromarray(masked_rgb.clip(0, 255).astype(np.uint8)).convert("RGB")
        mask_image = pil_mask.convert("RGB")

        seed = resolved_options.get("seed")
        generator = None
        if seed is not None:
            generator = torch.Generator(device).manual_seed(int(seed))

        guidance_scale = float(
            resolved_options.get("guidance_scale", 10.0 if task in {"object-removal", "image-outpainting"} else 7.5)
        )
        inference_kwargs: dict[str, Any] = {
            "promptA": prompt_a,
            "promptB": prompt_b,
            "promptU": prompt_text,
            "tradoff": float(resolved_options.get("fitting_degree", 1.0)),
            "tradoff_nag": float(resolved_options.get("fitting_degree", 1.0)),
            "image": masked_image,
            "mask": mask_image,
            "num_inference_steps": int(resolved_options.get("steps", resolved_options.get("num_inference_steps", 45))),
            "brushnet_conditioning_scale": float(resolved_options.get("brushnet_conditioning_scale", 1.0)),
            "negative_promptA": negative_prompt_a,
            "negative_promptB": negative_prompt_b,
            "negative_promptU": negative_prompt_text,
            "guidance_scale": guidance_scale,
            "width": working_width,
            "height": working_height,
        }
        if generator is not None:
            inference_kwargs["generator"] = generator

        output = pipeline(**inference_kwargs).images[0].convert("RGB")
        if resized_for_model:
            resampling = getattr(Image, "Resampling", Image)
            output = output.resize((original_width, original_height), resampling.LANCZOS)
        restored_rgb = np.array(output)

        composite_back = bool(resolved_options.get("composite_back", True))
        blend_mask_blur_radius = float(resolved_options.get("blend_mask_blur_radius", 4.0))
        if composite_back:
            original_rgb = image[:, :, ::-1]
            restored_rgb = _blend_restored_rgb_with_original(
                original_rgb,
                restored_rgb,
                mask,
                blur_radius=blend_mask_blur_radius,
            )

        if device == "cuda" and bool(resolved_options.get("clear_cuda_cache", True)) and hasattr(torch.cuda, "empty_cache"):
            torch.cuda.empty_cache()

        latency_ms = (perf_counter() - start) * 1000.0
        return {
            "restored": restored_rgb[:, :, ::-1],
            "latency_ms": latency_ms,
            "meta": {
                "engine": "powerpaint_v2_1",
                "session_mode": session_mode or "inprocess",
                "checkpoint_dir": str(checkpoint_dir),
                "base_model_path": base_model_path,
                "backbone_source": backbone_source,
                "source_dir": str(source_dir) if source_dir else None,
                "device": device,
                "torch_dtype": torch_dtype_name,
                "task": task,
                "original_shape": [int(original_height), int(original_width)],
                "working_shape": [int(working_height), int(working_width)],
                "resized_for_model": resized_for_model,
                "input_mask_blur_radius": input_mask_blur_radius,
                "blend_mask_blur_radius": blend_mask_blur_radius,
                "composite_back": composite_back,
                "prompt_used": prompt_text,
                "negative_prompt_used": negative_prompt_text,
                "promptA": prompt_a,
                "promptB": prompt_b,
                "negative_promptA": negative_prompt_a,
                "negative_promptB": negative_prompt_b,
                "restore_options": dict(resolved_options),
            },
        }


def restore_with_brushnet(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    prompt: str | None = None,
    negative_prompt: str | None = None,
    options: dict[str, Any] | None = None,
    session_mode: str | None = None,
) -> dict[str, Any]:
    resolved_options = dict(options or {})
    source_dir = _resolve_brushnet_source_dir(resolved_options)

    with _brushnet_source_context(source_dir):
        import torch
        from diffusers import BrushNetModel, StableDiffusionBrushNetPipeline, UniPCMultistepScheduler

        start = perf_counter()
        device = _resolve_brushnet_device(torch, resolved_options)
        torch_dtype, torch_dtype_name = _resolve_brushnet_dtype(torch, device, resolved_options)
        brushnet_model_path = _resolve_brushnet_model_path(resolved_options)
        local_files_only = bool(resolved_options.get("local_files_only", False))
        base_model_path = _resolve_brushnet_base_model_path(
            resolved_options,
            brushnet_model_path=brushnet_model_path,
            local_files_only=local_files_only,
        )

        pipeline_meta = _load_brushnet_pipeline(
            brushnet_model_path=brushnet_model_path,
            base_model_path=base_model_path,
            source_dir=source_dir,
            device=device,
            torch_dtype=torch_dtype,
            torch_dtype_name=torch_dtype_name,
            local_files_only=local_files_only,
            enable_model_cpu_offload=bool(resolved_options.get("enable_model_cpu_offload", device == "cuda")),
            use_safetensors=bool(resolved_options.get("use_safetensors", True)),
            brushnet_type=BrushNetModel,
            pipeline_type=StableDiffusionBrushNetPipeline,
            scheduler_type=UniPCMultistepScheduler,
        )
        pipeline = pipeline_meta["pipeline"]

        if bool(resolved_options.get("enable_attention_slicing", True)) and hasattr(pipeline, "enable_attention_slicing"):
            pipeline.enable_attention_slicing()

        prompt_text = _resolve_brushnet_prompt(prompt, resolved_options)
        negative_prompt_text = _resolve_brushnet_negative_prompt(negative_prompt, resolved_options)

        original_height, original_width = image.shape[:2]
        pil_image = Image.fromarray(image[:, :, ::-1]).convert("RGB")
        pil_mask = Image.fromarray(mask.astype(np.uint8)).convert("L")

        target_longest_side = int(resolved_options.get("resize_longest_side", 768))
        allow_upscale = bool(resolved_options.get("allow_upscale", False))
        resized_width, resized_height = _resize_to_longest_side(
            original_width,
            original_height,
            target_longest_side,
            allow_upscale=allow_upscale,
        )
        working_width, working_height = _floor_align_size(
            resized_width,
            resized_height,
            max(1, int(resolved_options.get("align_to_multiple_of", 8))),
        )
        resized_for_model = working_width != original_width or working_height != original_height

        if resized_for_model:
            resampling = getattr(Image, "Resampling", Image)
            pil_image = pil_image.resize((working_width, working_height), resampling.LANCZOS)
            pil_mask = pil_mask.resize((working_width, working_height), resampling.NEAREST)

        input_mask_blur_radius = float(resolved_options.get("input_mask_blur_radius", 0.0))
        if input_mask_blur_radius > 0:
            pil_mask = pil_mask.filter(ImageFilter.GaussianBlur(radius=input_mask_blur_radius))

        masked_rgb = np.array(pil_image, dtype=np.float32)
        mask_array = np.array(pil_mask, dtype=np.float32) / 255.0
        masked_rgb *= 1.0 - mask_array[:, :, None]
        masked_image = Image.fromarray(masked_rgb.clip(0, 255).astype(np.uint8)).convert("RGB")
        mask_image = pil_mask.convert("RGB")

        seed = resolved_options.get("seed")
        generator = None
        if seed is not None:
            generator = torch.Generator(device).manual_seed(int(seed))

        inference_kwargs: dict[str, Any] = {
            "prompt": prompt_text,
            "image": masked_image,
            "mask": mask_image,
            "num_inference_steps": int(resolved_options.get("steps", resolved_options.get("num_inference_steps", 50))),
            "guidance_scale": float(resolved_options.get("guidance_scale", 7.5)),
            "brushnet_conditioning_scale": float(resolved_options.get("brushnet_conditioning_scale", 1.0)),
            "height": working_height,
            "width": working_width,
        }
        if negative_prompt_text:
            inference_kwargs["negative_prompt"] = negative_prompt_text
        if generator is not None:
            inference_kwargs["generator"] = generator
        if "eta" in resolved_options:
            inference_kwargs["eta"] = float(resolved_options["eta"])
        if "guess_mode" in resolved_options:
            inference_kwargs["guess_mode"] = bool(resolved_options["guess_mode"])
        if "control_guidance_start" in resolved_options:
            inference_kwargs["control_guidance_start"] = float(resolved_options["control_guidance_start"])
        if "control_guidance_end" in resolved_options:
            inference_kwargs["control_guidance_end"] = float(resolved_options["control_guidance_end"])
        if "clip_skip" in resolved_options:
            inference_kwargs["clip_skip"] = int(resolved_options["clip_skip"])

        output = pipeline(**inference_kwargs).images[0].convert("RGB")
        if resized_for_model:
            resampling = getattr(Image, "Resampling", Image)
            output = output.resize((original_width, original_height), resampling.LANCZOS)
        restored_rgb = np.array(output)

        composite_back = bool(resolved_options.get("composite_back", True))
        blend_mask_blur_radius = float(resolved_options.get("blend_mask_blur_radius", 4.0))
        if composite_back:
            original_rgb = image[:, :, ::-1]
            restored_rgb = _blend_restored_rgb_with_original(
                original_rgb,
                restored_rgb,
                mask,
                blur_radius=blend_mask_blur_radius,
            )

        if device == "cuda" and bool(resolved_options.get("clear_cuda_cache", True)) and hasattr(torch.cuda, "empty_cache"):
            torch.cuda.empty_cache()

        latency_ms = (perf_counter() - start) * 1000.0
        return {
            "restored": restored_rgb[:, :, ::-1],
            "latency_ms": latency_ms,
            "meta": {
                "engine": "brushnet",
                "session_mode": session_mode or "inprocess",
                "brushnet_model_path": brushnet_model_path,
                "base_model_path": base_model_path,
                "source_dir": str(source_dir) if source_dir else None,
                "device": device,
                "torch_dtype": torch_dtype_name,
                "original_shape": [int(original_height), int(original_width)],
                "working_shape": [int(working_height), int(working_width)],
                "resized_for_model": resized_for_model,
                "input_mask_blur_radius": input_mask_blur_radius,
                "blend_mask_blur_radius": blend_mask_blur_radius,
                "composite_back": composite_back,
                "prompt_used": prompt_text,
                "negative_prompt_used": negative_prompt_text,
                "restore_options": dict(resolved_options),
            },
        }


def _fit_restored_to_original_size(restored: np.ndarray, original_height: int, original_width: int) -> np.ndarray:
    current_height, current_width = restored.shape[:2]
    if current_height == original_height and current_width == original_width:
        return restored
    return restored[:original_height, :original_width]


def _read_optional_string_option(options: dict[str, Any], key: str) -> str | None:
    value = options.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _resolve_brushnet_source_dir(options: dict[str, Any]) -> Path | None:
    source_dir = _read_optional_string_option(options, "source_dir") or os.getenv("NO_WATERMAR_BRUSHNET_SOURCE_DIR")
    if not source_dir:
        return None
    path = Path(source_dir).expanduser()
    if not path.exists():
        raise ValueError(f"BrushNet source_dir does not exist: {path}")
    return path.resolve()


def _resolve_brushnet_model_path(options: dict[str, Any]) -> str:
    value = _read_optional_string_option(options, "brushnet_model_path") or os.getenv("NO_WATERMAR_BRUSHNET_MODEL")
    if not value:
        raise ValueError(
            "BrushNet requires restore_options.brushnet_model_path or the NO_WATERMAR_BRUSHNET_MODEL environment variable."
        )
    candidate = Path(value).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    return value


def _resolve_brushnet_base_model_path(
    options: dict[str, Any],
    *,
    brushnet_model_path: str,
    local_files_only: bool,
) -> str:
    explicit = _read_optional_string_option(options, "base_model_path") or os.getenv("NO_WATERMAR_BRUSHNET_BASE_MODEL_PATH")
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.exists():
            return str(candidate.resolve())
        return explicit

    brushnet_candidate = Path(brushnet_model_path).expanduser()
    if brushnet_candidate.exists():
        sibling_base_model = brushnet_candidate.parent / "realisticVisionV60B1_v51VAE"
        if sibling_base_model.exists():
            return str(sibling_base_model.resolve())

    if local_files_only:
        raise ValueError(
            "BrushNet base model path is missing. Set restore_options.base_model_path or NO_WATERMAR_BRUSHNET_BASE_MODEL_PATH."
        )
    return "runwayml/stable-diffusion-v1-5"


def _resolve_powerpaint_source_dir(options: dict[str, Any]) -> Path | None:
    source_dir = _read_optional_string_option(options, "source_dir") or os.getenv("NO_WATERMAR_POWERPAINT_SOURCE_DIR")
    if not source_dir:
        return None
    path = Path(source_dir).expanduser()
    if not path.exists():
        raise ValueError(f"PowerPaint source_dir does not exist: {path}")
    return path.resolve()


def _resolve_powerpaint_checkpoint_dir(options: dict[str, Any]) -> Path:
    checkpoint_dir = _read_optional_string_option(options, "checkpoint_dir") or os.getenv(
        "NO_WATERMAR_POWERPAINT_CHECKPOINT_DIR"
    )
    if not checkpoint_dir:
        raise ValueError(
            "PowerPaint v2.1 requires restore_options.checkpoint_dir or the NO_WATERMAR_POWERPAINT_CHECKPOINT_DIR environment variable."
        )
    path = Path(checkpoint_dir).expanduser()
    if not path.exists():
        raise ValueError(f"PowerPaint checkpoint_dir does not exist: {path}")
    return path.resolve()


def _resolve_powerpaint_base_model_path(options: dict[str, Any], checkpoint_dir: Path) -> str:
    value = (
        _read_optional_string_option(options, "base_model_path")
        or os.getenv("NO_WATERMAR_POWERPAINT_BASE_MODEL_PATH")
        or str(checkpoint_dir / "realisticVisionV60B1_v51VAE")
    )
    candidate = Path(value).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    if value == str(checkpoint_dir / "realisticVisionV60B1_v51VAE"):
        raise ValueError(
            "PowerPaint base model path is missing. Expected checkpoint_dir/realisticVisionV60B1_v51VAE or set restore_options.base_model_path."
        )
    return value


def _resolve_powerpaint_backbone_source(
    options: dict[str, Any],
    *,
    base_model_path: str,
    local_files_only: bool,
) -> str:
    explicit_source = _read_optional_string_option(options, "backbone_source") or _read_optional_string_option(
        options, "brushnet_backbone_source"
    )
    if explicit_source:
        return explicit_source
    if local_files_only:
        return base_model_path
    return "runwayml/stable-diffusion-v1-5"


def _resolve_powerpaint_device(torch_module: Any, options: dict[str, Any]) -> str:
    requested_device = str(options.get("device") or os.getenv("NO_WATERMAR_POWERPAINT_DEVICE") or "auto").strip().lower()
    if requested_device in {"", "auto"}:
        if torch_module.cuda.is_available():
            return "cuda"
        return "cpu"
    return requested_device


def _resolve_brushnet_device(torch_module: Any, options: dict[str, Any]) -> str:
    requested_device = str(options.get("device") or os.getenv("NO_WATERMAR_BRUSHNET_DEVICE") or "auto").strip().lower()
    if requested_device in {"", "auto"}:
        if torch_module.cuda.is_available():
            return "cuda"
        return "cpu"
    return requested_device


def _resolve_powerpaint_dtype(torch_module: Any, device: str, options: dict[str, Any]) -> tuple[Any, str]:
    requested_dtype = str(options.get("torch_dtype") or os.getenv("NO_WATERMAR_POWERPAINT_TORCH_DTYPE") or "auto").strip().lower()
    if requested_dtype in {"", "auto"}:
        resolved_name = "float16" if device == "cuda" else "float32"
    else:
        resolved_name = requested_dtype

    dtype_map = {
        "float16": torch_module.float16,
        "float32": torch_module.float32,
        "bfloat16": torch_module.bfloat16,
    }
    try:
        return dtype_map[resolved_name], resolved_name
    except KeyError as exc:
        raise ValueError(f"Unsupported PowerPaint torch_dtype option: {requested_dtype}") from exc


def _resolve_brushnet_dtype(torch_module: Any, device: str, options: dict[str, Any]) -> tuple[Any, str]:
    requested_dtype = str(options.get("torch_dtype") or os.getenv("NO_WATERMAR_BRUSHNET_TORCH_DTYPE") or "auto").strip().lower()
    if requested_dtype in {"", "auto"}:
        resolved_name = "float16" if device == "cuda" else "float32"
    else:
        resolved_name = requested_dtype

    dtype_map = {
        "float16": torch_module.float16,
        "float32": torch_module.float32,
        "bfloat16": torch_module.bfloat16,
    }
    try:
        return dtype_map[resolved_name], resolved_name
    except KeyError as exc:
        raise ValueError(f"Unsupported BrushNet torch_dtype option: {requested_dtype}") from exc


def _load_brushnet_pipeline(
    *,
    brushnet_model_path: str,
    base_model_path: str,
    source_dir: Path | None,
    device: str,
    torch_dtype: Any,
    torch_dtype_name: str,
    local_files_only: bool,
    enable_model_cpu_offload: bool,
    use_safetensors: bool,
    brushnet_type: Any,
    pipeline_type: Any,
    scheduler_type: Any,
) -> dict[str, Any]:
    cache_key = (
        brushnet_model_path,
        base_model_path,
        str(source_dir or ""),
        device,
        torch_dtype_name,
        str(local_files_only),
        str(enable_model_cpu_offload),
        str(use_safetensors),
    )
    cached = _BRUSHNET_PIPELINE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    brushnet = brushnet_type.from_pretrained(
        brushnet_model_path,
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
        use_safetensors=use_safetensors,
    )
    pipeline = pipeline_type.from_pretrained(
        base_model_path,
        brushnet=brushnet,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=False,
        safety_checker=None,
        local_files_only=local_files_only,
        use_safetensors=use_safetensors,
    )
    pipeline.scheduler = scheduler_type.from_config(pipeline.scheduler.config)

    if enable_model_cpu_offload and device == "cuda" and hasattr(pipeline, "enable_model_cpu_offload"):
        pipeline.enable_model_cpu_offload()
        pipeline = pipeline.to(device)
    else:
        pipeline = pipeline.to(device)

    payload = {
        "pipeline": pipeline,
        "brushnet_model_path": brushnet_model_path,
        "base_model_path": base_model_path,
        "device": device,
        "torch_dtype": torch_dtype_name,
        "local_files_only": local_files_only,
        "source_dir": str(source_dir) if source_dir else None,
    }
    _BRUSHNET_PIPELINE_CACHE[cache_key] = payload
    return payload


def _load_powerpaint_v21_pipeline(
    *,
    checkpoint_dir: Path,
    base_model_path: str,
    backbone_source: str,
    source_dir: Path | None,
    device: str,
    torch_dtype: Any,
    torch_dtype_name: str,
    local_files_only: bool,
    enable_model_cpu_offload: bool,
    torch_module: Any,
    scheduler_type: Any,
    text_encoder_type: Any,
    brushnet_type: Any,
    unet_type: Any,
    pipeline_type: Any,
    tokenizer_type: Any,
    add_tokens_fn: Any,
    load_model_fn: Any,
) -> dict[str, Any]:
    cache_key = (
        str(checkpoint_dir),
        base_model_path,
        backbone_source,
        str(source_dir or ""),
        device,
        torch_dtype_name,
        str(local_files_only),
        str(enable_model_cpu_offload),
    )
    cached = _POWERPAINT_PIPELINE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    unet = unet_type.from_pretrained(
        backbone_source,
        subfolder="unet",
        revision=None,
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
    )
    text_encoder_brushnet = text_encoder_type.from_pretrained(
        backbone_source,
        subfolder="text_encoder",
        revision=None,
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
    )
    brushnet = brushnet_type.from_unet(unet)
    pipeline = pipeline_type.from_pretrained(
        base_model_path,
        brushnet=brushnet,
        text_encoder_brushnet=text_encoder_brushnet,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=False,
        safety_checker=None,
        local_files_only=local_files_only,
    )
    pipeline.unet = unet_type.from_pretrained(
        base_model_path,
        subfolder="unet",
        revision=None,
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
    )
    pipeline.tokenizer = tokenizer_type(
        from_pretrained=base_model_path,
        subfolder="tokenizer",
        revision=None,
        torch_type=torch_dtype,
        local_files_only=local_files_only,
    )
    add_tokens_fn(
        tokenizer=pipeline.tokenizer,
        text_encoder=pipeline.text_encoder_brushnet,
        placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
        initialize_tokens=["a", "a", "a"],
        num_vectors_per_token=10,
    )

    brushnet_weights = checkpoint_dir / "PowerPaint_Brushnet" / "diffusion_pytorch_model.safetensors"
    text_encoder_weights = checkpoint_dir / "PowerPaint_Brushnet" / "pytorch_model.bin"
    if not brushnet_weights.exists():
        raise ValueError(f"PowerPaint brushnet weights are missing: {brushnet_weights}")
    if not text_encoder_weights.exists():
        raise ValueError(f"PowerPaint text encoder weights are missing: {text_encoder_weights}")

    load_model_fn(pipeline.brushnet, str(brushnet_weights))
    pipeline.text_encoder_brushnet.load_state_dict(torch_module.load(str(text_encoder_weights), map_location="cpu"), strict=False)
    pipeline.scheduler = scheduler_type.from_config(pipeline.scheduler.config)

    if enable_model_cpu_offload and device == "cuda" and hasattr(pipeline, "enable_model_cpu_offload"):
        pipeline.enable_model_cpu_offload()
        pipeline = pipeline.to(device)
    else:
        pipeline = pipeline.to(device)

    payload = {
        "pipeline": pipeline,
        "checkpoint_dir": str(checkpoint_dir),
        "base_model_path": base_model_path,
        "backbone_source": backbone_source,
        "device": device,
        "torch_dtype": torch_dtype_name,
        "local_files_only": local_files_only,
    }
    _POWERPAINT_PIPELINE_CACHE[cache_key] = payload
    return payload


def _resolve_powerpaint_prompt(prompt: str | None, options: dict[str, Any], task: str) -> str:
    if prompt and str(prompt).strip():
        return str(prompt).strip()
    default_prompt = _read_optional_string_option(options, "default_prompt")
    if default_prompt:
        return default_prompt
    if task == "object-removal":
        return "remove the watermark text and reconstruct the covered background naturally"
    return "reconstruct the covered background naturally"


def _resolve_brushnet_prompt(prompt: str | None, options: dict[str, Any]) -> str:
    if prompt and str(prompt).strip():
        return str(prompt).strip()
    default_prompt = _read_optional_string_option(options, "default_prompt")
    if default_prompt:
        return default_prompt
    return "remove the watermark text and reconstruct the covered background naturally"


def _resolve_powerpaint_negative_prompt(negative_prompt: str | None, options: dict[str, Any]) -> str:
    if negative_prompt and str(negative_prompt).strip():
        return str(negative_prompt).strip()
    default_negative_prompt = _read_optional_string_option(options, "default_negative_prompt")
    if default_negative_prompt:
        return default_negative_prompt
    return "text, logo, watermark, letters, words, blur, smear, artifacts, duplicated edges"


def _resolve_brushnet_negative_prompt(negative_prompt: str | None, options: dict[str, Any]) -> str:
    if negative_prompt and str(negative_prompt).strip():
        return str(negative_prompt).strip()
    default_negative_prompt = _read_optional_string_option(options, "default_negative_prompt")
    if default_negative_prompt:
        return default_negative_prompt
    return "text, logo, watermark, letters, words, blur, smear, artifacts, duplicated edges"


def _prepare_powerpaint_task_prompts(
    *,
    prompt_text: str,
    negative_prompt_text: str,
    task: str,
) -> tuple[str, str, str, str]:
    normalized_task = task.strip().lower()
    if normalized_task in {"object-removal", "image-outpainting"}:
        positive_prefix = f"{prompt_text} empty scene".strip()
        if normalized_task == "object-removal":
            positive_prefix = f"{positive_prefix} blur".strip()
        return (
            f"{positive_prefix} P_ctxt".strip(),
            f"{positive_prefix} P_ctxt".strip(),
            f"{negative_prompt_text} P_obj".strip(),
            f"{negative_prompt_text} P_obj".strip(),
        )
    if normalized_task == "shape-guided":
        return (
            f"{prompt_text} P_shape".strip(),
            f"{prompt_text} P_ctxt".strip(),
            f"{negative_prompt_text} P_shape".strip(),
            f"{negative_prompt_text} P_ctxt".strip(),
        )
    return (
        f"{prompt_text} P_obj".strip(),
        f"{prompt_text} P_obj".strip(),
        f"{negative_prompt_text} P_obj".strip(),
        f"{negative_prompt_text} P_obj".strip(),
    )


def _resize_to_longest_side(width: int, height: int, longest_side: int, *, allow_upscale: bool) -> tuple[int, int]:
    if longest_side <= 0:
        return width, height
    current_longest = max(width, height)
    if current_longest <= longest_side and not allow_upscale:
        return width, height
    scale = float(longest_side) / float(current_longest)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def _floor_align_size(width: int, height: int, align_multiple: int) -> tuple[int, int]:
    if align_multiple <= 1:
        return width, height
    aligned_width = max(align_multiple, width - (width % align_multiple))
    aligned_height = max(align_multiple, height - (height % align_multiple))
    return aligned_width, aligned_height


def _blend_restored_rgb_with_original(
    original_rgb: np.ndarray,
    restored_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    blur_radius: float,
) -> np.ndarray:
    mask_image = Image.fromarray(mask.astype(np.uint8)).convert("L")
    if blur_radius > 0:
        mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    alpha = np.asarray(mask_image, dtype=np.float32) / 255.0
    alpha = alpha[:, :, None]
    blended = restored_rgb.astype(np.float32) * alpha + original_rgb.astype(np.float32) * (1.0 - alpha)
    return blended.clip(0, 255).astype(np.uint8)


@contextlib.contextmanager
def _powerpaint_source_context(source_dir: Path | None):
    if source_dir is None:
        yield
        return

    source_dir_text = str(source_dir)
    removed = False
    if source_dir_text not in sys.path:
        sys.path.insert(0, source_dir_text)
        removed = True
    try:
        yield
    finally:
        if removed:
            with contextlib.suppress(ValueError):
                sys.path.remove(source_dir_text)


@contextlib.contextmanager
def _brushnet_source_context(source_dir: Path | None):
    if source_dir is None:
        yield
        return

    import_root = source_dir / "src" if (source_dir / "src").exists() else source_dir
    import_root_text = str(import_root)
    removed = False
    if import_root_text not in sys.path:
        sys.path.insert(0, import_root_text)
        removed = True
    try:
        yield
    finally:
        if removed:
            with contextlib.suppress(ValueError):
                sys.path.remove(import_root_text)


def _load_diffusers_inpaint_pipeline(
    *,
    model_id: str,
    resolved_options: dict[str, Any],
    torch_dtype: Any,
) -> tuple[Any, str, str, str, str]:
    from diffusers import (
        AutoPipelineForInpainting,
        FluxFillPipeline,
        StableDiffusionInpaintPipeline,
        StableDiffusionXLInpaintPipeline,
    )

    load_mode = _resolve_diffusers_load_mode(model_id, resolved_options)
    local_files_only = bool(resolved_options.get("local_files_only", False))
    use_safetensors = bool(resolved_options.get("use_safetensors", True))
    pipeline_class_name = _read_optional_string_option(resolved_options, "pipeline_class")
    requested_model_id = model_id
    hf_endpoint = _read_optional_string_option(resolved_options, "hf_endpoint") or os.getenv("HF_ENDPOINT")
    model_store_dir = resolve_diffusers_model_store_dir(
        _read_optional_string_option(resolved_options, "model_store_dir")
    )
    pretrained_pipeline_class_map = {
        "AutoPipelineForInpainting": AutoPipelineForInpainting,
        "FluxFillPipeline": FluxFillPipeline,
    }

    if load_mode == "pretrained":
        if pipeline_class_name:
            try:
                pipeline_class = pretrained_pipeline_class_map[pipeline_class_name]
            except KeyError as exc:
                raise ValueError(
                    "Unsupported diffusers pipeline_class option for pretrained loading: "
                    f"{pipeline_class_name}. Supported values: {', '.join(sorted(pretrained_pipeline_class_map))}."
                ) from exc
        else:
            pipeline_class = AutoPipelineForInpainting
            pipeline_class_name = "AutoPipelineForInpainting"

        resolved_model_source = model_id
        if not local_files_only and not Path(model_id).exists():
            from huggingface_hub import snapshot_download

            resolved_model_source = str(
                download_hf_repo_snapshot(
                    repo_id=model_id,
                    model_store_dir=model_store_dir,
                    endpoint=hf_endpoint,
                    snapshot_download=snapshot_download,
                )
            )

        with using_hf_project_store(model_store_dir, endpoint=hf_endpoint):
            pipeline = pipeline_class.from_pretrained(
                resolved_model_source,
                torch_dtype=torch_dtype,
                variant=_read_optional_string_option(resolved_options, "variant"),
                revision=_read_optional_string_option(resolved_options, "revision"),
                local_files_only=local_files_only or resolved_model_source != model_id,
                use_safetensors=use_safetensors,
            )
        if resolved_model_source != requested_model_id:
            resolved_options.setdefault("requested_model_id", requested_model_id)
            resolved_options.setdefault("model_store_dir", str(model_store_dir))
        return pipeline, "diffusers_autopipeline", load_mode, pipeline_class_name, resolved_model_source

    if load_mode != "single_file":
        raise ValueError(f"Unsupported diffusers load_mode option: {load_mode}")

    pipeline_class_name = pipeline_class_name or "StableDiffusionInpaintPipeline"
    pipeline_class_map = {
        "FluxFillPipeline": FluxFillPipeline,
        "StableDiffusionInpaintPipeline": StableDiffusionInpaintPipeline,
        "StableDiffusionXLInpaintPipeline": StableDiffusionXLInpaintPipeline,
    }
    try:
        pipeline_class = pipeline_class_map[pipeline_class_name]
    except KeyError as exc:
        raise ValueError(
            "Unsupported diffusers pipeline_class option for single_file loading: "
            f"{pipeline_class_name}. Supported values: {', '.join(sorted(pipeline_class_map))}."
        ) from exc

    model_source = (
        _read_optional_string_option(resolved_options, "single_file_url")
        or _read_optional_string_option(resolved_options, "single_file_path")
        or model_id
    )
    single_file_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "local_files_only": local_files_only,
        "use_safetensors": use_safetensors,
    }
    config = _read_optional_string_option(resolved_options, "config")
    if config:
        single_file_kwargs["config"] = config
    original_config = _read_optional_string_option(resolved_options, "original_config")
    if original_config:
        single_file_kwargs["original_config"] = original_config
    if "extract_ema" in resolved_options:
        single_file_kwargs["extract_ema"] = bool(resolved_options["extract_ema"])

    with using_hf_project_store(model_store_dir, endpoint=hf_endpoint):
        pipeline = pipeline_class.from_single_file(model_source, **single_file_kwargs)
    return pipeline, "diffusers_single_file", load_mode, pipeline_class_name, model_source


def _resolve_diffusers_load_mode(model_id: str, options: dict[str, Any]) -> str:
    requested_mode = _read_optional_string_option(options, "load_mode")
    if requested_mode:
        return requested_mode.lower()
    if _looks_like_single_file_source(model_id):
        return "single_file"
    if _read_optional_string_option(options, "single_file_url") or _read_optional_string_option(options, "single_file_path"):
        return "single_file"
    return "pretrained"


def _looks_like_single_file_source(model_id: str) -> bool:
    lower_model_id = model_id.lower()
    if lower_model_id.startswith(("http://", "https://")):
        return lower_model_id.endswith((".ckpt", ".safetensors", ".bin"))
    return Path(model_id).suffix.lower() in {".ckpt", ".safetensors", ".bin"}


def _resolve_diffusers_device(torch_module: Any, options: dict[str, Any]) -> str:
    requested_device = str(options.get("device") or os.getenv("NO_WATERMAR_DIFFUSERS_DEVICE") or "auto").strip().lower()
    if requested_device in {"", "auto"}:
        if torch_module.cuda.is_available():
            return "cuda"
        return "cpu"
    return requested_device


def _resolve_diffusers_dtype(
    torch_module: Any,
    device: str,
    options: dict[str, Any],
) -> tuple[Any, str]:
    requested_dtype = str(options.get("torch_dtype") or os.getenv("NO_WATERMAR_DIFFUSERS_TORCH_DTYPE") or "auto").strip().lower()
    if requested_dtype in {"", "auto"}:
        resolved_name = "float16" if device == "cuda" else "float32"
    else:
        resolved_name = requested_dtype

    dtype_map = {
        "float16": torch_module.float16,
        "float32": torch_module.float32,
        "bfloat16": torch_module.bfloat16,
    }
    try:
        return dtype_map[resolved_name], resolved_name
    except KeyError as exc:
        raise ValueError(f"Unsupported diffusers torch_dtype option: {requested_dtype}") from exc


def _maybe_enable_diffusers_layerwise_casting(
    pipeline: Any,
    *,
    torch_module: Any,
    options: dict[str, Any],
) -> dict[str, Any] | None:
    if not bool(options.get("enable_layerwise_casting", False)):
        return None

    transformer = getattr(pipeline, "transformer", None)
    if transformer is None or not hasattr(transformer, "enable_layerwise_casting"):
        raise ValueError(
            "Diffusers layerwise casting requires a pipeline with a transformer that exposes enable_layerwise_casting()."
        )

    storage_dtype_name = str(options.get("layerwise_storage_dtype") or "float8_e4m3fn").strip().lower()
    compute_dtype_name = str(options.get("layerwise_compute_dtype") or options.get("torch_dtype") or "bfloat16").strip().lower()
    storage_dtype = _resolve_named_torch_dtype(torch_module, storage_dtype_name, option_name="layerwise_storage_dtype")
    compute_dtype = _resolve_named_torch_dtype(torch_module, compute_dtype_name, option_name="layerwise_compute_dtype")

    kwargs: dict[str, Any] = {
        "storage_dtype": storage_dtype,
        "compute_dtype": compute_dtype,
    }
    if "layerwise_non_blocking" in options:
        kwargs["non_blocking"] = bool(options["layerwise_non_blocking"])
    transformer.enable_layerwise_casting(**kwargs)
    return {
        "enabled": True,
        "target": "transformer",
        "storage_dtype": storage_dtype_name,
        "compute_dtype": compute_dtype_name,
        "non_blocking": bool(kwargs.get("non_blocking", False)),
    }


def _resolve_named_torch_dtype(torch_module: Any, requested_dtype: str, *, option_name: str) -> Any:
    normalized = requested_dtype.strip().lower()
    dtype_map = {
        "float16": getattr(torch_module, "float16", None),
        "float32": getattr(torch_module, "float32", None),
        "bfloat16": getattr(torch_module, "bfloat16", None),
        "float8_e4m3fn": getattr(torch_module, "float8_e4m3fn", None),
        "float8_e5m2": getattr(torch_module, "float8_e5m2", None),
    }
    resolved = dtype_map.get(normalized)
    if resolved is None:
        supported = ", ".join(name for name, value in dtype_map.items() if value is not None)
        raise ValueError(f"Unsupported {option_name} option: {requested_dtype}. Supported values: {supported}.")
    return resolved


def _align_diffusers_size(width: int, height: int, align_multiple: int) -> tuple[int, int]:
    if align_multiple <= 1:
        return width, height
    aligned_width = max(align_multiple, ((width + align_multiple - 1) // align_multiple) * align_multiple)
    aligned_height = max(align_multiple, ((height + align_multiple - 1) // align_multiple) * align_multiple)
    return aligned_width, aligned_height


def _flatten_paddleocr_result(raw_result: Any) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    if not raw_result:
        return lines

    if isinstance(raw_result, list) and raw_result and hasattr(raw_result[0], "get") and raw_result[0].get("dt_polys") is not None:
        for page in raw_result:
            polygons = page.get("dt_polys") or []
            texts = page.get("rec_texts") or []
            scores = page.get("rec_scores") or []
            for polygon, text, score in zip(polygons, texts, scores):
                normalized_polygon = [(int(point[0]), int(point[1])) for point in polygon]
                lines.append(
                    {
                        "polygon": normalized_polygon,
                        "text": str(text),
                        "score": float(score),
                    }
                )
        return lines

    entries = raw_result[0] if isinstance(raw_result, list) and raw_result and isinstance(raw_result[0], list) else raw_result
    for entry in entries:
        if not isinstance(entry, list) or len(entry) < 2:
            continue
        polygon = entry[0]
        text_info = entry[1]
        if not isinstance(text_info, (list, tuple)) or len(text_info) < 2:
            continue
        text = str(text_info[0])
        score = float(text_info[1])
        normalized_polygon = [(int(point[0]), int(point[1])) for point in polygon]
        lines.append({"polygon": normalized_polygon, "text": text, "score": score})
    return lines


def _classify_text_region(
    polygon: list[tuple[int, int]],
    width: int,
    height: int,
    source_category: str,
) -> str | None:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    center_x = sum(xs) / max(1, len(xs))
    center_y = sum(ys) / max(1, len(ys))

    if source_category == "cover_heavy":
        return "cover_text"

    if source_category == "landscape_regular":
        if center_y >= height * 0.82 and center_x <= width * 0.62:
            return "bottom_left"
        if center_y >= height * 0.80 and center_x >= width * 0.74:
            return "bottom_right"
        return None

    if center_y >= height * 0.88 and center_x <= width * 0.72:
        return "bottom_left"
    if center_y >= height * 0.84 and center_x >= width * 0.74:
        return "bottom_right"
    return None


def _looks_like_watermark_text(text: str, region: str, watermark_tokens: tuple[str, ...] | None = None) -> bool:
    normalized = "".join(text.lower().split())
    if watermark_tokens is None:
        watermark_tokens = get_watermark_keyword_settings().tokens
    if any(token in normalized for token in watermark_tokens):
        return True
    if region in {"bottom_left", "bottom_right", "cover_text"} and len(normalized) >= 4:
        if any(token in normalized for token in ("com", "net", "org", "cn", "eu")):
            return True
        if any(token in normalized for token in ("copyright", "issuedby", "xiuren", "imiss", "bestgiri", "girisexy")):
            return True
    if region == "bottom_right" and len(normalized) >= 4 and "com" in normalized:
        return True
    return False


def _expand_box(x1: int, y1: int, x2: int, y2: int, width: int, height: int, region: str) -> tuple[int, int, int, int]:
    if region == "bottom_left":
        pad_x = 16
        pad_y = 8
    elif region == "bottom_right":
        pad_x = 12
        pad_y = 8
    else:
        pad_x = 10
        pad_y = 6

    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )
