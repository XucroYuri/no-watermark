from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .benchmark_metrics import OCR_RESIDUAL_METRIC_KEYS
from .benchmark_models import BenchmarkDatasetItem
from .benchmark_providers import PaddleOCRMaskProvider, ProviderUnavailableError
from .io_utils import write_image


def score_ocr_residual(
    item: BenchmarkDatasetItem,
    restored: np.ndarray,
    *,
    hints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    provider = PaddleOCRMaskProvider()
    hints = dict(hints or {})

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        temp_input_path = tmp_root / item.relative_path.with_suffix(".jpg")
        write_image(temp_input_path, restored)
        restored_height, restored_width = restored.shape[:2]
        temp_item = BenchmarkDatasetItem(
            item_id=item.item_id,
            dataset_id=item.dataset_id,
            source_path=item.source_path,
            input_path=temp_input_path,
            relative_path=item.relative_path,
            width=restored_width,
            height=restored_height,
            source_category=item.source_category,
            benchmark_category=item.benchmark_category,
            prompt=item.prompt,
            prompt_path=item.prompt_path,
            seed_mask_path=None,
            seed_overlay_path=None,
            notes=dict(item.notes),
        )

        try:
            mask_result = provider.detect(temp_item, restored, hints=hints)
        except ProviderUnavailableError as exc:
            return {
                "metrics": _empty_metrics(),
                "meta": {
                    "status": "unavailable",
                    "note": str(exc),
                    "session_mode": _read_session_mode(hints),
                    "latency_ms": None,
                },
            }
        except Exception as exc:
            return {
                "metrics": _empty_metrics(),
                "meta": {
                    "status": "error",
                    "note": str(exc),
                    "session_mode": _read_session_mode(hints),
                    "latency_ms": None,
                },
            }

    matched_texts = list(mask_result.meta.get("matched_texts") or [])
    scores = [float(entry.get("score", 0.0)) for entry in matched_texts]
    return {
        "metrics": {
            "ocr_residual_hits": len(matched_texts),
            "ocr_residual_score": round(sum(scores), 6),
            "ocr_residual_max_score": round(max(scores), 6) if scores else 0.0,
        },
        "meta": {
            "status": "ok",
            "provider": mask_result.provider_name,
            "latency_ms": round(mask_result.latency_ms, 3),
            "matched_texts": matched_texts,
            "session_mode": mask_result.meta.get("session_mode"),
        },
    }


def _empty_metrics() -> dict[str, Any]:
    return {key: None for key in OCR_RESIDUAL_METRIC_KEYS}


def _read_session_mode(hints: dict[str, Any]) -> str | None:
    ocr_context = hints.get("paddleocr_context")
    return getattr(ocr_context, "resolved_mode", None)
