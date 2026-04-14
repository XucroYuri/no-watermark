from __future__ import annotations

import contextlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import numpy as np

from .benchmark_paddleocr_session import (
    OCR_RESOLVED_MODE_INPROCESS,
    OCR_RESOLVED_MODE_ONESHOT,
    OCR_RESOLVED_MODE_PERSISTENT,
    PaddleOCRExecutionContext,
    resolve_paddleocr_sidecar_python,
)
from .benchmark_restore_session import (
    RESTORE_RESOLVED_MODE_ONESHOT,
    RESTORE_RESOLVED_MODE_PERSISTENT,
    PowerPaintExecutionContext,
)
from .benchmark_models import BenchmarkDatasetItem, MaskResult, RestoreResult
from .detector import detect_watermarks
from .io_utils import read_image, read_mask, write_image
from .models import ScanItem
from .provider_runtime import (
    probe_current_module,
    probe_current_runtime,
    probe_python_module,
    probe_python_runtime,
    summarize_probe,
)
from .restorer import crop_image_to_remove_corner_watermark, restore_regular_image


class ProviderUnavailableError(RuntimeError):
    pass


class MaskProvider(Protocol):
    name: str

    def detect(
        self,
        item: BenchmarkDatasetItem,
        image: np.ndarray,
        *,
        hints: dict[str, Any] | None = None,
    ) -> MaskResult:
        ...


class RestoreProvider(Protocol):
    name: str

    def restore(
        self,
        item: BenchmarkDatasetItem,
        image: np.ndarray,
        mask: np.ndarray,
        *,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        meta: dict[str, Any] | None = None,
        ) -> RestoreResult:
        ...


PROVIDER_PUBLIC_METADATA: dict[str, dict[str, Any]] = {
    "rule_based_roi": {
        "support_tier": "stable",
        "validated_platforms": ["windows", "linux"],
        "recommended_entrypoint": "no-watermar batch apply",
    },
    "seed_manifest": {
        "support_tier": "stable",
        "validated_platforms": ["windows", "linux"],
        "recommended_entrypoint": "no-watermar benchmark run",
    },
    "paddleocr": {
        "support_tier": "stable",
        "validated_platforms": ["windows"],
        "recommended_entrypoint": "no-watermar batch apply",
    },
    "edgesam": {
        "support_tier": "planned",
        "validated_platforms": [],
        "recommended_entrypoint": None,
    },
    "watermark_segmentation": {
        "support_tier": "planned",
        "validated_platforms": [],
        "recommended_entrypoint": None,
    },
    "telea": {
        "support_tier": "stable",
        "validated_platforms": ["windows", "linux"],
        "recommended_entrypoint": "no-watermar batch apply",
    },
    "corner_crop": {
        "support_tier": "stable",
        "validated_platforms": ["windows", "linux"],
        "recommended_entrypoint": "no-watermar batch apply",
    },
    "noop": {
        "support_tier": "stable",
        "validated_platforms": ["windows", "linux"],
        "recommended_entrypoint": "no-watermar benchmark run",
    },
    "lama": {
        "support_tier": "stable",
        "validated_platforms": ["windows"],
        "recommended_entrypoint": "no-watermar batch apply",
    },
    "diffusers_inpaint": {
        "support_tier": "experimental",
        "validated_platforms": ["windows"],
        "recommended_entrypoint": "no-watermar benchmark run",
    },
    "powerpaint_v2_1": {
        "support_tier": "experimental",
        "validated_platforms": ["windows"],
        "recommended_entrypoint": "no-watermar benchmark run",
    },
    "brushnet": {
        "support_tier": "experimental",
        "validated_platforms": ["windows"],
        "recommended_entrypoint": "no-watermar benchmark run",
    },
}


def get_provider_public_metadata(name: str) -> dict[str, Any]:
    entry = PROVIDER_PUBLIC_METADATA.get(name) or {}
    return {
        "support_tier": str(entry.get("support_tier") or "planned"),
        "validated_platforms": list(entry.get("validated_platforms") or []),
        "recommended_entrypoint": entry.get("recommended_entrypoint"),
    }


@dataclass(frozen=True, slots=True)
class ProviderDescriptor:
    name: str
    kind: str
    summary: str
    implemented: bool
    execution_modes: list[str]
    default_mode: str | None
    required_env_vars: list[str]
    runtime_available: bool
    runtime_note: str | None = None
    runtime_probe: dict[str, Any] | None = None
    support_tier: str = "planned"
    validated_platforms: list[str] = field(default_factory=list)
    recommended_entrypoint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "summary": self.summary,
            "implemented": self.implemented,
            "execution_modes": self.execution_modes,
            "default_mode": self.default_mode,
            "required_env_vars": self.required_env_vars,
            "runtime_available": self.runtime_available,
            "runtime_note": self.runtime_note,
            "runtime_probe": self.runtime_probe,
            "support_tier": self.support_tier,
            "validated_platforms": self.validated_platforms,
            "recommended_entrypoint": self.recommended_entrypoint,
        }


class RuleBasedMaskProvider:
    name = "rule_based_roi"

    def detect(
        self,
        item: BenchmarkDatasetItem,
        image: np.ndarray,
        *,
        hints: dict[str, Any] | None = None,
    ) -> MaskResult:
        start = perf_counter()
        scan_item = ScanItem(
            source_path=item.source_path,
            relative_path=item.relative_path,
            width=item.width,
            height=item.height,
            category=item.source_category,
        )
        mask, detection = detect_watermarks(scan_item, image)
        latency_ms = (perf_counter() - start) * 1000.0
        return MaskResult(
            provider_name=self.name,
            mask=mask,
            confidence=detection.confidence,
            boxes=detection.boxes,
            latency_ms=latency_ms,
            meta={"source_category": detection.category},
        )


class SeedManifestMaskProvider:
    name = "seed_manifest"

    def detect(
        self,
        item: BenchmarkDatasetItem,
        image: np.ndarray,
        *,
        hints: dict[str, Any] | None = None,
    ) -> MaskResult:
        start = perf_counter()
        if item.seed_mask_path and item.seed_mask_path.exists():
            mask = read_mask(item.seed_mask_path)
            confidence = 1.0 if np.any(mask) else 0.0
            meta = {"seed_mask_path": str(item.seed_mask_path)}
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            confidence = 0.0
            meta = {"seed_mask_path": None, "note": "missing seed mask"}

        latency_ms = (perf_counter() - start) * 1000.0
        return MaskResult(
            provider_name=self.name,
            mask=mask,
            confidence=confidence,
            boxes=[],
            latency_ms=latency_ms,
            meta=meta,
        )


class PaddleOCRMaskProvider:
    name = "paddleocr"

    def detect(
        self,
        item: BenchmarkDatasetItem,
        image: np.ndarray,
        *,
        hints: dict[str, Any] | None = None,
    ) -> MaskResult:
        hints = hints or {}
        score_threshold = float(hints.get("score_threshold", 0.45))
        ocr_context = hints.get("paddleocr_context")
        if isinstance(ocr_context, PaddleOCRExecutionContext):
            try:
                payload, mask = ocr_context.detect(
                    image_path=item.input_path,
                    source_category=item.source_category,
                    score_threshold=score_threshold,
                )
            except RuntimeError as exc:
                raise ProviderUnavailableError(str(exc)) from exc

            return MaskResult(
                provider_name=self.name,
                mask=mask,
                confidence=float(payload.get("confidence", 0.0)),
                boxes=list(payload.get("boxes", [])),
                latency_ms=float(payload.get("latency_ms", 0.0)),
                meta=dict(payload.get("meta") or {}),
            )

        if _module_available("paddleocr"):
            from .provider_workers import detect_mask_with_paddleocr

            payload = detect_mask_with_paddleocr(
                image,
                item.source_category,
                score_threshold=score_threshold,
                session_mode=OCR_RESOLVED_MODE_INPROCESS,
            )
            return MaskResult(
                provider_name=self.name,
                mask=payload["mask"],
                confidence=float(payload["confidence"]),
                boxes=list(payload["boxes"]),
                latency_ms=float(payload["latency_ms"]),
                meta=dict(payload["meta"]),
            )

        python_executable = os.getenv("NO_WATERMAR_PADDLEOCR_PYTHON")
        if not python_executable:
            raise ProviderUnavailableError(
                "PaddleOCR is unavailable in the current environment. "
                "Set NO_WATERMAR_PADDLEOCR_PYTHON to a Python 3.8-3.12 environment with paddleocr installed."
            )

        payload, mask = _run_mask_sidecar(
            python_executable=python_executable,
            script_path=_project_root() / "tools" / "sidecars" / "paddleocr_mask.py",
            image_path=item.input_path,
            source_category=item.source_category,
            score_threshold=score_threshold,
        )
        return MaskResult(
            provider_name=self.name,
            mask=mask,
            confidence=float(payload.get("confidence", 0.0)),
            boxes=list(payload.get("boxes", [])),
            latency_ms=float(payload.get("latency_ms", 0.0)),
            meta=dict(payload.get("meta") or {}),
        )


class TeleaRestoreProvider:
    name = "telea"

    def restore(
        self,
        item: BenchmarkDatasetItem,
        image: np.ndarray,
        mask: np.ndarray,
        *,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> RestoreResult:
        start = perf_counter()
        restored = restore_regular_image(image, mask)
        latency_ms = (perf_counter() - start) * 1000.0
        restore_options = dict((meta or {}).get("restore_options") or {})
        return RestoreResult(
            provider_name=self.name,
            restored=restored,
            latency_ms=latency_ms,
            peak_vram_mb=None,
            meta={
                "prompt_used": bool(prompt),
                "negative_prompt_used": bool(negative_prompt),
                "restore_options": restore_options,
            },
        )


class CornerCropRestoreProvider:
    name = "corner_crop"

    def restore(
        self,
        item: BenchmarkDatasetItem,
        image: np.ndarray,
        mask: np.ndarray,
        *,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> RestoreResult:
        start = perf_counter()
        restore_options = dict((meta or {}).get("restore_options") or {})
        payload = crop_image_to_remove_corner_watermark(
            image,
            mask,
            boxes=list((meta or {}).get("detection_boxes") or []),
            options=restore_options,
        )
        latency_ms = (perf_counter() - start) * 1000.0
        restore_meta = dict(payload.get("meta") or {})
        restore_meta["prompt_used"] = bool(prompt)
        restore_meta["negative_prompt_used"] = bool(negative_prompt)
        restore_meta["restore_options"] = restore_options
        return RestoreResult(
            provider_name=self.name,
            restored=payload["restored"],
            latency_ms=latency_ms,
            peak_vram_mb=None,
            meta=restore_meta,
        )


class NoopRestoreProvider:
    name = "noop"

    def restore(
        self,
        item: BenchmarkDatasetItem,
        image: np.ndarray,
        mask: np.ndarray,
        *,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> RestoreResult:
        start = perf_counter()
        restored = image.copy()
        latency_ms = (perf_counter() - start) * 1000.0
        return RestoreResult(
            provider_name=self.name,
            restored=restored,
            latency_ms=latency_ms,
            peak_vram_mb=None,
            meta={"note": "mask-only benchmark"},
        )


class LamaRestoreProvider:
    name = "lama"

    def restore(
        self,
        item: BenchmarkDatasetItem,
        image: np.ndarray,
        mask: np.ndarray,
        *,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> RestoreResult:
        if _lama_module_available():
            from .provider_workers import restore_with_simple_lama

            payload = restore_with_simple_lama(
                image,
                mask,
                prompt=prompt,
                negative_prompt=negative_prompt,
                options=dict((meta or {}).get("restore_options") or {}),
            )
            return RestoreResult(
                provider_name=self.name,
                restored=payload["restored"],
                latency_ms=float(payload["latency_ms"]),
                peak_vram_mb=None,
                meta=dict(payload["meta"]),
            )

        python_executable = os.getenv("NO_WATERMAR_LAMA_PYTHON")
        if not python_executable:
            raise ProviderUnavailableError(
                "LaMa sidecar is not configured. "
                "Set NO_WATERMAR_LAMA_PYTHON to a Python environment with simple-lama-inpainting or another compatible LaMa wrapper installed."
            )

        payload, restored = _run_restore_sidecar(
            python_executable=python_executable,
            script_path=_project_root() / "tools" / "sidecars" / "lama_restore.py",
            image_path=item.input_path,
            mask=mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            options=dict((meta or {}).get("restore_options") or {}),
        )
        return RestoreResult(
            provider_name=self.name,
            restored=restored,
            latency_ms=float(payload.get("latency_ms", 0.0)),
            peak_vram_mb=None,
            meta=dict(payload.get("meta") or {}),
        )


class DiffusersInpaintRestoreProvider:
    name = "diffusers_inpaint"

    def restore(
        self,
        item: BenchmarkDatasetItem,
        image: np.ndarray,
        mask: np.ndarray,
        *,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> RestoreResult:
        restore_options = dict((meta or {}).get("restore_options") or {})
        if _diffusers_module_available():
            from .provider_workers import restore_with_diffusers_inpaint

            payload = restore_with_diffusers_inpaint(
                image,
                mask,
                prompt=prompt,
                negative_prompt=negative_prompt,
                options=restore_options,
            )
            return RestoreResult(
                provider_name=self.name,
                restored=payload["restored"],
                latency_ms=float(payload["latency_ms"]),
                peak_vram_mb=None,
                meta=dict(payload["meta"]),
            )

        python_executable = os.getenv("NO_WATERMAR_DIFFUSERS_PYTHON")
        if not python_executable:
            raise ProviderUnavailableError(
                "Diffusers inpaint sidecar is not configured. "
                "Set NO_WATERMAR_DIFFUSERS_PYTHON to a Python environment with diffusers and torch installed. "
                "Also provide restore_options.model_id or NO_WATERMAR_DIFFUSERS_MODEL."
            )

        payload, restored = _run_restore_sidecar(
            python_executable=python_executable,
            script_path=_project_root() / "tools" / "sidecars" / "diffusers_restore.py",
            image_path=item.input_path,
            mask=mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            options=restore_options,
        )
        return RestoreResult(
            provider_name=self.name,
            restored=restored,
            latency_ms=float(payload.get("latency_ms", 0.0)),
            peak_vram_mb=None,
            meta=dict(payload.get("meta") or {}),
        )


class PowerPaintV21RestoreProvider:
    name = "powerpaint_v2_1"

    def restore(
        self,
        item: BenchmarkDatasetItem,
        image: np.ndarray,
        mask: np.ndarray,
        *,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> RestoreResult:
        restore_options = dict((meta or {}).get("restore_options") or {})
        powerpaint_context = (meta or {}).get("powerpaint_context")
        if _powerpaint_module_available():
            from .provider_workers import restore_with_powerpaint_v2_1

            payload = restore_with_powerpaint_v2_1(
                image,
                mask,
                prompt=prompt,
                negative_prompt=negative_prompt,
                options=restore_options,
                session_mode="inprocess",
            )
            return RestoreResult(
                provider_name=self.name,
                restored=payload["restored"],
                latency_ms=float(payload["latency_ms"]),
                peak_vram_mb=None,
                meta=dict(payload["meta"]),
            )

        if isinstance(powerpaint_context, PowerPaintExecutionContext):
            if powerpaint_context.resolved_mode in {RESTORE_RESOLVED_MODE_PERSISTENT, RESTORE_RESOLVED_MODE_ONESHOT}:
                try:
                    payload, restored = powerpaint_context.restore(
                        image_path=item.input_path,
                        mask=mask,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        options=restore_options,
                    )
                except RuntimeError as exc:
                    raise ProviderUnavailableError(str(exc)) from exc
                return RestoreResult(
                    provider_name=self.name,
                    restored=restored,
                    latency_ms=float(payload.get("latency_ms", 0.0)),
                    peak_vram_mb=None,
                    meta=dict(payload.get("meta") or {}),
                )
            if powerpaint_context.resolved_mode not in {"inprocess", None}:
                raise ProviderUnavailableError(
                    powerpaint_context.note or "PowerPaint execution context is unavailable."
                )

        python_executable = os.getenv("NO_WATERMAR_POWERPAINT_PYTHON")
        if not python_executable:
            raise ProviderUnavailableError(
                "PowerPaint v2.1 sidecar is not configured. "
                "Set NO_WATERMAR_POWERPAINT_PYTHON to a Python environment with the PowerPaint package importable. "
                "Also provide restore_options.checkpoint_dir or NO_WATERMAR_POWERPAINT_CHECKPOINT_DIR."
            )

        payload, restored = _run_restore_sidecar(
            python_executable=python_executable,
            script_path=_project_root() / "tools" / "sidecars" / "powerpaint_restore.py",
            image_path=item.input_path,
            mask=mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            options=restore_options,
        )
        return RestoreResult(
            provider_name=self.name,
            restored=restored,
            latency_ms=float(payload.get("latency_ms", 0.0)),
            peak_vram_mb=None,
            meta=dict(payload.get("meta") or {}),
        )


class BrushNetRestoreProvider:
    name = "brushnet"

    def restore(
        self,
        item: BenchmarkDatasetItem,
        image: np.ndarray,
        mask: np.ndarray,
        *,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> RestoreResult:
        restore_options = dict((meta or {}).get("restore_options") or {})
        should_try_inprocess = _brushnet_module_available() or bool(restore_options.get("source_dir"))
        if should_try_inprocess:
            try:
                from .provider_workers import restore_with_brushnet

                payload = restore_with_brushnet(
                    image,
                    mask,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    options=restore_options,
                    session_mode="inprocess",
                )
                return RestoreResult(
                    provider_name=self.name,
                    restored=payload["restored"],
                    latency_ms=float(payload["latency_ms"]),
                    peak_vram_mb=None,
                    meta=dict(payload["meta"]),
                )
            except Exception as exc:
                if not os.getenv("NO_WATERMAR_BRUSHNET_PYTHON"):
                    raise ProviderUnavailableError(str(exc)) from exc

        python_executable = os.getenv("NO_WATERMAR_BRUSHNET_PYTHON")
        if not python_executable:
            raise ProviderUnavailableError(
                "BrushNet sidecar is not configured. "
                "Set NO_WATERMAR_BRUSHNET_PYTHON to a Python environment with the BrushNet diffusers fork importable, "
                "or pair it with NO_WATERMAR_BRUSHNET_SOURCE_DIR. "
                "Also provide restore_options.brushnet_model_path or NO_WATERMAR_BRUSHNET_MODEL."
            )

        payload, restored = _run_restore_sidecar(
            python_executable=python_executable,
            script_path=_project_root() / "tools" / "sidecars" / "brushnet_restore.py",
            image_path=item.input_path,
            mask=mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            options=restore_options,
        )
        return RestoreResult(
            provider_name=self.name,
            restored=restored,
            latency_ms=float(payload.get("latency_ms", 0.0)),
            peak_vram_mb=None,
            meta=dict(payload.get("meta") or {}),
        )


MASK_PROVIDER_REGISTRY: dict[str, type[MaskProvider]] = {
    "rule_based_roi": RuleBasedMaskProvider,
    "seed_manifest": SeedManifestMaskProvider,
    "paddleocr": PaddleOCRMaskProvider,
}

RESTORE_PROVIDER_REGISTRY: dict[str, type[RestoreProvider]] = {
    "telea": TeleaRestoreProvider,
    "corner_crop": CornerCropRestoreProvider,
    "noop": NoopRestoreProvider,
    "lama": LamaRestoreProvider,
    "diffusers_inpaint": DiffusersInpaintRestoreProvider,
    "powerpaint_v2_1": PowerPaintV21RestoreProvider,
    "brushnet": BrushNetRestoreProvider,
}


def create_mask_provider(name: str) -> MaskProvider:
    try:
        provider_type = MASK_PROVIDER_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown mask provider: {name}") from exc
    return provider_type()


def create_restore_provider(name: str) -> RestoreProvider:
    try:
        provider_type = RESTORE_PROVIDER_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown restore provider: {name}") from exc
    return provider_type()


def list_provider_descriptors() -> dict[str, list[dict[str, Any]]]:
    paddle_available, paddle_note, paddle_probe = _describe_paddleocr_runtime()
    lama_available, lama_note, lama_probe = _describe_lama_runtime()
    diffusers_available, diffusers_note, diffusers_probe = _describe_diffusers_runtime()
    powerpaint_available, powerpaint_note, powerpaint_probe = _describe_powerpaint_runtime()
    brushnet_available, brushnet_note, brushnet_probe = _describe_brushnet_runtime()
    mask_descriptors = [
        _descriptor("rule_based_roi", "mask", "Current ROI + morphology detector", True, ["inprocess"], "inprocess", [], True),
        _descriptor("seed_manifest", "mask", "Loads prepared seed masks from benchmark dataset", True, ["inprocess"], "inprocess", [], True),
        _descriptor(
            "paddleocr",
            "mask",
            "OCR-backed detector via local import or NO_WATERMAR_PADDLEOCR_PYTHON sidecar",
            True,
            ["inprocess", OCR_RESOLVED_MODE_ONESHOT, OCR_RESOLVED_MODE_PERSISTENT],
            OCR_RESOLVED_MODE_PERSISTENT,
            ["NO_WATERMAR_PADDLEOCR_PYTHON"],
            paddle_available,
            paddle_note,
            paddle_probe,
        ),
        _descriptor("edgesam", "mask", "Planned prompt-guided segmentation provider", False, [], None, [], False, "Planned provider."),
        _descriptor("watermark_segmentation", "mask", "Planned dedicated watermark segmentation provider", False, [], None, [], False, "Planned provider."),
    ]
    restore_descriptors = [
        _descriptor("telea", "restore", "Current OpenCV Telea baseline", True, ["inprocess"], "inprocess", [], True),
        _descriptor(
            "corner_crop",
            "restore",
            "Crops the nearest image edge to remove a detected corner watermark",
            True,
            ["inprocess"],
            "inprocess",
            [],
            True,
        ),
        _descriptor("noop", "restore", "Mask-only benchmark placeholder", True, ["inprocess"], "inprocess", [], True),
        _descriptor(
            "lama",
            "restore",
            "LaMa sidecar via local import or NO_WATERMAR_LAMA_PYTHON",
            True,
            [OCR_RESOLVED_MODE_ONESHOT],
            OCR_RESOLVED_MODE_ONESHOT,
            ["NO_WATERMAR_LAMA_PYTHON"],
            lama_available,
            lama_note,
            lama_probe,
        ),
        _descriptor(
            "diffusers_inpaint",
            "restore",
            "Prompt-driven HuggingFace diffusers inpainting via local import or NO_WATERMAR_DIFFUSERS_PYTHON",
            True,
            ["inprocess", OCR_RESOLVED_MODE_ONESHOT],
            OCR_RESOLVED_MODE_ONESHOT,
            ["NO_WATERMAR_DIFFUSERS_PYTHON", "NO_WATERMAR_DIFFUSERS_MODEL"],
            diffusers_available,
            diffusers_note,
            diffusers_probe,
        ),
        _descriptor(
            "powerpaint_v2_1",
            "restore",
            "PowerPaint v2.1 object-removal inpainting via local import or NO_WATERMAR_POWERPAINT_PYTHON",
            True,
            ["inprocess", RESTORE_RESOLVED_MODE_ONESHOT, RESTORE_RESOLVED_MODE_PERSISTENT],
            RESTORE_RESOLVED_MODE_PERSISTENT,
            ["NO_WATERMAR_POWERPAINT_PYTHON", "NO_WATERMAR_POWERPAINT_CHECKPOINT_DIR"],
            powerpaint_available,
            powerpaint_note,
            powerpaint_probe,
        ),
        _descriptor(
            "brushnet",
            "restore",
            "BrushNet SD1.5 inpainting via upstream diffusers fork or NO_WATERMAR_BRUSHNET_PYTHON sidecar",
            True,
            ["inprocess", OCR_RESOLVED_MODE_ONESHOT],
            OCR_RESOLVED_MODE_ONESHOT,
            ["NO_WATERMAR_BRUSHNET_PYTHON", "NO_WATERMAR_BRUSHNET_MODEL"],
            brushnet_available,
            brushnet_note,
            brushnet_probe,
        ),
    ]
    return {
        "mask_providers": [descriptor.to_dict() for descriptor in mask_descriptors],
        "restore_providers": [descriptor.to_dict() for descriptor in restore_descriptors],
    }


def _descriptor(
    name: str,
    kind: str,
    summary: str,
    implemented: bool,
    execution_modes: list[str],
    default_mode: str | None,
    required_env_vars: list[str],
    runtime_available: bool,
    runtime_note: str | None = None,
    runtime_probe: dict[str, Any] | None = None,
) -> ProviderDescriptor:
    public_metadata = get_provider_public_metadata(name)
    return ProviderDescriptor(
        name=name,
        kind=kind,
        summary=summary,
        implemented=implemented,
        execution_modes=execution_modes,
        default_mode=default_mode,
        required_env_vars=required_env_vars,
        runtime_available=runtime_available,
        runtime_note=runtime_note,
        runtime_probe=runtime_probe,
        support_tier=str(public_metadata["support_tier"]),
        validated_platforms=list(public_metadata["validated_platforms"]),
        recommended_entrypoint=public_metadata["recommended_entrypoint"],
    )


def probe_provider_runtimes() -> dict[str, Any]:
    descriptors = list_provider_descriptors()
    return {
        "mask_providers": descriptors["mask_providers"],
        "restore_providers": descriptors["restore_providers"],
    }


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _module_available_any(module_names: list[str]) -> bool:
    return any(_module_available(name) for name in module_names)


def _module_available_all(module_names: list[str]) -> bool:
    return all(_module_available(name) for name in module_names)


def _lama_module_available() -> bool:
    return _module_available_any(["simple_lama_inpainting", "simple_lama"])


def _diffusers_module_available() -> bool:
    return _module_available_all(["diffusers", "torch"])


def _powerpaint_module_available() -> bool:
    return _module_available_all(["powerpaint", "torch", "diffusers", "transformers", "safetensors"])


def _brushnet_module_available() -> bool:
    return bool(_probe_current_brushnet_runtime().get("ok"))


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_brushnet_source_dir_from_env() -> Path | None:
    source_dir = os.getenv("NO_WATERMAR_BRUSHNET_SOURCE_DIR")
    if not source_dir:
        return None
    return Path(source_dir).expanduser()


@contextlib.contextmanager
def _brushnet_import_context(source_dir: Path | None):
    if source_dir is None:
        yield
        return

    import_root = source_dir / "src" if (source_dir / "src").exists() else source_dir
    import_root_text = str(import_root.resolve())
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


def _build_brushnet_probe_payload(*, python_executable: str | None, source_dir: Path | None = None) -> dict[str, Any]:
    return {
        "module_name": "diffusers",
        "import_target": "diffusers",
        "required_symbols": ["BrushNetModel", "StableDiffusionBrushNetPipeline"],
        "python_executable": python_executable,
        "python_version": sys.version.split()[0] if python_executable == sys.executable else None,
        "source_dir": str(source_dir.resolve()) if source_dir and source_dir.exists() else (str(source_dir) if source_dir else None),
    }


def _probe_current_brushnet_runtime() -> dict[str, Any]:
    source_dir = _resolve_brushnet_source_dir_from_env()
    payload = _build_brushnet_probe_payload(python_executable=sys.executable, source_dir=source_dir)
    if source_dir is not None and not source_dir.exists():
        payload.update(
            {
                "ok": False,
                "module_found": False,
                "importable": False,
                "version": None,
                "error": f"NO_WATERMAR_BRUSHNET_SOURCE_DIR points to a missing path: {source_dir}",
            }
        )
        return payload

    with _brushnet_import_context(source_dir):
        spec = importlib.util.find_spec("diffusers")
        if spec is None:
            payload.update(
                {
                    "ok": False,
                    "module_found": False,
                    "importable": False,
                    "version": None,
                    "error": "Module not found: diffusers",
                }
            )
            return payload

        try:
            diffusers_module = importlib.import_module("diffusers")
        except Exception as exc:
            payload.update(
                {
                    "ok": False,
                    "module_found": True,
                    "importable": False,
                    "version": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            return payload

        missing_symbols = [
            name
            for name in ("BrushNetModel", "StableDiffusionBrushNetPipeline")
            if not hasattr(diffusers_module, name)
        ]
        if missing_symbols:
            payload.update(
                {
                    "ok": False,
                    "module_found": True,
                    "importable": True,
                    "version": getattr(diffusers_module, "__version__", None),
                    "error": f"BrushNet support is missing from diffusers: {', '.join(missing_symbols)}",
                }
            )
            return payload

        payload.update(
            {
                "ok": True,
                "module_found": True,
                "importable": True,
                "version": getattr(diffusers_module, "__version__", None),
                "error": None,
            }
        )
        return payload


def _probe_python_brushnet_runtime(python_executable: str, source_dir: Path | None = None) -> dict[str, Any]:
    python_path = Path(python_executable)
    if not python_path.exists():
        payload = _build_brushnet_probe_payload(python_executable=str(python_path), source_dir=source_dir)
        payload.update(
            {
                "ok": False,
                "module_found": False,
                "importable": False,
                "version": None,
                "error": f"Interpreter not found: {python_path}",
                "python_version": None,
            }
        )
        return payload

    script = "\n".join(
        [
            "import importlib",
            "import importlib.util",
            "import json",
            "import pathlib",
            "import sys",
            "source_dir_arg = sys.argv[1]",
            "source_dir = pathlib.Path(source_dir_arg).expanduser() if source_dir_arg else None",
            "payload = {",
            "  'module_name': 'diffusers',",
            "  'import_target': 'diffusers',",
            "  'required_symbols': ['BrushNetModel', 'StableDiffusionBrushNetPipeline'],",
            "  'python_executable': sys.executable,",
            "  'python_version': sys.version.split()[0],",
            "  'source_dir': str(source_dir) if source_dir else None,",
            "}",
            "if source_dir is not None:",
            "  if not source_dir.exists():",
            "    payload.update({'ok': False, 'module_found': False, 'importable': False, 'version': None, 'error': f'NO_WATERMAR_BRUSHNET_SOURCE_DIR points to a missing path: {source_dir}'})",
            "    print(json.dumps(payload, ensure_ascii=False))",
            "    raise SystemExit(0)",
            "  import_root = source_dir / 'src' if (source_dir / 'src').exists() else source_dir",
            "  sys.path.insert(0, str(import_root.resolve()))",
            "spec = importlib.util.find_spec('diffusers')",
            "if spec is None:",
            "  payload.update({'ok': False, 'module_found': False, 'importable': False, 'version': None, 'error': 'Module not found: diffusers'})",
            "  print(json.dumps(payload, ensure_ascii=False))",
            "  raise SystemExit(0)",
            "try:",
            "  diffusers_module = importlib.import_module('diffusers')",
            "except Exception as exc:",
            "  payload.update({'ok': False, 'module_found': True, 'importable': False, 'version': None, 'error': f'{type(exc).__name__}: {exc}'})",
            "  print(json.dumps(payload, ensure_ascii=False))",
            "  raise SystemExit(0)",
            "missing_symbols = [name for name in ('BrushNetModel', 'StableDiffusionBrushNetPipeline') if not hasattr(diffusers_module, name)]",
            "if missing_symbols:",
            "  payload.update({'ok': False, 'module_found': True, 'importable': True, 'version': getattr(diffusers_module, '__version__', None), 'error': 'BrushNet support is missing from diffusers: ' + ', '.join(missing_symbols)})",
            "else:",
            "  payload.update({'ok': True, 'module_found': True, 'importable': True, 'version': getattr(diffusers_module, '__version__', None), 'error': None})",
            "print(json.dumps(payload, ensure_ascii=False))",
        ]
    )

    completed = subprocess.run(
        [str(python_path), "-c", script, str(source_dir) if source_dir else ""],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    stdout = (completed.stdout or "").strip()
    if not stdout:
        payload = _build_brushnet_probe_payload(python_executable=str(python_path), source_dir=source_dir)
        payload.update(
            {
                "ok": False,
                "module_found": False,
                "importable": False,
                "version": None,
                "error": (completed.stderr or "").strip() or f"Probe process returned exit code {completed.returncode}",
                "python_version": None,
            }
        )
        return payload

    try:
        payload = json.loads(stdout.splitlines()[-1])
    except json.JSONDecodeError:
        payload = _build_brushnet_probe_payload(python_executable=str(python_path), source_dir=source_dir)
        payload.update(
            {
                "ok": False,
                "module_found": False,
                "importable": False,
                "version": None,
                "error": (completed.stderr or "").strip() or f"Invalid probe output: {stdout}",
                "python_version": None,
            }
        )
        return payload

    if completed.returncode not in {0} and payload.get("ok", False):
        payload["ok"] = False
        payload["error"] = payload.get("error") or f"Probe process returned exit code {completed.returncode}"
    return payload


def _summarize_brushnet_probe(probe: dict[str, Any]) -> tuple[bool, str]:
    if probe.get("ok"):
        version = probe.get("version")
        version_suffix = f" (version {version})" if version else ""
        source_suffix = f" with source dir {probe.get('source_dir')}" if probe.get("source_dir") else ""
        return True, f"BrushNet-enabled diffusers is importable via {probe.get('python_executable')}{version_suffix}{source_suffix}."

    if not probe.get("module_found"):
        return False, f"BrushNet is unavailable: {probe.get('error')}"

    return False, f"BrushNet support failed via {probe.get('python_executable')}: {probe.get('error')}"


def _run_mask_sidecar(
    *,
    python_executable: str,
    script_path: Path,
    image_path: Path,
    source_category: str,
    score_threshold: float = 0.45,
) -> tuple[dict[str, Any], np.ndarray]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        output_mask = tmp_root / "mask.png"
        output_json = tmp_root / "result.json"
        _run_sidecar(
            python_executable=python_executable,
            script_path=script_path,
            arguments=[
                "--image",
                str(image_path),
                "--output-mask",
                str(output_mask),
                "--output-json",
                str(output_json),
                "--category",
                source_category,
                "--score-threshold",
                str(score_threshold),
            ],
        )
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        return payload, read_mask(output_mask)


def _run_restore_sidecar(
    *,
    python_executable: str,
    script_path: Path,
    image_path: Path,
    mask: np.ndarray,
    prompt: str | None = None,
    negative_prompt: str | None = None,
    options: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        mask_path = tmp_root / "mask.png"
        output_image = tmp_root / "restored.png"
        output_json = tmp_root / "result.json"
        write_image(mask_path, mask)
        arguments = [
            "--image",
            str(image_path),
            "--mask",
            str(mask_path),
            "--output-image",
            str(output_image),
            "--output-json",
            str(output_json),
        ]
        if prompt:
            arguments.extend(["--prompt", prompt])
        if negative_prompt:
            arguments.extend(["--negative-prompt", negative_prompt])
        if options:
            arguments.extend(["--options-json", json.dumps(options, ensure_ascii=False)])
        _run_sidecar(
            python_executable=python_executable,
            script_path=script_path,
            arguments=arguments,
        )
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        return payload, read_image(output_image)


def _run_sidecar(*, python_executable: str, script_path: Path, arguments: list[str]) -> None:
    if not script_path.exists():
        raise ProviderUnavailableError(f"Provider sidecar script not found: {script_path}")

    try:
        completed = subprocess.run(
            [python_executable, str(script_path), *arguments],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        raise ProviderUnavailableError(f"Unable to launch sidecar Python: {python_executable}") from exc

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        details = stderr or stdout or f"exit code {completed.returncode}"
        raise ProviderUnavailableError(f"Sidecar failed: {details}")


def _describe_paddleocr_runtime() -> tuple[bool, str, dict[str, Any]]:
    if _module_available("paddleocr"):
        probe = probe_current_runtime("paddleocr", required_modules=["paddle"])
        available, note = summarize_probe("PaddleOCR", probe)
        return available, note, probe

    python_executable, note = resolve_paddleocr_sidecar_python()
    if not python_executable:
        probe = {
            "ok": False,
            "module_name": "paddleocr",
            "import_target": "paddleocr",
            "module_found": False,
            "importable": False,
            "version": None,
            "error": note,
            "python_executable": None,
        }
        return False, note, probe

    probe = probe_python_runtime(python_executable, "paddleocr", required_modules=["paddle"])
    available, summarized = summarize_probe("PaddleOCR", probe)
    return available, summarized, probe


def _describe_lama_runtime() -> tuple[bool, str, dict[str, Any]]:
    if _module_available("simple_lama_inpainting"):
        probe = probe_current_module("simple_lama_inpainting")
        available, note = summarize_probe("simple_lama_inpainting", probe)
        return available, note, probe
    if _module_available("simple_lama"):
        probe = probe_current_module("simple_lama")
        available, note = summarize_probe("simple_lama", probe)
        return available, note, probe

    configured_python = os.getenv("NO_WATERMAR_LAMA_PYTHON")
    if configured_python:
        path = Path(configured_python)
        if path.exists():
            primary_probe = probe_python_module(str(path), "simple_lama_inpainting")
            if primary_probe.get("ok"):
                available, note = summarize_probe("simple_lama_inpainting", primary_probe)
                return available, note, primary_probe

            legacy_probe = probe_python_module(str(path), "simple_lama")
            if legacy_probe.get("ok"):
                available, note = summarize_probe("simple_lama", legacy_probe)
                return available, note, legacy_probe
            return False, summarize_probe("simple_lama_inpainting", primary_probe)[1], primary_probe
        probe = {
            "ok": False,
            "module_name": "simple_lama_inpainting",
            "import_target": "simple_lama_inpainting",
            "module_found": False,
            "importable": False,
            "version": None,
            "error": f"NO_WATERMAR_LAMA_PYTHON points to a missing interpreter: {path}",
            "python_executable": str(path),
        }
        return False, str(probe["error"]), probe

    probe = {
        "ok": False,
        "module_name": "simple_lama_inpainting",
        "import_target": "simple_lama_inpainting",
        "module_found": False,
        "importable": False,
        "version": None,
        "error": "NO_WATERMAR_LAMA_PYTHON is not set.",
        "python_executable": None,
    }
    return False, str(probe["error"]), probe


def _describe_diffusers_runtime() -> tuple[bool, str, dict[str, Any]]:
    if _diffusers_module_available():
        probe = probe_current_module("diffusers")
        available, note = summarize_probe("diffusers", probe)
        return available, f"{note} Model id still comes from restore_options.model_id or NO_WATERMAR_DIFFUSERS_MODEL.", probe

    configured_python = os.getenv("NO_WATERMAR_DIFFUSERS_PYTHON")
    if configured_python:
        path = Path(configured_python)
        if path.exists():
            probe = probe_python_module(str(path), "diffusers")
            available, note = summarize_probe("diffusers", probe)
            if available:
                note = f"{note} Model id still comes from restore_options.model_id or NO_WATERMAR_DIFFUSERS_MODEL."
            return available, note, probe
        probe = {
            "ok": False,
            "module_name": "diffusers",
            "import_target": "diffusers",
            "module_found": False,
            "importable": False,
            "version": None,
            "error": f"NO_WATERMAR_DIFFUSERS_PYTHON points to a missing interpreter: {path}",
            "python_executable": str(path),
        }
        return False, str(probe["error"]), probe

    probe = {
        "ok": False,
        "module_name": "diffusers",
        "import_target": "diffusers",
        "module_found": False,
        "importable": False,
        "version": None,
        "error": "NO_WATERMAR_DIFFUSERS_PYTHON is not set.",
        "python_executable": None,
    }
    return False, str(probe["error"]), probe


def _describe_powerpaint_runtime() -> tuple[bool, str, dict[str, Any]]:
    if _powerpaint_module_available():
        probe = probe_current_module("powerpaint")
        available, note = summarize_probe("powerpaint", probe)
        if available:
            note = (
                f"{note} Checkpoint dir still comes from restore_options.checkpoint_dir or "
                "NO_WATERMAR_POWERPAINT_CHECKPOINT_DIR."
            )
        return available, note, probe

    configured_python = os.getenv("NO_WATERMAR_POWERPAINT_PYTHON")
    if configured_python:
        path = Path(configured_python)
        if path.exists():
            probe = probe_python_module(str(path), "powerpaint")
            available, note = summarize_probe("powerpaint", probe)
            if available:
                note = (
                    f"{note} Checkpoint dir still comes from restore_options.checkpoint_dir or "
                    "NO_WATERMAR_POWERPAINT_CHECKPOINT_DIR."
                )
            return available, note, probe
        probe = {
            "ok": False,
            "module_name": "powerpaint",
            "import_target": "powerpaint",
            "module_found": False,
            "importable": False,
            "version": None,
            "error": f"NO_WATERMAR_POWERPAINT_PYTHON points to a missing interpreter: {path}",
            "python_executable": str(path),
        }
        return False, str(probe["error"]), probe

    probe = {
        "ok": False,
        "module_name": "powerpaint",
        "import_target": "powerpaint",
        "module_found": False,
        "importable": False,
        "version": None,
        "error": "NO_WATERMAR_POWERPAINT_PYTHON is not set.",
        "python_executable": None,
    }
    return False, str(probe["error"]), probe


def _describe_brushnet_runtime() -> tuple[bool, str, dict[str, Any]]:
    current_probe = _probe_current_brushnet_runtime()
    available, note = _summarize_brushnet_probe(current_probe)
    if available:
        note = (
            f"{note} BrushNet model path still comes from restore_options.brushnet_model_path or "
            "NO_WATERMAR_BRUSHNET_MODEL."
        )
        return available, note, current_probe

    configured_python = os.getenv("NO_WATERMAR_BRUSHNET_PYTHON")
    source_dir = _resolve_brushnet_source_dir_from_env()
    if configured_python:
        path = Path(configured_python)
        if path.exists():
            probe = _probe_python_brushnet_runtime(str(path), source_dir=source_dir)
            available, note = _summarize_brushnet_probe(probe)
            if available:
                note = (
                    f"{note} BrushNet model path still comes from restore_options.brushnet_model_path or "
                    "NO_WATERMAR_BRUSHNET_MODEL."
                )
            return available, note, probe

        probe = _build_brushnet_probe_payload(python_executable=str(path), source_dir=source_dir)
        probe.update(
            {
                "ok": False,
                "module_found": False,
                "importable": False,
                "version": None,
                "error": f"NO_WATERMAR_BRUSHNET_PYTHON points to a missing interpreter: {path}",
                "python_version": None,
            }
        )
        return False, str(probe["error"]), probe

    probe = _build_brushnet_probe_payload(python_executable=None, source_dir=source_dir)
    probe.update(
        {
            "ok": False,
            "module_found": False,
            "importable": False,
            "version": None,
            "error": "NO_WATERMAR_BRUSHNET_PYTHON is not set.",
            "python_version": None,
        }
    )
    return False, str(probe["error"]), probe
