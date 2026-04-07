from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


BenchmarkDatasetId = str


@dataclass(slots=True)
class BenchmarkDatasetItem:
    item_id: str
    dataset_id: BenchmarkDatasetId
    source_path: Path
    input_path: Path
    relative_path: Path
    width: int
    height: int
    source_category: str
    benchmark_category: str
    prompt: str
    prompt_path: Path
    seed_mask_path: Path | None = None
    seed_overlay_path: Path | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "dataset_id": self.dataset_id,
            "source_path": str(self.source_path),
            "input_path": str(self.input_path),
            "relative_path": str(self.relative_path),
            "width": self.width,
            "height": self.height,
            "source_category": self.source_category,
            "benchmark_category": self.benchmark_category,
            "prompt": self.prompt,
            "prompt_path": str(self.prompt_path),
            "seed_mask_path": str(self.seed_mask_path) if self.seed_mask_path else None,
            "seed_overlay_path": str(self.seed_overlay_path) if self.seed_overlay_path else None,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkDatasetItem":
        return cls(
            item_id=str(data["item_id"]),
            dataset_id=str(data["dataset_id"]),
            source_path=Path(data["source_path"]),
            input_path=Path(data["input_path"]),
            relative_path=Path(data["relative_path"]),
            width=int(data["width"]),
            height=int(data["height"]),
            source_category=str(data["source_category"]),
            benchmark_category=str(data["benchmark_category"]),
            prompt=str(data["prompt"]),
            prompt_path=Path(data["prompt_path"]),
            seed_mask_path=Path(data["seed_mask_path"]) if data.get("seed_mask_path") else None,
            seed_overlay_path=Path(data["seed_overlay_path"]) if data.get("seed_overlay_path") else None,
            notes=dict(data.get("notes") or {}),
        )


@dataclass(slots=True)
class MaskResult:
    provider_name: str
    mask: np.ndarray
    confidence: float
    boxes: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def mask_nonzero(self) -> int:
        return int(np.count_nonzero(self.mask))

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "mask_nonzero": self.mask_nonzero,
            "confidence": round(self.confidence, 4),
            "boxes": self.boxes,
            "latency_ms": round(self.latency_ms, 3),
            "meta": self.meta,
        }


@dataclass(slots=True)
class RestoreResult:
    provider_name: str
    restored: np.ndarray
    latency_ms: float = 0.0
    peak_vram_mb: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "latency_ms": round(self.latency_ms, 3),
            "peak_vram_mb": None if self.peak_vram_mb is None else round(self.peak_vram_mb, 3),
            "meta": self.meta,
        }


@dataclass(slots=True)
class BenchmarkItemResult:
    item: BenchmarkDatasetItem
    status: str
    note: str = ""
    mask_provider: str = ""
    restore_provider: str = ""
    mask_path: Path | None = None
    overlay_path: Path | None = None
    restored_path: Path | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    manual_score: str | None = None
    mask_result: MaskResult | None = None
    restore_result: RestoreResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "item": self.item.to_dict(),
            "status": self.status,
            "note": self.note,
            "mask_provider": self.mask_provider,
            "restore_provider": self.restore_provider,
            "mask_path": str(self.mask_path) if self.mask_path else None,
            "overlay_path": str(self.overlay_path) if self.overlay_path else None,
            "restored_path": str(self.restored_path) if self.restored_path else None,
            "metrics": self.metrics,
            "manual_score": self.manual_score,
            "mask_result": self.mask_result.to_dict() if self.mask_result else None,
            "restore_result": self.restore_result.to_dict() if self.restore_result else None,
        }
