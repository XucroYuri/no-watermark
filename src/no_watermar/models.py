from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ImageCategory = str


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            return value
    return value


@dataclass(slots=True)
class ScanItem:
    source_path: Path
    relative_path: Path
    width: int
    height: int
    category: ImageCategory

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "relative_path": str(self.relative_path),
            "width": self.width,
            "height": self.height,
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ScanItem":
        source_path = payload.get("source_path")
        relative_path = payload.get("relative_path")
        width = payload.get("width")
        height = payload.get("height")
        category = payload.get("category")

        if not isinstance(source_path, str) or not source_path:
            raise ValueError("ScanItem field 'source_path' must be a non-empty string.")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError("ScanItem field 'relative_path' must be a non-empty string.")
        if not isinstance(width, int):
            raise ValueError("ScanItem field 'width' must be an integer.")
        if not isinstance(height, int):
            raise ValueError("ScanItem field 'height' must be an integer.")
        if not isinstance(category, str) or not category:
            raise ValueError("ScanItem field 'category' must be a non-empty string.")

        return cls(
            source_path=Path(source_path),
            relative_path=Path(relative_path),
            width=width,
            height=height,
            category=category,
        )


@dataclass(slots=True)
class DetectionResult:
    category: ImageCategory
    mask_nonzero: int
    confidence: float
    boxes: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "mask_nonzero": self.mask_nonzero,
            "confidence": round(self.confidence, 4),
            "boxes": _json_safe(self.boxes),
        }


@dataclass(slots=True)
class ProcessResult:
    item: ScanItem
    status: str
    note: str = ""
    mask_provider: str = ""
    restore_provider: str = ""
    mask_path: Path | None = None
    overlay_path: Path | None = None
    restored_path: Path | None = None
    detection: DetectionResult | None = None
    mask_meta: dict[str, Any] = field(default_factory=dict)
    restore_meta: dict[str, Any] = field(default_factory=dict)
    mask_latency_ms: float | None = None
    restore_latency_ms: float | None = None

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
            "detection": self.detection.to_dict() if self.detection else None,
            "mask_meta": _json_safe(self.mask_meta),
            "restore_meta": _json_safe(self.restore_meta),
            "mask_latency_ms": None if self.mask_latency_ms is None else round(self.mask_latency_ms, 3),
            "restore_latency_ms": None if self.restore_latency_ms is None else round(self.restore_latency_ms, 3),
        }
