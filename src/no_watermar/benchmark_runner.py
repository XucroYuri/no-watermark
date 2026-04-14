from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .benchmark_dataset import DATASET_COVER, DATASET_REGULAR, load_dataset_items
from .benchmark_metrics import BENCHMARK_METRIC_KEYS, OCR_RESIDUAL_METRIC_KEYS
from .benchmark_models import BenchmarkItemResult, MaskResult, RestoreResult
from .benchmark_paddleocr_session import OCR_SESSION_MODE_AUTO, create_paddleocr_execution_context
from .benchmark_restore_session import (
    RESTORE_RESOLVED_MODE_UNAVAILABLE,
    PowerPaintExecutionContext,
    create_powerpaint_execution_context,
)
from .benchmark_providers import ProviderUnavailableError, create_mask_provider, create_restore_provider
from .benchmark_scoring import score_ocr_residual
from .detector import build_overlay
from .io_utils import ensure_dir, read_image, write_image, write_json


def run_benchmark(
    benchmark_root: Path,
    *,
    dataset_id: str = DATASET_REGULAR,
    mask_provider_name: str = "rule_based_roi",
    restore_provider_name: str = "telea",
    limit: int | None = None,
    ocr_session_mode: str = OCR_SESSION_MODE_AUTO,
    restore_prompt: str | None = None,
    restore_negative_prompt: str | None = None,
    restore_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    benchmark_root = benchmark_root.resolve()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_root = ensure_dir(benchmark_root / "runs" / run_id)
    reports_root = ensure_dir(run_root / "reports")

    mask_provider = create_mask_provider(mask_provider_name)
    restore_provider = create_restore_provider(restore_provider_name)
    dataset_ids = _resolve_dataset_ids(dataset_id)
    dataset_items_by_id: dict[str, list[Any]] = {}
    for current_dataset_id in dataset_ids:
        dataset_items = load_dataset_items(benchmark_root, current_dataset_id)
        if limit is not None:
            dataset_items = dataset_items[:limit]
        if not dataset_items:
            raise ValueError(
                f"Dataset '{current_dataset_id}' contains no benchmark items. "
                "Run prepare with a populated input root or adjust the requested dataset/limit."
            )
        dataset_items_by_id[current_dataset_id] = dataset_items

    ocr_context = create_paddleocr_execution_context(ocr_session_mode)
    restore_execution_context = _build_restore_execution_context(
        restore_provider_name=restore_provider_name,
        restore_options=restore_options,
    )

    all_results: list[BenchmarkItemResult] = []
    dataset_summaries: list[dict[str, Any]] = []

    try:
        for current_dataset_id in dataset_ids:
            dataset_items = dataset_items_by_id[current_dataset_id]

            dataset_root = ensure_dir(run_root / current_dataset_id / f"{mask_provider_name}__{restore_provider_name}")
            masks_root = ensure_dir(dataset_root / "masks")
            overlays_root = ensure_dir(dataset_root / "overlays")
            restored_root = ensure_dir(dataset_root / "restored")

            dataset_results: list[BenchmarkItemResult] = []
            for item in dataset_items:
                image = read_image(item.input_path)
                mask_path = masks_root / item.relative_path.with_suffix(".png")
                overlay_path = overlays_root / item.relative_path.with_suffix(".jpg")
                provider_hints = {
                    "seed_mask_path": item.seed_mask_path,
                    "paddleocr_context": ocr_context,
                }

                try:
                    mask_result = mask_provider.detect(item, image, hints=provider_hints)
                except ProviderUnavailableError as exc:
                    empty_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    write_image(mask_path, empty_mask)
                    write_image(overlay_path, build_overlay(image, empty_mask))
                    dataset_results.append(
                        BenchmarkItemResult(
                            item=item,
                            status="mask_provider_unavailable",
                            note=str(exc),
                            mask_provider=mask_provider_name,
                            restore_provider=restore_provider_name,
                            mask_path=mask_path,
                            overlay_path=overlay_path,
                            metrics=_build_mask_metrics(image, empty_mask),
                        )
                    )
                    continue
                except Exception as exc:
                    empty_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    write_image(mask_path, empty_mask)
                    write_image(overlay_path, build_overlay(image, empty_mask))
                    dataset_results.append(
                        BenchmarkItemResult(
                            item=item,
                            status="mask_provider_error",
                            note=str(exc),
                            mask_provider=mask_provider_name,
                            restore_provider=restore_provider_name,
                            mask_path=mask_path,
                            overlay_path=overlay_path,
                            metrics=_build_mask_metrics(image, empty_mask),
                        )
                    )
                    continue

                write_image(mask_path, mask_result.mask)
                write_image(overlay_path, build_overlay(image, mask_result.mask))

                if mask_result.mask_nonzero == 0:
                    dataset_results.append(
                        BenchmarkItemResult(
                            item=item,
                            status="no_mask",
                            note="mask provider returned an empty mask",
                            mask_provider=mask_provider_name,
                            restore_provider=restore_provider_name,
                            mask_path=mask_path,
                            overlay_path=overlay_path,
                            metrics=_build_mask_metrics(image, mask_result.mask, mask_result=mask_result),
                            mask_result=mask_result,
                        )
                    )
                    continue

                try:
                    restore_meta = {"detection_boxes": list(mask_result.boxes)}
                    if restore_options:
                        restore_meta["restore_options"] = dict(restore_options)
                    if restore_execution_context is not None:
                        restore_meta["powerpaint_context"] = restore_execution_context
                    restore_result = restore_provider.restore(
                        item,
                        image,
                        mask_result.mask,
                        prompt=restore_prompt or item.prompt,
                        negative_prompt=restore_negative_prompt,
                        meta=restore_meta,
                    )
                except ProviderUnavailableError as exc:
                    dataset_results.append(
                        BenchmarkItemResult(
                            item=item,
                            status="restore_provider_unavailable",
                            note=str(exc),
                            mask_provider=mask_provider_name,
                            restore_provider=restore_provider_name,
                            mask_path=mask_path,
                            overlay_path=overlay_path,
                            metrics=_build_mask_metrics(image, mask_result.mask, mask_result=mask_result),
                            mask_result=mask_result,
                        )
                    )
                    continue
                except Exception as exc:
                    dataset_results.append(
                        BenchmarkItemResult(
                            item=item,
                            status="restore_provider_error",
                            note=str(exc),
                            mask_provider=mask_provider_name,
                            restore_provider=restore_provider_name,
                            mask_path=mask_path,
                            overlay_path=overlay_path,
                            metrics=_build_mask_metrics(image, mask_result.mask, mask_result=mask_result),
                            mask_result=mask_result,
                        )
                    )
                    continue

                restored_path = restored_root / item.relative_path.with_suffix(".jpg")
                write_image(restored_path, restore_result.restored)
                ocr_residual = score_ocr_residual(
                    item,
                    restore_result.restored,
                    hints={"paddleocr_context": ocr_context},
                )
                restore_result.meta["ocr_residual"] = dict(ocr_residual["meta"])

                default_status = "masked_only" if restore_provider_name == "noop" else "restored"
                status = str((restore_result.meta or {}).get("result_status") or default_status)
                note = str((restore_result.meta or {}).get("result_note") or "")
                dataset_results.append(
                    BenchmarkItemResult(
                        item=item,
                        status=status,
                        note=note,
                        mask_provider=mask_provider_name,
                        restore_provider=restore_provider_name,
                        mask_path=mask_path,
                        overlay_path=overlay_path,
                        restored_path=restored_path,
                        metrics=_build_metrics(
                            image,
                            restore_result.restored,
                            mask_result.mask,
                            mask_result=mask_result,
                            restore_result=restore_result,
                            ocr_residual_metrics=ocr_residual["metrics"],
                            ocr_residual_meta=ocr_residual["meta"],
                        ),
                        mask_result=mask_result,
                        restore_result=restore_result,
                    )
                )

            report_stem = f"{current_dataset_id}_{mask_provider_name}__{restore_provider_name}"
            dataset_summary = {
                "run_id": run_id,
                "dataset_id": current_dataset_id,
                "mask_provider": mask_provider_name,
                "restore_provider": restore_provider_name,
                "ocr_session_mode_requested": ocr_session_mode,
                "ocr_session_mode_resolved": ocr_context.resolved_mode,
                "ocr_session_note": ocr_context.note,
                "restore_session_mode_requested": (
                    restore_execution_context.requested_mode if restore_execution_context is not None else None
                ),
                "restore_session_mode_resolved": (
                    restore_execution_context.resolved_mode if restore_execution_context is not None else None
                ),
                "restore_session_note": restore_execution_context.note if restore_execution_context is not None else None,
                "restore_prompt": restore_prompt,
                "restore_negative_prompt": restore_negative_prompt,
                "restore_options": dict(restore_options or {}),
                "item_count": len(dataset_items),
                "status_counts": dict(Counter(result.status for result in dataset_results)),
                "results": [result.to_dict() for result in dataset_results],
                "output_root": str(dataset_root),
            }
            write_json(reports_root / f"{report_stem}.json", dataset_summary)
            _write_csv_report(reports_root / f"{report_stem}.csv", dataset_results)
            dataset_summaries.append(
                {
                    "dataset_id": current_dataset_id,
                    "item_count": len(dataset_items),
                    "status_counts": dataset_summary["status_counts"],
                    "report_json": str(reports_root / f"{report_stem}.json"),
                    "report_csv": str(reports_root / f"{report_stem}.csv"),
                }
            )
            all_results.extend(dataset_results)
    finally:
        ocr_context.close()
        if isinstance(restore_execution_context, PowerPaintExecutionContext):
            restore_execution_context.close()

    summary = {
        "run_id": run_id,
        "benchmark_root": str(benchmark_root),
        "dataset_id": dataset_id,
        "resolved_dataset_ids": dataset_ids,
        "mask_provider": mask_provider_name,
        "restore_provider": restore_provider_name,
        "limit": limit,
        "ocr_session_mode_requested": ocr_session_mode,
        "ocr_session_mode_resolved": ocr_context.resolved_mode,
        "ocr_session_note": ocr_context.note,
        "restore_session_mode_requested": (
            restore_execution_context.requested_mode if restore_execution_context is not None else None
        ),
        "restore_session_mode_resolved": (
            restore_execution_context.resolved_mode if restore_execution_context is not None else None
        ),
        "restore_session_note": restore_execution_context.note if restore_execution_context is not None else None,
        "restore_prompt": restore_prompt,
        "restore_negative_prompt": restore_negative_prompt,
        "restore_options": dict(restore_options or {}),
        "status_counts": dict(Counter(result.status for result in all_results)),
        "dataset_summaries": dataset_summaries,
        "reports_root": str(reports_root),
    }
    write_json(reports_root / "summary.json", summary)
    return summary


def _resolve_dataset_ids(dataset_id: str) -> list[str]:
    if dataset_id == "all":
        return [DATASET_REGULAR, DATASET_COVER]
    if dataset_id not in {DATASET_REGULAR, DATASET_COVER}:
        raise ValueError(f"Unsupported dataset_id: {dataset_id}")
    return [dataset_id]


def _build_mask_metrics(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    mask_result: MaskResult | None = None,
) -> dict[str, Any]:
    height, width = image.shape[:2]
    mask_nonzero = int(np.count_nonzero(mask))
    metrics = _empty_metrics()
    metrics.update(
        {
            "mask_nonzero": mask_nonzero,
            "mask_ratio": round(float(mask_nonzero) / float(max(1, width * height)), 6),
            "changed_nonzero": 0,
            "mean_abs_diff": 0.0,
            "edge_delta": 0.0,
        }
    )
    if mask_result is not None:
        metrics["mask_latency_ms"] = round(float(mask_result.latency_ms), 3)
    return metrics


def _build_metrics(
    original: np.ndarray,
    restored: np.ndarray,
    mask: np.ndarray,
    *,
    mask_result: MaskResult | None = None,
    restore_result: RestoreResult | None = None,
    ocr_residual_metrics: dict[str, Any] | None = None,
    ocr_residual_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    height, width = original.shape[:2]
    mask_nonzero = int(np.count_nonzero(mask))
    metrics = _empty_metrics()
    metrics.update(
        {
            "mask_nonzero": mask_nonzero,
            "mask_ratio": round(float(mask_nonzero) / float(max(1, width * height)), 6),
        }
    )

    if original.shape[:2] == restored.shape[:2]:
        diff = cv2.absdiff(original, restored)
        changed_nonzero = int(np.count_nonzero(np.any(diff > 0, axis=2)))
        mean_abs_diff = float(diff.mean())
        edge_original = cv2.Canny(original, 80, 160)
        edge_restored = cv2.Canny(restored, 80, 160)
        edge_delta = float(cv2.absdiff(edge_original, edge_restored).mean())
        metrics.update(
            {
                "changed_nonzero": changed_nonzero,
                "mean_abs_diff": round(mean_abs_diff, 6),
                "edge_delta": round(edge_delta, 6),
            }
        )

    if mask_result is not None:
        metrics["mask_latency_ms"] = round(float(mask_result.latency_ms), 3)
    if restore_result is not None:
        metrics["restore_latency_ms"] = round(float(restore_result.latency_ms), 3)
    if ocr_residual_metrics:
        metrics.update(ocr_residual_metrics)
    else:
        metrics.update({key: None for key in OCR_RESIDUAL_METRIC_KEYS})
    if ocr_residual_meta and ocr_residual_meta.get("latency_ms") is not None:
        metrics["ocr_residual_latency_ms"] = round(float(ocr_residual_meta["latency_ms"]), 3)
    return metrics


def _empty_metrics() -> dict[str, Any]:
    return {key: None for key in BENCHMARK_METRIC_KEYS}


def _write_csv_report(path: Path, results: list[BenchmarkItemResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "item_id",
                "dataset_id",
                "relative_path",
                "status",
                "mask_provider",
                "restore_provider",
                "mask_nonzero",
                "mask_ratio",
                "changed_nonzero",
                "mean_abs_diff",
                "edge_delta",
                "mask_latency_ms",
                "restore_latency_ms",
                "restore_session_mode",
                "ocr_residual_latency_ms",
                "ocr_residual_hits",
                "ocr_residual_score",
                "ocr_residual_max_score",
                "mask_session_mode",
                "ocr_residual_session_mode",
                "mask_path",
                "overlay_path",
                "restored_path",
                "note",
            ],
        )
        writer.writeheader()
        for result in results:
            metrics = result.metrics
            writer.writerow(
                {
                    "item_id": result.item.item_id,
                    "dataset_id": result.item.dataset_id,
                    "relative_path": str(result.item.relative_path),
                    "status": result.status,
                    "mask_provider": result.mask_provider,
                    "restore_provider": result.restore_provider,
                    "mask_nonzero": metrics.get("mask_nonzero", 0),
                    "mask_ratio": metrics.get("mask_ratio", 0.0),
                    "changed_nonzero": metrics.get("changed_nonzero", 0),
                    "mean_abs_diff": metrics.get("mean_abs_diff", 0.0),
                    "edge_delta": metrics.get("edge_delta", 0.0),
                    "mask_latency_ms": metrics.get("mask_latency_ms", ""),
                    "restore_latency_ms": metrics.get("restore_latency_ms", ""),
                    "restore_session_mode": ((result.restore_result.meta if result.restore_result else {}) or {}).get(
                        "session_mode",
                        "",
                    ),
                    "ocr_residual_latency_ms": metrics.get("ocr_residual_latency_ms", ""),
                    "ocr_residual_hits": metrics.get("ocr_residual_hits", ""),
                    "ocr_residual_score": metrics.get("ocr_residual_score", ""),
                    "ocr_residual_max_score": metrics.get("ocr_residual_max_score", ""),
                    "mask_session_mode": ((result.mask_result.meta if result.mask_result else {}) or {}).get("session_mode", ""),
                    "ocr_residual_session_mode": (
                        ((result.restore_result.meta if result.restore_result else {}) or {}).get("ocr_residual", {}) or {}
                    ).get("session_mode", ""),
                    "mask_path": str(result.mask_path) if result.mask_path else "",
                    "overlay_path": str(result.overlay_path) if result.overlay_path else "",
                    "restored_path": str(result.restored_path) if result.restored_path else "",
                    "note": result.note,
                }
            )


def _build_restore_execution_context(
    *,
    restore_provider_name: str,
    restore_options: dict[str, Any] | None,
) -> PowerPaintExecutionContext | None:
    if restore_provider_name != "powerpaint_v2_1":
        return None

    context = create_powerpaint_execution_context(restore_options)
    if context.resolved_mode == RESTORE_RESOLVED_MODE_UNAVAILABLE:
        return context
    return context
