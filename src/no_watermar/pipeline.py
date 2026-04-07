from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from .batch_run import (
    collect_pending_batch_run_items,
    create_batch_run_artifacts,
    finalize_batch_run,
    load_batch_run_manifest,
    load_batch_run_summary,
    mark_batch_run_resuming,
    update_batch_run_progress,
)
from .benchmark_models import BenchmarkDatasetItem, MaskResult
from .benchmark_paddleocr_session import (
    OCR_RESOLVED_MODE_UNAVAILABLE,
    OCR_SESSION_MODE_AUTO,
    PaddleOCRExecutionContext,
    create_paddleocr_execution_context,
)
from .benchmark_restore_session import (
    RESTORE_RESOLVED_MODE_UNAVAILABLE,
    PowerPaintExecutionContext,
    create_powerpaint_execution_context,
)
from .benchmark_providers import create_mask_provider, create_restore_provider
from .detector import build_overlay
from .io_utils import ensure_dir, read_image, write_image
from .models import DetectionResult, ProcessResult, ScanItem
from .scanner import build_scan_items


def run_pipeline(
    input_root: Path,
    output_root: Path,
    recursive: bool = True,
    limit: int | None = None,
    scan_only: bool = False,
    items: list[ScanItem] | None = None,
    mask_provider_name: str = "rule_based_roi",
    restore_provider_name: str = "telea",
    ocr_session_mode: str = OCR_SESSION_MODE_AUTO,
    restore_prompt: str | None = None,
    restore_negative_prompt: str | None = None,
    restore_options: dict[str, Any] | None = None,
    summary_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    resolved_items = list(items) if items is not None else build_scan_items(input_root, recursive=recursive, limit=limit)
    runtime = _build_pipeline_runtime(
        mask_provider_name=mask_provider_name,
        restore_provider_name=restore_provider_name,
        ocr_session_mode=ocr_session_mode,
        restore_prompt=restore_prompt,
        restore_negative_prompt=restore_negative_prompt,
        restore_options=restore_options,
    )
    resolved_summary_context = _merge_runtime_summary_context(summary_context, runtime)
    _, summary, run_paths = create_batch_run_artifacts(
        runs_root=output_root,
        run_id=run_id,
        input_root=input_root,
        recursive=recursive,
        limit=limit,
        scan_only=scan_only,
        items=resolved_items,
        summary_context=resolved_summary_context,
    )
    try:
        return _run_items_into_summary(
            summary=summary,
            run_root=run_paths["run_root"],
            items=resolved_items,
            scan_only=scan_only,
            mask_provider=runtime["mask_provider"],
            restore_provider=runtime["restore_provider"],
            provider_hints=runtime["provider_hints"],
        )
    finally:
        _close_pipeline_runtime(runtime)


def resume_pipeline(
    summary: dict[str, Any],
    *,
    summary_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary_path = Path(str(summary.get("summary_json") or summary.get("report_json"))).resolve()
    loaded_summary = load_batch_run_summary(summary_path)
    manifest = load_batch_run_manifest(Path(str(loaded_summary["manifest_json"])))
    pending_items = collect_pending_batch_run_items(loaded_summary, manifest)
    resume_command = str((summary_context or {}).get("command", "batch resume"))
    resume_mode = str((summary_context or {}).get("mode", "resume"))
    loaded_summary["image_count"] = int(manifest.get("item_count", loaded_summary.get("image_count", 0)))
    loaded_summary["pending_item_count"] = len(pending_items)
    initial_completed_item_count = int(loaded_summary.get("completed_item_count", len(loaded_summary.get("results", []))))
    initial_run_status = str(loaded_summary.get("run_status", "unknown"))
    initial_command = loaded_summary.get("command")
    initial_mode = loaded_summary.get("mode")
    if not pending_items:
        loaded_summary["command"] = resume_command
        loaded_summary["mode"] = resume_mode
        loaded_summary["resumed_item_count"] = 0
        loaded_summary["resumed_from_command"] = initial_command
        loaded_summary["resumed_from_mode"] = initial_mode
        loaded_summary["resumed_from_run_status"] = initial_run_status
        loaded_summary["run_status"] = "completed"
        warnings = loaded_summary.setdefault("warnings", [])
        if "No pending items remain for this run." not in warnings:
            warnings.append("No pending items remain for this run.")
        finalize_batch_run(loaded_summary)
        _write_csv_report(Path(str(loaded_summary["report_csv"])), loaded_summary.get("results", []))
        return loaded_summary

    runtime = _build_pipeline_runtime(
        mask_provider_name=str(loaded_summary.get("mask_provider") or "rule_based_roi"),
        restore_provider_name=str(loaded_summary.get("restore_provider") or "telea"),
        ocr_session_mode=str(loaded_summary.get("ocr_session_mode_requested") or OCR_SESSION_MODE_AUTO),
        restore_prompt=loaded_summary.get("restore_prompt"),
        restore_negative_prompt=loaded_summary.get("restore_negative_prompt"),
        restore_options=loaded_summary.get("restore_options") or {},
    )
    mark_batch_run_resuming(loaded_summary, command=resume_command, mode=resume_mode)
    if summary_context:
        for key, value in summary_context.items():
            if key in {"command", "mode"}:
                continue
            loaded_summary[key] = value
    loaded_summary.update(
        {
            "mask_provider": runtime["mask_provider_name"],
            "restore_provider": runtime["restore_provider_name"],
            "ocr_session_mode_requested": runtime["ocr_session_mode_requested"],
            "ocr_session_mode_resolved": runtime["ocr_session_mode_resolved"],
            "restore_session_mode_requested": runtime["restore_session_mode_requested"],
            "restore_session_mode_resolved": runtime["restore_session_mode_resolved"],
            "restore_session_note": runtime["restore_session_note"],
            "restore_prompt": runtime["restore_prompt"],
            "restore_negative_prompt": runtime["restore_negative_prompt"],
            "restore_options": dict(runtime["restore_options"]),
        }
    )
    warnings = loaded_summary.setdefault("warnings", [])
    for warning in runtime["warnings"]:
        if warning not in warnings:
            warnings.append(warning)
    try:
        resumed_summary = _run_items_into_summary(
            summary=loaded_summary,
            run_root=Path(str(loaded_summary["run_root"])),
            items=pending_items,
            scan_only=bool(loaded_summary.get("scan_only", False)),
            mask_provider=runtime["mask_provider"],
            restore_provider=runtime["restore_provider"],
            provider_hints=runtime["provider_hints"],
        )
    finally:
        _close_pipeline_runtime(runtime)
    resumed_summary["resumed_item_count"] = int(resumed_summary.get("completed_item_count", 0)) - initial_completed_item_count
    resumed_summary["resumed_from_run_status"] = initial_run_status
    return resumed_summary


def _run_items_into_summary(
    *,
    summary: dict[str, Any],
    run_root: Path,
    items: list[ScanItem],
    scan_only: bool,
    mask_provider: Any,
    restore_provider: Any,
    provider_hints: dict[str, Any],
) -> dict[str, Any]:
    resolved_run_root = ensure_dir(run_root.resolve())
    masks_root = ensure_dir(resolved_run_root / "masks")
    overlays_root = ensure_dir(resolved_run_root / "overlays")
    restored_root = ensure_dir(resolved_run_root / "restored")
    reports_root = ensure_dir(resolved_run_root / "reports")

    try:
        for item in items:
            result = _process_pipeline_item(
                item=item,
                scan_only=scan_only,
                masks_root=masks_root,
                overlays_root=overlays_root,
                restored_root=restored_root,
                mask_provider=mask_provider,
                restore_provider=restore_provider,
                provider_hints=provider_hints,
                restore_prompt=summary.get("restore_prompt"),
                restore_negative_prompt=summary.get("restore_negative_prompt"),
                restore_options=summary.get("restore_options") or {},
            )
            update_batch_run_progress(summary, result)
    except Exception as exc:
        summary["run_status"] = "interrupted"
        warnings = summary.setdefault("warnings", [])
        warning = f"Run interrupted: {type(exc).__name__}: {exc}"
        if warning not in warnings:
            warnings.append(warning)
        finalize_batch_run(summary, run_status="interrupted")
        _write_csv_report(Path(str(summary["report_csv"])), summary.get("results", []))
        raise

    finalize_batch_run(summary)
    _write_csv_report(reports_root / "report.csv", summary.get("results", []))
    return summary


def _process_pipeline_item(
    *,
    item: ScanItem,
    scan_only: bool,
    masks_root: Path,
    overlays_root: Path,
    restored_root: Path,
    mask_provider: Any,
    restore_provider: Any,
    provider_hints: dict[str, Any],
    restore_prompt: str | None,
    restore_negative_prompt: str | None,
    restore_options: dict[str, Any],
) -> ProcessResult:
    image = read_image(item.source_path)
    relative_png = item.relative_path.with_suffix(".png")
    relative_jpg = item.relative_path.with_suffix(".jpg")
    mask_provider_name = getattr(mask_provider, "name", "")
    restore_provider_name = getattr(restore_provider, "name", "")

    if scan_only:
        return ProcessResult(
            item=item,
            status="scanned",
            note="scan_only",
            mask_provider=mask_provider_name,
            restore_provider=restore_provider_name,
        )

    if item.category == "cover_heavy":
        return ProcessResult(
            item=item,
            status="skipped_cover",
            note="heavy watermark cover",
            mask_provider=mask_provider_name,
            restore_provider=restore_provider_name,
        )

    benchmark_item = _build_batch_benchmark_item(item)
    mask_result = mask_provider.detect(benchmark_item, image, hints=provider_hints)
    detection = _mask_result_to_detection(item, mask_result)
    mask_path = masks_root / relative_png
    overlay_path = overlays_root / relative_jpg
    restored_path = restored_root / relative_jpg

    write_image(mask_path, mask_result.mask)
    write_image(overlay_path, build_overlay(image, mask_result.mask))

    if mask_result.mask_nonzero == 0:
        return ProcessResult(
            item=item,
            status="no_mask",
            note="mask provider produced an empty mask",
            mask_provider=mask_result.provider_name,
            restore_provider=restore_provider_name,
            mask_path=mask_path,
            overlay_path=overlay_path,
            detection=detection,
            mask_meta=dict(mask_result.meta),
            mask_latency_ms=mask_result.latency_ms,
        )

    restore_meta = dict(mask_result.meta)
    restore_meta["detection_boxes"] = list(mask_result.boxes)
    if restore_options:
        restore_meta["restore_options"] = dict(restore_options)
    powerpaint_context = provider_hints.get("powerpaint_context")
    if powerpaint_context is not None:
        restore_meta["powerpaint_context"] = powerpaint_context
    restore_result = restore_provider.restore(
        benchmark_item,
        image,
        mask_result.mask,
        prompt=restore_prompt,
        negative_prompt=restore_negative_prompt,
        meta=restore_meta,
    )
    result_status = str((restore_result.meta or {}).get("result_status") or "restored")
    result_note = str((restore_result.meta or {}).get("result_note") or "")
    write_image(restored_path, restore_result.restored)
    return ProcessResult(
        item=item,
        status=result_status,
        note=result_note,
        mask_provider=mask_result.provider_name,
        restore_provider=restore_result.provider_name,
        mask_path=mask_path,
        overlay_path=overlay_path,
        restored_path=restored_path,
        detection=detection,
        mask_meta=dict(mask_result.meta),
        restore_meta=dict(restore_result.meta),
        mask_latency_ms=mask_result.latency_ms,
        restore_latency_ms=restore_result.latency_ms,
    )


def _write_csv_report(path: Path, results: list[ProcessResult] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_path",
                "relative_path",
                "category",
                "status",
                "note",
                "mask_provider",
                "restore_provider",
                "mask_nonzero",
                "confidence",
                "mask_latency_ms",
                "restore_latency_ms",
                "restore_session_mode",
                "mask_path",
                "overlay_path",
                "restored_path",
            ],
        )
        writer.writeheader()
        for result in results:
            normalized = result.to_dict() if isinstance(result, ProcessResult) else result
            if not isinstance(normalized, dict):
                continue
            item = normalized.get("item") or {}
            detection = normalized.get("detection") or {}
            writer.writerow(
                {
                    "source_path": str(item.get("source_path") or ""),
                    "relative_path": str(item.get("relative_path") or ""),
                    "category": str(item.get("category") or ""),
                    "status": str(normalized.get("status") or ""),
                    "note": str(normalized.get("note") or ""),
                    "mask_provider": str(normalized.get("mask_provider") or ""),
                    "restore_provider": str(normalized.get("restore_provider") or ""),
                    "mask_nonzero": int(detection.get("mask_nonzero", 0) or 0),
                    "confidence": f"{float(detection.get('confidence', 0.0) or 0.0):.4f}",
                    "mask_latency_ms": (
                        f"{float(normalized.get('mask_latency_ms', 0.0) or 0.0):.3f}"
                        if normalized.get("mask_latency_ms") is not None
                        else ""
                    ),
                    "restore_latency_ms": (
                        f"{float(normalized.get('restore_latency_ms', 0.0) or 0.0):.3f}"
                        if normalized.get("restore_latency_ms") is not None
                        else ""
                    ),
                    "restore_session_mode": str((normalized.get("restore_meta") or {}).get("session_mode") or ""),
                    "mask_path": str(normalized.get("mask_path") or ""),
                    "overlay_path": str(normalized.get("overlay_path") or ""),
                    "restored_path": str(normalized.get("restored_path") or ""),
                }
            )


def _merge_runtime_summary_context(
    summary_context: dict[str, Any] | None,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    resolved_summary_context = dict(summary_context or {})
    resolved_summary_context.setdefault("mask_provider", runtime["mask_provider_name"])
    resolved_summary_context.setdefault("restore_provider", runtime["restore_provider_name"])
    resolved_summary_context.setdefault("ocr_session_mode_requested", runtime["ocr_session_mode_requested"])
    resolved_summary_context.setdefault("ocr_session_mode_resolved", runtime["ocr_session_mode_resolved"])
    resolved_summary_context.setdefault("restore_prompt", runtime["restore_prompt"])
    resolved_summary_context.setdefault("restore_negative_prompt", runtime["restore_negative_prompt"])
    resolved_summary_context.setdefault("restore_options", dict(runtime["restore_options"]))
    resolved_summary_context.setdefault("restore_session_mode_requested", runtime["restore_session_mode_requested"])
    resolved_summary_context.setdefault("restore_session_mode_resolved", runtime["restore_session_mode_resolved"])
    resolved_summary_context.setdefault("restore_session_note", runtime["restore_session_note"])
    resolved_summary_context.setdefault("warnings", [])
    warnings = resolved_summary_context["warnings"]
    if not isinstance(warnings, list):
        warnings = []
        resolved_summary_context["warnings"] = warnings
    for warning in runtime["warnings"]:
        if warning not in warnings:
            warnings.append(warning)
    return resolved_summary_context


def _build_pipeline_runtime(
    *,
    mask_provider_name: str,
    restore_provider_name: str,
    ocr_session_mode: str,
    restore_prompt: str | None,
    restore_negative_prompt: str | None,
    restore_options: dict[str, Any] | None,
) -> dict[str, Any]:
    mask_provider = create_mask_provider(mask_provider_name)
    restore_provider = create_restore_provider(restore_provider_name)
    provider_hints: dict[str, Any] = {}
    warnings: list[str] = []
    resolved_ocr_session_mode: str | None = None
    ocr_context: PaddleOCRExecutionContext | None = None
    restore_context: PowerPaintExecutionContext | None = None
    resolved_restore_session_mode: str | None = None
    restore_session_note: str | None = None

    if mask_provider_name == "paddleocr":
        ocr_context = create_paddleocr_execution_context(ocr_session_mode)
        resolved_ocr_session_mode = ocr_context.resolved_mode
        if ocr_context.resolved_mode == OCR_RESOLVED_MODE_UNAVAILABLE:
            raise RuntimeError(ocr_context.note or "PaddleOCR execution context is unavailable.")
        if ocr_context.note and ocr_context.note not in warnings:
            warnings.append(ocr_context.note)
        provider_hints["paddleocr_context"] = ocr_context

    if restore_provider_name == "powerpaint_v2_1":
        restore_context = create_powerpaint_execution_context(restore_options)
        resolved_restore_session_mode = restore_context.resolved_mode
        restore_session_note = restore_context.note
        if restore_context.resolved_mode == RESTORE_RESOLVED_MODE_UNAVAILABLE:
            raise RuntimeError(restore_context.note or "PowerPaint restore execution context is unavailable.")
        if restore_context.note and restore_context.note not in warnings:
            warnings.append(restore_context.note)
        provider_hints["powerpaint_context"] = restore_context

    return {
        "mask_provider_name": mask_provider_name,
        "restore_provider_name": restore_provider_name,
        "ocr_session_mode_requested": ocr_session_mode,
        "ocr_session_mode_resolved": resolved_ocr_session_mode,
        "restore_session_mode_requested": (
            restore_context.requested_mode if restore_context is not None else None
        ),
        "restore_session_mode_resolved": resolved_restore_session_mode,
        "restore_session_note": restore_session_note,
        "restore_prompt": restore_prompt,
        "restore_negative_prompt": restore_negative_prompt,
        "restore_options": dict(restore_options or {}),
        "mask_provider": mask_provider,
        "restore_provider": restore_provider,
        "provider_hints": provider_hints,
        "ocr_context": ocr_context,
        "restore_context": restore_context,
        "warnings": warnings,
    }


def _close_pipeline_runtime(runtime: dict[str, Any]) -> None:
    ocr_context = runtime.get("ocr_context")
    if isinstance(ocr_context, PaddleOCRExecutionContext):
        ocr_context.close()
    restore_context = runtime.get("restore_context")
    if isinstance(restore_context, PowerPaintExecutionContext):
        restore_context.close()


def _build_batch_benchmark_item(item: ScanItem) -> BenchmarkDatasetItem:
    return BenchmarkDatasetItem(
        item_id=item.relative_path.as_posix(),
        dataset_id="batch_runtime",
        source_path=item.source_path,
        input_path=item.source_path,
        relative_path=item.relative_path,
        width=item.width,
        height=item.height,
        source_category=item.category,
        benchmark_category=item.category,
        prompt="",
        prompt_path=item.source_path,
        notes={"source": "batch_pipeline"},
    )


def _mask_result_to_detection(item: ScanItem, mask_result: MaskResult) -> DetectionResult:
    return DetectionResult(
        category=item.category,
        mask_nonzero=mask_result.mask_nonzero,
        confidence=mask_result.confidence,
        boxes=list(mask_result.boxes),
    )
