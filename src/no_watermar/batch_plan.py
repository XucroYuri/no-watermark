from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from .batch_run import persist_batch_run_summary
from .io_utils import ensure_dir, write_json
from .models import ScanItem
from .pipeline import run_pipeline
from .scan_manifest import load_scan_manifest
from .scanner import build_scan_items

BATCH_PLAN_VERSION = 1


def create_batch_plan(
    input_root: Path | None,
    output_root: Path,
    *,
    scan_manifest_path: Path | None = None,
    plans_root: Path | None = None,
    recursive: bool = True,
    limit: int | None = None,
    scan_only: bool = False,
    mask_provider: str = "rule_based_roi",
    restore_provider: str = "telea",
    ocr_session_mode: str = "auto",
    restore_prompt: str | None = None,
    restore_negative_prompt: str | None = None,
    restore_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    input_root, items, recursive, limit, input_mode, scan_manifest_summary = _resolve_batch_plan_input(
        input_root=input_root,
        scan_manifest_path=scan_manifest_path,
        recursive=recursive,
        limit=limit,
    )
    resolved_plans_root = _resolve_plans_root(output_root, plans_root)
    validate_batch_paths(input_root=input_root, output_root=output_root, plans_root=resolved_plans_root)

    category_counts = Counter(item.category for item in items)
    plan_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    plan_path = resolved_plans_root / f"{plan_id}.json"
    latest_path = resolved_plans_root / "latest.json"

    summary = {
        "command": "batch plan",
        "status": "ok",
        "mode": "plan",
        "plan_version": BATCH_PLAN_VERSION,
        "plan_id": plan_id,
        "plan_path": str(plan_path),
        "latest_plan_path": str(latest_path),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "plans_root": str(resolved_plans_root),
        "input_mode": input_mode,
        "scan_manifest_path": scan_manifest_summary.get("scan_manifest_path"),
        "source_scan_id": scan_manifest_summary.get("source_scan_id"),
        "recursive": recursive,
        "scan_only": scan_only,
        "limit": limit,
        "mask_provider": mask_provider,
        "restore_provider": restore_provider,
        "ocr_session_mode": ocr_session_mode,
        "restore_prompt": restore_prompt,
        "restore_negative_prompt": restore_negative_prompt,
        "restore_options": dict(restore_options or {}),
        "item_count": len(items),
        "category_counts": dict(category_counts),
        "items": [item.to_dict() for item in items],
        "warnings": [],
    }

    ensure_dir(resolved_plans_root)
    write_json(plan_path, summary)
    write_json(latest_path, summary)
    return summary


def apply_batch_plan(plan_path: Path) -> dict[str, Any]:
    resolved_plan_path = plan_path.resolve()
    plan = load_batch_plan(resolved_plan_path)
    return apply_loaded_batch_plan(plan, resolved_plan_path)


def apply_loaded_batch_plan(plan: dict[str, Any], plan_path: Path) -> dict[str, Any]:
    resolved_plan_path = plan_path.resolve()
    input_root = Path(_require_string(plan, "input_root"))
    output_root = Path(_require_string(plan, "output_root"))
    plans_root = Path(_require_string(plan, "plans_root"))
    recursive = _require_bool(plan, "recursive")
    scan_only = _require_bool(plan, "scan_only")
    mask_provider = str(plan.get("mask_provider") or "rule_based_roi")
    restore_provider = str(plan.get("restore_provider") or "telea")
    ocr_session_mode = str(plan.get("ocr_session_mode") or "auto")
    restore_prompt = _optional_string(plan, "restore_prompt")
    restore_negative_prompt = _optional_string(plan, "restore_negative_prompt")
    restore_options = _optional_dict(plan, "restore_options")
    plan_items = _load_scan_items(plan, source_name=f"batch plan {resolved_plan_path}")
    limit = plan.get("limit")
    if limit is not None and not isinstance(limit, int):
        raise ValueError("Batch plan field 'limit' must be an integer or null.")

    validate_batch_paths(input_root=input_root, output_root=output_root, plans_root=plans_root)
    summary = run_pipeline(
        input_root=input_root,
        output_root=output_root,
        recursive=recursive,
        limit=limit,
        scan_only=scan_only,
        items=plan_items,
        mask_provider_name=mask_provider,
        restore_provider_name=restore_provider,
        ocr_session_mode=ocr_session_mode,
        restore_prompt=restore_prompt,
        restore_negative_prompt=restore_negative_prompt,
        restore_options=restore_options,
        summary_context={
            "command": "batch apply",
            "mode": "planned",
            "plan_id": _require_string(plan, "plan_id"),
            "plan_path": str(resolved_plan_path),
            "input_mode": plan.get("input_mode", "scan"),
            "scan_manifest_path": plan.get("scan_manifest_path"),
            "source_scan_id": plan.get("source_scan_id"),
            "mask_provider": mask_provider,
            "restore_provider": restore_provider,
            "ocr_session_mode_requested": ocr_session_mode,
            "restore_prompt": restore_prompt,
            "restore_negative_prompt": restore_negative_prompt,
            "restore_options": restore_options,
            "warnings": [],
        },
    )

    planned_item_count = _require_int(plan, "item_count")
    if summary["image_count"] != planned_item_count:
        summary["warnings"].append(
            f"Planned item_count was {planned_item_count}, but apply discovered {summary['image_count']} items."
        )
        _rewrite_pipeline_report(summary)
    return summary


def summarize_batch_plan(plan_path: Path) -> dict[str, Any]:
    resolved_plan_path = plan_path.resolve()
    plan = load_batch_plan(resolved_plan_path)
    return {
        "plan_id": _require_string(plan, "plan_id"),
        "plan_path": str(resolved_plan_path),
        "input_root": _require_string(plan, "input_root"),
        "output_root": _require_string(plan, "output_root"),
        "plans_root": _require_string(plan, "plans_root"),
        "input_mode": plan.get("input_mode", "scan"),
        "scan_manifest_path": plan.get("scan_manifest_path"),
        "source_scan_id": plan.get("source_scan_id"),
        "item_count": _require_int(plan, "item_count"),
        "recursive": _require_bool(plan, "recursive"),
        "scan_only": _require_bool(plan, "scan_only"),
        "limit": plan.get("limit"),
        "mask_provider": str(plan.get("mask_provider") or "rule_based_roi"),
        "restore_provider": str(plan.get("restore_provider") or "telea"),
        "ocr_session_mode": str(plan.get("ocr_session_mode") or "auto"),
        "restore_prompt": _optional_string(plan, "restore_prompt"),
        "restore_negative_prompt": _optional_string(plan, "restore_negative_prompt"),
        "restore_options": _optional_dict(plan, "restore_options"),
    }


def load_batch_plan(plan_path: Path) -> dict[str, Any]:
    if not plan_path.exists():
        raise FileNotFoundError(f"Batch plan does not exist: {plan_path}")

    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Batch plan must be a JSON object: {plan_path}")
    if payload.get("plan_version") != BATCH_PLAN_VERSION:
        raise ValueError(
            f"Unsupported batch plan version at {plan_path}: {payload.get('plan_version')!r}. "
            f"Expected {BATCH_PLAN_VERSION}."
        )
    return payload


def validate_batch_paths(
    *,
    input_root: Path,
    output_root: Path,
    plans_root: Path,
) -> None:
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root is not a directory: {input_root}")
    if _is_relative_to(output_root, input_root):
        raise ValueError(f"Output root must not be inside input root: {output_root}")
    if _is_relative_to(plans_root, input_root):
        raise ValueError(f"Plans root must not be inside input root: {plans_root}")


def _resolve_plans_root(output_root: Path, plans_root: Path | None) -> Path:
    if plans_root is not None:
        return plans_root.resolve()
    return (output_root.parent / "plans").resolve()


def _resolve_batch_plan_input(
    *,
    input_root: Path | None,
    scan_manifest_path: Path | None,
    recursive: bool,
    limit: int | None,
) -> tuple[Path, list[ScanItem], bool, int | None, str, dict[str, str | None]]:
    if scan_manifest_path is not None:
        manifest = load_scan_manifest(scan_manifest_path.resolve())
        resolved_input_root = Path(_require_string(manifest, "input_root"))
        manifest_recursive = _require_bool(manifest, "recursive")
        manifest_limit = manifest.get("limit")
        if manifest_limit is not None and not isinstance(manifest_limit, int):
            raise ValueError("Scan manifest field 'limit' must be an integer or null.")
        items = _load_scan_items(manifest, source_name=f"scan manifest {scan_manifest_path.resolve()}")
        return (
            resolved_input_root,
            items,
            manifest_recursive,
            manifest_limit,
            "scan_manifest",
            {
                "scan_manifest_path": str(scan_manifest_path.resolve()),
                "source_scan_id": _optional_string(manifest, "scan_id"),
            },
        )

    if input_root is None:
        raise ValueError("Batch plan requires either an input_root or a scan_manifest_path.")

    resolved_input_root = input_root.resolve()
    items = build_scan_items(resolved_input_root, recursive=recursive, limit=limit)
    return (
        resolved_input_root,
        items,
        recursive,
        limit,
        "scan",
        {
            "scan_manifest_path": None,
            "source_scan_id": None,
        },
    )


def _rewrite_pipeline_report(summary: dict[str, Any]) -> None:
    persist_batch_run_summary(summary)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _require_string(payload: dict[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Batch plan field '{field_name}' must be a non-empty string.")
    return value


def _require_int(payload: dict[str, Any], field_name: str) -> int:
    value = payload.get(field_name)
    if not isinstance(value, int):
        raise ValueError(f"Batch plan field '{field_name}' must be an integer.")
    return value


def _require_bool(payload: dict[str, Any], field_name: str) -> bool:
    value = payload.get(field_name)
    if not isinstance(value, bool):
        raise ValueError(f"Batch plan field '{field_name}' must be a boolean.")
    return value


def _optional_string(payload: dict[str, Any], field_name: str) -> str | None:
    value = payload.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"Batch plan field '{field_name}' must be a non-empty string or null.")
    return value


def _optional_dict(payload: dict[str, Any], field_name: str) -> dict[str, Any]:
    value = payload.get(field_name)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Batch plan field '{field_name}' must be a JSON object or null.")
    return dict(value)


def _load_scan_items(payload: dict[str, Any], *, source_name: str) -> list[ScanItem]:
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        raise ValueError(f"{source_name} field 'items' must be a list.")

    items: list[ScanItem] = []
    for index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, dict):
            raise ValueError(f"{source_name} item at index {index} must be a JSON object.")
        items.append(ScanItem.from_dict(raw_item))

    declared_item_count = _require_int(payload, "item_count")
    if declared_item_count != len(items):
        raise ValueError(
            f"{source_name} item_count was {declared_item_count}, but loaded {len(items)} items from the payload."
        )
    return items
