from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from .io_utils import ensure_dir, write_json
from .scanner import DEFAULT_EXCLUDED_DIRS, build_scan_items

SCAN_VERSION = 1


def summarize_scan_input(
    input_root: Path,
    *,
    recursive: bool = True,
    limit: int | None = None,
    excluded_dirs: set[str] | None = None,
    summary_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_input_root = input_root.resolve()
    resolved_excluded_dirs = _resolve_excluded_dirs(excluded_dirs)
    validate_scan_paths(input_root=resolved_input_root)

    items = build_scan_items(
        resolved_input_root,
        recursive=recursive,
        excluded_dirs=resolved_excluded_dirs,
        limit=limit,
    )
    summary = {
        "status": "ok",
        "scan_version": SCAN_VERSION,
        "input_root": str(resolved_input_root),
        "recursive": recursive,
        "limit": limit,
        "excluded_dirs": sorted(resolved_excluded_dirs),
        "item_count": len(items),
        "category_counts": dict(Counter(item.category for item in items)),
        "items": [item.to_dict() for item in items],
        "warnings": [],
    }
    if summary_context:
        summary.update(summary_context)
    return summary


def create_scan_manifest(
    input_root: Path,
    *,
    scans_root: Path,
    recursive: bool = True,
    limit: int | None = None,
    excluded_dirs: set[str] | None = None,
) -> dict[str, Any]:
    resolved_input_root = input_root.resolve()
    resolved_scans_root = scans_root.resolve()
    validate_scan_paths(input_root=resolved_input_root, scans_root=resolved_scans_root)

    scan_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    manifest_path = resolved_scans_root / f"{scan_id}.json"
    latest_manifest_path = resolved_scans_root / "latest.json"
    summary = summarize_scan_input(
        resolved_input_root,
        recursive=recursive,
        limit=limit,
        excluded_dirs=excluded_dirs,
        summary_context={
            "command": "scan run",
            "mode": "manifest",
            "scan_id": scan_id,
            "scans_root": str(resolved_scans_root),
            "manifest_path": str(manifest_path),
            "latest_manifest_path": str(latest_manifest_path),
        },
    )

    ensure_dir(resolved_scans_root)
    write_json(manifest_path, summary)
    write_json(latest_manifest_path, summary)
    return summary


def load_scan_manifest(manifest_path: Path) -> dict[str, Any]:
    resolved_manifest_path = manifest_path.resolve()
    if not resolved_manifest_path.exists():
        raise FileNotFoundError(f"Scan manifest does not exist: {resolved_manifest_path}")

    payload = json.loads(resolved_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Scan manifest must be a JSON object: {resolved_manifest_path}")
    if payload.get("scan_version") != SCAN_VERSION:
        raise ValueError(
            f"Unsupported scan manifest version at {resolved_manifest_path}: {payload.get('scan_version')!r}. "
            f"Expected {SCAN_VERSION}."
        )
    return payload


def validate_scan_paths(*, input_root: Path, scans_root: Path | None = None) -> None:
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root is not a directory: {input_root}")
    if scans_root is not None and _is_relative_to(scans_root, input_root):
        raise ValueError(f"Scans root must not be inside input root: {scans_root}")


def _resolve_excluded_dirs(excluded_dirs: set[str] | None) -> set[str]:
    resolved = set(DEFAULT_EXCLUDED_DIRS)
    if excluded_dirs:
        resolved.update(excluded_dirs)
    return resolved


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True
