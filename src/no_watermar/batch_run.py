from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .io_utils import ensure_dir, write_json
from .models import ProcessResult, ScanItem

BATCH_RUN_VERSION = 1


def create_batch_run_artifacts(
    *,
    runs_root: Path,
    run_id: str,
    input_root: Path,
    recursive: bool,
    limit: int | None,
    scan_only: bool,
    items: list[ScanItem],
    summary_context: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path]]:
    resolved_runs_root = runs_root.resolve()
    resolved_input_root = input_root.resolve()
    paths = build_batch_run_paths(resolved_runs_root, run_id)
    ensure_dir(paths["run_root"])
    ensure_dir(paths["reports_root"])

    category_counts = dict(Counter(item.category for item in items))
    manifest = {
        "run_version": BATCH_RUN_VERSION,
        "run_id": run_id,
        "run_root": str(paths["run_root"]),
        "runs_root": str(resolved_runs_root),
        "input_root": str(resolved_input_root),
        "recursive": recursive,
        "scan_only": scan_only,
        "limit": limit,
        "item_count": len(items),
        "category_counts": category_counts,
        "items": [item.to_dict() for item in items],
    }
    summary = {
        "run_version": BATCH_RUN_VERSION,
        "status": "ok",
        "run_status": "running",
        "resume_count": 0,
        "run_id": run_id,
        "run_root": str(paths["run_root"]),
        "runs_root": str(resolved_runs_root),
        "input_root": str(resolved_input_root),
        "output_root": str(paths["run_root"]),
        "recursive": recursive,
        "scan_only": scan_only,
        "limit": limit,
        "image_count": len(items),
        "completed_item_count": 0,
        "pending_item_count": len(items),
        "category_counts": category_counts,
        "status_counts": {},
        "results": [],
        "manifest_json": str(paths["manifest_json"]),
        "summary_json": str(paths["summary_json"]),
        "results_jsonl": str(paths["results_jsonl"]),
        "latest_summary_json": str(paths["latest_summary_json"]),
        "report_json": str(paths["report_json"]),
        "report_csv": str(paths["report_csv"]),
        "warnings": [],
    }
    if summary_context:
        summary.update(summary_context)

    write_json(paths["manifest_json"], manifest)
    paths["results_jsonl"].write_text("", encoding="utf-8")
    persist_batch_run_summary(summary)
    return manifest, summary, paths


def build_batch_run_paths(runs_root: Path, run_id: str) -> dict[str, Path]:
    resolved_runs_root = runs_root.resolve()
    run_root = resolved_runs_root / run_id
    reports_root = run_root / "reports"
    return {
        "runs_root": resolved_runs_root,
        "run_root": run_root,
        "reports_root": reports_root,
        "manifest_json": run_root / "manifest.json",
        "summary_json": run_root / "summary.json",
        "results_jsonl": run_root / "results.jsonl",
        "latest_summary_json": resolved_runs_root / "latest.json",
        "report_json": reports_root / "report.json",
        "report_csv": reports_root / "report.csv",
    }


def update_batch_run_progress(summary: dict[str, Any], result: ProcessResult) -> None:
    results = summary.setdefault("results", [])
    results.append(result.to_dict())
    status_counts = Counter(entry.get("status") for entry in results if isinstance(entry, dict))
    image_count = int(summary.get("image_count", len(results)))
    completed_item_count = len(results)
    summary["status_counts"] = dict(status_counts)
    summary["completed_item_count"] = completed_item_count
    summary["pending_item_count"] = max(0, image_count - completed_item_count)
    append_batch_run_result(summary, result)
    persist_batch_run_summary(summary)


def finalize_batch_run(summary: dict[str, Any], *, run_status: str = "completed") -> None:
    summary["run_status"] = run_status
    persist_batch_run_summary(summary)


def append_batch_run_result(summary: dict[str, Any], result: ProcessResult) -> None:
    results_jsonl = Path(_require_string(summary, "results_jsonl"))
    results_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with results_jsonl.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result.to_dict(), ensure_ascii=False))
        handle.write("\n")


def persist_batch_run_summary(summary: dict[str, Any]) -> None:
    summary_json = Path(_require_string(summary, "summary_json"))
    report_json = Path(_require_string(summary, "report_json"))
    latest_summary_json = Path(_require_string(summary, "latest_summary_json"))
    write_json(summary_json, summary)
    write_json(report_json, summary)
    write_json(latest_summary_json, summary)


def load_batch_run_summary(summary_path: Path) -> dict[str, Any]:
    resolved_path = summary_path.resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Batch run summary does not exist: {resolved_path}")
    try:
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fallback_path = _summary_fallback_path(resolved_path)
        if fallback_path is not None:
            return load_batch_run_summary(fallback_path)
        raise exc
    if not isinstance(payload, dict):
        raise ValueError(f"Batch run summary must be a JSON object: {resolved_path}")
    return _normalize_batch_run_summary(payload, resolved_path)


def load_batch_run_manifest(manifest_path: Path) -> dict[str, Any]:
    resolved_path = manifest_path.resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Batch run manifest does not exist: {resolved_path}")
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Batch run manifest must be a JSON object: {resolved_path}")
    return payload


def load_batch_run_items(manifest: dict[str, Any]) -> list[ScanItem]:
    raw_items = manifest.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("Batch run manifest field 'items' must be a list.")

    items: list[ScanItem] = []
    for index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, dict):
            raise ValueError(f"Batch run manifest item at index {index} must be a JSON object.")
        items.append(ScanItem.from_dict(raw_item))

    item_count = manifest.get("item_count")
    if not isinstance(item_count, int):
        raise ValueError("Batch run manifest field 'item_count' must be an integer.")
    if item_count != len(items):
        raise ValueError(f"Batch run manifest item_count was {item_count}, but loaded {len(items)} items.")
    return items


def collect_pending_batch_run_items(summary: dict[str, Any], manifest: dict[str, Any]) -> list[ScanItem]:
    items = load_batch_run_items(manifest)
    completed_keys = collect_completed_item_keys(summary)
    return [item for item in items if _scan_item_key(item) not in completed_keys]


def collect_completed_item_keys(summary: dict[str, Any]) -> set[str]:
    results = summary.get("results")
    if not isinstance(results, list):
        return set()
    keys: set[str] = set()
    for result in results:
        if not isinstance(result, dict):
            continue
        key = _result_item_key(result)
        if key:
            keys.add(key)
    return keys


def mark_batch_run_resuming(summary: dict[str, Any], *, command: str = "batch resume", mode: str = "resume") -> dict[str, Any]:
    previous_command = summary.get("command")
    previous_mode = summary.get("mode")
    previous_run_status = summary.get("run_status", "unknown")
    summary["resume_count"] = int(summary.get("resume_count", 0)) + 1
    summary["resumed_from_command"] = previous_command
    summary["resumed_from_mode"] = previous_mode
    summary["resumed_from_run_status"] = previous_run_status
    summary["command"] = command
    summary["mode"] = mode
    summary["run_status"] = "running"
    summary["resumed_item_count"] = 0
    persist_batch_run_summary(summary)
    return summary


def resolve_batch_run_summary(
    *,
    runs_root: Path,
    latest: bool = False,
    run_id: str | None = None,
    run_dir: Path | None = None,
    summary_path: Path | None = None,
) -> tuple[dict[str, Any], str, Path]:
    selector_count = int(latest) + int(run_id is not None) + int(run_dir is not None) + int(summary_path is not None)
    if selector_count > 1:
        raise ValueError("Select only one batch report source: --latest, --run-id, --run-dir, or --summary.")

    resolved_runs_root = runs_root.resolve()
    if summary_path is not None:
        resolved_summary_path = summary_path.resolve()
        return load_batch_run_summary(resolved_summary_path), "summary_path", resolved_summary_path

    if run_dir is not None:
        resolved_run_dir = run_dir.resolve()
        return _load_batch_run_from_run_dir(resolved_run_dir), "run_dir", resolved_run_dir

    if run_id is not None:
        resolved_run_dir = (resolved_runs_root / run_id).resolve()
        return _load_batch_run_from_run_dir(resolved_run_dir), "run_id", resolved_run_dir

    return _load_latest_batch_run_summary(resolved_runs_root)


def _load_latest_batch_run_summary(runs_root: Path) -> tuple[dict[str, Any], str, Path]:
    latest_summary_path = runs_root.resolve() / "latest.json"
    if latest_summary_path.exists():
        try:
            return load_batch_run_summary(latest_summary_path), "latest", latest_summary_path
        except (json.JSONDecodeError, ValueError):
            pass

    run_dirs = [path for path in runs_root.iterdir() if path.is_dir()] if runs_root.exists() else []
    if not run_dirs:
        raise FileNotFoundError(f"No batch runs found under: {runs_root}")
    resolved_run_dir = sorted(run_dirs, key=lambda path: path.name)[-1]
    return _load_batch_run_from_run_dir(resolved_run_dir), "latest", resolved_run_dir


def _load_batch_run_from_run_dir(run_dir: Path) -> dict[str, Any]:
    if not run_dir.exists():
        raise FileNotFoundError(f"Batch run directory does not exist: {run_dir}")
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Batch run path is not a directory: {run_dir}")

    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        return load_batch_run_summary(summary_path)

    legacy_report_path = run_dir / "reports" / "report.json"
    if legacy_report_path.exists():
        return load_batch_run_summary(legacy_report_path)

    raise FileNotFoundError(f"Batch run summary was not found under: {run_dir}")


def _summary_fallback_path(summary_path: Path) -> Path | None:
    if summary_path.name != "summary.json":
        return None
    fallback_path = summary_path.parent / "reports" / "report.json"
    if fallback_path.exists():
        return fallback_path.resolve()
    return None


def _normalize_batch_run_summary(payload: dict[str, Any], source_path: Path) -> dict[str, Any]:
    normalized = dict(payload)
    run_root = _infer_run_root(normalized, source_path)
    runs_root = run_root.parent
    normalized.setdefault("run_version", BATCH_RUN_VERSION)
    normalized.setdefault("status", "ok")
    normalized.setdefault("run_status", "completed")
    normalized.setdefault("run_root", str(run_root))
    normalized.setdefault("runs_root", str(runs_root))
    normalized.setdefault("summary_json", str(run_root / "summary.json"))
    normalized.setdefault("manifest_json", str(run_root / "manifest.json"))
    normalized.setdefault("results_jsonl", str(run_root / "results.jsonl"))
    normalized.setdefault("latest_summary_json", str(runs_root / "latest.json"))
    normalized.setdefault("report_json", str(run_root / "reports" / "report.json"))
    normalized.setdefault("report_csv", str(run_root / "reports" / "report.csv"))
    normalized.setdefault("resume_count", 0)
    normalized["results"] = _reconcile_results(normalized)
    completed_item_count = len(normalized["results"])
    image_count = normalized.get("image_count")
    if not isinstance(image_count, int):
        image_count = completed_item_count
        normalized["image_count"] = image_count
    normalized.setdefault("completed_item_count", completed_item_count)
    normalized.setdefault("pending_item_count", max(0, image_count - completed_item_count))
    status_counts = normalized.get("status_counts")
    if not isinstance(status_counts, dict):
        normalized["status_counts"] = dict(Counter(entry.get("status") for entry in normalized["results"] if isinstance(entry, dict)))
    warnings = normalized.get("warnings")
    if not isinstance(warnings, list):
        normalized["warnings"] = []
    return normalized


def _infer_run_root(payload: dict[str, Any], source_path: Path) -> Path:
    run_root_value = payload.get("run_root")
    if isinstance(run_root_value, str) and run_root_value:
        return Path(run_root_value).resolve()
    if source_path.name == "summary.json":
        return source_path.parent.resolve()
    if source_path.name == "report.json":
        return source_path.parent.parent.resolve()
    return source_path.parent.resolve()


def _require_string(payload: dict[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Batch run field '{field_name}' must be a non-empty string.")
    return value


def _reconcile_results(summary: dict[str, Any]) -> list[dict[str, Any]]:
    summary_results = summary.get("results")
    resolved_summary_results = [entry for entry in summary_results if isinstance(entry, dict)] if isinstance(summary_results, list) else []
    results_jsonl_path = Path(str(summary.get("results_jsonl"))).resolve() if summary.get("results_jsonl") else None
    jsonl_results = _load_results_jsonl(results_jsonl_path) if results_jsonl_path else []
    if len(jsonl_results) >= len(resolved_summary_results):
        return jsonl_results
    return resolved_summary_results


def _load_results_jsonl(results_jsonl_path: Path) -> list[dict[str, Any]]:
    if not results_jsonl_path.exists():
        return []

    records: list[dict[str, Any]] = []
    for index, raw_line in enumerate(results_jsonl_path.read_text(encoding="utf-8").splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {results_jsonl_path}:{index + 1}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Batch run results entry at {results_jsonl_path}:{index + 1} must be a JSON object.")
        records.append(payload)
    return records


def _result_item_key(result: dict[str, Any]) -> str | None:
    item = result.get("item")
    if not isinstance(item, dict):
        return None
    relative_path = item.get("relative_path")
    if isinstance(relative_path, str) and relative_path:
        return relative_path
    source_path = item.get("source_path")
    if isinstance(source_path, str) and source_path:
        return source_path
    return None


def _scan_item_key(item: ScanItem) -> str:
    return str(item.relative_path)
