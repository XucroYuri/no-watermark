from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from .benchmark_metrics import BENCHMARK_METRIC_KEYS
from .io_utils import ensure_dir, write_json


def aggregate_benchmark_reports(
    reports_root: Path,
    *,
    dataset_id: str = "all",
    mask_provider: str | None = None,
    restore_provider: str | None = None,
    run_after: str | None = None,
    run_before: str | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    reports_root = reports_root.resolve()
    report_paths = _find_report_paths(reports_root)
    report_entries = [_load_report(path) for path in report_paths]
    report_entries = _filter_entries(
        report_entries,
        dataset_id=dataset_id,
        mask_provider=mask_provider,
        restore_provider=restore_provider,
        run_after=run_after,
        run_before=run_before,
    )

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in report_entries:
        key = (entry["dataset_id"], entry["mask_provider"], entry["restore_provider"])
        grouped[key].append(entry)

    groups = [
        _build_group_summary(key, entries)
        for key, entries in sorted(grouped.items())
    ]

    summary = {
        "reports_root": str(reports_root),
        "filters": {
            "dataset_id": dataset_id,
            "mask_provider": mask_provider,
            "restore_provider": restore_provider,
            "run_after": run_after,
            "run_before": run_before,
        },
        "report_count": len(report_entries),
        "group_count": len(groups),
        "groups": groups,
    }

    if output_dir is not None:
        output_dir = ensure_dir(output_dir.resolve())
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        stem = _build_output_stem(
            dataset_id=dataset_id,
            mask_provider=mask_provider,
            restore_provider=restore_provider,
            stamp=stamp,
        )
        output_json = output_dir / f"{stem}.json"
        output_csv = output_dir / f"{stem}.csv"
        write_json(output_json, summary)
        _write_aggregate_csv(output_csv, groups)
        summary["output_json"] = str(output_json)
        summary["output_csv"] = str(output_csv)

    return summary


def _find_report_paths(reports_root: Path) -> list[Path]:
    if not reports_root.exists():
        return []
    return sorted(
        path
        for path in reports_root.rglob("*.json")
        if path.name != "summary.json"
    )


def _load_report(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "results" not in data:
        raise ValueError(f"Unsupported benchmark report format: {path}")

    dataset_id = str(data.get("dataset_id") or "")
    mask_provider = str(data.get("mask_provider") or _infer_provider(data, "mask_provider"))
    restore_provider = str(data.get("restore_provider") or _infer_provider(data, "restore_provider"))
    run_id = str(data.get("run_id") or _infer_run_id(path))
    return {
        "path": path,
        "run_id": run_id,
        "dataset_id": dataset_id,
        "mask_provider": mask_provider,
        "restore_provider": restore_provider,
        "results": list(data.get("results") or []),
        "status_counts": dict(data.get("status_counts") or {}),
    }


def _infer_provider(data: dict[str, Any], key: str) -> str:
    results = list(data.get("results") or [])
    if not results:
        return ""
    return str(results[0].get(key) or "")


def _infer_run_id(path: Path) -> str:
    for parent in path.parents:
        name = parent.name
        if _looks_like_run_id(name):
            return name
    return ""


def _looks_like_run_id(value: str) -> bool:
    return len(value) >= 16 and value[:8].isdigit() and "-" in value


def _filter_entries(
    report_entries: list[dict[str, Any]],
    *,
    dataset_id: str,
    mask_provider: str | None,
    restore_provider: str | None,
    run_after: str | None,
    run_before: str | None,
) -> list[dict[str, Any]]:
    filtered = report_entries
    if dataset_id != "all":
        filtered = [entry for entry in filtered if entry["dataset_id"] == dataset_id]
    if mask_provider:
        filtered = [entry for entry in filtered if entry["mask_provider"] == mask_provider]
    if restore_provider:
        filtered = [entry for entry in filtered if entry["restore_provider"] == restore_provider]
    if run_after:
        filtered = [entry for entry in filtered if entry["run_id"] and entry["run_id"] >= run_after]
    if run_before:
        filtered = [entry for entry in filtered if entry["run_id"] and entry["run_id"] <= run_before]
    return filtered


def _build_group_summary(key: tuple[str, str, str], entries: list[dict[str, Any]]) -> dict[str, Any]:
    dataset_id, mask_provider, restore_provider = key
    results = [result for entry in entries for result in entry["results"]]
    status_counts = Counter()
    for entry in entries:
        status_counts.update({str(name): int(count) for name, count in entry["status_counts"].items()})

    mean_metrics = {
        metric_key: _mean_metric(results, metric_key)
        for metric_key in BENCHMARK_METRIC_KEYS
    }
    confidences = [
        confidence
        for result in results
        if (confidence := _read_confidence(result)) is not None
    ]

    run_ids = [entry["run_id"] for entry in entries if entry.get("run_id")]
    return {
        "dataset_id": dataset_id,
        "mask_provider": mask_provider,
        "restore_provider": restore_provider,
        "run_count": len(entries),
        "item_count": len(results),
        "status_counts": dict(status_counts),
        "mean_metrics": mean_metrics,
        "mean_confidence": round(mean(confidences), 6) if confidences else None,
        "mean_mask_latency_ms": mean_metrics.get("mask_latency_ms"),
        "mean_restore_latency_ms": mean_metrics.get("restore_latency_ms"),
        "mean_ocr_residual_latency_ms": mean_metrics.get("ocr_residual_latency_ms"),
        "run_ids": run_ids,
        "reports": [str(entry["path"]) for entry in entries],
    }


def _mean_metric(results: list[dict[str, Any]], metric_key: str) -> float | None:
    values = [
        value
        for result in results
        if (value := _read_numeric_metric(result, metric_key)) is not None
    ]
    return round(mean(values), 6) if values else None


def _read_numeric_metric(result: dict[str, Any], metric_key: str) -> float | None:
    value = result.get("metrics", {}).get(metric_key)
    if value is None:
        return None
    return float(value)


def _read_confidence(result: dict[str, Any]) -> float | None:
    mask_result = result.get("mask_result") or {}
    confidence = mask_result.get("confidence")
    if confidence is None:
        return None
    return float(confidence)


def _build_output_stem(
    *,
    dataset_id: str,
    mask_provider: str | None,
    restore_provider: str | None,
    stamp: str,
) -> str:
    parts = ["aggregate", dataset_id]
    if mask_provider:
        parts.append(mask_provider)
    if restore_provider:
        parts.append(restore_provider)
    parts.append(stamp)
    return "_".join(parts)


def _write_aggregate_csv(path: Path, groups: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset_id",
        "mask_provider",
        "restore_provider",
        "run_count",
        "item_count",
        "mean_confidence",
        "status_counts",
        "run_ids",
    ]
    fieldnames.extend(BENCHMARK_METRIC_KEYS)
    fieldnames.extend(
        [
            "mean_mask_latency_ms",
            "mean_restore_latency_ms",
            "mean_ocr_residual_latency_ms",
            "reports",
        ]
    )

    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group in groups:
            row = {
                "dataset_id": group["dataset_id"],
                "mask_provider": group["mask_provider"],
                "restore_provider": group["restore_provider"],
                "run_count": group["run_count"],
                "item_count": group["item_count"],
                "mean_confidence": group["mean_confidence"],
                "status_counts": json.dumps(group["status_counts"], ensure_ascii=False),
                "run_ids": json.dumps(group["run_ids"], ensure_ascii=False),
                "mean_mask_latency_ms": group["mean_mask_latency_ms"],
                "mean_restore_latency_ms": group["mean_restore_latency_ms"],
                "mean_ocr_residual_latency_ms": group["mean_ocr_residual_latency_ms"],
                "reports": json.dumps(group["reports"], ensure_ascii=False),
            }
            for metric_key in BENCHMARK_METRIC_KEYS:
                row[metric_key] = group["mean_metrics"].get(metric_key)
            writer.writerow(row)
