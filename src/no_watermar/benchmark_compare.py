from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from .benchmark_metrics import BENCHMARK_METRIC_KEYS
from .io_utils import ensure_dir, read_mask, write_json


def compare_benchmark_reports(
    baseline_report: Path,
    candidate_report: Path,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    baseline_report = baseline_report.resolve()
    candidate_report = candidate_report.resolve()
    baseline_data = json.loads(baseline_report.read_text(encoding="utf-8"))
    candidate_data = json.loads(candidate_report.read_text(encoding="utf-8"))

    baseline_index = _index_results(baseline_data.get("results", []))
    candidate_index = _index_results(candidate_data.get("results", []))
    common_item_ids = sorted(set(baseline_index) & set(candidate_index))

    comparison_rows: list[dict[str, Any]] = []
    status_pair_counts: Counter[str] = Counter()

    for item_id in common_item_ids:
        baseline_result = baseline_index[item_id]
        candidate_result = candidate_index[item_id]
        status_pair = f"{baseline_result['status']}__{candidate_result['status']}"
        status_pair_counts[status_pair] += 1

        row: dict[str, Any] = {
            "item_id": item_id,
            "relative_path": baseline_result["item"]["relative_path"],
            "baseline_status": baseline_result["status"],
            "candidate_status": candidate_result["status"],
            "baseline_confidence": _read_confidence(baseline_result),
            "candidate_confidence": _read_confidence(candidate_result),
        }

        for metric_key in BENCHMARK_METRIC_KEYS:
            baseline_value = _read_numeric_metric(baseline_result, metric_key)
            candidate_value = _read_numeric_metric(candidate_result, metric_key)
            row[f"baseline_{metric_key}"] = baseline_value
            row[f"candidate_{metric_key}"] = candidate_value
            if baseline_value is None or candidate_value is None:
                row[f"delta_{metric_key}"] = None
            else:
                row[f"delta_{metric_key}"] = round(candidate_value - baseline_value, 6)

        mask_overlap = _compute_mask_overlap(
            baseline_result.get("mask_path"),
            candidate_result.get("mask_path"),
        )
        row.update(mask_overlap)
        comparison_rows.append(row)

    summary = {
        "dataset_id": baseline_data.get("dataset_id"),
        "baseline_report": str(baseline_report),
        "candidate_report": str(candidate_report),
        "baseline_summary": _build_report_summary(baseline_data.get("results", [])),
        "candidate_summary": _build_report_summary(candidate_data.get("results", [])),
        "common_item_count": len(common_item_ids),
        "baseline_only_count": len(set(baseline_index) - set(candidate_index)),
        "candidate_only_count": len(set(candidate_index) - set(baseline_index)),
        "status_pair_counts": dict(status_pair_counts),
        "delta_summary": _build_delta_summary(comparison_rows),
        "mask_overlap_summary": _build_mask_overlap_summary(comparison_rows),
        "items": comparison_rows,
    }

    if output_dir is not None:
        output_dir = ensure_dir(output_dir.resolve())
        stem = f"{baseline_report.stem}__vs__{candidate_report.stem}"
        output_json = output_dir / f"{stem}.json"
        output_csv = output_dir / f"{stem}.csv"
        write_json(output_json, summary)
        _write_compare_csv(output_csv, comparison_rows)
        summary["output_json"] = str(output_json)
        summary["output_csv"] = str(output_csv)

    return summary


def _index_results(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for result in results:
        item = result.get("item") or {}
        item_id = str(item.get("item_id") or "")
        if item_id:
            indexed[item_id] = result
    return indexed


def _read_confidence(result: dict[str, Any]) -> float | None:
    mask_result = result.get("mask_result") or {}
    confidence = mask_result.get("confidence")
    if confidence is None:
        return None
    return round(float(confidence), 6)


def _compute_mask_overlap(baseline_mask_path: str | None, candidate_mask_path: str | None) -> dict[str, Any]:
    if not baseline_mask_path or not candidate_mask_path:
        return {"mask_iou": None, "mask_precision": None, "mask_recall": None}

    baseline_path = Path(baseline_mask_path)
    candidate_path = Path(candidate_mask_path)
    if not baseline_path.exists() or not candidate_path.exists():
        return {"mask_iou": None, "mask_precision": None, "mask_recall": None}

    baseline_mask = read_mask(baseline_path) > 0
    candidate_mask = read_mask(candidate_path) > 0
    intersection = int(np.count_nonzero(baseline_mask & candidate_mask))
    baseline_nonzero = int(np.count_nonzero(baseline_mask))
    candidate_nonzero = int(np.count_nonzero(candidate_mask))
    union = int(np.count_nonzero(baseline_mask | candidate_mask))

    if union == 0:
        return {"mask_iou": 1.0, "mask_precision": 1.0, "mask_recall": 1.0}

    precision = float(intersection) / float(max(1, candidate_nonzero))
    recall = float(intersection) / float(max(1, baseline_nonzero))
    iou = float(intersection) / float(union)
    return {
        "mask_iou": round(iou, 6),
        "mask_precision": round(precision, 6),
        "mask_recall": round(recall, 6),
    }


def _build_report_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(str(result.get("status") or "") for result in results)
    metric_summary: dict[str, Any] = {}
    for metric_key in BENCHMARK_METRIC_KEYS:
        values = [
            value
            for result in results
            if (value := _read_numeric_metric(result, metric_key)) is not None
        ]
        metric_summary[metric_key] = round(mean(values), 6) if values else None

    confidences = []
    for result in results:
        confidence = _read_confidence(result)
        if confidence is not None:
            confidences.append(confidence)

    return {
        "item_count": len(results),
        "status_counts": dict(status_counts),
        "mean_metrics": metric_summary,
        "mean_confidence": round(mean(confidences), 6) if confidences else None,
    }


def _build_delta_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for metric_key in BENCHMARK_METRIC_KEYS:
        values = [
            float(row[f"delta_{metric_key}"])
            for row in rows
            if row.get(f"delta_{metric_key}") is not None
        ]
        summary[metric_key] = _summarize_numeric(values)
    return summary


def _build_mask_overlap_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ious = [float(row["mask_iou"]) for row in rows if row.get("mask_iou") is not None]
    precisions = [float(row["mask_precision"]) for row in rows if row.get("mask_precision") is not None]
    recalls = [float(row["mask_recall"]) for row in rows if row.get("mask_recall") is not None]
    return {
        "paired_count": len(ious),
        "mean_iou": round(mean(ious), 6) if ious else None,
        "mean_precision": round(mean(precisions), 6) if precisions else None,
        "mean_recall": round(mean(recalls), 6) if recalls else None,
    }


def _summarize_numeric(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "min": None, "max": None}
    return {
        "mean": round(mean(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def _read_numeric_metric(result: dict[str, Any], metric_key: str) -> float | None:
    value = result.get("metrics", {}).get(metric_key)
    if value is None:
        return None
    return float(value)


def _write_compare_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "item_id",
        "relative_path",
        "baseline_status",
        "candidate_status",
        "baseline_confidence",
        "candidate_confidence",
    ]
    for metric_key in BENCHMARK_METRIC_KEYS:
        fieldnames.extend(
            [
                f"baseline_{metric_key}",
                f"candidate_{metric_key}",
                f"delta_{metric_key}",
            ]
        )
    fieldnames.extend(["mask_iou", "mask_precision", "mask_recall"])

    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
