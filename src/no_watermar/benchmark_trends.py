from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .benchmark_metrics import BENCHMARK_METRIC_KEYS
from .io_utils import ensure_dir, write_json

METRIC_PREFERENCES: dict[str, str | None] = {
    "mask_nonzero": None,
    "mask_ratio": None,
    "changed_nonzero": None,
    "mean_abs_diff": "lower",
    "edge_delta": "lower",
    "mask_latency_ms": "lower",
    "restore_latency_ms": "lower",
    "ocr_residual_latency_ms": "lower",
    "ocr_residual_hits": "lower",
    "ocr_residual_score": "lower",
    "ocr_residual_max_score": "lower",
    "mean_confidence": "higher",
}

HEADLINE_METRICS = [
    "mean_abs_diff",
    "edge_delta",
    "mask_latency_ms",
    "restore_latency_ms",
    "ocr_residual_latency_ms",
    "ocr_residual_score",
    "ocr_residual_hits",
    "mean_confidence",
]


def build_benchmark_trend_snapshot(
    *,
    comparison: Path | None = None,
    aggregation: Path | None = None,
    comparisons_root: Path | None = None,
    aggregations_root: Path | None = None,
    dataset_id: str | None = None,
    baseline_mask_provider: str | None = None,
    baseline_restore_provider: str | None = None,
    candidate_mask_provider: str | None = None,
    candidate_restore_provider: str | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    comparison_path, comparison_data, baseline_latest, candidate_latest = _resolve_comparison_source(
        comparison=comparison,
        comparisons_root=comparisons_root,
        dataset_id=dataset_id,
        baseline_mask_provider=baseline_mask_provider,
        baseline_restore_provider=baseline_restore_provider,
        candidate_mask_provider=candidate_mask_provider,
        candidate_restore_provider=candidate_restore_provider,
    )

    baseline_aggregation = _resolve_aggregation_source(
        aggregation=aggregation,
        aggregations_root=aggregations_root,
        dataset_id=dataset_id or baseline_latest["dataset_id"],
        mask_provider=baseline_mask_provider or baseline_latest["mask_provider"],
        restore_provider=baseline_restore_provider or baseline_latest["restore_provider"],
        required=True,
    )
    baseline_aggregation_path, _, baseline_group = baseline_aggregation

    candidate_aggregation = _resolve_aggregation_source(
        aggregation=None,
        aggregations_root=aggregations_root,
        dataset_id=dataset_id or candidate_latest["dataset_id"],
        mask_provider=candidate_mask_provider or candidate_latest["mask_provider"],
        restore_provider=candidate_restore_provider or candidate_latest["restore_provider"],
        required=False,
    )

    candidate_aggregate_reference: dict[str, Any] | None = None
    if candidate_aggregation is not None:
        candidate_aggregation_path, _, candidate_group = candidate_aggregation
        candidate_aggregate_reference = {
            "path": str(candidate_aggregation_path),
            "group": candidate_group,
            "candidate_vs_aggregate": _build_trend_section(
                reference_group=candidate_group,
                candidate_latest=candidate_latest,
            ),
        }

    snapshot_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    dataset_value = dataset_id or baseline_latest["dataset_id"] or candidate_latest["dataset_id"]
    summary = {
        "snapshot_id": snapshot_id,
        "dataset_id": dataset_value,
        "comparison": {
            "path": str(comparison_path),
            "common_item_count": comparison_data.get("common_item_count"),
            "baseline_only_count": comparison_data.get("baseline_only_count"),
            "candidate_only_count": comparison_data.get("candidate_only_count"),
            "status_pair_counts": dict(comparison_data.get("status_pair_counts") or {}),
            "mask_overlap_summary": dict(comparison_data.get("mask_overlap_summary") or {}),
            "delta_summary": dict(comparison_data.get("delta_summary") or {}),
        },
        "baseline_latest": baseline_latest,
        "candidate_latest": candidate_latest,
        "baseline_aggregate": {
            "path": str(baseline_aggregation_path),
            "group": baseline_group,
            "candidate_vs_aggregate": _build_trend_section(
                reference_group=baseline_group,
                candidate_latest=candidate_latest,
            ),
        },
        "candidate_aggregate": candidate_aggregate_reference,
    }

    if output_dir is not None:
        output_dir = ensure_dir(output_dir.resolve())
        stem = _build_snapshot_stem(
            dataset_id=summary["dataset_id"],
            baseline_mask_provider=baseline_latest["mask_provider"],
            baseline_restore_provider=baseline_latest["restore_provider"],
            candidate_mask_provider=candidate_latest["mask_provider"],
            candidate_restore_provider=candidate_latest["restore_provider"],
            snapshot_id=snapshot_id,
        )
        output_json = output_dir / f"{stem}.json"
        output_markdown = output_dir / f"{stem}.md"
        latest_json = output_dir / "latest.json"
        latest_markdown = output_dir / "latest.md"
        write_json(output_json, summary)
        output_markdown.write_text(_render_trend_markdown(summary), encoding="utf-8")
        write_json(latest_json, summary)
        latest_markdown.write_text(_render_trend_markdown(summary), encoding="utf-8")
        summary["output_json"] = str(output_json)
        summary["output_markdown"] = str(output_markdown)
        summary["latest_json"] = str(latest_json)
        summary["latest_markdown"] = str(latest_markdown)

    return summary


def _resolve_comparison_source(
    *,
    comparison: Path | None,
    comparisons_root: Path | None,
    dataset_id: str | None,
    baseline_mask_provider: str | None,
    baseline_restore_provider: str | None,
    candidate_mask_provider: str | None,
    candidate_restore_provider: str | None,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    if comparison is not None:
        comparison_path = comparison.resolve()
        comparison_data = _load_json(comparison_path)
        baseline_latest, candidate_latest = _build_latest_report_contexts(comparison_data)
        _validate_comparison_filters(
            baseline_latest=baseline_latest,
            candidate_latest=candidate_latest,
            dataset_id=dataset_id,
            baseline_mask_provider=baseline_mask_provider,
            baseline_restore_provider=baseline_restore_provider,
            candidate_mask_provider=candidate_mask_provider,
            candidate_restore_provider=candidate_restore_provider,
        )
        return comparison_path, comparison_data, baseline_latest, candidate_latest

    if comparisons_root is None:
        raise ValueError("comparisons_root is required when --comparison is not provided.")

    root = comparisons_root.resolve()
    for path in _iter_json_files(root):
        comparison_data = _load_json(path)
        if not _looks_like_compare_summary(comparison_data):
            continue
        baseline_latest, candidate_latest = _build_latest_report_contexts(comparison_data)
        if _comparison_matches(
            baseline_latest=baseline_latest,
            candidate_latest=candidate_latest,
            dataset_id=dataset_id,
            baseline_mask_provider=baseline_mask_provider,
            baseline_restore_provider=baseline_restore_provider,
            candidate_mask_provider=candidate_mask_provider,
            candidate_restore_provider=candidate_restore_provider,
        ):
            return path, comparison_data, baseline_latest, candidate_latest

    raise FileNotFoundError(
        f"No comparison summary matched under {root} for dataset={dataset_id!r}, "
        f"baseline={baseline_mask_provider!r}/{baseline_restore_provider!r}, "
        f"candidate={candidate_mask_provider!r}/{candidate_restore_provider!r}."
    )


def _resolve_aggregation_source(
    *,
    aggregation: Path | None,
    aggregations_root: Path | None,
    dataset_id: str | None,
    mask_provider: str | None,
    restore_provider: str | None,
    required: bool,
) -> tuple[Path, dict[str, Any], dict[str, Any]] | None:
    if aggregation is not None:
        aggregation_path = aggregation.resolve()
        aggregation_data = _load_json(aggregation_path)
        group = _select_aggregation_group(
            aggregation_data,
            dataset_id=dataset_id,
            mask_provider=mask_provider,
            restore_provider=restore_provider,
        )
        if group is None:
            raise ValueError(
                "Aggregation file does not contain a matching group for "
                f"dataset={dataset_id!r}, mask_provider={mask_provider!r}, restore_provider={restore_provider!r}."
            )
        return aggregation_path, aggregation_data, group

    if aggregations_root is None:
        if required:
            raise ValueError("aggregations_root is required when --aggregation is not provided.")
        return None

    root = aggregations_root.resolve()
    for path in _iter_json_files(root):
        aggregation_data = _load_json(path)
        group = _select_aggregation_group(
            aggregation_data,
            dataset_id=dataset_id,
            mask_provider=mask_provider,
            restore_provider=restore_provider,
        )
        if group is not None:
            return path, aggregation_data, group

    if required:
        raise FileNotFoundError(
            f"No aggregation summary matched under {root} for dataset={dataset_id!r}, "
            f"mask_provider={mask_provider!r}, restore_provider={restore_provider!r}."
        )
    return None


def _build_latest_report_contexts(comparison_data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline_report = _load_report_identity(Path(str(comparison_data["baseline_report"])))
    candidate_report = _load_report_identity(Path(str(comparison_data["candidate_report"])))
    baseline_report["summary"] = dict(comparison_data.get("baseline_summary") or {})
    candidate_report["summary"] = dict(comparison_data.get("candidate_summary") or {})
    return baseline_report, candidate_report


def _load_report_identity(path: Path) -> dict[str, Any]:
    report_path = path.resolve()
    data = _load_json(report_path)
    if "results" not in data:
        raise ValueError(f"Unsupported benchmark report format: {report_path}")
    return {
        "path": str(report_path),
        "run_id": str(data.get("run_id") or ""),
        "dataset_id": str(data.get("dataset_id") or ""),
        "mask_provider": str(data.get("mask_provider") or ""),
        "restore_provider": str(data.get("restore_provider") or ""),
        "item_count": len(list(data.get("results") or [])),
        "status_counts": dict(data.get("status_counts") or {}),
    }


def _build_trend_section(
    *,
    reference_group: dict[str, Any],
    candidate_latest: dict[str, Any],
) -> dict[str, Any]:
    reference_metrics = dict(reference_group.get("mean_metrics") or {})
    candidate_metrics = dict((candidate_latest.get("summary") or {}).get("mean_metrics") or {})
    metric_deltas = {
        metric_key: _build_metric_snapshot(
            metric_key,
            reference_metrics.get(metric_key),
            candidate_metrics.get(metric_key),
        )
        for metric_key in BENCHMARK_METRIC_KEYS
    }
    confidence_delta = _build_metric_snapshot(
        "mean_confidence",
        reference_group.get("mean_confidence"),
        (candidate_latest.get("summary") or {}).get("mean_confidence"),
    )
    return {
        "reference_run_count": reference_group.get("run_count"),
        "reference_item_count": reference_group.get("item_count"),
        "candidate_run_id": candidate_latest.get("run_id"),
        "metric_deltas": metric_deltas,
        "confidence_delta": confidence_delta,
        "headlines": _build_headlines(metric_deltas, confidence_delta),
    }


def _build_metric_snapshot(metric_key: str, baseline_value: Any, candidate_value: Any) -> dict[str, Any]:
    baseline_number = _coerce_optional_float(baseline_value)
    candidate_number = _coerce_optional_float(candidate_value)
    delta = None
    if baseline_number is not None and candidate_number is not None:
        delta = round(candidate_number - baseline_number, 6)
    preference = METRIC_PREFERENCES.get(metric_key)
    return {
        "metric_key": metric_key,
        "preference": preference,
        "baseline_value": baseline_number,
        "candidate_value": candidate_number,
        "delta": delta,
        "assessment": _assess_metric_delta(preference, delta),
    }


def _build_headlines(metric_deltas: dict[str, dict[str, Any]], confidence_delta: dict[str, Any]) -> list[dict[str, Any]]:
    headlines: list[dict[str, Any]] = []
    for metric_key in HEADLINE_METRICS:
        snapshot = confidence_delta if metric_key == "mean_confidence" else metric_deltas.get(metric_key)
        if snapshot is None or snapshot.get("delta") is None:
            continue
        if snapshot.get("assessment") == "flat":
            continue
        headlines.append(dict(snapshot))
    return headlines


def _assess_metric_delta(preference: str | None, delta: float | None) -> str:
    if delta is None:
        return "unavailable"
    if abs(delta) < 1e-9:
        return "flat"
    if preference == "lower":
        return "improved" if delta < 0 else "regressed"
    if preference == "higher":
        return "improved" if delta > 0 else "regressed"
    return "changed"


def _validate_comparison_filters(
    *,
    baseline_latest: dict[str, Any],
    candidate_latest: dict[str, Any],
    dataset_id: str | None,
    baseline_mask_provider: str | None,
    baseline_restore_provider: str | None,
    candidate_mask_provider: str | None,
    candidate_restore_provider: str | None,
) -> None:
    if not _comparison_matches(
        baseline_latest=baseline_latest,
        candidate_latest=candidate_latest,
        dataset_id=dataset_id,
        baseline_mask_provider=baseline_mask_provider,
        baseline_restore_provider=baseline_restore_provider,
        candidate_mask_provider=candidate_mask_provider,
        candidate_restore_provider=candidate_restore_provider,
    ):
        raise ValueError("Explicit comparison file does not match the requested dataset/provider filters.")


def _comparison_matches(
    *,
    baseline_latest: dict[str, Any],
    candidate_latest: dict[str, Any],
    dataset_id: str | None,
    baseline_mask_provider: str | None,
    baseline_restore_provider: str | None,
    candidate_mask_provider: str | None,
    candidate_restore_provider: str | None,
) -> bool:
    if dataset_id is not None and (
        baseline_latest["dataset_id"] != dataset_id or candidate_latest["dataset_id"] != dataset_id
    ):
        return False
    if baseline_mask_provider is not None and baseline_latest["mask_provider"] != baseline_mask_provider:
        return False
    if baseline_restore_provider is not None and baseline_latest["restore_provider"] != baseline_restore_provider:
        return False
    if candidate_mask_provider is not None and candidate_latest["mask_provider"] != candidate_mask_provider:
        return False
    if candidate_restore_provider is not None and candidate_latest["restore_provider"] != candidate_restore_provider:
        return False
    return True


def _select_aggregation_group(
    aggregation_data: dict[str, Any],
    *,
    dataset_id: str | None,
    mask_provider: str | None,
    restore_provider: str | None,
) -> dict[str, Any] | None:
    groups = list(aggregation_data.get("groups") or [])
    if not groups:
        return None
    matches = [
        group
        for group in groups
        if (dataset_id is None or str(group.get("dataset_id") or "") == dataset_id)
        and (mask_provider is None or str(group.get("mask_provider") or "") == mask_provider)
        and (restore_provider is None or str(group.get("restore_provider") or "") == restore_provider)
    ]
    if matches:
        return matches[0]
    if dataset_id is None and mask_provider is None and restore_provider is None and len(groups) == 1:
        return groups[0]
    return None


def _looks_like_compare_summary(data: dict[str, Any]) -> bool:
    return "baseline_report" in data and "candidate_report" in data and "delta_summary" in data


def _iter_json_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        (
            path
            for path in root.glob("*.json")
            if path.is_file() and path.name != "latest.json"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _build_snapshot_stem(
    *,
    dataset_id: str,
    baseline_mask_provider: str,
    baseline_restore_provider: str,
    candidate_mask_provider: str,
    candidate_restore_provider: str,
    snapshot_id: str,
) -> str:
    return (
        f"trend_{dataset_id}_{baseline_mask_provider}__{baseline_restore_provider}"
        f"__vs__{candidate_mask_provider}__{candidate_restore_provider}_{snapshot_id}"
    )


def _render_trend_markdown(summary: dict[str, Any]) -> str:
    baseline_latest = summary["baseline_latest"]
    candidate_latest = summary["candidate_latest"]
    baseline_group = summary["baseline_aggregate"]["group"]
    baseline_trend = summary["baseline_aggregate"]["candidate_vs_aggregate"]

    lines = [
        "# Benchmark Trend Snapshot",
        "",
        f"- Snapshot ID: `{summary['snapshot_id']}`",
        f"- Dataset: `{summary['dataset_id']}`",
        (
            "- Baseline aggregate: "
            f"`{baseline_group['mask_provider']} + {baseline_group['restore_provider']}` "
            f"across `{baseline_group['run_count']}` runs / `{baseline_group['item_count']}` items"
        ),
        (
            "- Candidate latest run: "
            f"`{candidate_latest['mask_provider']} + {candidate_latest['restore_provider']}` "
            f"(run `{candidate_latest['run_id']}`)"
        ),
        f"- Comparison source: `{summary['comparison']['path']}`",
        f"- Aggregation source: `{summary['baseline_aggregate']['path']}`",
        "",
        "## Headlines",
        "",
    ]

    if baseline_trend["headlines"]:
        for headline in baseline_trend["headlines"]:
            lines.append(
                "- "
                f"`{headline['metric_key']}`: {headline['assessment']} "
                f"(baseline `{headline['baseline_value']}`, candidate `{headline['candidate_value']}`, delta `{headline['delta']}`)"
            )
    else:
        lines.append("- No headline deltas were available.")

    lines.extend(
        [
            "",
            "## Latest Pairing",
            "",
            f"- Baseline latest run: `{baseline_latest['mask_provider']} + {baseline_latest['restore_provider']}` (`{baseline_latest['run_id']}`)",
            f"- Candidate latest run: `{candidate_latest['mask_provider']} + {candidate_latest['restore_provider']}` (`{candidate_latest['run_id']}`)",
            f"- Paired items: `{summary['comparison']['common_item_count']}`",
            f"- Status pairs: `{summary['comparison']['status_pair_counts']}`",
            f"- Mask overlap: `{summary['comparison']['mask_overlap_summary']}`",
            "",
            "## Candidate Vs Baseline Aggregate",
            "",
            "| Metric | Baseline Aggregate | Candidate Latest | Delta | Assessment |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )

    for metric_key in BENCHMARK_METRIC_KEYS:
        metric = baseline_trend["metric_deltas"][metric_key]
        lines.append(
            f"| `{metric_key}` | {metric['baseline_value']} | {metric['candidate_value']} | "
            f"{metric['delta']} | {metric['assessment']} |"
        )

    confidence = baseline_trend["confidence_delta"]
    lines.append(
        f"| `mean_confidence` | {confidence['baseline_value']} | {confidence['candidate_value']} | "
        f"{confidence['delta']} | {confidence['assessment']} |"
    )

    if summary["candidate_aggregate"] is not None:
        candidate_group = summary["candidate_aggregate"]["group"]
        candidate_trend = summary["candidate_aggregate"]["candidate_vs_aggregate"]
        lines.extend(
            [
                "",
                "## Candidate Vs Candidate Aggregate",
                "",
                (
                    "- Candidate aggregate reference: "
                    f"`{candidate_group['mask_provider']} + {candidate_group['restore_provider']}` "
                    f"across `{candidate_group['run_count']}` runs / `{candidate_group['item_count']}` items"
                ),
                "",
                "| Metric | Candidate Aggregate | Candidate Latest | Delta | Assessment |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
        )
        for metric_key in BENCHMARK_METRIC_KEYS:
            metric = candidate_trend["metric_deltas"][metric_key]
            lines.append(
                f"| `{metric_key}` | {metric['baseline_value']} | {metric['candidate_value']} | "
                f"{metric['delta']} | {metric['assessment']} |"
            )
        confidence = candidate_trend["confidence_delta"]
        lines.append(
            f"| `mean_confidence` | {confidence['baseline_value']} | {confidence['candidate_value']} | "
            f"{confidence['delta']} | {confidence['assessment']} |"
        )

    lines.append("")
    return "\n".join(lines)
