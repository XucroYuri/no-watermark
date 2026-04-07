from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .io_utils import ensure_dir, write_json


def build_review_bundle(
    *,
    report_paths: list[Path],
    output_dir: Path,
    provider_labels: list[str] | None = None,
    compare_paths: list[Path] | None = None,
    trend_paths: list[Path] | None = None,
) -> dict[str, Any]:
    if len(report_paths) < 2:
        raise ValueError("review bundles require at least two report files")

    compare_paths = compare_paths or []
    trend_paths = trend_paths or []
    reports = [_load_report(path) for path in report_paths]
    first_report = reports[0]
    dataset_id = str(first_report.get("dataset_id", "unknown"))
    if provider_labels is not None and len(provider_labels) != len(report_paths):
        raise ValueError("provider_labels must match the number of report paths")
    derived_labels = provider_labels or [_provider_label(report) for report in reports]
    provider_labels = _dedupe_provider_labels(derived_labels, reports)

    output_dir = output_dir.resolve()
    ensure_dir(output_dir)

    results_by_provider: dict[str, dict[str, dict[str, Any]]] = {}
    provider_summaries: dict[str, dict[str, Any]] = {}
    item_order: list[str] = []
    item_display_names: dict[str, str] = {}

    for report_path, report, provider_label in zip(report_paths, reports, provider_labels):
        provider_results: dict[str, dict[str, Any]] = {}
        for result in report.get("results", []):
            item = result.get("item", {})
            item_id = str(item.get("item_id") or item.get("relative_path") or "").strip()
            if not item_id:
                continue
            provider_results[item_id] = result
            item_display_names.setdefault(item_id, str(item.get("relative_path") or item_id))
            if provider_label == provider_labels[0]:
                item_order.append(item_id)
        results_by_provider[provider_label] = provider_results
        provider_summaries[provider_label] = _summarize_report(report_path, report)

    common_item_ids = set(item_order)
    for provider_label in provider_labels[1:]:
        common_item_ids &= set(results_by_provider[provider_label])
    ordered_item_ids = [item_id for item_id in item_order if item_id in common_item_ids]
    if not ordered_item_ids:
        raise ValueError("review bundles require at least one common item across all reports")

    review_bundle = {
        "dataset_id": dataset_id,
        "provider_labels": provider_labels,
        "item_count": len(ordered_item_ids),
        "report_artifacts": _copy_artifacts(
            artifact_paths=report_paths,
            output_dir=output_dir / "artifacts" / "reports",
            root=output_dir,
        ),
        "comparison_artifacts": _copy_artifacts(
            artifact_paths=compare_paths,
            output_dir=output_dir / "artifacts" / "comparisons",
            root=output_dir,
        ),
        "trend_artifacts": _copy_artifacts(
            artifact_paths=trend_paths,
            output_dir=output_dir / "artifacts" / "trends",
            root=output_dir,
        ),
        "providers": provider_summaries,
        "items": [],
    }

    first_provider = provider_labels[0]
    for item_id in ordered_item_ids:
        reference_result = results_by_provider[first_provider][item_id]
        reference_item = dict(reference_result.get("item", {}))
        item_dir = output_dir / "items" / item_id
        source_dir = ensure_dir(item_dir / "source")

        input_path = _path_or_none(reference_item.get("input_path"))
        seed_mask_path = _path_or_none(reference_item.get("seed_mask_path"))
        seed_overlay_path = _path_or_none(reference_item.get("seed_overlay_path"))
        prompt_path = _path_or_none(reference_item.get("prompt_path"))

        source_entry = {
            "input": _copy_file_if_exists(
                input_path,
                source_dir / f"input{(input_path.suffix if input_path is not None else '.jpg')}",
                root=output_dir,
            ),
            "seed_mask": _copy_file_if_exists(seed_mask_path, source_dir / "seed_mask.png", root=output_dir),
            "seed_overlay": _copy_file_if_exists(seed_overlay_path, source_dir / "seed_overlay.jpg", root=output_dir),
            "prompt": _copy_file_if_exists(prompt_path, source_dir / "prompt.txt", root=output_dir),
        }

        provider_entries: dict[str, dict[str, Any]] = {}
        for provider_label in provider_labels:
            result = results_by_provider[provider_label][item_id]
            provider_dir = ensure_dir(item_dir / provider_label)
            mask_path = _path_or_none(result.get("mask_path"))
            overlay_path = _path_or_none(result.get("overlay_path"))
            restored_path = _path_or_none(result.get("restored_path"))
            provider_entries[provider_label] = {
                "status": result.get("status"),
                "note": result.get("note", ""),
                "confidence": _extract_confidence(result),
                "metrics": result.get("metrics", {}),
                "mask": _copy_file_if_exists(
                    mask_path,
                    provider_dir / f"mask{(mask_path.suffix if mask_path is not None else '.png')}",
                    root=output_dir,
                ),
                "overlay": _copy_file_if_exists(
                    overlay_path,
                    provider_dir / f"overlay{(overlay_path.suffix if overlay_path is not None else '.jpg')}",
                    root=output_dir,
                ),
                "restored": _copy_file_if_exists(
                    restored_path,
                    provider_dir / f"restored{(restored_path.suffix if restored_path is not None else '.jpg')}",
                    root=output_dir,
                ),
            }

        item_entry = {
            "item_id": item_id,
            "relative_path": item_display_names.get(item_id, item_id),
            "source_category": reference_item.get("source_category"),
            "benchmark_category": reference_item.get("benchmark_category"),
            "prompt": reference_item.get("prompt"),
            "source": source_entry,
            "providers": provider_entries,
        }
        write_json(item_dir / "item.json", item_entry)
        review_bundle["items"].append(item_entry)

    write_json(output_dir / "review.json", review_bundle)
    (output_dir / "README.md").write_text(_render_review_markdown(review_bundle), encoding="utf-8")
    return review_bundle


def _load_report(path: Path) -> dict[str, Any]:
    candidate = path.resolve()
    if not candidate.exists():
        raise ValueError(f"Review report does not exist: {candidate}")
    return json.loads(candidate.read_text(encoding="utf-8"))


def _provider_label(report: dict[str, Any]) -> str:
    restore_provider = str(report.get("restore_provider") or "").strip()
    if restore_provider:
        return restore_provider
    return str(report.get("mask_provider") or "provider")


def _dedupe_provider_labels(labels: list[str], reports: list[dict[str, Any]]) -> list[str]:
    totals = Counter(labels)
    seen: Counter[str] = Counter()
    deduped: list[str] = []
    for label, report in zip(labels, reports):
        seen[label] += 1
        if totals[label] == 1:
            deduped.append(label)
            continue
        run_id = str(report.get("run_id") or "").strip()
        suffix = run_id or str(seen[label])
        deduped.append(f"{label}@{suffix}")
    return deduped


def _summarize_report(report_path: Path, report: dict[str, Any]) -> dict[str, Any]:
    results = list(report.get("results", []))
    status_counts = report.get("status_counts") or dict(Counter(result.get("status", "unknown") for result in results))
    metric_samples: dict[str, list[float]] = defaultdict(list)
    confidence_samples: list[float] = []
    for result in results:
        for key, value in dict(result.get("metrics", {})).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metric_samples[key].append(float(value))
        confidence = _extract_confidence(result)
        if confidence is not None:
            confidence_samples.append(confidence)
    mean_metrics = {
        key: round(sum(values) / len(values), 6)
        for key, values in sorted(metric_samples.items())
        if values
    }
    mean_confidence = round(sum(confidence_samples) / len(confidence_samples), 6) if confidence_samples else None
    return {
        "report_path": str(report_path.resolve()),
        "item_count": len(results),
        "status_counts": status_counts,
        "mean_metrics": mean_metrics,
        "mean_confidence": mean_confidence,
    }


def _extract_confidence(result: dict[str, Any]) -> float | None:
    value = dict(result.get("mask_result", {})).get("confidence")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _copy_artifacts(*, artifact_paths: list[Path], output_dir: Path, root: Path) -> list[str]:
    ensure_dir(output_dir)
    name_counts = Counter(Path(path).name for path in artifact_paths)
    copied: list[str] = []
    for artifact_path in artifact_paths:
        source_path = Path(artifact_path)
        if name_counts[source_path.name] == 1:
            destination = output_dir / source_path.name
        else:
            destination = output_dir / _artifact_relative_path(source_path, skip_root_name=output_dir.name)
        copied_path = _copy_file_if_exists(
            source_path,
            destination,
            root=root,
        )
        if copied_path is not None:
            copied.append(copied_path)
    return copied


def _copy_file_if_exists(source: Path | None, destination: Path, *, root: Path) -> str | None:
    if source is None:
        return None
    resolved = source.resolve()
    if not resolved.exists() or not resolved.is_file():
        return None
    ensure_dir(destination.parent)
    shutil.copy2(resolved, destination)
    return destination.relative_to(root).as_posix()


def _path_or_none(value: Any) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    return Path(text).expanduser()


def _artifact_relative_path(source: Path, *, skip_root_name: str | None = None) -> Path:
    parts = [parent.name for parent in reversed(source.parents[:2]) if parent.name]
    if skip_root_name and parts and parts[0] == skip_root_name:
        parts = parts[1:]
    if not parts:
        return Path(source.name)
    return Path(*parts) / source.name


def _render_review_markdown(review_bundle: dict[str, Any]) -> str:
    lines = [
        "# Review Bundle",
        "",
        f"- Dataset: `{review_bundle['dataset_id']}`",
        f"- Providers: {', '.join(f'`{provider}`' for provider in review_bundle['provider_labels'])}",
        f"- Common item count: `{review_bundle['item_count']}`",
        "",
        "## Provider Summary",
        "",
    ]
    for provider_label in review_bundle["provider_labels"]:
        provider_summary = review_bundle["providers"][provider_label]
        mean_metrics = provider_summary["mean_metrics"]
        metric_bits: list[str] = []
        for metric_name in ("mean_abs_diff", "edge_delta", "restore_latency_ms", "mask_latency_ms"):
            if metric_name in mean_metrics:
                metric_bits.append(f"`{metric_name} = {mean_metrics[metric_name]}`")
        lines.append(f"### `{provider_label}`")
        lines.append("")
        lines.append(f"- Report: `{Path(provider_summary['report_path']).name}`")
        lines.append(f"- Item count: `{provider_summary['item_count']}`")
        lines.append(f"- Status counts: `{provider_summary['status_counts']}`")
        if provider_summary["mean_confidence"] is not None:
            lines.append(f"- Mean confidence: `{provider_summary['mean_confidence']}`")
        if metric_bits:
            lines.append(f"- Means: {', '.join(metric_bits)}")
        lines.append("")
    if review_bundle["comparison_artifacts"]:
        lines.extend(["## Comparison Artifacts", ""])
        for artifact in review_bundle["comparison_artifacts"]:
            lines.append(f"- `{artifact}`")
        lines.append("")
    if review_bundle["trend_artifacts"]:
        lines.extend(["## Trend Artifacts", ""])
        for artifact in review_bundle["trend_artifacts"]:
            lines.append(f"- `{artifact}`")
        lines.append("")
    lines.extend(["## Items", ""])
    for item in review_bundle["items"]:
        lines.append(f"### `{item['item_id']}`")
        lines.append("")
        lines.append(f"- Relative path: `{item['relative_path']}`")
        if item.get("source_category"):
            lines.append(f"- Source category: `{item['source_category']}`")
        if item.get("benchmark_category"):
            lines.append(f"- Benchmark category: `{item['benchmark_category']}`")
        for provider_label in review_bundle["provider_labels"]:
            provider_entry = item["providers"][provider_label]
            metric_bits = []
            for metric_name in ("mean_abs_diff", "edge_delta", "restore_latency_ms", "mask_latency_ms"):
                if metric_name in provider_entry["metrics"]:
                    metric_bits.append(f"{metric_name}={provider_entry['metrics'][metric_name]}")
            metric_text = ", ".join(metric_bits)
            line = (
                f"- `{provider_label}`: status=`{provider_entry['status']}`, restored=`{provider_entry['restored']}`"
            )
            if metric_text:
                line += f", {metric_text}"
            lines.append(line)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
