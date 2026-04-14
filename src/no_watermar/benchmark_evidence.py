from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .benchmark_aggregate import aggregate_benchmark_reports
from .benchmark_compare import compare_benchmark_reports
from .benchmark_trends import build_benchmark_trend_snapshot
from .io_utils import ensure_dir, write_json

DEFAULT_BASELINE_MASK_PROVIDER = "seed_manifest"
DEFAULT_BASELINE_RESTORE_PROVIDER = "telea"
DEFAULT_CANDIDATE_MASK_PROVIDER = "paddleocr"
DEFAULT_CANDIDATE_RESTORE_PROVIDER = "telea"
DEFAULT_OPTIONAL_MASK_PROVIDER = "seed_manifest"
DEFAULT_OPTIONAL_RESTORE_PROVIDER = "lama"


def build_stable_baseline_evidence(
    *,
    benchmark_root: Path,
    dataset_id: str,
    baseline_mask_provider: str = DEFAULT_BASELINE_MASK_PROVIDER,
    baseline_restore_provider: str = DEFAULT_BASELINE_RESTORE_PROVIDER,
    candidate_mask_provider: str = DEFAULT_CANDIDATE_MASK_PROVIDER,
    candidate_restore_provider: str = DEFAULT_CANDIDATE_RESTORE_PROVIDER,
    optional_mask_provider: str = DEFAULT_OPTIONAL_MASK_PROVIDER,
    optional_restore_provider: str = DEFAULT_OPTIONAL_RESTORE_PROVIDER,
    include_optional: bool = True,
    minimum_run_count: int = 3,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    benchmark_root = benchmark_root.resolve()
    reports_root = benchmark_root / "runs"
    aggregations_root = ensure_dir(benchmark_root / "aggregations")
    comparisons_root = ensure_dir(benchmark_root / "comparisons")
    trends_root = ensure_dir(benchmark_root / "trends")

    baseline_pair = _build_pair_summary(
        reports_root=reports_root,
        aggregations_root=aggregations_root,
        dataset_id=dataset_id,
        mask_provider=baseline_mask_provider,
        restore_provider=baseline_restore_provider,
    )
    candidate_pair = _build_pair_summary(
        reports_root=reports_root,
        aggregations_root=aggregations_root,
        dataset_id=dataset_id,
        mask_provider=candidate_mask_provider,
        restore_provider=candidate_restore_provider,
    )
    release_blocking = _build_evidence_section(
        section_name="release_blocking",
        baseline_pair=baseline_pair,
        candidate_pair=candidate_pair,
        dataset_id=dataset_id,
        minimum_run_count=minimum_run_count,
        comparisons_root=comparisons_root,
        aggregations_root=aggregations_root,
        trends_root=trends_root,
    )

    if include_optional:
        optional_pair = _build_pair_summary(
            reports_root=reports_root,
            aggregations_root=aggregations_root,
            dataset_id=dataset_id,
            mask_provider=optional_mask_provider,
            restore_provider=optional_restore_provider,
        )
        optional_section = _build_evidence_section(
            section_name="optional_stable",
            baseline_pair=baseline_pair,
            candidate_pair=optional_pair,
            dataset_id=dataset_id,
            minimum_run_count=minimum_run_count,
            comparisons_root=comparisons_root,
            aggregations_root=aggregations_root,
            trends_root=trends_root,
        )
    else:
        optional_section = {
            "status": "skipped",
            "ready": False,
            "issues": [],
            "baseline": baseline_pair,
            "candidate": None,
            "comparison_json": None,
            "comparison_csv": None,
            "trend_json": None,
            "trend_markdown": None,
            "headlines": [],
        }

    overall_status = _resolve_overall_status(release_blocking, optional_section)
    summary: dict[str, Any] = {
        "evidence_id": datetime.now().strftime("%Y%m%d-%H%M%S-%f"),
        "status": overall_status,
        "benchmark_root": str(benchmark_root),
        "dataset_id": dataset_id,
        "minimum_run_count": minimum_run_count,
        "release_blocking": release_blocking,
        "optional_stable": optional_section,
    }

    resolved_output_dir = ensure_dir((output_dir or (benchmark_root / "evidence")).resolve())
    stem = (
        f"stable_baseline_{dataset_id}_{baseline_mask_provider}__{baseline_restore_provider}"
        f"__vs__{candidate_mask_provider}__{candidate_restore_provider}_{summary['evidence_id']}"
    )
    output_json = resolved_output_dir / f"{stem}.json"
    output_markdown = resolved_output_dir / f"{stem}.md"
    latest_json = resolved_output_dir / "latest.json"
    latest_markdown = resolved_output_dir / "latest.md"
    write_json(output_json, summary)
    output_markdown.write_text(_render_evidence_markdown(summary), encoding="utf-8")
    write_json(latest_json, summary)
    latest_markdown.write_text(_render_evidence_markdown(summary), encoding="utf-8")
    summary["output_json"] = str(output_json)
    summary["output_markdown"] = str(output_markdown)
    summary["latest_json"] = str(latest_json)
    summary["latest_markdown"] = str(latest_markdown)
    write_json(output_json, summary)
    write_json(latest_json, summary)
    return summary


def _build_pair_summary(
    *,
    reports_root: Path,
    aggregations_root: Path,
    dataset_id: str,
    mask_provider: str,
    restore_provider: str,
) -> dict[str, Any]:
    aggregation = aggregate_benchmark_reports(
        reports_root=reports_root,
        dataset_id=dataset_id,
        mask_provider=mask_provider,
        restore_provider=restore_provider,
        output_dir=aggregations_root,
    )
    group = _first_matching_group(aggregation, dataset_id=dataset_id, mask_provider=mask_provider, restore_provider=restore_provider)
    latest_report = _find_latest_report(
        reports_root=reports_root,
        dataset_id=dataset_id,
        mask_provider=mask_provider,
        restore_provider=restore_provider,
    )

    return {
        "dataset_id": dataset_id,
        "mask_provider": mask_provider,
        "restore_provider": restore_provider,
        "latest_report": latest_report["path"] if latest_report else None,
        "latest_run_id": latest_report["run_id"] if latest_report else None,
        "aggregate_json": aggregation.get("output_json"),
        "aggregate_csv": aggregation.get("output_csv"),
        "run_count": int(group.get("run_count") or 0) if group else 0,
        "item_count": int(group.get("item_count") or 0) if group else 0,
        "aggregate_group": group,
    }


def _build_evidence_section(
    *,
    section_name: str,
    baseline_pair: dict[str, Any],
    candidate_pair: dict[str, Any],
    dataset_id: str,
    minimum_run_count: int,
    comparisons_root: Path,
    aggregations_root: Path,
    trends_root: Path,
) -> dict[str, Any]:
    issues = _collect_pair_issues(
        pair=baseline_pair,
        minimum_run_count=minimum_run_count,
        pair_role="baseline",
    )
    issues.extend(
        _collect_pair_issues(
            pair=candidate_pair,
            minimum_run_count=minimum_run_count,
            pair_role="candidate",
        )
    )

    comparison_json: str | None = None
    comparison_csv: str | None = None
    trend_json: str | None = None
    trend_markdown: str | None = None
    headlines: list[dict[str, Any]] = []

    if baseline_pair["latest_report"] and candidate_pair["latest_report"]:
        comparison = compare_benchmark_reports(
            Path(str(baseline_pair["latest_report"])),
            Path(str(candidate_pair["latest_report"])),
            output_dir=comparisons_root,
        )
        comparison_json = comparison.get("output_json")
        comparison_csv = comparison.get("output_csv")
        if baseline_pair["aggregate_group"] is not None and candidate_pair["aggregate_group"] is not None:
            trend = build_benchmark_trend_snapshot(
                comparison=Path(str(comparison_json)) if comparison_json else None,
                aggregations_root=aggregations_root,
                dataset_id=dataset_id,
                baseline_mask_provider=str(baseline_pair["mask_provider"]),
                baseline_restore_provider=str(baseline_pair["restore_provider"]),
                candidate_mask_provider=str(candidate_pair["mask_provider"]),
                candidate_restore_provider=str(candidate_pair["restore_provider"]),
                output_dir=trends_root,
            )
            trend_json = trend.get("output_json")
            trend_markdown = trend.get("output_markdown")
            headlines = list((trend.get("baseline_aggregate") or {}).get("candidate_vs_aggregate", {}).get("headlines") or [])
        else:
            issues.append(
                {
                    "pair_role": "candidate",
                    "issue_code": "missing_aggregate_group",
                    "detail": (
                        f"{candidate_pair['mask_provider']} + {candidate_pair['restore_provider']} "
                        "does not have an aggregate group yet."
                    ),
                }
            )
    else:
        if not baseline_pair["latest_report"]:
            issues.append(
                {
                    "pair_role": "baseline",
                    "issue_code": "missing_latest_report",
                    "detail": (
                        f"{baseline_pair['mask_provider']} + {baseline_pair['restore_provider']} "
                        "does not have a benchmark report yet."
                    ),
                }
            )
        if not candidate_pair["latest_report"]:
            issues.append(
                {
                    "pair_role": "candidate",
                    "issue_code": "missing_latest_report",
                    "detail": (
                        f"{candidate_pair['mask_provider']} + {candidate_pair['restore_provider']} "
                        "does not have a benchmark report yet."
                    ),
                }
            )

    ready = len(issues) == 0 and trend_json is not None
    status = "ready" if ready else "action_required"
    return {
        "status": status,
        "ready": ready,
        "baseline": baseline_pair,
        "candidate": candidate_pair,
        "comparison_json": comparison_json,
        "comparison_csv": comparison_csv,
        "trend_json": trend_json,
        "trend_markdown": trend_markdown,
        "headlines": headlines,
        "issues": _unique_issue_list(issues),
        "section_name": section_name,
    }


def _collect_pair_issues(
    *,
    pair: dict[str, Any],
    minimum_run_count: int,
    pair_role: str,
) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    if pair["aggregate_group"] is None:
        issues.append(
            {
                "pair_role": pair_role,
                "issue_code": "missing_aggregate_group",
                "detail": f"{pair['mask_provider']} + {pair['restore_provider']} has not produced any aggregate group yet.",
            }
        )
        return issues

    if int(pair["run_count"]) < minimum_run_count:
        issues.append(
            {
                "pair_role": pair_role,
                "issue_code": "run_count_below_threshold",
                "detail": (
                    f"{pair['mask_provider']} + {pair['restore_provider']} only has {pair['run_count']} runs; "
                    f"{minimum_run_count} are required."
                ),
            }
        )
    return issues


def _find_latest_report(
    *,
    reports_root: Path,
    dataset_id: str,
    mask_provider: str,
    restore_provider: str,
) -> dict[str, str] | None:
    best_match: dict[str, str] | None = None
    for path in reports_root.rglob("*.json"):
        if path.name == "summary.json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(data.get("dataset_id") or "") != dataset_id:
            continue
        if str(data.get("mask_provider") or "") != mask_provider:
            continue
        if str(data.get("restore_provider") or "") != restore_provider:
            continue
        run_id = str(data.get("run_id") or "")
        candidate = {"path": str(path.resolve()), "run_id": run_id}
        if best_match is None or candidate["run_id"] > best_match["run_id"]:
            best_match = candidate
    return best_match


def _first_matching_group(
    aggregation: dict[str, Any],
    *,
    dataset_id: str,
    mask_provider: str,
    restore_provider: str,
) -> dict[str, Any] | None:
    for group in list(aggregation.get("groups") or []):
        if (
            str(group.get("dataset_id") or "") == dataset_id
            and str(group.get("mask_provider") or "") == mask_provider
            and str(group.get("restore_provider") or "") == restore_provider
        ):
            return group
    return None


def _unique_issue_list(issues: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for issue in issues:
        key = (issue["pair_role"], issue["issue_code"], issue["detail"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(issue)
    return deduped


def _resolve_overall_status(release_blocking: dict[str, Any], optional_stable: dict[str, Any]) -> str:
    if release_blocking.get("ready") and optional_stable.get("status") in {"ready", "skipped"}:
        return "ready"
    if release_blocking.get("ready"):
        return "release_blocking_ready"
    return "action_required"


def _render_evidence_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Stable Baseline Evidence",
        "",
        f"- Evidence ID: `{summary['evidence_id']}`",
        f"- Dataset: `{summary['dataset_id']}`",
        f"- Minimum run count: `{summary['minimum_run_count']}`",
        f"- Status: `{summary['status']}`",
        "",
        "## Release-Blocking Stable Path",
        "",
    ]
    lines.extend(_render_section_markdown(summary["release_blocking"]))
    lines.extend(["", "## Optional Stable Path", ""])
    if summary["optional_stable"]["status"] == "skipped":
        lines.append("- Optional stable evidence was skipped for this run.")
    else:
        lines.extend(_render_section_markdown(summary["optional_stable"]))
    lines.append("")
    return "\n".join(lines)


def _render_section_markdown(section: dict[str, Any]) -> list[str]:
    baseline = section["baseline"]
    candidate = section["candidate"]
    lines = [
        f"- Ready: `{section['ready']}`",
        f"- Baseline pair: `{baseline['mask_provider']} + {baseline['restore_provider']}` across `{baseline['run_count']}` runs",
        f"- Candidate pair: `{candidate['mask_provider']} + {candidate['restore_provider']}` across `{candidate['run_count']}` runs",
    ]
    if section.get("trend_json"):
        lines.append(f"- Trend JSON: `{section['trend_json']}`")
    if section.get("headlines"):
        lines.append("")
        lines.append("### Headlines")
        lines.append("")
        for headline in section["headlines"]:
            lines.append(
                "- "
                f"`{headline['metric_key']}` {headline['assessment']} "
                f"(baseline `{headline['baseline_value']}`, candidate `{headline['candidate_value']}`, delta `{headline['delta']}`)"
            )
    if section.get("issues"):
        lines.append("")
        lines.append("### Issues")
        lines.append("")
        for issue in section["issues"]:
            lines.append(f"- `{issue['issue_code']}`: {issue['detail']}")
    return lines
