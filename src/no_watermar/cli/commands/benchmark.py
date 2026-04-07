from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ...benchmark_aggregate import aggregate_benchmark_reports
from ...benchmark_compare import compare_benchmark_reports
from ...benchmark_dataset import DATASET_COVER, DATASET_REGULAR, prepare_benchmark_dataset
from ...benchmark_paddleocr_session import OCR_SESSION_MODE_AUTO, OCR_SESSION_MODE_CHOICES
from ...benchmark_providers import list_provider_descriptors, probe_provider_runtimes
from ...benchmark_runner import run_benchmark
from ...benchmark_trends import build_benchmark_trend_snapshot
from ...config import dataset_profile_to_dict, provider_profile_to_dict, resolve_dataset_profile, resolve_provider_profile
from ...io_utils import print_json


def configure_benchmark_group(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("benchmark", help="Run benchmark workflows.")
    benchmark_subparsers = parser.add_subparsers(dest="benchmark_command", required=True)

    prepare_parser = benchmark_subparsers.add_parser("prepare", help="Build benchmark dataset manifests and seed masks.")
    prepare_parser.add_argument("--dataset-profile", default=None, help="Optional named dataset profile from no-watermar.toml.")
    prepare_parser.add_argument("--input", type=Path, default=None, help="Input root to scan.")
    prepare_parser.add_argument("--benchmark-root", type=Path, default=_default_benchmark_root(), help="Benchmark root.")
    prepare_parser.add_argument("--no-recursive", action="store_true", help="Only scan the top-level directory.")
    prepare_parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of images to include.")
    prepare_parser.set_defaults(handler=_handle_prepare)

    run_parser = benchmark_subparsers.add_parser("run", help="Run a benchmark dataset with selected providers.")
    run_parser.add_argument("--dataset-profile", default=None, help="Optional named dataset profile from no-watermar.toml.")
    run_parser.add_argument("--provider-profile", default=None, help="Optional named provider profile from no-watermar.toml.")
    run_parser.add_argument("--benchmark-root", type=Path, default=_default_benchmark_root(), help="Benchmark root.")
    run_parser.add_argument(
        "--dataset",
        default=None,
        choices=[DATASET_REGULAR, DATASET_COVER, "all"],
        help="Dataset manifest to run.",
    )
    run_parser.add_argument("--mask-provider", default=None, help="Mask provider name.")
    run_parser.add_argument("--restore-provider", default=None, help="Restore provider name.")
    run_parser.add_argument("--limit", type=int, default=None, help="Optional limit per dataset run.")
    run_parser.add_argument(
        "--ocr-session-mode",
        default=None,
        choices=OCR_SESSION_MODE_CHOICES,
        help="OCR execution mode for PaddleOCR-backed detection and OCR residual scoring.",
    )
    run_parser.set_defaults(handler=_handle_run)

    compare_parser = benchmark_subparsers.add_parser("compare", help="Compare two benchmark report JSON files.")
    compare_parser.add_argument("--baseline-report", type=Path, required=True, help="Baseline report JSON.")
    compare_parser.add_argument("--candidate-report", type=Path, required=True, help="Candidate report JSON.")
    compare_parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_benchmark_root() / "comparisons",
        help="Directory for comparison outputs.",
    )
    compare_parser.set_defaults(handler=_handle_compare)

    aggregate_parser = benchmark_subparsers.add_parser("aggregate", help="Aggregate benchmark reports across runs.")
    aggregate_parser.add_argument(
        "--reports-root",
        type=Path,
        default=_default_benchmark_root() / "runs",
        help="Root directory containing benchmark run reports.",
    )
    aggregate_parser.add_argument(
        "--dataset",
        default=None,
        choices=[DATASET_REGULAR, DATASET_COVER, "all"],
        help="Optional dataset filter.",
    )
    aggregate_parser.add_argument("--dataset-profile", default=None, help="Optional named dataset profile from no-watermar.toml.")
    aggregate_parser.add_argument("--provider-profile", default=None, help="Optional named provider profile from no-watermar.toml.")
    aggregate_parser.add_argument("--mask-provider", default=None, help="Optional mask provider filter.")
    aggregate_parser.add_argument("--restore-provider", default=None, help="Optional restore provider filter.")
    aggregate_parser.add_argument("--run-after", default=None, help="Only include runs whose run_id is >= this value.")
    aggregate_parser.add_argument("--run-before", default=None, help="Only include runs whose run_id is <= this value.")
    aggregate_parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_benchmark_root() / "aggregations",
        help="Directory for aggregation outputs.",
    )
    aggregate_parser.set_defaults(handler=_handle_aggregate)

    trends_parser = benchmark_subparsers.add_parser(
        "trends",
        help="Build a trend snapshot that merges comparison and aggregation outputs.",
    )
    trends_parser.add_argument("--benchmark-root", type=Path, default=_default_benchmark_root(), help="Benchmark root.")
    trends_parser.add_argument("--comparison", type=Path, default=None, help="Explicit comparison JSON to snapshot.")
    trends_parser.add_argument("--aggregation", type=Path, default=None, help="Explicit aggregation JSON for the baseline reference.")
    trends_parser.add_argument("--comparisons-root", type=Path, default=None, help="Directory containing comparison JSON files.")
    trends_parser.add_argument("--aggregations-root", type=Path, default=None, help="Directory containing aggregation JSON files.")
    trends_parser.add_argument("--dataset-profile", default=None, help="Optional named dataset profile from no-watermar.toml.")
    trends_parser.add_argument("--dataset", default=None, help="Optional dataset filter when auto-resolving sources.")
    trends_parser.add_argument("--baseline-provider-profile", default=None, help="Optional baseline provider profile from no-watermar.toml.")
    trends_parser.add_argument("--baseline-mask-provider", default=None, help="Optional baseline mask provider filter.")
    trends_parser.add_argument("--baseline-restore-provider", default=None, help="Optional baseline restore provider filter.")
    trends_parser.add_argument("--candidate-provider-profile", default=None, help="Optional candidate provider profile from no-watermar.toml.")
    trends_parser.add_argument("--candidate-mask-provider", default=None, help="Optional candidate mask provider filter.")
    trends_parser.add_argument("--candidate-restore-provider", default=None, help="Optional candidate restore provider filter.")
    trends_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for trend snapshot outputs. Defaults to <benchmark-root>/trends.",
    )
    trends_parser.set_defaults(handler=_handle_trends)

    list_parser = benchmark_subparsers.add_parser("list-providers", help="List implemented and planned provider slots.")
    list_parser.set_defaults(handler=_handle_list_providers)

    probe_parser = benchmark_subparsers.add_parser("probe-providers", help="Probe provider runtimes and sidecar interpreter importability.")
    probe_parser.set_defaults(handler=_handle_probe_providers)


def _handle_prepare(args: argparse.Namespace) -> int:
    dataset_profile = resolve_dataset_profile(args.dataset_profile, start_dir=Path.cwd()) if args.dataset_profile else None
    summary = prepare_benchmark_dataset(
        input_root=args.input or (dataset_profile.input_root if dataset_profile and dataset_profile.input_root else _default_input_root()),
        benchmark_root=args.benchmark_root,
        recursive=False if args.no_recursive else (
            dataset_profile.recursive if dataset_profile and dataset_profile.recursive is not None else True
        ),
        limit=args.limit if args.limit is not None else (dataset_profile.limit if dataset_profile else None),
    )
    if dataset_profile is not None:
        summary["dataset_profile"] = dataset_profile.name
        summary["dataset_profile_config"] = dataset_profile_to_dict(dataset_profile)
    print_json(summary)
    return 0


def _handle_run(args: argparse.Namespace) -> int:
    dataset_profile = resolve_dataset_profile(args.dataset_profile, start_dir=Path.cwd()) if args.dataset_profile else None
    provider_profile = resolve_provider_profile(args.provider_profile, start_dir=Path.cwd()) if args.provider_profile else None
    provider_execution = _resolve_provider_profile_args(
        provider_profile=provider_profile,
        mask_provider=args.mask_provider,
        restore_provider=args.restore_provider,
        ocr_session_mode=args.ocr_session_mode,
    )
    summary = run_benchmark(
        benchmark_root=args.benchmark_root,
        dataset_id=args.dataset or (
            dataset_profile.benchmark_dataset if dataset_profile and dataset_profile.benchmark_dataset else DATASET_REGULAR
        ),
        mask_provider_name=provider_execution["mask_provider"],
        restore_provider_name=provider_execution["restore_provider"],
        limit=args.limit if args.limit is not None else (dataset_profile.limit if dataset_profile else None),
        ocr_session_mode=provider_execution["ocr_session_mode"],
        restore_prompt=provider_execution["restore_prompt"],
        restore_negative_prompt=provider_execution["restore_negative_prompt"],
        restore_options=provider_execution["restore_options"],
    )
    if dataset_profile is not None:
        summary["dataset_profile"] = dataset_profile.name
        summary["dataset_profile_config"] = dataset_profile_to_dict(dataset_profile)
    if provider_profile is not None:
        summary["provider_profile"] = provider_profile.name
        summary["provider_profile_config"] = provider_profile_to_dict(provider_profile)
    print_json(summary)
    return 0


def _handle_compare(args: argparse.Namespace) -> int:
    summary = compare_benchmark_reports(
        baseline_report=args.baseline_report,
        candidate_report=args.candidate_report,
        output_dir=args.output_dir,
    )
    print_json(summary)
    return 0


def _handle_aggregate(args: argparse.Namespace) -> int:
    dataset_profile = resolve_dataset_profile(args.dataset_profile, start_dir=Path.cwd()) if args.dataset_profile else None
    provider_profile = resolve_provider_profile(args.provider_profile, start_dir=Path.cwd()) if args.provider_profile else None
    summary = aggregate_benchmark_reports(
        reports_root=args.reports_root,
        dataset_id=args.dataset or (
            dataset_profile.benchmark_dataset if dataset_profile and dataset_profile.benchmark_dataset else "all"
        ),
        mask_provider=args.mask_provider or (
            provider_profile.mask_provider if provider_profile and provider_profile.mask_provider else None
        ),
        restore_provider=args.restore_provider or (
            provider_profile.restore_provider if provider_profile and provider_profile.restore_provider else None
        ),
        run_after=args.run_after,
        run_before=args.run_before,
        output_dir=args.output_dir,
    )
    if dataset_profile is not None:
        summary["dataset_profile"] = dataset_profile.name
        summary["dataset_profile_config"] = dataset_profile_to_dict(dataset_profile)
    if provider_profile is not None:
        summary["provider_profile"] = provider_profile.name
        summary["provider_profile_config"] = provider_profile_to_dict(provider_profile)
    print_json(summary)
    return 0


def _handle_trends(args: argparse.Namespace) -> int:
    dataset_profile = resolve_dataset_profile(args.dataset_profile, start_dir=Path.cwd()) if args.dataset_profile else None
    baseline_provider_profile = (
        resolve_provider_profile(args.baseline_provider_profile, start_dir=Path.cwd())
        if args.baseline_provider_profile
        else None
    )
    candidate_provider_profile = (
        resolve_provider_profile(args.candidate_provider_profile, start_dir=Path.cwd())
        if args.candidate_provider_profile
        else None
    )
    summary = build_benchmark_trend_snapshot(
        comparison=args.comparison,
        aggregation=args.aggregation,
        comparisons_root=args.comparisons_root or (args.benchmark_root / "comparisons"),
        aggregations_root=args.aggregations_root or (args.benchmark_root / "aggregations"),
        dataset_id=args.dataset or (
            dataset_profile.benchmark_dataset if dataset_profile and dataset_profile.benchmark_dataset else None
        ),
        baseline_mask_provider=args.baseline_mask_provider or (
            baseline_provider_profile.mask_provider if baseline_provider_profile else None
        ),
        baseline_restore_provider=args.baseline_restore_provider or (
            baseline_provider_profile.restore_provider if baseline_provider_profile else None
        ),
        candidate_mask_provider=args.candidate_mask_provider or (
            candidate_provider_profile.mask_provider if candidate_provider_profile else None
        ),
        candidate_restore_provider=args.candidate_restore_provider or (
            candidate_provider_profile.restore_provider if candidate_provider_profile else None
        ),
        output_dir=args.output_dir or (args.benchmark_root / "trends"),
    )
    _attach_profile_summary(
        summary,
        dataset_profile=dataset_profile,
        provider_profile=None,
        baseline_provider_profile=baseline_provider_profile,
        candidate_provider_profile=candidate_provider_profile,
    )
    print_json(summary)
    return 0


def _handle_list_providers(args: argparse.Namespace) -> int:
    del args
    print_json(list_provider_descriptors())
    return 0


def _handle_probe_providers(args: argparse.Namespace) -> int:
    del args
    print_json(probe_provider_runtimes())
    return 0


def _resolve_provider_profile_args(
    *,
    provider_profile: Any,
    mask_provider: str | None,
    restore_provider: str | None,
    ocr_session_mode: str | None,
) -> dict[str, Any]:
    resolved_mask_provider = mask_provider or (
        provider_profile.mask_provider if provider_profile and provider_profile.mask_provider else "rule_based_roi"
    )
    resolved_restore_provider = restore_provider or (
        provider_profile.restore_provider if provider_profile and provider_profile.restore_provider else "telea"
    )
    resolved_ocr_session_mode = ocr_session_mode or (
        provider_profile.ocr_session_mode if provider_profile and provider_profile.ocr_session_mode else OCR_SESSION_MODE_AUTO
    )
    return {
        "mask_provider": resolved_mask_provider,
        "restore_provider": resolved_restore_provider,
        "ocr_session_mode": resolved_ocr_session_mode,
        "restore_prompt": provider_profile.restore_prompt if provider_profile else None,
        "restore_negative_prompt": provider_profile.restore_negative_prompt if provider_profile else None,
        "restore_options": dict(provider_profile.restore_options) if provider_profile else {},
    }


def _attach_profile_summary(
    summary: dict[str, Any],
    *,
    dataset_profile: Any = None,
    provider_profile: Any = None,
    baseline_provider_profile: Any = None,
    candidate_provider_profile: Any = None,
) -> None:
    if dataset_profile is not None:
        summary["dataset_profile"] = dataset_profile.name
        summary["dataset_profile_config"] = dataset_profile_to_dict(dataset_profile)
    if provider_profile is not None:
        summary["provider_profile"] = provider_profile.name
        summary["provider_profile_config"] = provider_profile_to_dict(provider_profile)
    if baseline_provider_profile is not None:
        summary["baseline_provider_profile"] = baseline_provider_profile.name
        summary["baseline_provider_profile_config"] = provider_profile_to_dict(baseline_provider_profile)
    if candidate_provider_profile is not None:
        summary["candidate_provider_profile"] = candidate_provider_profile.name
        summary["candidate_provider_profile_config"] = provider_profile_to_dict(candidate_provider_profile)


def _default_input_root() -> Path:
    return Path(__file__).resolve().parents[4] / "inputs"


def _default_benchmark_root() -> Path:
    return Path(__file__).resolve().parents[4] / "benchmarks"
