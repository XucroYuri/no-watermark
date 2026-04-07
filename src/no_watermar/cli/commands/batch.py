from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ...batch_run import resolve_batch_run_summary
from ...batch_plan import apply_loaded_batch_plan, create_batch_plan, load_batch_plan, summarize_batch_plan
from ...benchmark_paddleocr_session import OCR_SESSION_MODE_AUTO, OCR_SESSION_MODE_CHOICES
from ...config import (
    dataset_profile_to_dict,
    provider_profile_to_dict,
    resolve_dataset_profile,
    resolve_provider_profile,
)
from ..confirm import confirm_batch_apply_plan
from ...io_utils import print_json
from ...pipeline import resume_pipeline, run_pipeline


def configure_batch_group(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("batch", help="Run batch processing workflows.")
    batch_subparsers = parser.add_subparsers(dest="batch_command", required=True)

    plan_parser = batch_subparsers.add_parser("plan", help="Create a batch execution plan without writing masks or restores.")
    plan_parser.add_argument("--dataset-profile", default=None, help="Optional named dataset profile from no-watermar.toml.")
    plan_parser.add_argument("--provider-profile", default=None, help="Optional named provider profile from no-watermar.toml.")
    plan_parser.add_argument("--input", type=Path, default=None, help="Input root to scan.")
    plan_parser.add_argument("--scan-manifest", type=Path, default=None, help="Optional scan manifest JSON to use as the plan input contract.")
    plan_parser.add_argument("--output", type=Path, default=_default_output_root(), help="Output root for pipeline runs.")
    plan_parser.add_argument("--plans-root", type=Path, default=None, help="Directory for persisted batch plan JSON files.")
    plan_parser.add_argument("--no-recursive", action="store_true", help="Only scan the top-level directory.")
    plan_parser.add_argument("--scan-only", action="store_true", help="Plan a scan-only run.")
    plan_parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of images to process.")
    plan_parser.add_argument("--mask-provider", default=None, help="Mask provider name for the planned batch run.")
    plan_parser.add_argument("--restore-provider", default=None, help="Restore provider name for the planned batch run.")
    plan_parser.add_argument(
        "--ocr-session-mode",
        default=None,
        choices=OCR_SESSION_MODE_CHOICES,
        help="OCR execution mode when the planned batch run uses the paddleocr mask provider.",
    )
    plan_parser.set_defaults(handler=_handle_plan)

    apply_parser = batch_subparsers.add_parser("apply", help="Run the current batch pipeline implementation.")
    apply_parser.add_argument("--plan", type=Path, default=None, help="Optional batch plan JSON to apply.")
    apply_parser.add_argument("--dataset-profile", default=None, help="Optional named dataset profile from no-watermar.toml.")
    apply_parser.add_argument("--provider-profile", default=None, help="Optional named provider profile from no-watermar.toml.")
    apply_parser.add_argument("--input", type=Path, default=None, help="Input root to scan.")
    apply_parser.add_argument("--output", type=Path, default=None, help="Output root for pipeline runs.")
    apply_parser.add_argument("--no-recursive", action="store_true", help="Only scan the top-level directory.")
    apply_parser.add_argument("--scan-only", action="store_true", help="Only build the scan manifest without writing masks/restores.")
    apply_parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of images to process.")
    apply_parser.add_argument("--mask-provider", default=None, help="Mask provider name for the direct batch run.")
    apply_parser.add_argument("--restore-provider", default=None, help="Restore provider name for the direct batch run.")
    apply_parser.add_argument(
        "--ocr-session-mode",
        default=None,
        choices=OCR_SESSION_MODE_CHOICES,
        help="OCR execution mode when the direct batch run uses the paddleocr mask provider.",
    )
    apply_parser.add_argument("--yes", action="store_true", help="Skip interactive confirmation for planned apply runs.")
    apply_parser.add_argument("--no-input", action="store_true", help="Disable interactive prompts. Requires --yes for planned apply runs.")
    apply_parser.set_defaults(handler=_handle_apply)

    report_parser = batch_subparsers.add_parser("report", help="Load a persisted batch run summary.")
    report_parser.add_argument("--run-id", default=None, help="Existing run id under the runs root.")
    report_parser.add_argument("--run-dir", type=Path, default=None, help="Explicit batch run directory.")
    report_parser.add_argument("--summary", type=Path, default=None, help="Explicit summary.json or legacy report.json path.")
    report_parser.add_argument("--runs-root", type=Path, default=_default_output_root(), help="Batch runs root used for --run-id or --latest resolution.")
    report_parser.add_argument("--latest", action="store_true", help="Load the latest available batch run summary.")
    report_parser.set_defaults(handler=_handle_report)

    resume_parser = batch_subparsers.add_parser("resume", help="Continue an interrupted or partial batch run without rescanning inputs.")
    resume_parser.add_argument("--run-id", default=None, help="Existing run id under the runs root.")
    resume_parser.add_argument("--run-dir", type=Path, default=None, help="Explicit batch run directory.")
    resume_parser.add_argument("--summary", type=Path, default=None, help="Explicit summary.json path.")
    resume_parser.add_argument("--runs-root", type=Path, default=_default_output_root(), help="Batch runs root used for --run-id or --latest resolution.")
    resume_parser.add_argument("--latest", action="store_true", help="Resume the latest available batch run.")
    resume_parser.set_defaults(handler=_handle_resume)


def _handle_plan(args: argparse.Namespace) -> int:
    try:
        dataset_profile = resolve_dataset_profile(args.dataset_profile, start_dir=Path.cwd()) if args.dataset_profile else None
        provider_profile = resolve_provider_profile(args.provider_profile, start_dir=Path.cwd()) if args.provider_profile else None
        if args.scan_manifest is not None and (
            args.dataset_profile is not None or args.input is not None or args.no_recursive or args.limit is not None
        ):
            raise ValueError("When --scan-manifest is provided, do not also pass --dataset-profile, --input, --no-recursive, or --limit.")
        input_root, recursive, limit = _resolve_dataset_profile_args(
            dataset_profile=dataset_profile,
            input_root=args.input,
            no_recursive=args.no_recursive,
            limit=args.limit,
        )
        provider_execution = _resolve_provider_profile_args(
            provider_profile=provider_profile,
            mask_provider=args.mask_provider,
            restore_provider=args.restore_provider,
            ocr_session_mode=args.ocr_session_mode,
        )
        summary = create_batch_plan(
            input_root=input_root if args.scan_manifest is None else None,
            output_root=args.output,
            scan_manifest_path=args.scan_manifest,
            plans_root=args.plans_root,
            recursive=recursive,
            limit=limit,
            scan_only=args.scan_only,
            mask_provider=provider_execution["mask_provider"],
            restore_provider=provider_execution["restore_provider"],
            ocr_session_mode=provider_execution["ocr_session_mode"],
            restore_prompt=provider_execution["restore_prompt"],
            restore_negative_prompt=provider_execution["restore_negative_prompt"],
            restore_options=provider_execution["restore_options"],
        )
    except BATCH_COMMAND_ERRORS as exc:
        print_json(_batch_error_payload("batch plan", args, exc))
        return 2

    if dataset_profile is not None:
        summary["dataset_profile"] = dataset_profile.name
        summary["dataset_profile_config"] = dataset_profile_to_dict(dataset_profile)
    if provider_profile is not None:
        summary["provider_profile"] = provider_profile.name
        summary["provider_profile_config"] = provider_profile_to_dict(provider_profile)
    print_json(summary)
    return 0


def _handle_apply(args: argparse.Namespace) -> int:
    try:
        if args.plan is not None:
            if (
                args.dataset_profile is not None
                or args.input is not None
                or args.output is not None
                or args.no_recursive
                or args.scan_only
                or args.limit is not None
                or args.provider_profile is not None
                or args.mask_provider is not None
                or args.restore_provider is not None
                or args.ocr_session_mode is not None
            ):
                raise ValueError("When --plan is provided, do not also pass direct input/output or execution flags.")
            plan = load_batch_plan(args.plan.resolve())
            plan_summary = summarize_batch_plan(args.plan.resolve())
            confirmation_mode = confirm_batch_apply_plan(
                plan_summary,
                yes=args.yes,
                no_input=args.no_input,
            )
            summary = apply_loaded_batch_plan(plan, args.plan.resolve())
            summary["confirmation_mode"] = confirmation_mode
            summary["confirmation_required"] = True
        else:
            if args.yes or args.no_input:
                raise ValueError("--yes and --no-input are only supported with --plan in the current batch apply flow.")
            dataset_profile = resolve_dataset_profile(args.dataset_profile, start_dir=Path.cwd()) if args.dataset_profile else None
            provider_profile = resolve_provider_profile(args.provider_profile, start_dir=Path.cwd()) if args.provider_profile else None
            input_root, recursive, limit = _resolve_dataset_profile_args(
                dataset_profile=dataset_profile,
                input_root=args.input,
                no_recursive=args.no_recursive,
                limit=args.limit,
            )
            provider_execution = _resolve_provider_profile_args(
                provider_profile=provider_profile,
                mask_provider=args.mask_provider,
                restore_provider=args.restore_provider,
                ocr_session_mode=args.ocr_session_mode,
            )
            summary = run_pipeline(
                input_root=input_root,
                output_root=args.output or _default_output_root(),
                recursive=recursive,
                limit=limit,
                scan_only=args.scan_only,
                mask_provider_name=provider_execution["mask_provider"],
                restore_provider_name=provider_execution["restore_provider"],
                ocr_session_mode=provider_execution["ocr_session_mode"],
                restore_prompt=provider_execution["restore_prompt"],
                restore_negative_prompt=provider_execution["restore_negative_prompt"],
                restore_options=provider_execution["restore_options"],
                summary_context={
                    "command": "batch apply",
                    "mode": "direct",
                    "confirmation_mode": "compatibility_direct",
                    "confirmation_required": False,
                    "dataset_profile": dataset_profile.name if dataset_profile else None,
                    "provider_profile": provider_profile.name if provider_profile else None,
                    "warnings": [],
                },
            )
            if dataset_profile is not None:
                summary["dataset_profile"] = dataset_profile.name
                summary["dataset_profile_config"] = dataset_profile_to_dict(dataset_profile)
            if provider_profile is not None:
                summary["provider_profile"] = provider_profile.name
                summary["provider_profile_config"] = provider_profile_to_dict(provider_profile)
    except BATCH_COMMAND_ERRORS as exc:
        print_json(_batch_error_payload("batch apply", args, exc))
        return 2

    print_json(summary)
    return 0


def _handle_report(args: argparse.Namespace) -> int:
    try:
        summary, lookup_mode, resolved_source = resolve_batch_run_summary(
            runs_root=args.runs_root,
            latest=args.latest or not any((args.run_id, args.run_dir, args.summary)),
            run_id=args.run_id,
            run_dir=args.run_dir,
            summary_path=args.summary,
        )
    except BATCH_COMMAND_ERRORS as exc:
        print_json(_batch_error_payload("batch report", args, exc))
        return 2

    payload = dict(summary)
    payload["command"] = "batch report"
    payload["status"] = "ok"
    payload["source_command"] = summary.get("command")
    payload["lookup_mode"] = lookup_mode
    payload["resolved_source"] = str(resolved_source)
    print_json(payload)
    return 0


def _handle_resume(args: argparse.Namespace) -> int:
    try:
        summary, lookup_mode, resolved_source = resolve_batch_run_summary(
            runs_root=args.runs_root,
            latest=args.latest or not any((args.run_id, args.run_dir, args.summary)),
            run_id=args.run_id,
            run_dir=args.run_dir,
            summary_path=args.summary,
        )
        resumed = resume_pipeline(
            summary,
            summary_context={
                "command": "batch resume",
                "mode": "resume",
            },
        )
    except BATCH_COMMAND_ERRORS as exc:
        print_json(_batch_error_payload("batch resume", args, exc))
        return 2

    payload = dict(resumed)
    payload["command"] = "batch resume"
    payload["status"] = "ok"
    payload["lookup_mode"] = lookup_mode
    payload["resolved_source"] = str(resolved_source)
    print_json(payload)
    return 0


BATCH_COMMAND_ERRORS = (OSError, RuntimeError, ValueError, json.JSONDecodeError, KeyError)


def _resolve_dataset_profile_args(
    *,
    dataset_profile: Any,
    input_root: Path | None,
    no_recursive: bool,
    limit: int | None,
) -> tuple[Path, bool, int | None]:
    resolved_input_root = input_root or (
        dataset_profile.input_root if dataset_profile and dataset_profile.input_root else _default_input_root()
    )
    recursive = False if no_recursive else (
        dataset_profile.recursive if dataset_profile and dataset_profile.recursive is not None else True
    )
    resolved_limit = limit if limit is not None else (dataset_profile.limit if dataset_profile else None)
    return resolved_input_root, recursive, resolved_limit


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


def _batch_error_payload(command: str, args: argparse.Namespace, error: Exception) -> dict[str, Any]:
    plan_path = str(args.plan.resolve()) if getattr(args, "plan", None) else None
    scan_manifest_path = str(args.scan_manifest.resolve()) if getattr(args, "scan_manifest", None) else None
    input_root = str(args.input.resolve()) if getattr(args, "input", None) else None
    output_root = str(args.output.resolve()) if getattr(args, "output", None) else None
    return {
        "command": command,
        "status": "error",
        "plan_path": plan_path,
        "scan_manifest_path": scan_manifest_path,
        "input_root": input_root,
        "output_root": output_root,
        "error": str(error),
    }


def _default_input_root() -> Path:
    return Path(__file__).resolve().parents[4] / "inputs"


def _default_output_root() -> Path:
    return Path(__file__).resolve().parents[4] / "runtime" / "runs"
