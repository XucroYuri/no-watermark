from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ...config import dataset_profile_to_dict, resolve_dataset_profile
from ...io_utils import print_json
from ...scan_manifest import create_scan_manifest, summarize_scan_input

SCAN_COMMAND_ERRORS = (OSError, ValueError, json.JSONDecodeError)


def configure_scan_group(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("scan", help="Inspect or persist discovery-only scan results.")
    scan_subparsers = parser.add_subparsers(dest="scan_command", required=True)

    show_parser = scan_subparsers.add_parser("show", help="Show a discovery summary without writing artifacts.")
    _add_scan_read_args(show_parser)
    show_parser.set_defaults(handler=_handle_show)

    run_parser = scan_subparsers.add_parser("run", help="Persist a scan manifest under the scans runtime root.")
    _add_scan_read_args(run_parser)
    run_parser.add_argument("--scans-root", type=Path, default=_default_scans_root(), help="Directory for persisted scan manifests.")
    run_parser.set_defaults(handler=_handle_run)


def _handle_show(args: argparse.Namespace) -> int:
    try:
        input_root, recursive, limit, dataset_profile = _resolve_scan_profile_args(args)
        summary = summarize_scan_input(
            input_root=input_root,
            recursive=recursive,
            limit=limit,
            summary_context={
                "command": "scan show",
                "mode": "show",
                "dataset_profile": dataset_profile.name if dataset_profile else None,
            },
        )
    except SCAN_COMMAND_ERRORS as exc:
        print_json(_scan_error_payload("scan show", args.input, None, exc))
        return 2

    if dataset_profile is not None:
        summary["dataset_profile"] = dataset_profile.name
        summary["dataset_profile_config"] = dataset_profile_to_dict(dataset_profile)
    print_json(summary)
    return 0


def _handle_run(args: argparse.Namespace) -> int:
    try:
        input_root, recursive, limit, dataset_profile = _resolve_scan_profile_args(args)
        summary = create_scan_manifest(
            input_root=input_root,
            scans_root=args.scans_root,
            recursive=recursive,
            limit=limit,
        )
    except SCAN_COMMAND_ERRORS as exc:
        print_json(_scan_error_payload("scan run", args.input, args.scans_root, exc))
        return 2

    if dataset_profile is not None:
        summary["dataset_profile"] = dataset_profile.name
        summary["dataset_profile_config"] = dataset_profile_to_dict(dataset_profile)
    print_json(summary)
    return 0


def _add_scan_read_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-profile", default=None, help="Optional named dataset profile from no-watermar.toml.")
    parser.add_argument("--input", type=Path, default=None, help="Input root to scan.")
    parser.add_argument("--no-recursive", action="store_true", help="Only scan the top-level directory.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of discovered images to include.")


def _resolve_scan_profile_args(
    args: argparse.Namespace,
) -> tuple[Path, bool, int | None, Any]:
    dataset_profile = resolve_dataset_profile(args.dataset_profile, start_dir=Path.cwd()) if args.dataset_profile else None
    input_root = args.input or (dataset_profile.input_root if dataset_profile and dataset_profile.input_root else _default_input_root())
    recursive = False if args.no_recursive else (
        dataset_profile.recursive if dataset_profile and dataset_profile.recursive is not None else True
    )
    limit = args.limit if args.limit is not None else (dataset_profile.limit if dataset_profile else None)
    return input_root, recursive, limit, dataset_profile


def _scan_error_payload(command: str, input_root: Path | None, scans_root: Path | None, error: Exception) -> dict[str, Any]:
    return {
        "command": command,
        "status": "error",
        "input_root": str(input_root.resolve()) if input_root else None,
        "scans_root": str(scans_root.resolve()) if scans_root else None,
        "error": str(error),
    }


def _default_input_root() -> Path:
    return Path(__file__).resolve().parents[4] / "inputs"


def _default_scans_root() -> Path:
    return Path(__file__).resolve().parents[4] / "runtime" / "scans"
