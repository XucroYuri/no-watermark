from __future__ import annotations

from collections.abc import Callable
from typing import Any


def confirm_batch_apply_plan(
    plan_summary: dict[str, Any],
    *,
    yes: bool,
    no_input: bool,
    prompt: Callable[[str], str] | None = None,
) -> str:
    if yes:
        return "yes_flag"
    if no_input:
        raise ValueError("batch apply --plan requires --yes when --no-input is set.")

    if prompt is None:
        prompt = input

    answer = prompt(_render_batch_apply_prompt(plan_summary))
    if answer.strip().lower() not in {"y", "yes"}:
        raise ValueError("Batch apply aborted by user.")
    return "interactive"


def _render_batch_apply_prompt(plan_summary: dict[str, Any]) -> str:
    plan_id = plan_summary.get("plan_id", "unknown")
    input_mode = plan_summary.get("input_mode", "scan")
    input_root = plan_summary.get("input_root", "")
    output_root = plan_summary.get("output_root", "")
    scan_manifest_path = plan_summary.get("scan_manifest_path")
    item_count = plan_summary.get("item_count", 0)
    scan_only = plan_summary.get("scan_only", False)
    recursive = plan_summary.get("recursive", True)
    limit = plan_summary.get("limit")
    mask_provider = plan_summary.get("mask_provider")
    restore_provider = plan_summary.get("restore_provider")
    ocr_session_mode = plan_summary.get("ocr_session_mode")
    restore_prompt = plan_summary.get("restore_prompt")
    restore_negative_prompt = plan_summary.get("restore_negative_prompt")
    restore_options = plan_summary.get("restore_options")

    lines = [
        "Batch apply confirmation:",
        f"  plan_id: {plan_id}",
        f"  input_mode: {input_mode}",
        f"  input_root: {input_root}",
        f"  output_root: {output_root}",
        f"  item_count: {item_count}",
        f"  recursive: {recursive}",
        f"  scan_only: {scan_only}",
        f"  limit: {limit}",
    ]
    if mask_provider:
        lines.append(f"  mask_provider: {mask_provider}")
    if restore_provider:
        lines.append(f"  restore_provider: {restore_provider}")
    if ocr_session_mode:
        lines.append(f"  ocr_session_mode: {ocr_session_mode}")
    if restore_prompt:
        lines.append(f"  restore_prompt: {restore_prompt}")
    if restore_negative_prompt:
        lines.append(f"  restore_negative_prompt: {restore_negative_prompt}")
    if restore_options:
        lines.append(f"  restore_options: {restore_options}")
    if scan_manifest_path:
        lines.append(f"  scan_manifest_path: {scan_manifest_path}")
    lines.append("Proceed with batch apply? [y/N]: ")
    return "\n".join(lines)
