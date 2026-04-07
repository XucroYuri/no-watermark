from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from no_watermar.provider_workers import restore_with_powerpaint_v2_1


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "serve":
        return _serve()
    return _run_cli(argv)


def _run_cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="PowerPaint v2.1 restore sidecar.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output-image", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--options-json", default=None)
    args = parser.parse_args(argv)

    payload = _run_restore_request(
        image_path=args.image,
        mask_path=args.mask,
        output_image_path=args.output_image,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        options=json.loads(args.options_json) if args.options_json else {},
        session_mode="oneshot-sidecar",
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def _serve() -> int:
    print(json.dumps({"ok": True, "provider": "powerpaint_v2_1"}), flush=True)
    for line in sys.stdin:
        request_line = line.strip()
        if not request_line:
            continue
        try:
            request = json.loads(request_line)
        except json.JSONDecodeError as exc:
            print(json.dumps({"ok": False, "error": f"Invalid JSON request: {exc}"}), flush=True)
            continue

        op = str(request.get("op") or "").strip().lower()
        request_id = str(request.get("request_id") or "")
        if op == "shutdown":
            print(json.dumps({"ok": True, "request_id": request_id, "payload": {"shutdown": True}}), flush=True)
            return 0
        if op != "restore":
            print(json.dumps({"ok": False, "request_id": request_id, "error": f"Unsupported op: {op}"}), flush=True)
            continue

        captured_stdout = io.StringIO()
        captured_stderr = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
                payload = _run_restore_request(
                    image_path=Path(str(request["image_path"])),
                    mask_path=Path(str(request["mask_path"])),
                    output_image_path=Path(str(request["output_image_path"])),
                    prompt=request.get("prompt"),
                    negative_prompt=request.get("negative_prompt"),
                    options=dict(request.get("options") or {}),
                    session_mode="persistent-sidecar",
                )
            print(json.dumps({"ok": True, "request_id": request_id, "payload": payload}, ensure_ascii=False), flush=True)
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "request_id": request_id,
                        "error": _build_error_message(exc, captured_stdout.getvalue(), captured_stderr.getvalue()),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return 0


def _run_restore_request(
    *,
    image_path: Path,
    mask_path: Path,
    output_image_path: Path,
    prompt: str | None,
    negative_prompt: str | None,
    options: dict[str, object],
    session_mode: str,
) -> dict[str, object]:
    image = _read_color_image(image_path)
    mask = _read_gray_image(mask_path)
    result = restore_with_powerpaint_v2_1(
        image,
        mask,
        prompt=prompt,
        negative_prompt=negative_prompt,
        options=options,
        session_mode=session_mode,
    )
    _write_image(output_image_path, result["restored"])
    return {
        "latency_ms": result["latency_ms"],
        "meta": result["meta"],
    }


def _read_color_image(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to decode color image: {path}")
    return image


def _read_gray_image(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to decode grayscale image: {path}")
    return image


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix.lower() or ".png", image)
    if not ok:
        raise ValueError(f"Unable to encode image for: {path}")
    encoded.tofile(str(path))


def _build_error_message(exc: Exception, captured_stdout: str, captured_stderr: str) -> str:
    details = f"{type(exc).__name__}: {exc}"
    log_snippets = [text.strip() for text in (captured_stdout, captured_stderr) if text and text.strip()]
    if not log_snippets:
        return details
    combined_logs = " | ".join(log_snippets)
    return f"{details} | logs: {combined_logs}"


if __name__ == "__main__":
    raise SystemExit(main())
