from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from no_watermar.provider_workers import create_paddleocr_engine, detect_mask_with_paddleocr


def main() -> int:
    _configure_stdio_encoding()
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        return _serve()

    parser = argparse.ArgumentParser(description="PaddleOCR sidecar mask detector.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-mask", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.45)
    args = parser.parse_args()

    image = _read_image(args.image)
    result = detect_mask_with_paddleocr(
        image,
        args.category,
        score_threshold=args.score_threshold,
        session_mode="oneshot-sidecar",
    )
    _write_payload(args.output_mask, args.output_json, result)
    return 0


def _serve() -> int:
    with redirect_stdout(sys.stderr):
        engine = create_paddleocr_engine()

    _emit_response({"event": "ready", "ok": True})

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            _emit_response({"ok": False, "error": f"Invalid JSON request: {exc}"})
            continue

        request_id = request.get("request_id")
        op = request.get("op")

        if op == "shutdown":
            _emit_response({"request_id": request_id, "ok": True})
            return 0

        if op != "detect":
            _emit_response({"request_id": request_id, "ok": False, "error": f"Unsupported op: {op}"})
            continue

        try:
            image_path = Path(request["image_path"])
            output_mask_path = Path(request["output_mask_path"])
            category = str(request["category"])
            score_threshold = float(request.get("score_threshold", 0.45))
            image = _read_image(image_path)
            with redirect_stdout(sys.stderr):
                result = detect_mask_with_paddleocr(
                    image,
                    category,
                    score_threshold=score_threshold,
                    ocr_engine=engine,
                    session_mode="persistent-sidecar",
                )
            _write_image(output_mask_path, result["mask"])
            payload = {
                "confidence": result["confidence"],
                "latency_ms": result["latency_ms"],
                "boxes": result["boxes"],
                "meta": result["meta"],
            }
            _emit_response({"request_id": request_id, "ok": True, "payload": payload})
        except Exception as exc:
            _emit_response({"request_id": request_id, "ok": False, "error": str(exc)})

    return 0


def _read_image(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to decode image: {path}")
    return image


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix.lower() or ".png", image)
    if not ok:
        raise ValueError(f"Unable to encode image for: {path}")
    encoded.tofile(str(path))


def _write_payload(output_mask_path: Path, output_json_path: Path, result: dict[str, object]) -> None:
    _write_image(output_mask_path, result["mask"])
    payload = {
        "confidence": result["confidence"],
        "latency_ms": result["latency_ms"],
        "boxes": result["boxes"],
        "meta": result["meta"],
    }
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _emit_response(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _configure_stdio_encoding() -> None:
    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
