from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import read_mask

OCR_SESSION_MODE_AUTO = "auto"
OCR_SESSION_MODE_PERSISTENT = "persistent"
OCR_SESSION_MODE_ONESHOT = "oneshot"
OCR_SESSION_MODE_CHOICES = [
    OCR_SESSION_MODE_AUTO,
    OCR_SESSION_MODE_PERSISTENT,
    OCR_SESSION_MODE_ONESHOT,
]

OCR_RESOLVED_MODE_PERSISTENT = "persistent-sidecar"
OCR_RESOLVED_MODE_ONESHOT = "oneshot-sidecar"
OCR_RESOLVED_MODE_INPROCESS = "inprocess"
OCR_RESOLVED_MODE_UNAVAILABLE = "unavailable"


def build_sidecar_process_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


@dataclass(slots=True)
class PaddleOCRExecutionContext:
    requested_mode: str
    resolved_mode: str
    python_executable: str | None = None
    note: str | None = None
    session: "PaddleOCRPersistentSession | None" = None

    def detect(
        self,
        *,
        image_path: Path,
        source_category: str,
        score_threshold: float = 0.45,
    ) -> tuple[dict[str, Any], np.ndarray]:
        if self.resolved_mode == OCR_RESOLVED_MODE_PERSISTENT:
            if self.session is None:
                raise RuntimeError(self.note or "Persistent PaddleOCR session is unavailable.")
            return self.session.detect(
                image_path=image_path,
                source_category=source_category,
                score_threshold=score_threshold,
            )

        if self.resolved_mode == OCR_RESOLVED_MODE_ONESHOT:
            if not self.python_executable:
                raise RuntimeError(self.note or "PaddleOCR sidecar interpreter is unavailable.")
            return run_paddleocr_mask_oneshot(
                python_executable=self.python_executable,
                image_path=image_path,
                source_category=source_category,
                score_threshold=score_threshold,
            )

        raise RuntimeError(self.note or "PaddleOCR sidecar execution is unavailable.")

    def close(self) -> None:
        if self.session is not None:
            self.session.close()
            self.session = None


class PaddleOCRPersistentSession:
    def __init__(self, *, python_executable: str, script_path: Path) -> None:
        self._python_executable = python_executable
        self._script_path = script_path
        self._process: subprocess.Popen[str] | None = None
        self._request_counter = 0

    def start(self) -> None:
        if self._process is not None:
            return

        process = subprocess.Popen(
            [self._python_executable, str(self._script_path), "serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=build_sidecar_process_env(),
        )
        ready_line = process.stdout.readline() if process.stdout else ""
        if not ready_line:
            stderr = process.stderr.read().strip() if process.stderr else ""
            process.kill()
            process.wait(timeout=5)
            raise RuntimeError(stderr or "PaddleOCR persistent sidecar exited before becoming ready.")

        try:
            ready_payload = json.loads(ready_line)
        except json.JSONDecodeError as exc:
            stderr = process.stderr.read().strip() if process.stderr else ""
            process.kill()
            process.wait(timeout=5)
            raise RuntimeError(stderr or f"Invalid PaddleOCR sidecar handshake: {ready_line!r}") from exc

        if not ready_payload.get("ok", False):
            stderr = process.stderr.read().strip() if process.stderr else ""
            process.kill()
            process.wait(timeout=5)
            raise RuntimeError(str(ready_payload.get("error") or stderr or "PaddleOCR persistent sidecar startup failed."))

        self._process = process

    def detect(
        self,
        *,
        image_path: Path,
        source_category: str,
        score_threshold: float = 0.45,
    ) -> tuple[dict[str, Any], np.ndarray]:
        if self._process is None or self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("PaddleOCR persistent sidecar is not running.")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            output_mask = tmp_root / "mask.png"
            request_id = self._next_request_id()
            request = {
                "op": "detect",
                "request_id": request_id,
                "image_path": str(image_path),
                "output_mask_path": str(output_mask),
                "category": source_category,
                "score_threshold": score_threshold,
            }
            self._process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            self._process.stdin.flush()

            response_line = self._process.stdout.readline()
            if not response_line:
                stderr = self._process.stderr.read().strip() if self._process.stderr else ""
                raise RuntimeError(stderr or "PaddleOCR persistent sidecar closed unexpectedly.")

            try:
                response = json.loads(response_line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid PaddleOCR sidecar response: {response_line!r}") from exc

            if response.get("request_id") != request_id:
                raise RuntimeError("Mismatched PaddleOCR sidecar response id.")
            if not response.get("ok", False):
                raise RuntimeError(str(response.get("error") or "PaddleOCR sidecar request failed."))
            if not output_mask.exists():
                raise RuntimeError("PaddleOCR sidecar did not produce an output mask.")

            payload = dict(response.get("payload") or {})
            return payload, read_mask(output_mask)

    def close(self) -> None:
        if self._process is None:
            return

        try:
            if self._process.stdin is not None:
                shutdown_request = {"op": "shutdown", "request_id": self._next_request_id()}
                self._process.stdin.write(json.dumps(shutdown_request, ensure_ascii=False) + "\n")
                self._process.stdin.flush()
        except OSError:
            pass

        try:
            self._process.terminate()
            self._process.wait(timeout=5)
        except Exception:
            self._process.kill()
            self._process.wait(timeout=5)
        finally:
            self._process = None

    def _next_request_id(self) -> str:
        self._request_counter += 1
        return f"req-{self._request_counter:06d}"


def create_paddleocr_execution_context(requested_mode: str) -> PaddleOCRExecutionContext:
    if requested_mode not in OCR_SESSION_MODE_CHOICES:
        raise ValueError(f"Unsupported OCR session mode: {requested_mode}")

    python_executable, runtime_note = resolve_paddleocr_sidecar_python()
    script_path = _project_root() / "tools" / "sidecars" / "paddleocr_mask.py"

    if requested_mode == OCR_SESSION_MODE_ONESHOT:
        if python_executable:
            return PaddleOCRExecutionContext(
                requested_mode=requested_mode,
                resolved_mode=OCR_RESOLVED_MODE_ONESHOT,
                python_executable=python_executable,
                note=runtime_note,
            )
        return PaddleOCRExecutionContext(
            requested_mode=requested_mode,
            resolved_mode=OCR_RESOLVED_MODE_UNAVAILABLE,
            python_executable=None,
            note=runtime_note or "PaddleOCR sidecar interpreter is unavailable.",
        )

    if requested_mode in {OCR_SESSION_MODE_AUTO, OCR_SESSION_MODE_PERSISTENT}:
        if not python_executable:
            resolved_mode = OCR_RESOLVED_MODE_ONESHOT if requested_mode == OCR_SESSION_MODE_AUTO else OCR_RESOLVED_MODE_UNAVAILABLE
            return PaddleOCRExecutionContext(
                requested_mode=requested_mode,
                resolved_mode=resolved_mode if python_executable else OCR_RESOLVED_MODE_UNAVAILABLE,
                python_executable=python_executable,
                note=runtime_note or "PaddleOCR sidecar interpreter is unavailable.",
            )

        session = PaddleOCRPersistentSession(python_executable=python_executable, script_path=script_path)
        try:
            session.start()
        except Exception as exc:
            if requested_mode == OCR_SESSION_MODE_PERSISTENT:
                return PaddleOCRExecutionContext(
                    requested_mode=requested_mode,
                    resolved_mode=OCR_RESOLVED_MODE_UNAVAILABLE,
                    python_executable=python_executable,
                    note=f"Unable to start persistent PaddleOCR sidecar: {exc}",
                )
            return PaddleOCRExecutionContext(
                requested_mode=requested_mode,
                resolved_mode=OCR_RESOLVED_MODE_ONESHOT,
                python_executable=python_executable,
                note=f"Persistent PaddleOCR sidecar unavailable, falling back to oneshot: {exc}",
            )

        return PaddleOCRExecutionContext(
            requested_mode=requested_mode,
            resolved_mode=OCR_RESOLVED_MODE_PERSISTENT,
            python_executable=python_executable,
            note=runtime_note,
            session=session,
        )

    raise ValueError(f"Unsupported OCR session mode: {requested_mode}")


def resolve_paddleocr_sidecar_python() -> tuple[str | None, str]:
    configured_python = os.getenv("NO_WATERMAR_PADDLEOCR_PYTHON")
    if configured_python:
        path = Path(configured_python)
        if path.exists():
            return str(path), f"Using NO_WATERMAR_PADDLEOCR_PYTHON: {path}"
        return None, f"NO_WATERMAR_PADDLEOCR_PYTHON points to a missing interpreter: {path}"

    if _module_available("paddleocr"):
        return sys.executable, "Using current Python environment as the PaddleOCR sidecar interpreter."

    return None, "PaddleOCR is not importable in the current environment and NO_WATERMAR_PADDLEOCR_PYTHON is not set."


def run_paddleocr_mask_oneshot(
    *,
    python_executable: str,
    image_path: Path,
    source_category: str,
    score_threshold: float = 0.45,
) -> tuple[dict[str, Any], np.ndarray]:
    script_path = _project_root() / "tools" / "sidecars" / "paddleocr_mask.py"
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        output_mask = tmp_root / "mask.png"
        output_json = tmp_root / "result.json"
        completed = subprocess.run(
            [
                python_executable,
                str(script_path),
                "--image",
                str(image_path),
                "--output-mask",
                str(output_mask),
                "--output-json",
                str(output_json),
                "--category",
                source_category,
                "--score-threshold",
                str(score_threshold),
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=build_sidecar_process_env(),
        )
        if completed.returncode != 0:
            details = (completed.stderr or "").strip() or (completed.stdout or "").strip() or f"exit code {completed.returncode}"
            raise RuntimeError(f"Sidecar failed: {details}")

        payload = json.loads(output_json.read_text(encoding="utf-8"))
        return payload, read_mask(output_mask)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None
