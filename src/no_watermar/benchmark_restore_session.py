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

from .benchmark_paddleocr_session import build_sidecar_process_env
from .io_utils import read_image

RESTORE_SESSION_MODE_AUTO = "auto"
RESTORE_SESSION_MODE_PERSISTENT = "persistent"
RESTORE_SESSION_MODE_ONESHOT = "oneshot"
RESTORE_SESSION_MODE_CHOICES = [
    RESTORE_SESSION_MODE_AUTO,
    RESTORE_SESSION_MODE_PERSISTENT,
    RESTORE_SESSION_MODE_ONESHOT,
]

RESTORE_RESOLVED_MODE_PERSISTENT = "persistent-sidecar"
RESTORE_RESOLVED_MODE_ONESHOT = "oneshot-sidecar"
RESTORE_RESOLVED_MODE_INPROCESS = "inprocess"
RESTORE_RESOLVED_MODE_UNAVAILABLE = "unavailable"


@dataclass(slots=True)
class PowerPaintExecutionContext:
    requested_mode: str
    resolved_mode: str
    python_executable: str | None = None
    note: str | None = None
    session: "PowerPaintPersistentSession | None" = None

    def restore(
        self,
        *,
        image_path: Path,
        mask: np.ndarray,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], np.ndarray]:
        if self.resolved_mode == RESTORE_RESOLVED_MODE_PERSISTENT:
            if self.session is None:
                raise RuntimeError(self.note or "Persistent PowerPaint session is unavailable.")
            return self.session.restore(
                image_path=image_path,
                mask=mask,
                prompt=prompt,
                negative_prompt=negative_prompt,
                options=options,
            )

        if self.resolved_mode == RESTORE_RESOLVED_MODE_ONESHOT:
            if not self.python_executable:
                raise RuntimeError(self.note or "PowerPaint sidecar interpreter is unavailable.")
            return run_powerpaint_restore_oneshot(
                python_executable=self.python_executable,
                image_path=image_path,
                mask=mask,
                prompt=prompt,
                negative_prompt=negative_prompt,
                options=options,
            )

        raise RuntimeError(self.note or "PowerPaint restore execution is unavailable.")

    def close(self) -> None:
        if self.session is not None:
            self.session.close()
            self.session = None


class PowerPaintPersistentSession:
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
            raise RuntimeError(stderr or "PowerPaint persistent sidecar exited before becoming ready.")

        try:
            ready_payload = json.loads(ready_line)
        except json.JSONDecodeError as exc:
            stderr = process.stderr.read().strip() if process.stderr else ""
            process.kill()
            process.wait(timeout=5)
            raise RuntimeError(stderr or f"Invalid PowerPaint sidecar handshake: {ready_line!r}") from exc

        if not ready_payload.get("ok", False):
            stderr = process.stderr.read().strip() if process.stderr else ""
            process.kill()
            process.wait(timeout=5)
            raise RuntimeError(str(ready_payload.get("error") or stderr or "PowerPaint persistent sidecar startup failed."))

        self._process = process

    def restore(
        self,
        *,
        image_path: Path,
        mask: np.ndarray,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], np.ndarray]:
        if self._process is None or self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("PowerPaint persistent sidecar is not running.")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            mask_path = tmp_root / "mask.png"
            output_image = tmp_root / "restored.png"
            _write_mask(mask_path, mask)
            request_id = self._next_request_id()
            request = {
                "op": "restore",
                "request_id": request_id,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "output_image_path": str(output_image),
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "options": dict(options or {}),
            }
            self._process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            self._process.stdin.flush()

            response_line = self._process.stdout.readline()
            if not response_line:
                stderr = self._process.stderr.read().strip() if self._process.stderr else ""
                raise RuntimeError(stderr or "PowerPaint persistent sidecar closed unexpectedly.")

            try:
                response = json.loads(response_line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid PowerPaint sidecar response: {response_line!r}") from exc

            if response.get("request_id") != request_id:
                raise RuntimeError("Mismatched PowerPaint sidecar response id.")
            if not response.get("ok", False):
                raise RuntimeError(str(response.get("error") or "PowerPaint sidecar request failed."))
            if not output_image.exists():
                raise RuntimeError("PowerPaint sidecar did not produce an output image.")

            payload = dict(response.get("payload") or {})
            return payload, read_image(output_image)

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


def create_powerpaint_execution_context(restore_options: dict[str, Any] | None = None) -> PowerPaintExecutionContext:
    restore_options = dict(restore_options or {})
    requested_mode = _normalize_restore_session_mode(
        restore_options.get("session_mode") or restore_options.get("execution_mode") or RESTORE_SESSION_MODE_AUTO
    )

    if requested_mode == RESTORE_SESSION_MODE_AUTO and _powerpaint_module_available():
        return PowerPaintExecutionContext(
            requested_mode=requested_mode,
            resolved_mode=RESTORE_RESOLVED_MODE_INPROCESS,
            python_executable=sys.executable,
            note="Using current Python environment for inprocess PowerPaint execution.",
        )

    python_executable, runtime_note = resolve_powerpaint_sidecar_python()
    script_path = _project_root() / "tools" / "sidecars" / "powerpaint_restore.py"

    if requested_mode == RESTORE_SESSION_MODE_ONESHOT:
        if python_executable:
            return PowerPaintExecutionContext(
                requested_mode=requested_mode,
                resolved_mode=RESTORE_RESOLVED_MODE_ONESHOT,
                python_executable=python_executable,
                note=runtime_note,
            )
        return PowerPaintExecutionContext(
            requested_mode=requested_mode,
            resolved_mode=RESTORE_RESOLVED_MODE_UNAVAILABLE,
            note=runtime_note or "PowerPaint sidecar interpreter is unavailable.",
        )

    if requested_mode in {RESTORE_SESSION_MODE_AUTO, RESTORE_SESSION_MODE_PERSISTENT}:
        if not python_executable:
            fallback_mode = RESTORE_RESOLVED_MODE_UNAVAILABLE
            return PowerPaintExecutionContext(
                requested_mode=requested_mode,
                resolved_mode=fallback_mode,
                python_executable=python_executable,
                note=runtime_note or "PowerPaint sidecar interpreter is unavailable.",
            )

        session = PowerPaintPersistentSession(python_executable=python_executable, script_path=script_path)
        try:
            session.start()
        except Exception as exc:
            if requested_mode == RESTORE_SESSION_MODE_PERSISTENT:
                return PowerPaintExecutionContext(
                    requested_mode=requested_mode,
                    resolved_mode=RESTORE_RESOLVED_MODE_UNAVAILABLE,
                    python_executable=python_executable,
                    note=f"Unable to start persistent PowerPaint sidecar: {exc}",
                )
            return PowerPaintExecutionContext(
                requested_mode=requested_mode,
                resolved_mode=RESTORE_RESOLVED_MODE_ONESHOT,
                python_executable=python_executable,
                note=f"Persistent PowerPaint sidecar unavailable, falling back to oneshot: {exc}",
            )

        return PowerPaintExecutionContext(
            requested_mode=requested_mode,
            resolved_mode=RESTORE_RESOLVED_MODE_PERSISTENT,
            python_executable=python_executable,
            note=runtime_note,
            session=session,
        )

    raise ValueError(f"Unsupported restore session mode: {requested_mode}")


def resolve_powerpaint_sidecar_python() -> tuple[str | None, str]:
    configured_python = os.getenv("NO_WATERMAR_POWERPAINT_PYTHON")
    if configured_python:
        path = Path(configured_python)
        if path.exists():
            return str(path), f"Using NO_WATERMAR_POWERPAINT_PYTHON: {path}"
        return None, f"NO_WATERMAR_POWERPAINT_PYTHON points to a missing interpreter: {path}"

    if _powerpaint_module_available():
        return sys.executable, "Using current Python environment as the PowerPaint sidecar interpreter."

    return None, "PowerPaint is not importable in the current environment and NO_WATERMAR_POWERPAINT_PYTHON is not set."


def run_powerpaint_restore_oneshot(
    *,
    python_executable: str,
    image_path: Path,
    mask: np.ndarray,
    prompt: str | None = None,
    negative_prompt: str | None = None,
    options: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    script_path = _project_root() / "tools" / "sidecars" / "powerpaint_restore.py"
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        mask_path = tmp_root / "mask.png"
        output_image = tmp_root / "restored.png"
        output_json = tmp_root / "result.json"
        _write_mask(mask_path, mask)
        arguments = [
            "--image",
            str(image_path),
            "--mask",
            str(mask_path),
            "--output-image",
            str(output_image),
            "--output-json",
            str(output_json),
        ]
        if prompt:
            arguments.extend(["--prompt", prompt])
        if negative_prompt:
            arguments.extend(["--negative-prompt", negative_prompt])
        if options:
            arguments.extend(["--options-json", json.dumps(options, ensure_ascii=False)])

        completed = subprocess.run(
            [python_executable, str(script_path), *arguments],
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
        return payload, read_image(output_image)


def _normalize_restore_session_mode(value: object) -> str:
    raw_mode = str(value or RESTORE_SESSION_MODE_AUTO).strip().lower()
    aliases = {
        "persistent-sidecar": RESTORE_SESSION_MODE_PERSISTENT,
        "oneshot-sidecar": RESTORE_SESSION_MODE_ONESHOT,
    }
    normalized = aliases.get(raw_mode, raw_mode)
    if normalized not in RESTORE_SESSION_MODE_CHOICES:
        raise ValueError(
            "Unsupported restore session mode: "
            f"{value}. Supported values: {', '.join(RESTORE_SESSION_MODE_CHOICES)}."
        )
    return normalized


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _powerpaint_module_available() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in ["powerpaint", "torch", "diffusers", "transformers", "safetensors"])


def _write_mask(path: Path, mask: np.ndarray) -> None:
    from .io_utils import write_image

    write_image(path, mask)
