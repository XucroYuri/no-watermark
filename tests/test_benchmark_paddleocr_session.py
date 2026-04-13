from __future__ import annotations

import io
import unittest
from pathlib import Path
import sys
from unittest.mock import Mock, patch

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from no_watermar.benchmark_paddleocr_session import (
    OCR_RESOLVED_MODE_ONESHOT,
    OCR_RESOLVED_MODE_UNAVAILABLE,
    OCR_SESSION_MODE_AUTO,
    OCR_SESSION_MODE_PERSISTENT,
    PaddleOCRPersistentSession,
    build_sidecar_process_env,
    create_paddleocr_execution_context,
)


class BenchmarkPaddleOCRSessionTests(unittest.TestCase):
    def test_auto_mode_falls_back_to_oneshot_when_persistent_start_fails(self) -> None:
        with patch(
            "no_watermar.benchmark_paddleocr_session.resolve_paddleocr_sidecar_python",
            return_value=("C:\\mock\\python.exe", "mock interpreter"),
        ):
            with patch(
                "no_watermar.benchmark_paddleocr_session.PaddleOCRPersistentSession.start",
                side_effect=RuntimeError("mock start failure"),
            ):
                context = create_paddleocr_execution_context(OCR_SESSION_MODE_AUTO)

        self.assertEqual(context.resolved_mode, OCR_RESOLVED_MODE_ONESHOT)
        self.assertEqual(context.python_executable, "C:\\mock\\python.exe")
        self.assertIn("falling back to oneshot", context.note or "")

    def test_persistent_mode_reports_unavailable_when_start_fails(self) -> None:
        with patch(
            "no_watermar.benchmark_paddleocr_session.resolve_paddleocr_sidecar_python",
            return_value=("C:\\mock\\python.exe", "mock interpreter"),
        ):
            with patch(
                "no_watermar.benchmark_paddleocr_session.PaddleOCRPersistentSession.start",
                side_effect=RuntimeError("mock start failure"),
            ):
                context = create_paddleocr_execution_context(OCR_SESSION_MODE_PERSISTENT)

        self.assertEqual(context.resolved_mode, OCR_RESOLVED_MODE_UNAVAILABLE)
        self.assertIn("Unable to start persistent PaddleOCR sidecar", context.note or "")

    def test_build_sidecar_process_env_forces_utf8_stdio(self) -> None:
        env = build_sidecar_process_env()

        self.assertEqual(env["PYTHONUTF8"], "1")
        self.assertEqual(env["PYTHONIOENCODING"], "utf-8")

    def test_persistent_session_start_uses_utf8_sidecar_env(self) -> None:
        process = Mock()
        process.stdout = io.StringIO('{"event":"ready","ok":true}\n')
        process.stderr = io.StringIO("")
        process.stdin = Mock()

        with patch("no_watermar.benchmark_paddleocr_session.subprocess.Popen", return_value=process) as popen:
            session = PaddleOCRPersistentSession(
                python_executable="C:\\mock\\python.exe",
                script_path=Path("C:\\mock\\paddleocr_mask.py"),
            )
            session.start()

        self.assertIsNotNone(session._process)
        self.assertEqual(popen.call_args.kwargs["env"]["PYTHONUTF8"], "1")
        self.assertEqual(popen.call_args.kwargs["env"]["PYTHONIOENCODING"], "utf-8")


if __name__ == "__main__":
    unittest.main()
