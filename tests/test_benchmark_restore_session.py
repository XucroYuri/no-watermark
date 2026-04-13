from __future__ import annotations

import unittest
from unittest.mock import patch

from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from no_watermar.benchmark_restore_session import (
    RESTORE_RESOLVED_MODE_INPROCESS,
    RESTORE_RESOLVED_MODE_ONESHOT,
    RESTORE_RESOLVED_MODE_UNAVAILABLE,
    create_powerpaint_execution_context,
)


class BenchmarkRestoreSessionTests(unittest.TestCase):
    def test_create_powerpaint_execution_context_prefers_inprocess_when_available(self) -> None:
        with patch("no_watermar.benchmark_restore_session._powerpaint_module_available", return_value=True):
            context = create_powerpaint_execution_context({})

        self.assertEqual(context.requested_mode, "auto")
        self.assertEqual(context.resolved_mode, RESTORE_RESOLVED_MODE_INPROCESS)
        self.assertIsNotNone(context.python_executable)

    def test_create_powerpaint_execution_context_auto_falls_back_to_oneshot(self) -> None:
        with patch("no_watermar.benchmark_restore_session._powerpaint_module_available", return_value=False):
            with patch(
                "no_watermar.benchmark_restore_session.resolve_powerpaint_sidecar_python",
                return_value=("C:\\mock\\powerpaint\\python.exe", "mock sidecar"),
            ):
                with patch(
                    "no_watermar.benchmark_restore_session.PowerPaintPersistentSession.start",
                    side_effect=RuntimeError("boom"),
                ):
                    context = create_powerpaint_execution_context({"session_mode": "auto"})

        self.assertEqual(context.resolved_mode, RESTORE_RESOLVED_MODE_ONESHOT)
        self.assertIn("falling back to oneshot", context.note or "")

    def test_create_powerpaint_execution_context_persistent_reports_unavailable_on_failure(self) -> None:
        with patch("no_watermar.benchmark_restore_session._powerpaint_module_available", return_value=False):
            with patch(
                "no_watermar.benchmark_restore_session.resolve_powerpaint_sidecar_python",
                return_value=("C:\\mock\\powerpaint\\python.exe", "mock sidecar"),
            ):
                with patch(
                    "no_watermar.benchmark_restore_session.PowerPaintPersistentSession.start",
                    side_effect=RuntimeError("boom"),
                ):
                    context = create_powerpaint_execution_context({"session_mode": "persistent"})

        self.assertEqual(context.resolved_mode, RESTORE_RESOLVED_MODE_UNAVAILABLE)
        self.assertIn("Unable to start persistent PowerPaint sidecar", context.note or "")


if __name__ == "__main__":
    unittest.main()
