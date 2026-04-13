from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from no_watermar.provider_runtime import probe_current_module, probe_python_module, summarize_probe


class ProviderRuntimeTests(unittest.TestCase):
    def test_probe_current_module_reports_importable_module(self) -> None:
        probe = probe_current_module("json")
        available, note = summarize_probe("json", probe)

        self.assertTrue(probe["ok"])
        self.assertTrue(probe["module_found"])
        self.assertTrue(probe["importable"])
        self.assertTrue(available)
        self.assertIn("importable", note)

    def test_probe_python_module_reports_missing_module(self) -> None:
        probe = probe_python_module(sys.executable, "no_watermar_missing_module_for_test")
        available, note = summarize_probe("missing", probe)

        self.assertFalse(probe["ok"])
        self.assertFalse(probe["module_found"])
        self.assertFalse(probe["importable"])
        self.assertFalse(available)
        self.assertIn("Module not found", probe["error"])
        self.assertIn("unavailable", note)


if __name__ == "__main__":
    unittest.main()
