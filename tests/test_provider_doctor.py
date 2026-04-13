from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from no_watermar.provider_doctor import build_provider_doctor_report


class ProviderDoctorTests(unittest.TestCase):
    def test_build_provider_doctor_report_includes_sidecar_env_file_hints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            env_path = root / ".env"
            tools_sidecars = root / "tools" / "sidecars"
            tools_sidecars.mkdir(parents=True, exist_ok=True)
            (tools_sidecars / "paddleocr_mask.py").write_text("print('ok')\n", encoding="utf-8")
            (tools_sidecars / "lama_restore.py").write_text("print('ok')\n", encoding="utf-8")
            (tools_sidecars / "powerpaint_restore.py").write_text("print('ok')\n", encoding="utf-8")
            (tools_sidecars / "brushnet_restore.py").write_text("print('ok')\n", encoding="utf-8")
            env_path.write_text(
                "\n".join(
                    [
                        "NO_WATERMAR_PADDLEOCR_PYTHON=.\\.venvs\\paddleocr\\Scripts\\python.exe",
                        "NO_WATERMAR_LAMA_PYTHON=.\\.venvs\\lama\\Scripts\\python.exe",
                    ]
                ),
                encoding="utf-8",
            )
            descriptors = {
                "mask_providers": [
                    {
                        "name": "rule_based_roi",
                        "implemented": True,
                        "runtime_available": True,
                    },
                    {
                        "name": "paddleocr",
                        "implemented": True,
                        "runtime_available": False,
                        "runtime_note": "Module not found: paddleocr",
                        "runtime_probe": {
                            "ok": False,
                            "error": "Module not found: paddleocr",
                        },
                        "default_mode": "persistent-sidecar",
                        "execution_modes": ["oneshot-sidecar", "persistent-sidecar"],
                    },
                ],
                "restore_providers": [
                    {
                        "name": "telea",
                        "implemented": True,
                        "runtime_available": True,
                    },
                    {
                        "name": "lama",
                        "implemented": True,
                        "runtime_available": False,
                        "runtime_note": "Module not found: simple_lama_inpainting",
                        "runtime_probe": {
                            "ok": False,
                            "error": "Module not found: simple_lama_inpainting",
                        },
                        "default_mode": "oneshot-sidecar",
                        "execution_modes": ["oneshot-sidecar"],
                    },
                    {
                        "name": "powerpaint_v2_1",
                        "implemented": True,
                        "runtime_available": False,
                        "runtime_note": "Module not found: powerpaint",
                        "runtime_probe": {
                            "ok": False,
                            "error": "Module not found: powerpaint",
                        },
                        "default_mode": "oneshot-sidecar",
                        "execution_modes": ["oneshot-sidecar"],
                    },
                    {
                        "name": "brushnet",
                        "implemented": True,
                        "runtime_available": False,
                        "runtime_note": "Module not found: diffusers",
                        "runtime_probe": {
                            "ok": False,
                            "error": "Module not found: diffusers",
                        },
                        "default_mode": "oneshot-sidecar",
                        "execution_modes": ["inprocess", "oneshot-sidecar"],
                    },
                ],
            }

            with patch("no_watermar.provider_doctor.list_provider_descriptors", return_value=descriptors):
                with patch.dict(os.environ, {}, clear=True):
                    report = build_provider_doctor_report(project_root=root)

            self.assertEqual(report["status"], "ok")
            self.assertTrue(report["env_file"]["exists"])
            self.assertEqual(report["summary"]["implemented_unavailable"], 4)
            self.assertEqual(len(report["sidecars"]), 5)
            paddle = next(sidecar for sidecar in report["sidecars"] if sidecar["provider_name"] == "paddleocr")
            diffusers = next(sidecar for sidecar in report["sidecars"] if sidecar["provider_name"] == "diffusers_inpaint")
            powerpaint = next(sidecar for sidecar in report["sidecars"] if sidecar["provider_name"] == "powerpaint_v2_1")
            brushnet = next(sidecar for sidecar in report["sidecars"] if sidecar["provider_name"] == "brushnet")
            self.assertEqual(paddle["configured_source"], "env_file")
            self.assertTrue(paddle["script_exists"])
            self.assertFalse(paddle["runtime_available"])
            self.assertIsNone(diffusers["configured_source"])
            self.assertFalse(powerpaint["runtime_available"])
            self.assertFalse(brushnet["runtime_available"])
            self.assertIn("not currently exported", "\n".join(report["warnings"]))
            self.assertTrue(any("NO_WATERMAR_PADDLEOCR_PYTHON" in item for item in report["recommendations"]))

    def test_build_provider_doctor_report_includes_compatibility_matrix_and_version_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            env_path = root / ".env"
            tools_sidecars = root / "tools" / "sidecars"
            venvs = root / ".venvs"
            paddle_python = venvs / "paddleocr" / "Scripts" / "python.exe"
            lama_python = venvs / "lama" / "Scripts" / "python.exe"
            powerpaint_python = venvs / "powerpaint" / "Scripts" / "python.exe"
            brushnet_python = venvs / "brushnet" / "Scripts" / "python.exe"
            tools_sidecars.mkdir(parents=True, exist_ok=True)
            paddle_python.parent.mkdir(parents=True, exist_ok=True)
            lama_python.parent.mkdir(parents=True, exist_ok=True)
            powerpaint_python.parent.mkdir(parents=True, exist_ok=True)
            brushnet_python.parent.mkdir(parents=True, exist_ok=True)
            paddle_python.write_text("", encoding="utf-8")
            lama_python.write_text("", encoding="utf-8")
            powerpaint_python.write_text("", encoding="utf-8")
            brushnet_python.write_text("", encoding="utf-8")
            (tools_sidecars / "paddleocr_mask.py").write_text("print('ok')\n", encoding="utf-8")
            (tools_sidecars / "lama_restore.py").write_text("print('ok')\n", encoding="utf-8")
            (tools_sidecars / "powerpaint_restore.py").write_text("print('ok')\n", encoding="utf-8")
            (tools_sidecars / "brushnet_restore.py").write_text("print('ok')\n", encoding="utf-8")
            env_path.write_text(
                "\n".join(
                    [
                        f"NO_WATERMAR_PADDLEOCR_PYTHON={paddle_python}",
                        f"NO_WATERMAR_LAMA_PYTHON={lama_python}",
                        f"NO_WATERMAR_POWERPAINT_PYTHON={powerpaint_python}",
                        f"NO_WATERMAR_BRUSHNET_PYTHON={brushnet_python}",
                    ]
                ),
                encoding="utf-8",
            )
            descriptors = {
                "mask_providers": [
                    {
                        "name": "paddleocr",
                        "implemented": True,
                        "runtime_available": True,
                        "runtime_note": "ok",
                        "runtime_probe": {"ok": True},
                        "default_mode": "persistent-sidecar",
                        "execution_modes": ["oneshot-sidecar", "persistent-sidecar"],
                    },
                ],
                "restore_providers": [
                    {
                        "name": "lama",
                        "implemented": True,
                        "runtime_available": False,
                        "runtime_note": "Module not found: simple_lama_inpainting",
                        "runtime_probe": {
                            "ok": False,
                            "error": "Module not found: simple_lama_inpainting",
                        },
                        "default_mode": "oneshot-sidecar",
                        "execution_modes": ["oneshot-sidecar"],
                    },
                    {
                        "name": "powerpaint_v2_1",
                        "implemented": True,
                        "runtime_available": False,
                        "runtime_note": "Module not found: powerpaint",
                        "runtime_probe": {
                            "ok": False,
                            "error": "Module not found: powerpaint",
                        },
                        "default_mode": "oneshot-sidecar",
                        "execution_modes": ["oneshot-sidecar"],
                    },
                    {
                        "name": "brushnet",
                        "implemented": True,
                        "runtime_available": False,
                        "runtime_note": "BrushNet support is missing from diffusers",
                        "runtime_probe": {
                            "ok": False,
                            "error": "BrushNet support is missing from diffusers",
                        },
                        "default_mode": "oneshot-sidecar",
                        "execution_modes": ["inprocess", "oneshot-sidecar"],
                    },
                ],
            }

            def fake_probe_python_info(python_executable: str | Path, *, timeout_ms: int = 20000) -> dict[str, object]:
                del timeout_ms
                path = str(python_executable)
                if "paddleocr" in path:
                    return {"ok": True, "python_executable": path, "python_version": "3.11.9", "error": None}
                if "powerpaint" in path:
                    return {"ok": True, "python_executable": path, "python_version": "3.12.4", "error": None}
                if "brushnet" in path:
                    return {"ok": True, "python_executable": path, "python_version": "3.12.8", "error": None}
                return {"ok": True, "python_executable": path, "python_version": "3.13.1", "error": None}

            with patch("no_watermar.provider_doctor.list_provider_descriptors", return_value=descriptors):
                with patch("no_watermar.provider_doctor.probe_python_info", side_effect=fake_probe_python_info):
                    with patch.dict(os.environ, {}, clear=True):
                        report = build_provider_doctor_report(project_root=root)

            self.assertIn("compatibility_matrix", report)
            self.assertEqual(report["compatibility_matrix"]["paddleocr"]["validated_python_versions"], ["3.8", "3.9", "3.10", "3.11", "3.12"])
            self.assertEqual(report["compatibility_matrix"]["lama"]["validated_python_versions"], ["3.12"])
            self.assertEqual(report["compatibility_matrix"]["diffusers_inpaint"]["validated_python_versions"], ["3.12"])
            self.assertEqual(report["compatibility_matrix"]["powerpaint_v2_1"]["validated_python_versions"], ["3.12"])
            self.assertEqual(report["compatibility_matrix"]["brushnet"]["validated_python_versions"], ["3.12"])
            paddle = next(sidecar for sidecar in report["sidecars"] if sidecar["provider_name"] == "paddleocr")
            lama = next(sidecar for sidecar in report["sidecars"] if sidecar["provider_name"] == "lama")
            powerpaint = next(sidecar for sidecar in report["sidecars"] if sidecar["provider_name"] == "powerpaint_v2_1")
            brushnet = next(sidecar for sidecar in report["sidecars"] if sidecar["provider_name"] == "brushnet")
            self.assertEqual(paddle["configured_python_version"], "3.11.9")
            self.assertEqual(paddle["compatibility"]["status"], "validated")
            self.assertEqual(lama["configured_python_version"], "3.13.1")
            self.assertEqual(lama["compatibility"]["status"], "unvalidated")
            self.assertEqual(powerpaint["configured_python_version"], "3.12.4")
            self.assertEqual(powerpaint["compatibility"]["status"], "validated")
            self.assertEqual(brushnet["configured_python_version"], "3.12.8")
            self.assertEqual(brushnet["compatibility"]["status"], "validated")
            self.assertTrue(any("validated Python version" in item for item in report["recommendations"]))


if __name__ == "__main__":
    unittest.main()
