from __future__ import annotations

import os
import sys
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class SetupToolingTests(unittest.TestCase):
    def test_example_config_includes_stable_optional_profile(self) -> None:
        content = (REPO_ROOT / "no-watermar.toml.example").read_text(encoding="utf-8")

        self.assertIn("[profiles.providers.lama_eval]", content)
        self.assertIn('[profiles.providers.ocr_corner_crop]', content)

    def test_bootstrap_sidecars_printonly_initializes_stable_public_config_when_missing(self) -> None:
        shell = shutil.which("powershell") or shutil.which("pwsh")
        if shell is None:
            self.skipTest("PowerShell is not available")

        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir)
            (project_root / "tools" / "setup").mkdir(parents=True, exist_ok=True)
            completed = subprocess.run(
                [
                    shell,
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(REPO_ROOT / "tools" / "setup" / "bootstrap-sidecars.ps1"),
                    "-ProjectRoot",
                    str(project_root),
                    "-StableOnly",
                    "-PrintOnly",
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, msg=completed.stderr or completed.stdout)
            self.assertIn("stable-public", completed.stdout)
            self.assertIn("Initialize stable-public config", completed.stdout)

    def test_bootstrap_sidecars_uses_repo_cli_python_for_config_init(self) -> None:
        shell = shutil.which("powershell") or shutil.which("pwsh")
        if shell is None:
            self.skipTest("PowerShell is not available")

        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir)
            shim_path = project_root / ("python-shim.cmd" if os.name == "nt" else "python-shim.sh")
            if os.name == "nt":
                shim_path.write_text(
                    "\n".join(
                        [
                            "@echo off",
                            'if "%1"=="-m" if "%2"=="venv" (',
                            f'  "{sys.executable}" %*',
                            "  exit /b %errorlevel%",
                            ")",
                            "echo shim blocked unexpected module invocation>&2",
                            "exit /b 123",
                            "",
                        ]
                    ),
                    encoding="utf-8",
                )
            else:
                shim_path.write_text(
                    "\n".join(
                        [
                            "#!/usr/bin/env sh",
                            'if [ "$1" = "-m" ] && [ "$2" = "venv" ]; then',
                            f'  exec "{sys.executable}" "$@"',
                            "fi",
                            'echo "shim blocked unexpected module invocation" >&2',
                            "exit 123",
                            "",
                        ]
                    ),
                    encoding="utf-8",
                )
                shim_path.chmod(0o755)

            completed = subprocess.run(
                [
                    shell,
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(REPO_ROOT / "tools" / "setup" / "bootstrap-sidecars.ps1"),
                    "-ProjectRoot",
                    str(project_root),
                    "-StableOnly",
                    "-PythonCommand",
                    str(shim_path),
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, msg=completed.stderr or completed.stdout)
            self.assertTrue((project_root / "no-watermar.toml").exists())
            self.assertNotIn("shim blocked unexpected module invocation", completed.stderr)

    def test_bootstrap_sidecars_fails_fast_when_external_command_returns_nonzero(self) -> None:
        shell = shutil.which("powershell") or shutil.which("pwsh")
        if shell is None:
            self.skipTest("PowerShell is not available")

        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir)
            shim_path = project_root / ("python-shim.cmd" if os.name == "nt" else "python-shim.sh")
            if os.name == "nt":
                shim_path.write_text(
                    "\n".join(
                        [
                            "@echo off",
                            "echo shim forced failure>&2",
                            "exit /b 123",
                            "",
                        ]
                    ),
                    encoding="utf-8",
                )
            else:
                shim_path.write_text(
                    "\n".join(
                        [
                            "#!/usr/bin/env sh",
                            'echo "shim forced failure" >&2',
                            "exit 123",
                            "",
                        ]
                    ),
                    encoding="utf-8",
                )
                shim_path.chmod(0o755)

            completed = subprocess.run(
                [
                    shell,
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(REPO_ROOT / "tools" / "setup" / "bootstrap-sidecars.ps1"),
                    "-ProjectRoot",
                    str(project_root),
                    "-StableOnly",
                    "-SkipConfigInit",
                    "-PythonCommand",
                    str(shim_path),
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("shim forced failure", completed.stderr)

    def test_validate_sidecars_passes_default_sidecar_paths_to_probe_and_doctor(self) -> None:
        shell = shutil.which("powershell") or shutil.which("pwsh")
        if shell is None:
            self.skipTest("PowerShell is not available")

        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir)
            paddle_python = project_root / ".venvs" / "paddleocr" / "Scripts" / "python.exe"
            lama_python = project_root / ".venvs" / "lama" / "Scripts" / "python.exe"
            paddle_python.parent.mkdir(parents=True, exist_ok=True)
            lama_python.parent.mkdir(parents=True, exist_ok=True)
            paddle_python.write_text("", encoding="utf-8")
            lama_python.write_text("", encoding="utf-8")
            (project_root / "benchmark.py").write_text("# probe shim target\n", encoding="utf-8")

            log_path = project_root / "python-invocations.log"
            shim_path = project_root / ("python.cmd" if os.name == "nt" else "python")
            if os.name == "nt":
                shim_path.write_text(
                    "\n".join(
                        [
                            "@echo off",
                            "setlocal",
                            f'>> "{log_path}" echo ARGS:%*',
                            f'>> "{log_path}" echo PADDLE:%NO_WATERMAR_PADDLEOCR_PYTHON%',
                            f'>> "{log_path}" echo LAMA:%NO_WATERMAR_LAMA_PYTHON%',
                            'if "%1"=="-m" (',
                            '  if not "%NO_WATERMAR_PADDLEOCR_PYTHON%"=="" if not "%NO_WATERMAR_LAMA_PYTHON%"=="" (',
                            '    echo {"stable_setup":{"status":"ready","release_blocking_ready":true,"optional_ready":true,"blocking_issues":[],"optional_issues":[],"recommended_commands":[]}}',
                            "  ) else (",
                            '    echo {"stable_setup":{"status":"action_required","release_blocking_ready":false,"optional_ready":false,"blocking_issues":[{"provider_name":"paddleocr","issue_code":"not_configured","detail":"missing env"}],"optional_issues":[{"provider_name":"lama","issue_code":"not_configured","detail":"missing env"}],"recommended_commands":[]}}',
                            "  )",
                            "  exit /b 0",
                            ")",
                            "echo probe ok",
                            "exit /b 0",
                            "",
                        ]
                    ),
                    encoding="utf-8",
                )
            else:
                shim_path.write_text(
                    "\n".join(
                        [
                            "#!/usr/bin/env sh",
                            f'printf "ARGS:%s\\n" "$*" >> "{log_path}"',
                            f'printf "PADDLE:%s\\n" "${{NO_WATERMAR_PADDLEOCR_PYTHON}}" >> "{log_path}"',
                            f'printf "LAMA:%s\\n" "${{NO_WATERMAR_LAMA_PYTHON}}" >> "{log_path}"',
                            'if [ "$1" = "-m" ]; then',
                            '  if [ -n "${NO_WATERMAR_PADDLEOCR_PYTHON}" ] && [ -n "${NO_WATERMAR_LAMA_PYTHON}" ]; then',
                            '    printf \'{"stable_setup":{"status":"ready","release_blocking_ready":true,"optional_ready":true,"blocking_issues":[],"optional_issues":[],"recommended_commands":[]}}\\n\'',
                            "  else",
                            '    printf \'{"stable_setup":{"status":"action_required","release_blocking_ready":false,"optional_ready":false,"blocking_issues":[{"provider_name":"paddleocr","issue_code":"not_configured","detail":"missing env"}],"optional_issues":[{"provider_name":"lama","issue_code":"not_configured","detail":"missing env"}],"recommended_commands":[]}}\\n\'',
                            "  fi",
                            "  exit 0",
                            "fi",
                            'printf "probe ok\\n"',
                            "exit 0",
                            "",
                        ]
                    ),
                    encoding="utf-8",
                )
                shim_path.chmod(0o755)

            env = os.environ.copy()
            env["PATH"] = str(project_root) + os.pathsep + env.get("PATH", "")

            completed = subprocess.run(
                [
                    shell,
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(REPO_ROOT / "tools" / "setup" / "validate-sidecars.ps1"),
                    "-ProjectRoot",
                    str(project_root),
                    "-StableOnly",
                    "-RunProbe",
                    "-RunDoctor",
                ],
                cwd=str(REPO_ROOT),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, msg=completed.stderr or completed.stdout)
            self.assertIn("FOUND paddleocr", completed.stdout)
            self.assertIn("FOUND lama", completed.stdout)
            self.assertIn("STABLE status=ready", completed.stdout)
            self.assertIn("STABLE release_blocking_ready=True", completed.stdout)
            self.assertIn("STABLE optional_ready=True", completed.stdout)

            log_text = log_path.read_text(encoding="utf-8")
            self.assertIn("ARGS:", log_text)
            self.assertIn("benchmark.py probe-providers", log_text)
            self.assertIn("-m no_watermar.cli providers doctor", log_text)
            self.assertIn(f"PADDLE:{paddle_python}", log_text)
            self.assertIn(f"LAMA:{lama_python}", log_text)


if __name__ == "__main__":
    unittest.main()
