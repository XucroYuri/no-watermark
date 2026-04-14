from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
