from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


class EvidencePackagingTests(unittest.TestCase):
    def test_package_evidence_script_exists(self) -> None:
        script_path = REPO_ROOT / "tools" / "releases" / "package-evidence.py"
        self.assertTrue(script_path.exists(), "evidence packaging helper should exist")
        content = script_path.read_text(encoding="utf-8")

        self.assertIn("zipfile", content)
        self.assertIn("latest.json", content)

    def test_package_evidence_script_writes_release_bundle_zip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            capture_script = REPO_ROOT / "tools" / "benchmark" / "capture-disposable-evidence.py"
            package_script = REPO_ROOT / "tools" / "releases" / "package-evidence.py"

            capture = subprocess.run(
                [
                    sys.executable,
                    str(capture_script),
                    "--workspace-root",
                    str(workspace_root),
                    "--repetitions",
                    "2",
                    "--minimum-run-count",
                    "2",
                    "--clean",
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(capture.returncode, 0, msg=capture.stderr or capture.stdout)
            capture_payload = json.loads(capture.stdout)
            evidence_root = workspace_root / "benchmarks" / "evidence"

            packaged = subprocess.run(
                [
                    sys.executable,
                    str(package_script),
                    "--evidence-root",
                    str(evidence_root),
                    "--output-dir",
                    str(workspace_root / "dist"),
                    "--bundle-name",
                    "disposable-evidence",
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(packaged.returncode, 0, msg=packaged.stderr or packaged.stdout)
            payload = json.loads(packaged.stdout)

            archive_path = Path(payload["archive_path"])
            self.assertTrue(archive_path.exists())
            self.assertEqual(payload["evidence_status"], "ready")
            self.assertEqual(payload["latest_evidence_id"], capture_payload["evidence"]["evidence_id"])

            with zipfile.ZipFile(archive_path) as bundle:
                names = set(bundle.namelist())

            self.assertIn("manifest.json", names)
            self.assertIn("latest.json", names)
            self.assertIn("latest.md", names)
            self.assertTrue(any(name.startswith("artifacts/comparisons/") for name in names))
            self.assertTrue(any(name.startswith("artifacts/trends/") for name in names))


if __name__ == "__main__":
    unittest.main()
