from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class ReleaseToolingTests(unittest.TestCase):
    def test_ci_workflow_runs_package_build_checks(self) -> None:
        workflow_path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
        content = workflow_path.read_text(encoding="utf-8")

        self.assertIn("python -m build", content)
        self.assertIn("python -m twine check dist/*", content)
        self.assertIn("windows-latest", content)
        self.assertIn("ubuntu-latest", content)

    def test_release_workflow_builds_and_publishes_tagged_release(self) -> None:
        workflow_path = REPO_ROOT / ".github" / "workflows" / "release.yml"
        self.assertTrue(workflow_path.exists(), "release workflow should exist")
        content = workflow_path.read_text(encoding="utf-8")

        self.assertIn("workflow_dispatch", content)
        self.assertIn("tags:", content)
        self.assertIn("v*", content)
        self.assertIn("python -m build", content)
        self.assertIn("python -m twine check dist/*", content)
        self.assertIn("softprops/action-gh-release", content)
        self.assertIn("pypa/gh-action-pypi-publish", content)
        self.assertIn("id-token: write", content)

    def test_local_release_build_script_exists(self) -> None:
        script_path = REPO_ROOT / "tools" / "releases" / "build-release.ps1"
        self.assertTrue(script_path.exists(), "local release build helper should exist")
        content = script_path.read_text(encoding="utf-8")

        self.assertIn("python -m build", content)
        self.assertIn("python -m twine check", content)
        self.assertIn("python -m unittest discover -s tests -v", content)
        self.assertIn("python -m no_watermar.cli --help", content)

    def test_release_docs_reference_stable_baseline_capture(self) -> None:
        checklist = (REPO_ROOT / "docs" / "releases" / "release-checklist.md").read_text(encoding="utf-8")
        benchmark_helpers = (REPO_ROOT / "tools" / "benchmark" / "README.md").read_text(encoding="utf-8")

        self.assertIn("capture-stable-baseline.ps1", checklist)
        self.assertIn("benchmark evidence", benchmark_helpers)
        self.assertIn("capture-stable-baseline.ps1", benchmark_helpers)

    def test_gitignore_excludes_release_build_artifacts(self) -> None:
        content = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")

        self.assertIn("/dist/", content)
        self.assertIn("/build/", content)


if __name__ == "__main__":
    unittest.main()
