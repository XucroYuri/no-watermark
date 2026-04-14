from __future__ import annotations

from pathlib import Path
import sys
import tomllib
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PackagingMetadataTests(unittest.TestCase):
    def test_public_version_is_consistent_across_metadata_and_docs(self) -> None:
        payload = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        project_version = payload["project"]["version"]
        init_module = (PROJECT_ROOT / "src" / "no_watermar" / "__init__.py").read_text(encoding="utf-8")
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        changelog = (PROJECT_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

        self.assertIn(f'__version__ = "{project_version}"', init_module)
        self.assertIn(f"- Current maturity: `{project_version}`", readme)
        self.assertIn(f"## [{project_version}] - ", changelog)

    def test_pyproject_declares_public_console_entrypoints_and_extras(self) -> None:
        payload = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        project = payload["project"]

        self.assertEqual(project["name"], "no-watermar")
        self.assertEqual(project["scripts"]["no-watermar"], "no_watermar.cli:main")
        self.assertEqual(project["scripts"]["no-watermar-benchmark"], "no_watermar.benchmark_cli:main")
        self.assertEqual(project["requires-python"], ">=3.11")
        self.assertEqual(project["license"], "MIT")
        self.assertEqual(project["license-files"], ["LICENSE"])

        optional_dependencies = project["optional-dependencies"]
        self.assertIn("ocr", optional_dependencies)
        self.assertIn("lama", optional_dependencies)
        self.assertIn("dev", optional_dependencies)
        self.assertIn("experimental", optional_dependencies)
        self.assertTrue(any("paddleocr" in dependency for dependency in optional_dependencies["ocr"]))
        self.assertTrue(any("simple-lama-inpainting" in dependency for dependency in optional_dependencies["lama"]))
        self.assertTrue(any("build" in dependency for dependency in optional_dependencies["dev"]))
        self.assertTrue(any("diffusers" in dependency for dependency in optional_dependencies["experimental"]))

    def test_pyproject_exposes_maturity_classifiers(self) -> None:
        payload = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        classifiers = payload["project"]["classifiers"]

        self.assertIn("Environment :: Console", classifiers)
        self.assertNotIn("License :: OSI Approved :: MIT License", classifiers)
        self.assertIn("Operating System :: Microsoft :: Windows", classifiers)
        self.assertIn("Operating System :: POSIX :: Linux", classifiers)
        self.assertIn("Programming Language :: Python :: 3.11", classifiers)
        self.assertIn("Programming Language :: Python :: 3.12", classifiers)

    def test_pyproject_uses_project_license_files_instead_of_setuptools_legacy_field(self) -> None:
        payload = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        setuptools_payload = payload.get("tool", {}).get("setuptools", {})

        self.assertNotIn("license-files", setuptools_payload)


if __name__ == "__main__":
    unittest.main()
