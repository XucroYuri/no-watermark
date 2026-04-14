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

from no_watermar.config import (
    dataset_profile_to_dict,
    discover_config_path,
    get_watermark_keyword_settings,
    init_project_config,
    list_config_template_names,
    provider_profile_to_dict,
    resolve_dataset_profile,
    resolve_provider_profile,
    validate_project_config,
)


def _assert_same_path(test_case: unittest.TestCase, actual: Path | str | None, expected: Path | str | None) -> None:
    test_case.assertIsNotNone(actual)
    test_case.assertIsNotNone(expected)
    test_case.assertEqual(Path(actual).resolve(), Path(expected).resolve())


class ConfigTests(unittest.TestCase):
    def test_discover_config_path_searches_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = root / "no-watermar.toml"
            config_path.write_text("[watermark_keywords]\n", encoding="utf-8")
            nested = root / "a" / "b"
            nested.mkdir(parents=True)

            with patch.dict(os.environ, {}, clear=True):
                _assert_same_path(self, discover_config_path(start_dir=nested), config_path)

    def test_get_watermark_keyword_settings_merges_presets_and_env_keywords(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = root / "no-watermar.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[watermark_keywords]",
                        'active_presets = ["brand"]',
                        "",
                        "[watermark_keywords.presets]",
                        'brand = ["Acme Studio", "@acme"]',
                        'stock = ["example.com", "rights reserved"]',
                    ]
                ),
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {
                    "NO_WATERMAR_CONFIG": str(config_path),
                    "NO_WATERMAR_WATERMARK_KEYWORD_PRESETS": "stock",
                    "NO_WATERMAR_WATERMARK_KEYWORDS": "flash sale",
                },
                clear=True,
            ):
                settings = get_watermark_keyword_settings()

            _assert_same_path(self, settings.config_path, config_path)
            self.assertEqual(settings.active_presets, ("brand", "stock"))
            self.assertIn("acmestudio", settings.tokens)
            self.assertIn("@acme", settings.tokens)
            self.assertIn("example.com", settings.tokens)
            self.assertIn("rightsreserved", settings.tokens)
            self.assertIn("flashsale", settings.tokens)

    def test_validate_project_config_reports_defaults_only_when_no_config_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)

            with patch.dict(os.environ, {}, clear=True):
                summary = validate_project_config(start_dir=root)

            self.assertEqual(summary["status"], "ok")
            self.assertEqual(summary["config"]["path"], None)
            self.assertEqual(summary["config"]["resolution_mode"], "defaults-only")
            self.assertTrue(summary["warnings"])

    def test_validate_project_config_reports_dataset_and_provider_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = root / "no-watermar.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[watermark_keywords]",
                        "",
                        "[profiles.datasets.smoke]",
                        'input = "./inputs/smoke"',
                        "recursive = false",
                        "limit = 2",
                        'benchmark_dataset = "regular_corner_text"',
                        "",
                        "[profiles.providers.ocr_telea]",
                        'mask_provider = "paddleocr"',
                        'restore_provider = "telea"',
                        'ocr_session_mode = "persistent"',
                        'restore_prompt = "remove the corner watermark and reconstruct the background naturally"',
                        'restore_negative_prompt = "smear, blur, melted text"',
                        "",
                        "[profiles.providers.ocr_telea.restore_options]",
                        "steps = 30",
                        "guidance_scale = 6.5",
                    ]
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                summary = validate_project_config(start_dir=root)

            smoke = summary["profiles"]["dataset_profiles"]["smoke"]
            ocr_telea = summary["profiles"]["provider_profiles"]["ocr_telea"]
            self.assertEqual(smoke["input_root"], str((root / "inputs" / "smoke").resolve()))
            self.assertFalse(smoke["recursive"])
            self.assertEqual(smoke["limit"], 2)
            self.assertEqual(smoke["benchmark_dataset"], "regular_corner_text")
            self.assertEqual(ocr_telea["mask_provider"], "paddleocr")
            self.assertEqual(ocr_telea["restore_provider"], "telea")
            self.assertEqual(ocr_telea["ocr_session_mode"], "persistent")
            self.assertEqual(
                ocr_telea["restore_prompt"],
                "remove the corner watermark and reconstruct the background naturally",
            )
            self.assertEqual(ocr_telea["restore_negative_prompt"], "smear, blur, melted text")
            self.assertEqual(ocr_telea["restore_options"]["steps"], 30)
            self.assertEqual(ocr_telea["restore_options"]["guidance_scale"], 6.5)

    def test_init_project_config_writes_selected_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)

            with patch.dict(os.environ, {}, clear=True):
                summary = init_project_config(start_dir=root, template_name="brand-social")

            config_path = root / "no-watermar.toml"
            self.assertEqual(summary["status"], "ok")
            _assert_same_path(self, summary["config_path"], config_path)
            self.assertEqual(summary["template"], "brand-social")
            self.assertFalse(summary["overwritten"])
            self.assertTrue(config_path.exists())
            self.assertIn('active_presets = ["brand_social"]', config_path.read_text(encoding="utf-8"))

    def test_init_project_config_writes_stable_public_template_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)

            with patch.dict(os.environ, {}, clear=True):
                summary = init_project_config(start_dir=root, template_name="stable-public")

            config_path = root / "no-watermar.toml"
            content = config_path.read_text(encoding="utf-8")
            self.assertEqual(summary["template"], "stable-public")
            self.assertIn("[profiles.datasets.local_smoke]", content)
            self.assertIn('[profiles.providers.seed_telea]', content)
            self.assertIn('[profiles.providers.ocr_telea]', content)
            self.assertIn('[profiles.providers.lama_eval]', content)
            self.assertIn('[profiles.providers.ocr_corner_crop]', content)

    def test_init_project_config_rejects_existing_file_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = root / "no-watermar.toml"
            config_path.write_text("existing\n", encoding="utf-8")

            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(FileExistsError, "already exists"):
                    init_project_config(start_dir=root)

    def test_list_config_template_names_includes_builtins(self) -> None:
        self.assertEqual(
            list_config_template_names(),
            ("default", "brand-social", "stock-marketplaces", "mixed-corner-text", "stable-public"),
        )

    def test_resolve_dataset_and_provider_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = root / "no-watermar.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[watermark_keywords]",
                        "",
                        "[profiles.datasets.local_quick]",
                        'input = "./inputs/local"',
                        "limit = 3",
                        "",
                        "[profiles.providers.seed_telea]",
                        'mask_provider = "seed_manifest"',
                        'restore_provider = "telea"',
                        'restore_prompt = "keep skin texture and garment detail"',
                        "",
                        "[profiles.providers.seed_telea.restore_options]",
                        "steps = 24",
                    ]
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                dataset_profile = resolve_dataset_profile("local_quick", start_dir=root)
                provider_profile = resolve_provider_profile("seed_telea", start_dir=root)

            self.assertEqual(dataset_profile_to_dict(dataset_profile)["input_root"], str((root / "inputs" / "local").resolve()))
            self.assertEqual(dataset_profile.limit, 3)
            self.assertEqual(provider_profile_to_dict(provider_profile)["mask_provider"], "seed_manifest")
            self.assertEqual(provider_profile.restore_provider, "telea")
            self.assertEqual(provider_profile.restore_prompt, "keep skin texture and garment detail")
            self.assertEqual(provider_profile.restore_options["steps"], 24)

    def test_get_watermark_keyword_settings_raises_for_unknown_presets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = root / "no-watermar.toml"
            config_path.write_text("[watermark_keywords]\n", encoding="utf-8")

            with patch.dict(
                os.environ,
                {
                    "NO_WATERMAR_CONFIG": str(config_path),
                    "NO_WATERMAR_WATERMARK_KEYWORD_PRESETS": "missing",
                },
                clear=True,
            ):
                with self.assertRaisesRegex(ValueError, "missing"):
                    get_watermark_keyword_settings()

    def test_resolve_dataset_profile_raises_for_unknown_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "no-watermar.toml").write_text("[watermark_keywords]\n", encoding="utf-8")

            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(ValueError, "Unknown dataset profile"):
                    resolve_dataset_profile("missing", start_dir=root)


if __name__ == "__main__":
    unittest.main()
