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

from no_watermar.env_loader import load_local_env, read_local_env_file, resolve_env_path_value


class EnvLoaderTests(unittest.TestCase):
    def test_resolve_env_path_value_normalizes_windows_relative_separators(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)

            resolved = resolve_env_path_value(
                ".\\.venvs\\lama\\Scripts\\python.exe",
                root,
            )

            self.assertEqual(
                resolved,
                str((root / ".venvs" / "lama" / "Scripts" / "python.exe").resolve()),
            )

    def test_read_local_env_file_resolves_relative_python_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            env_path = root / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "NO_WATERMAR_LAMA_PYTHON=.\\.venvs\\lama\\Scripts\\python.exe",
                        "NO_WATERMAR_CONFIG=.\\no-watermar.toml",
                    ]
                ),
                encoding="utf-8",
            )

            values = read_local_env_file(env_path)

            self.assertEqual(
                values["NO_WATERMAR_LAMA_PYTHON"],
                str((root / ".venvs" / "lama" / "Scripts" / "python.exe").resolve()),
            )
            self.assertEqual(values["NO_WATERMAR_CONFIG"], str((root / "no-watermar.toml").resolve()))

    def test_load_local_env_resolves_relative_python_paths_without_overwriting_existing_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            env_path = root / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "NO_WATERMAR_PADDLEOCR_PYTHON=.\\.venvs\\paddleocr\\Scripts\\python.exe",
                        "NO_WATERMAR_CONFIG=.\\no-watermar.toml",
                        "NO_WATERMAR_WATERMARK_KEYWORDS=alpha,beta",
                    ]
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"NO_WATERMAR_WATERMARK_KEYWORDS": "preset"}, clear=True):
                load_local_env(env_path)

                self.assertEqual(
                    os.environ["NO_WATERMAR_PADDLEOCR_PYTHON"],
                    str((root / ".venvs" / "paddleocr" / "Scripts" / "python.exe").resolve()),
                )
                self.assertEqual(
                    os.environ["NO_WATERMAR_CONFIG"],
                    str((root / "no-watermar.toml").resolve()),
                )
                self.assertEqual(os.environ["NO_WATERMAR_WATERMARK_KEYWORDS"], "preset")


if __name__ == "__main__":
    unittest.main()
