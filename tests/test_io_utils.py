from __future__ import annotations

import io
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from no_watermar.io_utils import print_json


class _FakeStdout:
    def __init__(self, *, encoding: str = "gbk") -> None:
        self.encoding = encoding
        self.buffer = io.BytesIO()
        self._calls = 0

    def write(self, text: str) -> int:
        self._calls += 1
        if self._calls == 1:
            raise UnicodeEncodeError(self.encoding, text, 0, 1, "fake encode failure")
        return len(text)

    def flush(self) -> None:
        return None


class IoUtilsTests(unittest.TestCase):
    def test_print_json_falls_back_when_stdout_encoding_rejects_characters(self) -> None:
        fake_stdout = _FakeStdout()

        with patch("sys.stdout", fake_stdout):
            print_json({"status": "error", "error": "含有中文和�替换字符"})

        payload = fake_stdout.buffer.getvalue().decode(fake_stdout.encoding)
        self.assertIn("含有中文", payload)
        self.assertIn("\\ufffd", payload)


if __name__ == "__main__":
    unittest.main()
