from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

import cv2
import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from no_watermar.scan_manifest import create_scan_manifest, load_scan_manifest, summarize_scan_input, validate_scan_paths


def _write_image(path: Path, shape: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full(shape, 255, dtype=np.uint8)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError("encode failed")
    encoded.tofile(str(path))


class ScanManifestTests(unittest.TestCase):
    def test_summarize_scan_input_reports_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_image(root / "sample.jpg", (120, 220, 3))

            summary = summarize_scan_input(root, recursive=False)

            self.assertEqual(summary["status"], "ok")
            self.assertEqual(summary["item_count"], 1)
            self.assertEqual(summary["category_counts"], {"landscape_regular": 1})
            self.assertIn("runtime", summary["excluded_dirs"])

    def test_create_scan_manifest_writes_versioned_and_latest_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_root = root / "inputs"
            scans_root = root / "runtime" / "scans"
            _write_image(input_root / "sample.jpg", (120, 220, 3))

            summary = create_scan_manifest(input_root, scans_root=scans_root, recursive=False)

            self.assertEqual(summary["command"], "scan run")
            self.assertTrue(Path(summary["manifest_path"]).exists())
            self.assertTrue(Path(summary["latest_manifest_path"]).exists())
            loaded = load_scan_manifest(Path(summary["manifest_path"]))
            self.assertEqual(loaded["scan_id"], summary["scan_id"])

    def test_validate_scan_paths_rejects_scans_root_inside_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_root = root / "inputs"
            input_root.mkdir()

            with self.assertRaisesRegex(ValueError, "Scans root must not be inside input root"):
                validate_scan_paths(input_root=input_root, scans_root=input_root / "runtime" / "scans")

    def test_load_scan_manifest_rejects_non_object_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "invalid.json"
            manifest_path.write_text(json.dumps(["not-an-object"]), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must be a JSON object"):
                load_scan_manifest(manifest_path)


if __name__ == "__main__":
    unittest.main()
