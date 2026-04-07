from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def read_image(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to decode image: {path}")
    return image


def read_mask(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.uint8)
    mask = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Unable to decode grayscale mask: {path}")
    return mask


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f"Unable to encode image for: {path}")
    encoded.tofile(str(path))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json_dumps(data)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            temp_path = Path(handle.name)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=_json_default)


def print_json(data: Any) -> None:
    _write_stdout_line(json_dumps(data))


def _write_stdout_line(text: str) -> None:
    payload = f"{text}\n"
    try:
        print(text)
        return
    except UnicodeEncodeError:
        stdout = sys.stdout
        buffer = getattr(stdout, "buffer", None)
        encoding = getattr(stdout, "encoding", None) or "utf-8"
        if buffer is not None:
            buffer.write(payload.encode(encoding, errors="backslashreplace"))
            buffer.flush()
            return
        stdout.write(payload.encode("ascii", errors="backslashreplace").decode("ascii"))
        stdout.flush()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
