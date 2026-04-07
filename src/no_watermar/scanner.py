from __future__ import annotations

import re
from pathlib import Path

import cv2

from .io_utils import IMAGE_SUFFIXES, read_image
from .models import ScanItem

DEFAULT_EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".venv",
    "venv",
    "env",
    "runtime",
    "docs",
    "src",
    "tests",
    "no-watermar",
}

_COVER_HINT_RE = re.compile(r"(?:^|[-_])(0|cover|封面)$", re.IGNORECASE)


def scan_image_paths(
    root: Path,
    recursive: bool = True,
    excluded_dirs: set[str] | None = None,
) -> list[Path]:
    excluded = set(DEFAULT_EXCLUDED_DIRS)
    if excluded_dirs:
        excluded.update(excluded_dirs)

    root = root.resolve()
    image_paths: list[Path] = []

    if recursive:
        for directory in sorted(root.rglob("*")):
            if not directory.is_file():
                continue
            if any(part in excluded for part in directory.parts[len(root.parts) :]):
                continue
            if directory.suffix.lower() in IMAGE_SUFFIXES:
                image_paths.append(directory)
    else:
        for path in sorted(root.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                image_paths.append(path)

    return image_paths


def classify_image(path: Path, image) -> str:
    height, width = image.shape[:2]
    if width > height:
        return "landscape_regular"

    stem = path.stem
    if _COVER_HINT_RE.search(stem):
        return "cover_heavy"

    if _looks_like_cover(image):
        return "cover_heavy"

    return "portrait_regular"


def build_scan_items(
    root: Path,
    recursive: bool = True,
    excluded_dirs: set[str] | None = None,
    limit: int | None = None,
) -> list[ScanItem]:
    root = root.resolve()
    items: list[ScanItem] = []
    for index, image_path in enumerate(scan_image_paths(root, recursive=recursive, excluded_dirs=excluded_dirs)):
        if limit is not None and index >= limit:
            break
        image = read_image(image_path)
        height, width = image.shape[:2]
        items.append(
            ScanItem(
                source_path=image_path,
                relative_path=image_path.relative_to(root),
                width=width,
                height=height,
                category=classify_image(image_path, image),
            )
        )
    return items


def _looks_like_cover(image) -> bool:
    height, width = image.shape[:2]
    if width >= height:
        return False

    lower_left = image[int(height * 0.72) : height, : int(width * 0.18)]
    gray = cv2.cvtColor(lower_left, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(edges.mean())
    dark_fraction = float((gray < 80).mean())
    bright_fraction = float((gray > 190).mean())

    return edge_density > 24.0 and dark_fraction > 0.45 and bright_fraction > 0.12
