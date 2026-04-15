#!/usr/bin/env python3
"""Eagle Library Batch Watermark Processor

Integrates with Eagle image library API and the no-watermar framework
to detect, crop, and delete watermarked images in batch.

Action rules:
  - DELETE: watermark is centered (box centre in centre 40% of image)
            OR watermark coverage > 50% of image area
  - CROP:   corner/edge watermark with coverage <= 50%
  - SKIP:   no watermark detected

Usage:
    python eagle_batch_watermark.py --dry-run --limit 10
    python eagle_batch_watermark.py --limit 100
    python eagle_batch_watermark.py              # process all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Bootstrap: add no-watermar src to path so we can import its modules
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, "E:/Code/Github/no-watermar/src")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from no_watermar.detector import (  # noqa: E402
    RoiSpec,
    _detect_text_mask,
    _expand_box,
    _mask_to_boxes,
    _merge_boxes,
    _resolve_roi,
    detect_watermarks,
)
from no_watermar.io_utils import read_image, write_image  # noqa: E402
from no_watermar.models import DetectionResult, ScanItem  # noqa: E402
from no_watermar.restorer import crop_image_to_remove_corner_watermark  # noqa: E402
from no_watermar.scanner import classify_image  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EAGLE_API_BASE = "http://localhost:41595"
EAGLE_LIBRARY_PATH = Path(r"E:\Eagle\图库搜集.library")
REPORT_PATH = PROJECT_ROOT / "runtime" / "eagle-batch-report.json"

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".webp",
    ".tiff",
    ".tif",
}
SUPPORTED_EXT_NO_DOT = {e.lstrip(".") for e in SUPPORTED_EXTENSIONS}

# Centre ROI specs – cover roughly 20%-80% of each axis (centre 60% x 60%).
PORTRAIT_CENTER_SPEC = RoiSpec(
    name="center",
    x1=0.20,
    y1=0.20,
    x2=0.80,
    y2=0.80,
    threshold=20,
    merge_all=True,
    fallback=(0.20, 0.20, 0.80, 0.80),
)
LANDSCAPE_CENTER_SPEC = RoiSpec(
    name="center",
    x1=0.20,
    y1=0.20,
    x2=0.80,
    y2=0.80,
    threshold=20,
    merge_all=True,
    fallback=(0.20, 0.20, 0.80, 0.80),
)

# "Centred" = bounding-box centre falls within the centre 40% of the image.
# Centre 40% spans [0.30, 0.70] on each axis.
CENTER_40_LO = 0.30
CENTER_40_HI = 0.70

COVERAGE_DELETE_THRESHOLD = 0.50
DELETE_BATCH_SIZE = 50
API_TIMEOUT_SEC = 30


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class EagleItem:
    id: str
    name: str
    ext: str
    width: int
    height: int


@dataclass
class WatermarkDecision:
    item_id: str
    item_name: str
    action: str  # "crop", "delete", "skip", "error"
    reason: str = ""
    coverage_ratio: float = 0.0
    is_centered: bool = False
    crop_bounds: dict[str, int] | None = None
    error: str = ""
    elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# Eagle API helpers (synchronous, urllib-based)
# ---------------------------------------------------------------------------
def _api_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Simple GET to Eagle native API."""
    query = ""
    if params:
        query = "?" + "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{EAGLE_API_BASE}{path}{query}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=API_TIMEOUT_SEC) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _api_post(path: str, body: dict[str, Any]) -> dict[str, Any]:
    """Simple POST to Eagle native API."""
    url = f"{EAGLE_API_BASE}{path}"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=API_TIMEOUT_SEC) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_all_items(
    limit: int | None = None,
    offset: int = 0,
    folder_ids: list[str] | None = None,
) -> list[EagleItem]:
    """Fetch items from Eagle API with pagination, filtered to image types.

    Args:
        limit: Maximum number of items to fetch (None = all).
        offset: Pagination offset.
        folder_ids: If provided, only fetch items belonging to these folder IDs.
    """
    items: list[EagleItem] = []
    page_size = 100

    # When folder_ids are given, query each folder separately and merge.
    if folder_ids:
        for fid in folder_ids:
            folder_items = _fetch_items_single_folder(
                fid, limit=limit, page_size=page_size
            )
            items.extend(folder_items)
            if limit is not None and len(items) >= limit:
                items = items[:limit]
                break
        return items

    # No folder filter – paginate the whole library.
    cur_offset = offset
    while True:
        fetch_count = page_size
        if limit is not None:
            remaining = limit - len(items)
            if remaining <= 0:
                break
            fetch_count = min(page_size, remaining)

        try:
            result = _api_get(
                "/api/item/list", {"limit": fetch_count, "offset": cur_offset}
            )
        except urllib.error.URLError as exc:
            print(f"  [ERROR] Cannot reach Eagle API: {exc}")
            break

        if result.get("status") != "success":
            print(f"  [ERROR] Eagle API error: {result}")
            break

        data = result.get("data", [])
        if not data:
            break

        for d in data:
            ext = (d.get("ext") or "").lower()
            if ext not in SUPPORTED_EXT_NO_DOT:
                continue
            items.append(
                EagleItem(
                    id=d["id"],
                    name=d.get("name", ""),
                    ext=ext,
                    width=d.get("width", 0),
                    height=d.get("height", 0),
                )
            )

        cur_offset += len(data)

        if len(data) < fetch_count:
            break
        if limit is not None and len(items) >= limit:
            break

    return items


def _fetch_items_single_folder(
    folder_id: str,
    limit: int | None = None,
    page_size: int = 100,
) -> list[EagleItem]:
    """Fetch all image items from a single Eagle folder."""
    items: list[EagleItem] = []
    cur_offset = 0

    while True:
        fetch_count = page_size
        if limit is not None:
            remaining = limit - len(items)
            if remaining <= 0:
                break
            fetch_count = min(page_size, remaining)

        try:
            result = _api_get(
                "/api/item/list",
                {"limit": fetch_count, "offset": cur_offset, "folders": folder_id},
            )
        except urllib.error.URLError as exc:
            print(f"  [ERROR] Cannot reach Eagle API for folder {folder_id}: {exc}")
            break

        if result.get("status") != "success":
            print(f"  [ERROR] Eagle API error for folder {folder_id}: {result}")
            break

        data = result.get("data", [])
        if not data:
            break

        for d in data:
            ext = (d.get("ext") or "").lower()
            if ext not in SUPPORTED_EXT_NO_DOT:
                continue
            items.append(
                EagleItem(
                    id=d["id"],
                    name=d.get("name", ""),
                    ext=ext,
                    width=d.get("width", 0),
                    height=d.get("height", 0),
                )
            )

        cur_offset += len(data)

        if len(data) < fetch_count:
            break
        if limit is not None and len(items) >= limit:
            break

    return items


def move_items_to_trash(ids: list[str], *, dry_run: bool = False) -> bool:
    """Move items to Eagle trash.  Returns True on success."""
    if not ids:
        return True
    if dry_run:
        preview = ids[:5]
        suffix = "..." if len(ids) > 5 else ""
        print(f"    [DRY-RUN] Would move {len(ids)} items to trash: {preview}{suffix}")
        return True
    try:
        resp = _api_post("/api/item/moveToTrash", {"ids": ids})
        ok = resp.get("status") == "success"
        if not ok:
            print(f"    [WARN] moveToTrash returned: {resp}")
        return ok
    except Exception as exc:
        print(f"    [ERROR] moveToTrash failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Image path resolution
# ---------------------------------------------------------------------------
def resolve_image_path(item: EagleItem) -> Path | None:
    """Resolve the filesystem path for an Eagle item's original image."""
    if not item.id:
        return None

    info_dir = EAGLE_LIBRARY_PATH / "images" / f"{item.id}.info"
    if not info_dir.is_dir():
        return None

    # Try the expected name.ext first
    if item.name and item.ext:
        expected = info_dir / f"{item.name}.{item.ext}"
        if expected.is_file():
            return expected

    # Fallback: scan info dir for any supported image file (skip thumbnails)
    for child in sorted(info_dir.iterdir()):
        if not child.is_file():
            continue
        if child.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if "_thumbnail" in child.stem or "_thumb" in child.stem:
            continue
        return child

    return None


# ---------------------------------------------------------------------------
# Centre watermark detection
# ---------------------------------------------------------------------------
def detect_center_watermark(
    image: np.ndarray,
    category: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Detect watermarks in the centre region of an image.

    Returns (center_mask, center_boxes) where *center_mask* is a full-image-
    size uint8 mask and *center_boxes* contains bounding-box dicts with
    absolute pixel coordinates.
    """
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    boxes: list[dict[str, Any]] = []

    # Choose spec based on orientation; cover_heavy is portrait-like.
    if category in ("portrait_regular", "cover_heavy"):
        spec = PORTRAIT_CENTER_SPEC
    else:
        spec = LANDSCAPE_CENTER_SPEC

    x1, y1, x2, y2 = _resolve_roi(spec, width, height)
    roi_bgr = image[y1:y2, x1:x2]
    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    local_mask = _detect_text_mask(roi_gray, roi_bgr, spec.threshold)
    roi_boxes = _mask_to_boxes(local_mask)

    if roi_boxes:
        merged = _merge_boxes(roi_boxes)
        for box in merged:
            expanded = _expand_box(
                box,
                roi_gray.shape[1],
                roi_gray.shape[0],
                margin_x=10,
                margin_y=6,
            )
            ax1 = x1 + expanded[0]
            ay1 = y1 + expanded[1]
            ax2 = x1 + expanded[2]
            ay2 = y1 + expanded[3]
            mask[ay1:ay2, ax1:ax2] = 255
            boxes.append(
                {"region": "center", "x1": ax1, "y1": ay1, "x2": ax2, "y2": ay2}
            )

    return mask, boxes


# ---------------------------------------------------------------------------
# Action determination
# ---------------------------------------------------------------------------
def is_box_centered(box: dict[str, Any], width: int, height: int) -> bool:
    """True when the bounding-box centre falls within the centre 40% of the image."""
    cx = (box["x1"] + box["x2"]) / 2.0
    cy = (box["y1"] + box["y2"]) / 2.0
    return (
        cx >= width * CENTER_40_LO
        and cx <= width * CENTER_40_HI
        and cy >= height * CENTER_40_LO
        and cy <= height * CENTER_40_HI
    )


def determine_action(
    corner_mask: np.ndarray,
    corner_detection: DetectionResult,
    center_mask: np.ndarray,
    center_boxes: list[dict[str, Any]],
    width: int,
    height: int,
) -> tuple[str, str, float, bool]:
    """Decide what to do with an image based on detection results.

    Returns (action, reason, coverage_ratio, is_centered).
    """
    total_pixels = width * height

    # Combine corner + centre masks
    combined_mask = cv2.bitwise_or(corner_mask, center_mask)
    combined_nonzero = int(np.count_nonzero(combined_mask))
    coverage_ratio = combined_nonzero / float(max(1, total_pixels))

    all_boxes = list(corner_detection.boxes) + center_boxes

    # 1. Any box whose centre is in the centre 40% → DELETE
    for box in all_boxes:
        if is_box_centered(box, width, height):
            bcx = (box["x1"] + box["x2"]) // 2
            bcy = (box["y1"] + box["y2"]) // 2
            return (
                "delete",
                f"centered watermark (box centre ({bcx},{bcy}) in centre 40%)",
                coverage_ratio,
                True,
            )

    # 2. Coverage > 50% → DELETE
    if coverage_ratio > COVERAGE_DELETE_THRESHOLD:
        return (
            "delete",
            f"coverage {coverage_ratio:.1%} > {COVERAGE_DELETE_THRESHOLD:.0%}",
            coverage_ratio,
            False,
        )

    # 3. Centre-region watermark detected but not "centred" by 40% rule,
    #    and no corner watermark → still DELETE (can't crop centre content)
    if center_boxes and corner_detection.mask_nonzero == 0:
        return (
            "delete",
            f"centre-region watermark not croppable, coverage {coverage_ratio:.1%}",
            coverage_ratio,
            True,
        )

    # 4. Corner/edge watermark with acceptable coverage → CROP
    if (
        corner_detection.mask_nonzero > 0
        and coverage_ratio <= COVERAGE_DELETE_THRESHOLD
    ):
        return (
            "crop",
            f"corner watermark, coverage {coverage_ratio:.1%}",
            coverage_ratio,
            False,
        )

    # 5. Nothing detected
    return ("skip", "no watermark detected", 0.0, False)


# ---------------------------------------------------------------------------
# Per-item processing
# ---------------------------------------------------------------------------
def process_single_image(
    image_path: Path,
    item: EagleItem,
    *,
    dry_run: bool = False,
) -> WatermarkDecision:
    """Detect watermarks on a single Eagle item and determine the action."""
    t0 = time.monotonic()
    result = WatermarkDecision(
        item_id=item.id,
        item_name=item.name,
        action="skip",
        reason="unknown",
    )

    try:
        # Read image
        image = read_image(image_path)
        height, width = image.shape[:2]

        # Classify orientation using no-watermar's classifier
        category = classify_image(image_path, image)

        # Build ScanItem for the existing detect_watermarks() call
        scan_item = ScanItem(
            source_path=image_path,
            relative_path=Path(image_path.name),
            width=width,
            height=height,
            category=category,
        )

        # --- Corner detection (existing framework) ---
        corner_mask, corner_detection = detect_watermarks(scan_item, image)

        # --- Centre detection (new) ---
        center_mask, center_boxes = detect_center_watermark(image, category)

        # --- Decide action ---
        action, reason, coverage, is_centered = determine_action(
            corner_mask,
            corner_detection,
            center_mask,
            center_boxes,
            width,
            height,
        )

        result.action = action
        result.reason = reason
        result.coverage_ratio = coverage
        result.is_centered = is_centered

        # --- Execute action ---
        if action == "crop" and not dry_run:
            crop_result = crop_image_to_remove_corner_watermark(
                image,
                corner_mask,
                boxes=corner_detection.boxes,
            )
            cropped = crop_result["restored"]
            write_image(image_path, cropped)
            result.crop_bounds = crop_result["meta"].get("crop_bounds")

    except Exception as exc:
        result.action = "error"
        result.reason = f"error: {exc}"
        result.error = str(exc)

    result.elapsed_ms = (time.monotonic() - t0) * 1000
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Eagle batch watermark detection, cropping, and deletion",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without executing deletions or crops",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images processed",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Start offset in item list (skip first N items)",
    )
    parser.add_argument(
        "--folder-ids",
        type=str,
        default=None,
        help="Comma-separated Eagle folder IDs to filter (e.g. 'MJNWUU6IX4RC8,MJNWV1U5HF4EI')",
    )
    args = parser.parse_args()

    dry_run: bool = args.dry_run
    limit: int | None = args.limit
    offset: int = args.offset
    folder_ids: list[str] | None = None
    if args.folder_ids:
        folder_ids = [fid.strip() for fid in args.folder_ids.split(",") if fid.strip()]

    print("=" * 60)
    print("Eagle Batch Watermark Processor")
    if dry_run:
        print("  [DRY RUN - no changes will be made]")
    print("=" * 60)
    print(f"  dry_run    = {dry_run}")
    print(f"  limit      = {limit}")
    print(f"  offset     = {offset}")
    print(f"  folder_ids = {folder_ids or '(all)'}")
    print()

    # ------------------------------------------------------------------
    # 1. Fetch items from Eagle API
    # ------------------------------------------------------------------
    print("[1/4] Fetching items from Eagle API ...")
    try:
        items = fetch_all_items(limit=limit, offset=offset, folder_ids=folder_ids)
    except Exception as exc:
        print(f"  [FATAL] Cannot connect to Eagle API at {EAGLE_API_BASE}: {exc}")
        print("  Make sure Eagle is running.")
        return 1

    print(f"  Fetched {len(items)} image items")
    if not items:
        print("  No items to process. Exiting.")
        return 0

    # ------------------------------------------------------------------
    # 2-3. Detect watermarks and determine actions
    # ------------------------------------------------------------------
    print(f"\n[2/4] Detecting watermarks on {len(items)} images ...")
    decisions: list[WatermarkDecision] = []
    to_delete_ids: list[str] = []
    counts: dict[str, int] = {"delete": 0, "crop": 0, "skip": 0, "error": 0}
    total = len(items)

    t0 = time.time()

    for idx, item in enumerate(items, 1):
        prefix = f"  [{idx}/{total}]"
        print(f"{prefix} {item.name}.{item.ext} ({item.id}) ...", end=" ", flush=True)

        # Resolve image path on disk
        image_path = resolve_image_path(item)
        if image_path is None:
            print("SKIP | file not found on disk")
            decisions.append(
                WatermarkDecision(
                    item_id=item.id,
                    item_name=item.name,
                    action="skip",
                    reason="file not found on disk",
                )
            )
            counts["skip"] += 1
            continue

        # Process the image
        decision = process_single_image(image_path, item, dry_run=dry_run)
        decisions.append(decision)

        # Accumulate delete IDs for batched API call
        if decision.action == "delete" and not dry_run:
            to_delete_ids.append(item.id)

        # Count
        if decision.action == "error":
            counts["error"] += 1
        else:
            counts[decision.action] = counts.get(decision.action, 0) + 1

        # Console feedback
        status_tag = decision.action.upper()
        if dry_run and decision.action in ("delete", "crop"):
            status_tag = f"WOULD-{status_tag}"
        print(
            f"{status_tag} | {decision.reason} | "
            f"{decision.coverage_ratio:.1%} | {decision.elapsed_ms:.0f}ms"
        )

        # Flush delete batch when buffer is full
        if len(to_delete_ids) >= DELETE_BATCH_SIZE:
            print(f"    >> Flushing delete batch ({len(to_delete_ids)} items) ...")
            move_items_to_trash(to_delete_ids, dry_run=dry_run)
            to_delete_ids.clear()

    # Flush remaining deletions
    if to_delete_ids:
        print(f"    >> Flushing final delete batch ({len(to_delete_ids)} items) ...")
        move_items_to_trash(to_delete_ids, dry_run=dry_run)

    elapsed = time.time() - t0

    # ------------------------------------------------------------------
    # 4. Summary & report
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total processed : {total}")
    print(f"  Delete          : {counts.get('delete', 0)}")
    print(f"  Crop            : {counts.get('crop', 0)}")
    print(f"  Skip            : {counts.get('skip', 0)}")
    print(f"  Error           : {counts.get('error', 0)}")
    print(f"  Elapsed         : {elapsed:.1f}s")
    if dry_run:
        print("  (dry-run mode — no actions were executed)")

    # Write JSON report
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "dry_run": dry_run,
            "limit": limit,
            "offset": offset,
            "folder_ids": folder_ids,
            "eagle_api_base": EAGLE_API_BASE,
            "eagle_library_path": str(EAGLE_LIBRARY_PATH),
        },
        "summary": {
            "total": total,
            "delete": counts.get("delete", 0),
            "crop": counts.get("crop", 0),
            "skip": counts.get("skip", 0),
            "error": counts.get("error", 0),
            "elapsed_seconds": round(elapsed, 1),
        },
        "items": [
            {
                "item_id": d.item_id,
                "item_name": d.item_name,
                "action": d.action,
                "reason": d.reason,
                "coverage_ratio": round(d.coverage_ratio, 6),
                "is_centered": d.is_centered,
                "crop_bounds": d.crop_bounds,
                "error": d.error,
                "elapsed_ms": round(d.elapsed_ms, 1),
            }
            for d in decisions
        ],
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=_json_default)
    print(f"\n  Report saved: {REPORT_PATH}")

    return 0


def _json_default(value: Any) -> Any:
    """JSON serializer for numpy types."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


if __name__ == "__main__":
    raise SystemExit(main())
