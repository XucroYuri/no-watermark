from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from .models import DetectionResult, ScanItem


@dataclass(slots=True)
class RoiSpec:
    name: str
    x1: float
    y1: float
    x2: float
    y2: float
    threshold: int
    merge_all: bool
    fallback: tuple[float, float, float, float]


PORTRAIT_SPECS = (
    RoiSpec("bottom_left", 0.00, 0.94, 0.70, 1.00, 24, True, (0.00, 0.952, 0.68, 1.00)),
    RoiSpec("bottom_right", 0.77, 0.90, 1.00, 1.00, 22, True, (0.77, 0.92, 1.00, 1.00)),
)

LANDSCAPE_SPECS = (
    RoiSpec("bottom_left", 0.00, 0.905, 0.58, 1.00, 24, True, (0.00, 0.915, 0.52, 1.00)),
    RoiSpec("bottom_right", 0.76, 0.88, 1.00, 1.00, 22, True, (0.76, 0.89, 1.00, 1.00)),
)


def detect_watermarks(item: ScanItem, image: np.ndarray) -> tuple[np.ndarray, DetectionResult]:
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    boxes: list[dict[str, int | str]] = []

    if item.category not in {"portrait_regular", "landscape_regular"}:
        return mask, DetectionResult(category=item.category, mask_nonzero=0, confidence=0.0, boxes=[])

    specs = PORTRAIT_SPECS if item.category == "portrait_regular" else LANDSCAPE_SPECS
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for spec in specs:
        x1, y1, x2, y2 = _resolve_roi(spec, width, height)
        roi_gray = gray[y1:y2, x1:x2]
        roi_bgr = image[y1:y2, x1:x2]
        local_mask = _detect_text_mask(roi_gray, roi_bgr, spec.threshold)
        roi_boxes = _mask_to_boxes(local_mask)

        if roi_boxes:
            merged = _merge_boxes(roi_boxes)
            for box in merged:
                expanded = _expand_box(box, roi_gray.shape[1], roi_gray.shape[0], margin_x=12 if spec.name == "bottom_left" else 8, margin_y=6)
                ax1 = x1 + expanded[0]
                ay1 = y1 + expanded[1]
                ax2 = x1 + expanded[2]
                ay2 = y1 + expanded[3]
                mask[ay1:ay2, ax1:ax2] = 255
                boxes.append({"region": spec.name, "x1": ax1, "y1": ay1, "x2": ax2, "y2": ay2})
        else:
            fx1, fy1, fx2, fy2 = _resolve_relative_box(spec.fallback, width, height)
            mask[fy1:fy2, fx1:fx2] = 255
            boxes.append({"region": spec.name, "x1": fx1, "y1": fy1, "x2": fx2, "y2": fy2})

    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    mask_nonzero = int(np.count_nonzero(mask))
    confidence = min(1.0, mask_nonzero / float(max(1, width * height * 0.08)))

    return mask, DetectionResult(
        category=item.category,
        mask_nonzero=mask_nonzero,
        confidence=confidence,
        boxes=boxes,
    )


def build_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    overlay[mask > 0] = (0, 0, 255)
    return overlay


def _resolve_roi(spec: RoiSpec, width: int, height: int) -> tuple[int, int, int, int]:
    return _resolve_relative_box((spec.x1, spec.y1, spec.x2, spec.y2), width, height)


def _resolve_relative_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x1 = int(width * box[0])
    y1 = int(height * box[1])
    x2 = int(width * box[2])
    y2 = int(height * box[3])
    return x1, y1, x2, y2


def _detect_text_mask(gray_roi: np.ndarray, bgr_roi: np.ndarray, threshold_value: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    top_hat = cv2.morphologyEx(gray_roi, cv2.MORPH_TOPHAT, kernel)
    _, bright_small = cv2.threshold(top_hat, threshold_value, 255, cv2.THRESH_BINARY)

    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    white_pixels = (((hsv[..., 1] < 70) & (hsv[..., 2] > 150)).astype(np.uint8) * 255)

    candidate = cv2.bitwise_and(bright_small, white_pixels)
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    return candidate


def _mask_to_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    boxes: list[tuple[int, int, int, int]] = []
    pixel_budget = mask.shape[0] * mask.shape[1]

    for index in range(1, component_count):
        x, y, width, height, area = stats[index]
        if area < 8 or area > pixel_budget * 0.25:
            continue
        if width < 2 or height < 2:
            continue
        boxes.append((x, y, x + width, y + height))

    return boxes


def _merge_boxes(boxes: Iterable[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    boxes = list(boxes)
    if not boxes:
        return []

    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    return [(x1, y1, x2, y2)]


def _expand_box(
    box: tuple[int, int, int, int],
    roi_width: int,
    roi_height: int,
    margin_x: int,
    margin_y: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, x1 - margin_x),
        max(0, y1 - margin_y),
        min(roi_width, x2 + margin_x),
        min(roi_height, y2 + margin_y),
    )
