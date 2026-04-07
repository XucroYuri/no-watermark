from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def restore_regular_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask is None or not np.any(mask):
        return image.copy()

    work_mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    restored = cv2.inpaint(image, work_mask, 5, cv2.INPAINT_TELEA)
    return restored


def crop_image_to_remove_corner_watermark(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    boxes: list[dict[str, Any]] | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if mask is None or not np.any(mask):
        return {
            "restored": image.copy(),
            "meta": {
                "operation": "corner_crop",
                "crop_applied": False,
                "result_status": "restored",
                "result_note": "mask was empty; crop was skipped",
                "original_shape": [int(image.shape[0]), int(image.shape[1])],
                "output_shape": [int(image.shape[0]), int(image.shape[1])],
                "restore_options": dict(options or {}),
            },
        }

    resolved_options = dict(options or {})
    height, width = image.shape[:2]
    source_box, source_regions = _resolve_crop_source_box(
        mask=mask,
        boxes=boxes or [],
        width=width,
        height=height,
        padding=max(0, int(resolved_options.get("crop_padding", 0))),
    )
    crop_bounds, crop_edge = _choose_crop_bounds(
        width=width,
        height=height,
        source_box=source_box,
        options=resolved_options,
    )
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_bounds
    cropped = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    removed_pixels = (height * width) - (cropped.shape[0] * cropped.shape[1])

    return {
        "restored": cropped,
        "meta": {
            "operation": "corner_crop",
            "crop_applied": True,
            "crop_edge": crop_edge,
            "crop_bounds": {
                "x1": int(crop_x1),
                "y1": int(crop_y1),
                "x2": int(crop_x2),
                "y2": int(crop_y2),
            },
            "source_box": {
                "x1": int(source_box[0]),
                "y1": int(source_box[1]),
                "x2": int(source_box[2]),
                "y2": int(source_box[3]),
            },
            "source_regions": source_regions,
            "original_shape": [int(height), int(width)],
            "output_shape": [int(cropped.shape[0]), int(cropped.shape[1])],
            "removed_pixels": int(removed_pixels),
            "removed_ratio": round(float(removed_pixels) / float(max(1, width * height)), 6),
            "result_status": "cropped",
            "result_note": f"cropped the {crop_edge} edge to remove the detected watermark corner",
            "restore_options": dict(resolved_options),
        },
    }


def _resolve_crop_source_box(
    *,
    mask: np.ndarray,
    boxes: list[dict[str, Any]],
    width: int,
    height: int,
    padding: int,
) -> tuple[tuple[int, int, int, int], list[str]]:
    valid_boxes: list[tuple[int, int, int, int]] = []
    regions: list[str] = []
    for box in boxes:
        try:
            x1 = int(box["x1"])
            y1 = int(box["y1"])
            x2 = int(box["x2"])
            y2 = int(box["y2"])
        except (KeyError, TypeError, ValueError):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        valid_boxes.append((x1, y1, x2, y2))
        region = str(box.get("region") or "").strip()
        if region and region not in regions:
            regions.append(region)

    if valid_boxes:
        x1 = max(0, min(box[0] for box in valid_boxes) - padding)
        y1 = max(0, min(box[1] for box in valid_boxes) - padding)
        x2 = min(width, max(box[2] for box in valid_boxes) + padding)
        y2 = min(height, max(box[3] for box in valid_boxes) + padding)
        return (x1, y1, x2, y2), regions

    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("corner crop requires a non-empty mask or detection boxes")

    x1 = max(0, int(xs.min()) - padding)
    y1 = max(0, int(ys.min()) - padding)
    x2 = min(width, int(xs.max()) + 1 + padding)
    y2 = min(height, int(ys.max()) + 1 + padding)
    return (x1, y1, x2, y2), regions


def _choose_crop_bounds(
    *,
    width: int,
    height: int,
    source_box: tuple[int, int, int, int],
    options: dict[str, Any],
) -> tuple[tuple[int, int, int, int], str]:
    x1, y1, x2, y2 = source_box
    edge_tolerance = int(options.get("edge_tolerance", max(8, round(min(width, height) * 0.08))))
    minimum_output_width = max(1, int(options.get("min_output_width", 1)))
    minimum_output_height = max(1, int(options.get("min_output_height", 1)))

    distances = {
        "left": x1,
        "right": max(0, width - x2),
        "top": y1,
        "bottom": max(0, height - y2),
    }

    candidate_edges: list[str] = []
    if distances["left"] <= edge_tolerance:
        candidate_edges.append("left")
    if distances["right"] <= edge_tolerance:
        candidate_edges.append("right")
    if distances["top"] <= edge_tolerance:
        candidate_edges.append("top")
    if distances["bottom"] <= edge_tolerance:
        candidate_edges.append("bottom")
    if not candidate_edges:
        nearest_edge = min(distances.items(), key=lambda item: item[1])[0]
        candidate_edges.append(nearest_edge)

    candidates: list[tuple[int, str, tuple[int, int, int, int]]] = []
    for edge in candidate_edges:
        if edge == "left":
            bounds = (x2, 0, width, height)
        elif edge == "right":
            bounds = (0, 0, x1, height)
        elif edge == "top":
            bounds = (0, y2, width, height)
        else:
            bounds = (0, 0, width, y1)

        crop_x1, crop_y1, crop_x2, crop_y2 = bounds
        kept_width = crop_x2 - crop_x1
        kept_height = crop_y2 - crop_y1
        if kept_width < minimum_output_width or kept_height < minimum_output_height:
            continue
        removed_area = (width * height) - (kept_width * kept_height)
        candidates.append((removed_area, edge, bounds))

    if not candidates:
        raise ValueError("corner crop could not produce a non-empty output for the detected watermark box")

    _, edge, bounds = min(candidates, key=lambda item: item[0])
    return bounds, edge
