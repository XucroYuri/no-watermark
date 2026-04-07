from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from .benchmark_models import BenchmarkDatasetItem
from .detector import build_overlay, detect_watermarks
from .io_utils import ensure_dir, read_image, write_image, write_json
from .scanner import build_scan_items

DATASET_REGULAR = "regular_corner_text"
DATASET_COVER = "cover_heavy"


def prepare_benchmark_dataset(
    input_root: Path,
    benchmark_root: Path,
    *,
    recursive: bool = True,
    limit: int | None = None,
) -> dict[str, Any]:
    input_root = input_root.resolve()
    benchmark_root = benchmark_root.resolve()
    datasets_root = ensure_dir(benchmark_root / "datasets")
    manifests_root = ensure_dir(benchmark_root / "manifests")

    items = build_scan_items(input_root, recursive=recursive, limit=limit)
    counter = Counter()
    grouped: dict[str, list[BenchmarkDatasetItem]] = {
        DATASET_REGULAR: [],
        DATASET_COVER: [],
    }

    for item in items:
        dataset_id = _map_source_category(item.category)
        counter[dataset_id] += 1
        dataset_root = ensure_dir(datasets_root / dataset_id)
        inputs_root = ensure_dir(dataset_root / "inputs")
        masks_root = ensure_dir(dataset_root / "masks_seed")
        overlays_root = ensure_dir(dataset_root / "overlays_seed")
        prompts_root = ensure_dir(dataset_root / "prompts")

        input_path = inputs_root / item.relative_path
        input_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item.source_path, input_path)

        prompt_path = prompts_root / item.relative_path.with_suffix(".txt")
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt = _build_prompt(item.category)
        prompt_path.write_text(prompt + "\n", encoding="utf-8")

        item_id = _build_item_id(dataset_id, counter[dataset_id])
        seed_mask_path: Path | None = None
        seed_overlay_path: Path | None = None
        notes: dict[str, Any]

        if dataset_id == DATASET_REGULAR:
            image = read_image(item.source_path)
            mask, detection = detect_watermarks(item, image)
            seed_mask_path = masks_root / item.relative_path.with_suffix(".png")
            seed_overlay_path = overlays_root / item.relative_path.with_suffix(".jpg")
            write_image(seed_mask_path, mask)
            write_image(seed_overlay_path, build_overlay(image, mask))
            notes = {
                "seed_mask_nonzero": detection.mask_nonzero,
                "seed_confidence": round(detection.confidence, 4),
            }
        else:
            notes = {"seed_mask_nonzero": 0, "seed_confidence": 0.0}

        grouped[dataset_id].append(
            BenchmarkDatasetItem(
                item_id=item_id,
                dataset_id=dataset_id,
                source_path=item.source_path,
                input_path=input_path,
                relative_path=item.relative_path,
                width=item.width,
                height=item.height,
                source_category=item.category,
                benchmark_category=dataset_id,
                prompt=prompt,
                prompt_path=prompt_path,
                seed_mask_path=seed_mask_path,
                seed_overlay_path=seed_overlay_path,
                notes=notes,
            )
        )

    datasets_summary = []
    for dataset_id, dataset_items in grouped.items():
        dataset_root = ensure_dir(datasets_root / dataset_id)
        manifest_path = dataset_root / "manifest.json"
        notes_path = dataset_root / "notes.csv"
        write_json(
            manifest_path,
            {
                "dataset_id": dataset_id,
                "item_count": len(dataset_items),
                "items": [dataset_item.to_dict() for dataset_item in dataset_items],
            },
        )
        _write_dataset_notes(notes_path, dataset_items)
        csv_name = "dataset_regular.csv" if dataset_id == DATASET_REGULAR else "dataset_cover.csv"
        _write_dataset_manifest_csv(manifests_root / csv_name, dataset_items)
        datasets_summary.append(
            {
                "dataset_id": dataset_id,
                "item_count": len(dataset_items),
                "manifest_path": str(manifest_path),
            }
        )

    summary = {
        "input_root": str(input_root),
        "benchmark_root": str(benchmark_root),
        "recursive": recursive,
        "limit": limit,
        "image_count": len(items),
        "dataset_counts": dict(counter),
        "datasets": datasets_summary,
    }
    write_json(manifests_root / "dataset_index.json", summary)
    return summary


def load_dataset_items(benchmark_root: Path, dataset_id: str) -> list[BenchmarkDatasetItem]:
    benchmark_root = benchmark_root.resolve()
    manifest_path = benchmark_root / "datasets" / dataset_id / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [BenchmarkDatasetItem.from_dict(item) for item in data.get("items", [])]


def _map_source_category(category: str) -> str:
    if category == "cover_heavy":
        return DATASET_COVER
    return DATASET_REGULAR


def _build_item_id(dataset_id: str, index: int) -> str:
    prefix = "regular" if dataset_id == DATASET_REGULAR else "cover"
    return f"{prefix}-{index:04d}"


def _build_prompt(source_category: str) -> str:
    if source_category == "cover_heavy":
        return "Remove the heavy cover watermark text and reconstruct the covered image content naturally."
    return "Remove the corner text watermark and reconstruct the local background naturally without changing the subject."


def _write_dataset_manifest_csv(path: Path, items: list[BenchmarkDatasetItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "item_id",
                "dataset_id",
                "source_path",
                "input_path",
                "relative_path",
                "source_category",
                "benchmark_category",
                "width",
                "height",
                "prompt",
                "prompt_path",
                "seed_mask_path",
                "seed_overlay_path",
            ],
        )
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "item_id": item.item_id,
                    "dataset_id": item.dataset_id,
                    "source_path": str(item.source_path),
                    "input_path": str(item.input_path),
                    "relative_path": str(item.relative_path),
                    "source_category": item.source_category,
                    "benchmark_category": item.benchmark_category,
                    "width": item.width,
                    "height": item.height,
                    "prompt": item.prompt,
                    "prompt_path": str(item.prompt_path),
                    "seed_mask_path": str(item.seed_mask_path) if item.seed_mask_path else "",
                    "seed_overlay_path": str(item.seed_overlay_path) if item.seed_overlay_path else "",
                }
            )


def _write_dataset_notes(path: Path, items: list[BenchmarkDatasetItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "item_id",
                "relative_path",
                "source_category",
                "prompt",
                "seed_mask_nonzero",
                "seed_confidence",
                "manual_score",
                "comment",
            ],
        )
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "item_id": item.item_id,
                    "relative_path": str(item.relative_path),
                    "source_category": item.source_category,
                    "prompt": item.prompt,
                    "seed_mask_nonzero": item.notes.get("seed_mask_nonzero", 0),
                    "seed_confidence": item.notes.get("seed_confidence", 0.0),
                    "manual_score": "",
                    "comment": "",
                }
            )
