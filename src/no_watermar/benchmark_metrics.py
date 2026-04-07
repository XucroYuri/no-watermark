from __future__ import annotations

BASE_METRIC_KEYS = [
    "mask_nonzero",
    "mask_ratio",
    "changed_nonzero",
    "mean_abs_diff",
    "edge_delta",
]

LATENCY_METRIC_KEYS = [
    "mask_latency_ms",
    "restore_latency_ms",
    "ocr_residual_latency_ms",
]

OCR_RESIDUAL_METRIC_KEYS = [
    "ocr_residual_hits",
    "ocr_residual_score",
    "ocr_residual_max_score",
]

BENCHMARK_METRIC_KEYS = BASE_METRIC_KEYS + LATENCY_METRIC_KEYS + OCR_RESIDUAL_METRIC_KEYS
