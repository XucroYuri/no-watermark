# Phase 02: Benchmark And Providers

## Goal

Turn model evaluation into a repeatable workflow.

## Deliverables

- Benchmark dataset preparation
- Seed mask generation
- Provider registry
- Graceful unavailable-provider handling
- OCR provider integration
- LaMa provider integration

## Current Status

- Benchmark CLI exists
- Baseline providers exist
- `paddleocr` and `lama` provider hooks exist
- Sidecar wiring exists
- Persistent sidecar optimization is pending

## Exit Criteria

- At least one OCR-backed mask provider benchmark
- At least one model-backed restore provider benchmark
- Comparable benchmark reports across providers
