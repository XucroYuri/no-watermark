# Architecture

## Overview

The project is organized around a stable, scriptable CLI surface plus optional heavyweight providers.
The public product target is a professional operator CLI: deterministic baseline workflows are the primary supported path, while heavyweight model providers stay behind explicit support tiers.

## Core Layers

### 1. Scanning

- Recursively discover candidate image files
- Skip project and workspace directories
- Build normalized `ScanItem` records
- Emit read-only summaries or persisted scan manifests under `runtime/scans/`

### 2. Detection

- Baseline rule-based ROI detection for repeatable watermark layouts
- Benchmark-oriented mask provider abstraction for OCR and segmentation providers

### 3. Restoration

- Baseline OpenCV restoration for fast local runs
- Optional corner-crop output path for direct edge removal after watermark detection
- Restore provider abstraction for LaMa, BrushNet, PowerPaint, and future models

### 4. Benchmarking

- Dataset preparation
- Seed mask generation
- Provider evaluation
- Report generation

### 5. CLI Surface

- Root command package under `src/no_watermar/cli/`
- Grouped command modules for `scan`, `batch`, `benchmark`, `config`, and `providers`
- `no-watermar` is the primary public entrypoint
- Confirmation helper for explicit write flows under `src/no_watermar/cli/confirm.py`
- Compatibility wrappers in `run.py`, `benchmark.py`, and `src/no_watermar/benchmark_cli.py` remain as legacy shims
- The root CLI module entrypoint auto-loads the repo-local `.env` file before command dispatch
- Scan manifests under `runtime/scans/` with `latest.json` plus versioned snapshots
- Batch planning artifacts under `runtime/plans/` with `latest.json` plus versioned plan snapshots
- Planned batch execution now consumes the item list stored in the plan so scan, plan, and apply can share one stable contract
- Batch runs now persist `summary.json`, `manifest.json`, `results.jsonl`, and `latest.json` under the runs root
- Interrupted runs can be resumed from persisted run state without rescanning the input directory
- Provider descriptors now include public support metadata such as `support_tier`, validated platforms, and a recommended operator entrypoint

### 6. Sidecars

Heavy providers may run outside the main environment:

- Different Python version
- Different framework stack
- Different GPU compatibility constraints

This keeps the core repository usable even when the local model stack is not installed.

## Public Support Policy

- Stable providers define the public support matrix and drive release smoke coverage.
- Experimental providers remain discoverable through the CLI and docs, but they must degrade gracefully and stay outside the default public smoke path.
- Planned providers may appear in descriptors for roadmap visibility, but they are not part of the supported contract.

## Key Modules

- `scanner.py`
- `scan_manifest.py`
- `provider_doctor.py`
- `detector.py`
- `restorer.py`
- `pipeline.py`
- `cli/app.py`
- `cli/commands/`
- `benchmark_dataset.py`
- `benchmark_runner.py`
- `benchmark_providers.py`
- `provider_workers.py`

## Design Constraints

- Local-first
- Batch-oriented
- Provider-driven extensibility
- Graceful degradation when model environments are missing
- Generated artifacts must remain outside source control
