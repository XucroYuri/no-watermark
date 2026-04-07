# Next Work And Delivery Plan

## Purpose

This document turns the current roadmap and TODO list into an execution-oriented plan for the next project cycle.

It focuses on:

- what should happen next
- why the order matters
- what artifacts each step must leave behind
- how progress should be recorded in the repository

## Current Baseline

The repository already has these validated foundations:

- recursive batch scanning and rule-based baseline restoration
- root-CLI `scan show` and `scan run` commands with persisted scan manifests
- root-CLI `batch plan` and confirmation-aware `batch apply --plan` flows
- `batch plan --scan-manifest` support plus planned apply reuse of persisted plan items
- `providers doctor` plus root-CLI `.env` auto-loading for provider diagnosis parity
- persisted batch run artifacts plus `batch report` lookup for latest and selected runs
- `batch resume` support for continuing interrupted runs without rescanning inputs
- config-backed dataset and provider profiles for scan, batch, and benchmark workflows
- local checkout launcher wrappers under `bin/` plus compatibility shims in `run.py` and `benchmark.py`
- benchmark dataset preparation and report generation
- OCR-backed benchmarking with persistent `paddleocr` sessions
- cross-run aggregation and report comparison
- benchmark trend snapshots that merge comparison and aggregation outputs
- release smoke automation for baseline, OCR-backed, and `lama`-backed paths
- a working `lama` sidecar path based on Python `3.12` and `simple_lama_inpainting`
- a documented sidecar compatibility matrix surfaced through `providers doctor`
- an implemented experimental `diffusers_inpaint` restore provider behind the prompt/options provider contract
- an implemented experimental `powerpaint_v2_1` restore provider behind the same prompt/options provider contract
- an implemented experimental `brushnet` restore provider behind the same prompt/options provider contract
- a validated local smoke path for `powerpaint_v2_1` on Python `3.12`
- a validated local smoke path for `brushnet` on Python `3.12` by reusing the PowerPaint sidecar environment plus a repo-local BrushNet source-dir clone
- a documented 2-image real-local benchmark snapshot for `ocr_telea` versus `ocr_powerpaint_v21`, showing that the current PowerPaint setup is runnable but not yet quality-competitive
- a validated persistent `powerpaint_v2_1` restore-session path, showing that later images can reuse the loaded model even though the current 2-image mean still trails `ocr_telea`

This means the next cycle should not return to baseline scaffolding work. It should focus on tightening the CLI contracts, producing more decision-useful benchmark evidence, and expanding the provider surface without losing reproducibility.

## Workstreams

### 1. CLI Contract Completion

Goal:
Finish the root-CLI contract so discovery, planning, execution, and diagnostics can be chained without fallback to ad hoc flags.

Primary tasks:

- keep the persisted run contract stable while future features reuse it

Required artifacts:

- stable report and resume contracts under `runtime/runs/`
- updated root CLI usage in `README.md` and `docs/DEVELOPMENT.md`
- refreshed `docs/ARCHITECTURE.md` describing read-only versus write-safe command boundaries

### 2. Benchmark Quality

Goal:
Build more decision-useful benchmark outputs instead of only producing raw run artifacts.

Primary tasks:

- run larger local sample slices for `seed_manifest + telea`, `paddleocr + telea`, and `seed_manifest + lama`
- capture documented trend snapshots from compare and aggregate outputs
- add stronger quality-oriented summaries on top of the current latency and OCR residual metrics

Required artifacts:

- at least one documented benchmark comparison summary for baseline versus model-backed restore
- updated benchmark operation notes in `README.md` and `docs/DEVELOPMENT.md`
- refreshed `docs/PROGRESS.md` with the latest validated conclusions

### 3. Provider Expansion

Goal:
Move beyond the current baseline plus `lama` pair and evaluate the next high-value providers.

Primary tasks:

- integrate `EdgeSAM` as a candidate mask provider
- integrate `watermark-segmentation` as a dedicated mask provider
- review the current 2-image `ocr_powerpaint_v21` versus `ocr_telea` snapshot side by side and capture a provider decision
- review the current 2-image `brushnet` tuning shortlist through the local review bundle against `ocr_telea` and the original `ocr_brushnet` baseline
- decide whether any tuned `brushnet` variant is subjectively strong enough to justify a wider slice despite the current metric regressions
- keep the repo-local `ocr_fluxfill_fp8` profile frozen until a larger-VRAM host or a validated low-memory smoke recipe exists
- decide whether `LaMa` needs a persistent sidecar path or should remain one-shot only

Required artifacts:

- provider adapter code under `src/no_watermar/`
- sidecar or setup notes under `docs/setup/`
- runtime probe compatibility reflected in `benchmark.py probe-providers`

### 4. Reproducible Setup

Goal:
Reduce ambiguity around local model environments so another developer can reproduce the current validated stack.

Primary tasks:

- document the validated `lama` sidecar recipe more explicitly
- record sidecar compatibility expectations by Python version
- tighten setup helpers so environment validation and benchmark validation stay aligned

Required artifacts:

- updated `docs/setup/windows-local-setup.md`
- updated `docs/setup/provider-sidecars.md`
- updated `tools/setup/README.md`

### 5. Release Readiness

Goal:
Push the repository toward a realistic `0.3.0` public milestone.

Primary tasks:

- keep smoke automation aligned with the validated provider matrix
- make release validation produce stable benchmark evidence
- keep changelog, progress, and public setup docs current after each material milestone

Required artifacts:

- updated `docs/releases/release-checklist.md`
- updated `CHANGELOG.md`
- updated `docs/releases/public-release-readiness.md`

## Execution Order

The next cycle should run in this order:

1. Finish the CLI contract around scan manifests, batch reporting, and provider diagnostics.
2. Run a benchmark quality pass on larger local sample slices.
3. Document the reproducible setup for the currently validated `lama` and `paddleocr` paths.
4. Expand to the next mask provider.
5. Expand to the next model-backed restore provider.
6. Run the `0.3.0` release readiness pass.

This order keeps the repository grounded in stable local operator contracts before adding more provider surface area.

## Immediate Priorities

These are the highest-priority next tasks:

1. Compare `seed_manifest + lama` against `seed_manifest + telea` on a larger local slice and capture the summary.
2. Review the current 2-image `brushnet` tuning shortlist through the local review bundle against `ocr_telea` and the original `ocr_brushnet` baseline.
3. Decide whether any tuned `brushnet` variant is subjectively strong enough to justify a wider slice despite the current metric regressions.
4. Review the latest persistent 2-image `ocr_powerpaint_v21` versus `ocr_telea` outputs side by side and capture the decision.
5. Turn the new single-run real-local `paddleocr + telea` benchmark snapshot into a repeated-run aggregate baseline.
6. Reduce persistent `paddleocr` mask and OCR residual latency on the same real-local slice.
7. Turn the current `lama`, `diffusers_inpaint`, `powerpaint_v2_1`, and `brushnet` sidecar recipes into more reproducible documented setup paths.

## Definition Of Done For The Next Cycle

The next cycle is complete when all of the following are true:

- scan, plan, apply, and provider diagnostics can be chained entirely through the root CLI
- planned runs can be inspected and continued through dedicated batch subcommands backed by persisted run state
- the repository has one documented larger-sample comparison between baseline and model-backed restore
- the validated `lama` setup path is clearly reproducible from repository docs
- one additional mask provider has been integrated behind the benchmark provider interface
- the release smoke workflow and release checklist reflect the current provider matrix without manual caveats

## Documentation Update Rules

When one of the above workstreams changes state, update these files in the same change set:

- `README.md` for user-facing entrypoints
- `docs/PROGRESS.md` for project history
- `TODO.md` for remaining work
- `CHANGELOG.md` for release-facing change tracking

If the change affects environment setup or provider compatibility, also update:

- `docs/setup/windows-local-setup.md`
- `docs/setup/provider-sidecars.md`
- `tools/setup/README.md`

If the change affects benchmark execution or release validation, also update:

- `docs/DEVELOPMENT.md`
- `docs/releases/release-checklist.md`
- `tools/benchmark/README.md`
