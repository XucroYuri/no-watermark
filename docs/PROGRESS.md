# Progress

## Completed

### Baseline Framework

- Recursive directory scanning and image classification
- Rule-based corner watermark mask generation
- Overlay export and Telea restoration baseline
- JSON and CSV reporting for batch runs

### Benchmark Scaffold

- Benchmark dataset preparation
- Seed mask generation and manifest export
- Provider registry and provider descriptors
- Graceful handling for unavailable external providers
- Report comparison command for baseline-versus-candidate benchmarking
- OCR residual scoring in benchmark reports and comparison summaries
- OCR session mode control for benchmark runs
- Cross-run benchmark aggregation with JSON and CSV outputs
- Aggregation filters for provider pair and run-id windows
- Latency metrics for mask, restore, and OCR residual stages
- Runtime import probes for configured sidecar interpreters

### Provider Integration Surface

- Sidecar entrypoints for `paddleocr` and `lama`
- Worker contract for external model execution
- Benchmark runner support for provider combinations
- Persistent `paddleocr` sidecar protocol for benchmark reuse
- Local `paddleocr` sidecar environment bootstrapped and import-verified
- `paddleocr + telea` benchmark smoke run completed successfully on local synthetic samples
- `paddleocr + telea` benchmark now runs through a repository-local `.env` configured persistent sidecar session
- `seed_manifest + telea` versus `paddleocr + telea` comparison report generated on local synthetic samples
- OCR residual metrics verified on sequential smoke benchmarks for both baseline and OCR-backed mask runs
- `seed_manifest + telea` benchmark reports now include per-item latency metrics
- Aggregate summaries now roll up repeated benchmark runs by dataset and provider pair
- Benchmark trend snapshots now merge the latest compare output with aggregate baselines into JSON and Markdown summaries
- Release smoke script now drives list, baseline run, OCR-backed run, compare, and aggregate in one local command
- `lama` sidecar rebuilt on Python `3.12` and validated with `simple_lama_inpainting`
- Model-backed `seed_manifest + lama` smoke benchmark now runs successfully through the release smoke script
- `providers doctor` now reports sidecar interpreter versions against a documented compatibility matrix
- A dedicated real-local 20-image benchmark root now captures `seed_manifest + telea` versus `paddleocr + telea` compare, aggregate, and trend artifacts

### Repository Generalization

- Removed sample-bound public documentation
- Moved defaults to local workspace directories
- Added open-source repository scaffolding and contributor docs
- Isolated local inputs, runtime outputs, and benchmark artifacts from version control
- Added repository-local `.env` loading for CLI sidecar configuration
- Added repo-local `no-watermar.toml` support for named OCR watermark keyword presets
- Added `config validate` output for effective keyword settings and config resolution
- Added `config show` and `config init` commands for local config inspection and bootstrapping
- Added reusable example preset bundles under `docs/examples/config/`
- Added a first root-CLI skeleton with grouped `scan`, `batch`, `benchmark`, `config`, and `providers` commands
- Added local checkout launcher wrappers under `bin/` while keeping `run.py` and `benchmark.py` as compatibility shims
- Added `scan show` and persisted `scan run` manifests under `runtime/scans/`
- Added `batch plan --scan-manifest` support so plans can reuse persisted scan manifests
- Added persisted `batch plan` artifacts and `batch apply --plan` replay support under `runtime/plans/`
- Added `providers doctor` for interpreter, sidecar script, `.env`, and runtime import diagnosis
- Added persisted batch run artifacts under `runtime/runs/` with `summary.json`, `manifest.json`, `results.jsonl`, and `latest.json`
- Added `batch report` for loading the latest or a selected persisted batch run summary
- Added `batch resume` for continuing interrupted runs from persisted manifests and item-state files
- Added dataset and provider config profiles that now feed scan, batch, benchmark run, aggregate, and trend commands
- Added batch provider selection so plans and direct batch runs can reuse provider profiles and OCR session modes
- Added a local `corner_crop` restore provider so batch runs can remove detected watermark corners directly instead of repairing them in place
- Added restore prompt, negative prompt, and structured restore options to provider profiles plus persisted batch and benchmark contracts
- Added an experimental `diffusers_inpaint` restore provider plus sidecar bootstrap, doctor, and config wiring for prompt-driven diffusion inpainting
- Added an experimental `powerpaint_v2_1` restore provider plus sidecar, setup, doctor, and config wiring for the official PowerPaint v2.1 object-removal pipeline
- Added an experimental `brushnet` restore provider plus sidecar, setup, doctor, and config wiring for the upstream BrushNet SD1.5 pipeline
- Validated a local `brushnet` smoke path on Python `3.12` by reusing the repository-local PowerPaint environment plus a BrushNet source-dir import and official segmentation checkpoint
- Captured a first 1-image real-local benchmark comparison for `ocr_telea` versus `ocr_brushnet`
- Expanded `ocr_brushnet` to a 2-image real-local slice and confirmed that it still trails `ocr_telea` while looking more competitive than the current persistent `ocr_powerpaint_v21` path on mean absolute difference and restore latency
- Stabilized the local BrushNet source-dir path under a repo-local ignored `models\brushnet-source` clone instead of a temporary workspace path
- Added a reusable review-bundle helper that copies benchmark reports, compare/trend artifacts, original inputs, seed artifacts, and per-provider outputs into one local side-by-side inspection directory
- Added explicit review-bundle labels so same-provider tuning variants can be compared side by side without name collisions
- Ran a 4-variant `brushnet` tuning sweep on the same 2-image slice and confirmed that the current high-resolution and boundary-focused variants all regress the objective metrics versus the original `ocr_brushnet` baseline
- Narrowed the next human-review pass to a shortlist bundle containing `telea`, the original `brushnet` baseline, `brushnet_tuned_edges`, and `brushnet_tuned_boundary_soft`
- Added `FluxFillPipeline` support plus FP8 layerwise-casting options to the `diffusers_inpaint` restore path for a lower-memory diffusion restore attempt
- Added a local `ocr_fluxfill_fp8` provider profile that targets `black-forest-labs/FLUX.1-Fill-dev` with BF16 compute plus FP8 transformer storage
- Froze the repo-local `ocr_fluxfill_fp8` profile on the current 16 GB local GPU while keeping the underlying `FluxFillPipeline` support in `diffusers_inpaint` for future larger-VRAM or better low-memory validation
- Validated a local `powerpaint_v2_1` sidecar environment on Python `3.12` with a local `JunhaoZhuang/PowerPaint-v2-1` checkpoint plus `realisticVisionV60B1_v51VAE`
- Validated real-local 1-image and 2-image `ocr_powerpaint_v21` smoke runs on the `round2_xiaomanyao_vol420` dataset profile
- Captured a real-local 2-image benchmark comparison, aggregation, and trend snapshot for `ocr_telea` versus `ocr_powerpaint_v21`
- Verified on that 2-image slice that the current `powerpaint_v2_1` path preserves the same OCR mask as `ocr_telea` but currently regresses `mean_abs_diff`, `edge_delta`, and restore latency
- Added a persistent `powerpaint_v2_1` restore-session path with JSONL sidecar reuse plus restore session metadata in batch and benchmark summaries
- Validated a real-local 2-image persistent `ocr_powerpaint_v21` rerun, and verified that it removes the stdout protocol blocker and speeds up the second image while still failing to beat `ocr_telea` or the earlier one-shot PowerPaint mean on that slice
- Validated a real-local 2-image `ocr_diffusers` smoke path on Python `3.12` using `from_single_file`, an explicit `original_config`, and the Hugging Face mirror endpoint
- Validated a larger real-local 4-image `ocr_diffusers` batch slice on the `round2_xiaomanyao_vol420` dataset profile, with 1 cover skipped and 3 regular portraits restored
- Captured a clean 4-image benchmark snapshot root for `ocr_telea` versus `ocr_diffusers` and then expanded the same comparison to a 10-image root
- Verified on both the 4-image and 10-image benchmark slices that the current generic `diffusers_inpaint` setup keeps the same OCR mask as `ocr_telea` but still regresses `mean_abs_diff` and `edge_delta` while heavily increasing restore latency
- Fixed persistent `paddleocr` sidecar I/O so UTF-8 Windows paths can be used in real local batch runs
- Rebuilt the local `paddleocr` sidecar on Python `3.12` so `providers doctor` now reports it as `validated`
- Captured comparable real-local batch baselines for `rule_based_roi + telea` and `paddleocr + telea` across 10-image and 20-image slices
- Changed planned batch apply to consume the item list stored in the plan instead of rescanning the input root
- Changed the root CLI module entrypoint to auto-load the repo-local `.env` file like the legacy wrappers
- Changed interrupted batch runs to persist `run_status = interrupted` plus partial result state for later resume
- Changed batch run JSON persistence to use atomic file replacement, and changed `batch report` / `batch resume` loading to fall back from a truncated `summary.json` to `reports/report.json`
- Added confirmation-aware `batch apply --plan` flows with interactive confirm, `--yes`, and `--no-input`
- Added public provider support metadata so descriptors and `providers doctor` now report `support_tier`, validated platforms, and a recommended entrypoint
- Added packaging metadata for public CLI distribution, including extras for OCR, LaMa, experimental providers, and release tooling
- Added tracked repository tests plus CI movement toward an install-first validation path instead of source-path-only execution
- Added a stable-only sidecar bootstrap and validation path, with `providers doctor` now reporting release-blocking stable readiness plus optional `lama` gaps
- Tightened the release smoke wrapper so it now stops early when the release-blocking stable OCR path is not ready
- Added release build automation with a local packaging helper, CI package checks, and a tag-driven GitHub Release plus PyPI workflow
- Added `benchmark evidence` plus `capture-stable-baseline.ps1` so repeated stable benchmark runs now collapse into one release-oriented JSON and Markdown evidence bundle

## In Progress

- Converting provider hooks into reproducible model environments
- Reducing OCR scoring overhead in real model-backed runs
- Formalizing release discipline for a reusable public project
- Hardening the Windows-first public CLI support matrix while keeping heavy model providers experimental
- Finishing the reproducible public `lama` recipe on top of the new stable bootstrap path
- Preparing the repository-side trusted publishing configuration needed for the first live automated release
- Preparing the first disposable or redistributable repeated-run benchmark corpus for public release evidence
- Reviewing the current 2-image `brushnet` tuning shortlist through the new local review bundle against `ocr_telea` and the original `ocr_brushnet` baseline
- Deciding whether any tuned `brushnet` variant is subjectively strong enough to justify a wider slice despite the current metric regressions

## Next Milestones

- Ship the first reproducible OCR-backed benchmark run with persistent sidecar reuse on real local samples
- Capture the first release-blocking stable evidence bundle on a disposable or redistributable benchmark slice
- Turn the current optional stable `lama` path into a one-command reproducible recipe with compatibility notes
- Turn the new diffusion-backed 10-image snapshot into a human-reviewed provider decision
- Turn the new persistent `ocr_powerpaint_v21` 2-image snapshot into a human-reviewed provider decision before scaling it to a larger slice

## Deferred

- Interactive review UI
- Manual mask editing workflow
- Cross-image patch retrieval
- Optimized inference backends such as ONNX or TensorRT
