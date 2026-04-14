# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog.

## [Unreleased]

### Added

- Added provider public-support metadata so CLI descriptors and `providers doctor` now report `support_tier`, validated platforms, and a recommended entrypoint
- Added tracked repository tests and public packaging metadata for install-first validation
- Added a `stable_setup` section to `providers doctor` so the public stable sidecar path now reports release-blocking readiness, optional stable gaps, and the next bootstrap or smoke commands
- Added a local `tools\releases\build-release.ps1` helper plus a tag-driven GitHub Release and PyPI workflow for packaging automation
- Added `benchmark evidence` and `tools\benchmark\capture-stable-baseline.ps1` so repeated stable benchmark runs now produce one release-ready evidence bundle with JSON and Markdown outputs
- Added a repo-native disposable benchmark fixture plus `tools\benchmark\capture-disposable-evidence.py`, and wired that sidecar-free evidence path into CI, release preflight, and release build artifacts
- Added a `stable-public` config template and automatic stable-config initialization during `bootstrap-sidecars.ps1 -StableOnly`
- Added `tools\releases\package-evidence.py` so disposable evidence can be archived as a release-ready zip bundle and attached to GitHub Releases
- Added bootstrap regression coverage so stable sidecar setup now keeps repository config initialization on the repo CLI interpreter and fails fast when external bootstrap commands return non-zero exit codes
- Added benchmark regression coverage so empty datasets now fail before OCR sidecars start

- Added persistent `paddleocr` sidecar session support for benchmark runs
- Added benchmark report aggregation across repeated runs
- Added aggregation filters for provider pairs and run-id windows
- Added latency metrics for mask, restore, and OCR residual stages
- Added unit coverage for OCR session fallback and report aggregation
- Added repository-local `.env` loading for CLI sidecar configuration
- Added a release smoke PowerShell wrapper for benchmark list, run, compare, and aggregate flows
- Added provider runtime import probes for configured sidecar interpreters
- Added unit coverage for restore-output size normalization in provider workers
- Added repo-local `no-watermar.toml` support for named OCR watermark keyword presets
- Added `config validate` CLI support for local configuration checks
- Added `config show` and `config init` CLI support for local configuration inspection and bootstrapping
- Added example watermark keyword preset bundles under `docs/examples/config/`
- Added a grouped root CLI skeleton for `scan`, `batch`, `benchmark`, `config`, and `providers`
- Added local checkout launcher wrappers under `bin/`
- Added `scan show` and `scan run` CLI support with persisted manifests under `runtime/scans/`
- Added `batch plan --scan-manifest` support for using persisted scan manifests as stable plan inputs
- Added persisted `batch plan` artifacts plus `batch apply --plan` replay support
- Added `providers doctor` for interpreter, sidecar script, `.env`, and runtime import diagnosis
- Added persisted batch run artifacts with `summary.json`, `manifest.json`, `results.jsonl`, and `latest.json`
- Added `batch report` for loading the latest or a selected persisted batch run summary
- Added `batch resume` for continuing interrupted runs from persisted run state
- Added confirmation-aware `batch apply --plan` support with `--yes` and `--no-input`
- Added `benchmark trends` for JSON and Markdown snapshots that merge the latest comparison output with aggregate baselines
- Added config-backed dataset and provider profiles for scan, batch, benchmark run, aggregate, and trend flows
- Added sidecar compatibility reporting in `providers doctor` plus per-provider Python overrides in the setup bootstrap script
- Added batch provider selection via provider profiles and explicit provider flags for plan/apply flows
- Added UTF-8 sidecar process defaults for `paddleocr` persistent sessions on Windows
- Added real-local batch baseline captures for `rule_based_roi + telea` versus `paddleocr + telea` using provider profiles
- Added a dedicated real-local benchmark snapshot for `seed_manifest + telea` versus `paddleocr + telea` with compare, aggregate, and trend outputs
- Added restore prompt, negative prompt, and structured restore options to provider profiles, batch plans, batch runs, and benchmark summaries
- Added a local `corner_crop` restore provider and provider-profile path for direct corner cropping after watermark detection
- Added an experimental `diffusers_inpaint` restore provider with sidecar setup, provider doctor coverage, and prompt-driven model option support
- Added single-file checkpoint loading support for `diffusers_inpaint`, including `pipeline_class` and `original_config` options for local or mirrored checkpoints
- Added an experimental `powerpaint_v2_1` restore provider with PowerPaint v2.1 object-removal sidecar support, setup helpers, provider doctor coverage, and config wiring
- Added an experimental `brushnet` restore provider with sidecar support, provider doctor coverage, and config wiring for the upstream BrushNet SD1.5 pipeline
- Added a validated local `brushnet` smoke path on Python `3.12` by reusing the repository-local PowerPaint environment plus a BrushNet source-dir import
- Added a 2-image real-local `ocr_brushnet` comparison against `ocr_telea` and persistent `ocr_powerpaint_v21`
- Added a validated local Python `3.12` smoke path for `powerpaint_v2_1`, including a local `JunhaoZhuang/PowerPaint-v2-1` checkpoint and private provider profile wiring
- Added a dedicated real-local 2-image benchmark snapshot for `ocr_telea` versus `ocr_powerpaint_v21` with compare, aggregate, and trend artifacts
- Added a persistent `powerpaint_v2_1` restore-session path with JSONL sidecar reuse plus restore session metadata in batch and benchmark summaries
- Added fallback loading for `batch report` and `batch resume` so a truncated `summary.json` can recover from `reports/report.json`
- Added dedicated real-local `slice4` and `slice10` benchmark roots for comparing `ocr_telea` against `ocr_diffusers`
- Added a reusable `build-review-bundle.py` helper plus `review_bundle` module for assembling side-by-side human review bundles from benchmark reports and copied artifacts
- Added explicit `--label` support to `build-review-bundle.py` so same-provider tuning variants can be compared without name collisions
- Added `FluxFillPipeline` support plus FP8 layerwise-casting options to the `diffusers_inpaint` restore path
- Added a local `ocr_fluxfill_fp8` profile for the first low-memory diffusion restore attempt

### Changed

- Changed CI to validate the editable package and CLI entrypoint across Windows and Linux runners
- Changed the public docs to center the professional CLI path and to separate stable providers from experimental restore providers
- Changed setup helpers and release smoke docs to start from a stable-only `paddleocr` plus optional `lama` bootstrap and validation path
- Changed the release smoke PowerShell wrapper to fail fast when the release-blocking stable OCR path is not ready
- Changed CI to run package build and `twine check` validation before the release workflow can publish artifacts
- Changed the release checklist and process docs to treat the stable evidence bundle as the default benchmark proof for release decisions
- Changed the local release helper to capture disposable stable evidence automatically before packaging
- Changed `run_benchmark` to reject empty prepared datasets before any provider sidecars are created

- Changed planned batch apply to consume the item list stored in the plan instead of rescanning the input root
- Changed the root CLI module entrypoint to auto-load the repo-local `.env` file like the legacy wrappers
- Changed interrupted batch runs to persist `run_status = interrupted` and partial result state for resume
- Changed batch runs and plans to persist the selected mask provider, restore provider, and OCR session mode
- Changed the `paddleocr` persistent sidecar protocol to force UTF-8 stdio for non-ASCII local paths
- Changed benchmark runs to reuse a shared OCR execution context across mask generation and OCR residual scoring
- Changed benchmark comparison summaries to include latency metrics alongside mask and residual metrics
- Expanded `list-providers` output with execution modes, required env vars, and runtime availability notes
- Changed release smoke flow to gate model-backed smoke coverage on real provider importability instead of interpreter existence alone
- Changed the `lama` integration to use the importable `simple_lama_inpainting` package path with output cropping back to the original image size
- Changed the local `lama` sidecar environment to a Python `3.12` stack that now supports model-backed smoke benchmarks
- Changed the local `paddleocr` sidecar environment from Python `3.13` to Python `3.12`, which now matches the documented validated set in `providers doctor`
- Changed the local `diffusers_inpaint` sidecar path to a validated Python `3.12` smoke setup using mirror-backed single-file Stable Diffusion inpainting weights
- Changed batch run JSON persistence to use atomic file replacement so interrupted writes no longer leave zero-byte state files as easily
- Changed the immediate restore-provider expansion path so `powerpaint_v2_1` is now the next implemented experimental diffusion restore target instead of a backlog-only option
- Changed `providers doctor` and setup docs so `powerpaint_v2_1` now records the first repository-local validated Python and package matrix instead of remaining completely unvalidated
- Changed the immediate next restore-provider milestone so the first `brushnet` smoke setup and comparison is now queued ahead of any larger `powerpaint_v2_1` slice expansion
- Changed `providers doctor` and setup docs so `brushnet` now records the first repository-local validated Python and package matrix instead of remaining purely experimental
- Changed the immediate BrushNet follow-up from runtime bring-up to human review and tuning decisions on the current 2-image slice
- Changed the immediate diffusion restore conclusion so generic `diffusers_inpaint` remains experimental and currently trails `ocr_telea` on the validated 4-image and 10-image local slices
- Changed the immediate `powerpaint_v2_1` evaluation path from sample-size expansion to human review plus restore-session reuse first
- Changed the immediate `powerpaint_v2_1` restore-session conclusion so persistent reuse is now working even though the current 2-image mean still trails both `ocr_telea` and the earlier one-shot PowerPaint slice
- Changed the local BrushNet smoke path to use a stable repo-local `models\brushnet-source` clone instead of a temporary source checkout
- Changed the immediate `brushnet` tuning status from open-ended prompt experimentation to a shortlist human-review pass after four tuned variants regressed the objective metrics versus the original `ocr_brushnet` baseline
- Froze the repo-local `ocr_fluxfill_fp8` profile on the current 16 GB local GPU while keeping the underlying `FluxFillPipeline` support in `diffusers_inpaint` for future re-evaluation on a larger-VRAM or better-validated low-memory path

## [0.2.0] - 2026-04-04

### Added

- Reorganized the repository as a generic open-source project
- Added project-level documentation, contribution guidance, roadmap, and TODO tracking
- Added benchmark data models, provider interfaces, runner, and CLI
- Added sidecar hooks for `paddleocr` and `lama`
- Added CI-ready repository scaffolding and Git metadata files

### Changed

- Removed sample-bound wording from the main project documentation
- Changed default CLI input roots from the parent sample folder to `./inputs`
- Replaced sample-specific OCR keyword defaults with generic watermark keyword heuristics plus env overrides
- Increased run IDs to microsecond precision to avoid collisions

### Removed

- Removed sample-specific planning documents from the public project surface
- Removed generated sample benchmark and runtime artifacts from the repository state

## [0.1.0] - 2026-04-04

### Added

- Initial MVP for recursive scanning, rule-based mask generation, overlay export, and Telea restoration
- Initial benchmark scaffold
- Initial provider placeholders for OCR and restoration models
