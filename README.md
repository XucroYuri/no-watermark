# no-watermar

`no-watermar` is a professional local-first CLI for batch watermark removal, restoration benchmarking, and repeatable review workflows over image sets that share reusable watermark layouts or corner text marks.

The repository is organized as a reusable framework, not as a single-case sample project. Private inputs, generated outputs, benchmark artifacts, and local model environments are intentionally kept out of version control.

## Status

- Current maturity: `0.3.0`
- Focus: public CLI delivery, stable evidence automation, release automation
- Default stack: Python, OpenCV, optional OCR / inpainting sidecars

## What It Does

- Recursively scan image directories
- Classify likely image layout categories
- Detect repeatable watermark regions
- Generate masks and overlays
- Run batch restoration baselines
- Support two no-watermark output modes: in-place repair or direct corner cropping
- Prepare benchmark datasets and benchmark reports
- Provide a provider abstraction for OCR, segmentation, and restoration models

## Public Support Matrix

The repository now distinguishes between stable and experimental provider paths.

- Release-blocking stable smoke path: `rule_based_roi`, `paddleocr`, and `telea`
- Stable optional extensions: `corner_crop` and `lama`
- Stable public platforms: Windows-first; Linux baseline for the lightweight CLI path
- Experimental providers: `diffusers_inpaint`, `powerpaint_v2_1`, and `brushnet`
- Planned providers: `edgesam` and `watermark_segmentation`

Use `.\bin\no-watermar.ps1 providers list` or `.\bin\no-watermar.ps1 providers doctor` to inspect each provider's `support_tier`, validated platforms, and recommended entrypoint. `providers doctor` now also reports `stable_setup`, which tells you whether the release-blocking stable sidecars are ready and which command to run next.

## Repository Layout

```text
bin/                 Local checkout launcher wrappers
src/no_watermar/     Core package
tests/               Unit tests
docs/                Project documentation
tools/sidecars/      Optional sidecar entrypoints for external model environments
inputs/              Local input images (not committed)
runtime/             Batch processing outputs (not committed)
benchmarks/          Benchmark datasets and reports (not committed)
```

## Quick Start

Install from a local checkout:

```powershell
python -m pip install .
```

For contributor workflows, install the editable package plus release tooling:

```powershell
python -m pip install -e .[dev]
powershell -ExecutionPolicy Bypass -File .\tools\releases\build-release.ps1 -CleanDist
```

Optional sidecar extras now map to the public support matrix:

```powershell
python -m pip install .[ocr]
python -m pip install .[lama]
python -m pip install .[experimental]
```

Bootstrap the public stable sidecar path on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\setup\bootstrap-sidecars.ps1 -StableOnly -InstallPackages
powershell -ExecutionPolicy Bypass -File .\tools\setup\validate-sidecars.ps1 -StableOnly -RunDoctor
```

The stable bootstrap path treats `paddleocr` as release-blocking for public smoke runs. `lama` stays on the same stable support track, but it remains optional unless you want the model-backed stable restore path as part of local validation. When no repo-local `no-watermar.toml` exists yet, the stable bootstrap also initializes it from the built-in `stable-public` template so `local_smoke`, `seed_telea`, `ocr_telea`, `lama_eval`, and `ocr_corner_crop` are available immediately.

Run the batch baseline against `.\inputs\`:

```powershell
.\bin\no-watermar.ps1 config init --template brand-social
.\bin\no-watermar.ps1 config show
.\bin\no-watermar.ps1 config validate
.\bin\no-watermar.ps1 scan show --input .\inputs
.\bin\no-watermar.ps1 scan show --dataset-profile local_smoke
.\bin\no-watermar.ps1 scan run --input .\inputs --scans-root .\runtime\scans
.\bin\no-watermar.ps1 batch plan --scan-manifest .\runtime\scans\latest.json --output .\runtime\runs
.\bin\no-watermar.ps1 batch plan --input .\inputs --output .\runtime\runs
.\bin\no-watermar.ps1 batch plan --dataset-profile local_smoke --output .\runtime\runs
.\bin\no-watermar.ps1 batch plan --dataset-profile local_smoke --provider-profile rule_telea --output .\runtime\runs
.\bin\no-watermar.ps1 batch apply --plan .\runtime\plans\latest.json --yes
.\bin\no-watermar.ps1 batch report --runs-root .\runtime\runs
.\bin\no-watermar.ps1 batch resume --runs-root .\runtime\runs
.\bin\no-watermar.ps1 batch apply --input .\inputs --output .\runtime\runs
.\bin\no-watermar.ps1 batch apply --dataset-profile local_smoke --provider-profile ocr_telea --output .\runtime\runs
.\bin\no-watermar.ps1 batch apply --dataset-profile local_smoke --provider-profile ocr_corner_crop --output .\runtime\runs
.\bin\no-watermar.ps1 batch apply --scan-only
```

Run the benchmark workflow:

```powershell
.\bin\no-watermar.ps1 providers list
.\bin\no-watermar.ps1 providers doctor
.\bin\no-watermar.ps1 providers probe
.\bin\no-watermar.ps1 benchmark prepare --input .\inputs
.\bin\no-watermar.ps1 benchmark prepare --dataset-profile local_smoke --benchmark-root .\benchmarks
.\bin\no-watermar.ps1 benchmark run --dataset-profile local_smoke --provider-profile seed_telea --benchmark-root .\benchmarks
.\bin\no-watermar.ps1 benchmark run --dataset regular_corner_text --mask-provider seed_manifest --restore-provider telea --ocr-session-mode auto
.\bin\no-watermar.ps1 benchmark compare --baseline-report .\benchmarks\path\baseline.json --candidate-report .\benchmarks\path\candidate.json
.\bin\no-watermar.ps1 benchmark aggregate --dataset-profile local_smoke --provider-profile seed_telea --reports-root .\benchmarks\runs
.\bin\no-watermar.ps1 benchmark trends --dataset-profile local_smoke --baseline-provider-profile seed_telea --candidate-provider-profile ocr_telea --benchmark-root .\benchmarks
.\bin\no-watermar.ps1 benchmark evidence --dataset-profile local_smoke --baseline-provider-profile seed_telea --candidate-provider-profile ocr_telea --optional-provider-profile lama_eval --benchmark-root .\benchmarks --minimum-run-count 3
.\bin\no-watermar.ps1 benchmark aggregate --reports-root .\benchmarks\runs --dataset regular_corner_text --mask-provider seed_manifest --restore-provider telea
.\bin\no-watermar.ps1 benchmark trends --dataset regular_corner_text --baseline-mask-provider seed_manifest --baseline-restore-provider telea --candidate-mask-provider paddleocr --candidate-restore-provider telea
powershell -ExecutionPolicy Bypass -File .\tools\benchmark\run-release-smoke.ps1 -Limit 1
powershell -ExecutionPolicy Bypass -File .\tools\benchmark\run-release-smoke.ps1 -Limit 1 -RequireLama
powershell -ExecutionPolicy Bypass -File .\tools\benchmark\capture-stable-baseline.ps1 -Repetitions 3
python .\tools\benchmark\capture-disposable-evidence.py --clean
```

`benchmark trends` reads the latest matching comparison and aggregation outputs, then writes a JSON and Markdown snapshot under `.\benchmarks\trends\`.
`benchmark evidence` turns repeated stable runs into one release-oriented JSON and Markdown bundle under `.\benchmarks\evidence\`, with `latest.json` and `latest.md` kept up to date for release review.
`capture-disposable-evidence.py` builds the repo-native redistributable synthetic corpus and a sidecar-free repeated evidence bundle under `.\runtime\disposable-evidence\benchmarks\evidence\`.
Dataset and provider profiles can live in `no-watermar.toml`; see [docs/CONFIGURATION.md](./docs/CONFIGURATION.md) and [docs/examples/config/benchmark-local-profiles.toml](./docs/examples/config/benchmark-local-profiles.toml).
Provider profiles can also carry `restore_prompt`, `restore_negative_prompt`, and structured `restore_options` for restore providers, including model-backed inpainting and direct corner cropping.
The current no-watermark flow now supports two result choices: repair the detected watermark region in place, or use the local `corner_crop` restore provider to crop away the watermark-bearing corner directly.
The first prompt-driven restore adapter now ships as `diffusers_inpaint`; it remains experimental and is not part of the default public support matrix.
The current local validated smoke path uses a Python `3.12` sidecar plus single-file Stable Diffusion inpainting weights loaded through `from_single_file`.
The current real-local 4-image and 10-image benchmark snapshots against `ocr_telea` show that this generic `diffusers_inpaint` configuration preserves the same OCR mask but still regresses `mean_abs_diff` and `edge_delta`, so it should be treated as an exploratory baseline rather than the current recommended restore path.
The repository still retains `FluxFillPipeline` support plus BF16 and FP8 layerwise-casting options for future low-memory diffusion experiments, but the repo-local `ocr_fluxfill_fp8` profile is currently frozen on the 16 GB local GPU because it has no validated smoke path and already depends on aggressive memory-saving and offload settings just to attempt bring-up.
The next restore path now also includes an experimental `powerpaint_v2_1` adapter that consumes the same prompt and `restore_options` contract while switching the restore engine to the official PowerPaint v2.1 object-removal pipeline.
The current local `powerpaint_v2_1` smoke path is now validated on Python `3.12` with the `JunhaoZhuang/PowerPaint-v2-1` checkpoint in local-only mode, and it has successfully restored 1-image and 2-image real-local `regular_corner_text` slices.
The first persistent `powerpaint_v2_1` restore-session pass is now also validated on a 2-image real-local slice, but it only reduced the second-image restore latency and did not improve the 2-image mean restore latency or the current quality regression against `ocr_telea`.
The next experimental restore adapter now also includes `brushnet`, wired behind the same prompt and `restore_options` contract while targeting the official BrushNet SD1.5 pipeline.
The first local `brushnet` smoke path is now also validated on Python `3.12` by reusing the repository-local PowerPaint environment together with a repo-local `NO_WATERMAR_BRUSHNET_SOURCE_DIR` clone under `.\models\brushnet-source`.
The first 2-image real-local `ocr_brushnet` comparison now shows a clearer position: it still trails `ocr_telea` on `mean_abs_diff`, `edge_delta`, and restore latency, but it already looks more promising than the current persistent `ocr_powerpaint_v21` path on mean absolute difference and end-to-end latency.

Legacy compatibility entrypoints still work, but the public CLI surface is `no-watermar`:

```powershell
python .\run.py --scan-only
python .\run.py scan show --input .\inputs
python .\run.py batch plan --input .\inputs
python .\run.py config validate
python .\benchmark.py probe-providers
```

## Provider Environments

The repository supports two execution styles for heavyweight providers:

- Direct import in the current Python environment
- Sidecar execution via a dedicated model environment

Current sidecar environment variables:

```powershell
$env:NO_WATERMAR_PADDLEOCR_PYTHON = ".\.venvs\paddleocr\Scripts\python.exe"
$env:NO_WATERMAR_LAMA_PYTHON = ".\.venvs\lama\Scripts\python.exe"
$env:NO_WATERMAR_DIFFUSERS_PYTHON = ".\.venvs\diffusers\Scripts\python.exe"
$env:NO_WATERMAR_POWERPAINT_PYTHON = ".\.venvs\powerpaint\Scripts\python.exe"
$env:NO_WATERMAR_BRUSHNET_PYTHON = ".\.venvs\brushnet\Scripts\python.exe"
$env:NO_WATERMAR_DIFFUSERS_MODEL = "runwayml/stable-diffusion-inpainting"
$env:NO_WATERMAR_POWERPAINT_CHECKPOINT_DIR = "D:\Models\PowerPaint-v2-1"
$env:NO_WATERMAR_BRUSHNET_MODEL = "D:\Models\BrushNet\segmentation_mask_brushnet_ckpt"
$env:NO_WATERMAR_BRUSHNET_SOURCE_DIR = ".\models\brushnet-source"
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HUB_DISABLE_XET = "1"
```

The root CLI module entrypoint, `benchmark.py`, and `run.py` all load the repository-local `.env` file automatically when present.

For the public stable setup path, prefer:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\setup\bootstrap-sidecars.ps1 -StableOnly -InstallPackages
powershell -ExecutionPolicy Bypass -File .\tools\setup\validate-sidecars.ps1 -StableOnly -RunDoctor
```

That flow keeps the release-blocking `paddleocr` setup explicit, reports whether the optional `lama` path is also ready, and lines up with the release smoke wrapper under `tools\benchmark`.

Current sidecar scripts:

- `tools/sidecars/paddleocr_mask.py`
- `tools/sidecars/lama_restore.py`
- `tools/sidecars/diffusers_restore.py`
- `tools/sidecars/powerpaint_restore.py`
- `tools/sidecars/brushnet_restore.py`

## Optional Local Config

You can define named OCR watermark keyword presets in a repo-local `no-watermar.toml` file.

- Discovery order: `NO_WATERMAR_CONFIG`, then `no-watermar.toml` searched from the current working directory upward
- Use `NO_WATERMAR_WATERMARK_KEYWORD_PRESETS` to activate additional named presets from the config file
- Use `NO_WATERMAR_WATERMARK_KEYWORDS` for one-off extra tokens
- Start from `./no-watermar.toml.example` and see [Configuration](./docs/CONFIGURATION.md) for the full format
- Create a starter file with `.\bin\no-watermar.ps1 config init --template default` or `.\bin\no-watermar.ps1 config init --template stable-public`
- Inspect the effective config with `.\bin\no-watermar.ps1 config show`
- Validate the effective config with `.\bin\no-watermar.ps1 config validate`
- See ready-made examples under [docs/examples/config](./docs/examples/config/README.md)

`paddleocr` now supports three benchmark-facing execution modes:

- `auto`: prefer a persistent sidecar, then fall back to one-shot execution
- `persistent`: require a long-lived sidecar session for the whole benchmark run
- `oneshot`: start a fresh sidecar process for each OCR request

`powerpaint_v2_1` restore profiles can also set `restore_options.session_mode`:

- `auto`: prefer a persistent restore sidecar, then fall back to one-shot execution
- `persistent`: require the long-lived restore sidecar for the whole run
- `oneshot`: always launch a fresh restore sidecar process per image

`brushnet` restore profiles reuse the same prompt contract and currently expect:

- `restore_options.brushnet_model_path` or `NO_WATERMAR_BRUSHNET_MODEL`
- optionally `restore_options.base_model_path` or `NO_WATERMAR_BRUSHNET_BASE_MODEL_PATH`
- optionally `restore_options.source_dir` or `NO_WATERMAR_BRUSHNET_SOURCE_DIR` when the sidecar interpreter does not have the BrushNet fork installed directly

`diffusers_inpaint` also supports pretrained pipeline classes such as `FluxFillPipeline`, plus:

- `enable_layerwise_casting = true`
- `layerwise_storage_dtype = "float8_e4m3fn"`
- `layerwise_compute_dtype = "bfloat16"`

This is the current repository path for FP8-style low-memory diffusion experiments.

## Project Docs

- [Project Docs](./docs/README.md)
- [Interactive CLI Redesign Plan](./docs/plans/interactive-cli-redesign-plan.md)
- [Next Work And Delivery Plan](./docs/plans/next-work-and-delivery-plan.md)
- [Architecture](./docs/ARCHITECTURE.md)
- [Development Guide](./docs/DEVELOPMENT.md)
- [Setup Guide](./docs/setup/README.md)
- [Configuration](./docs/CONFIGURATION.md)
- [History](./docs/HISTORY.md)
- [Roadmap](./ROADMAP.md)
- [TODO](./TODO.md)
- [Contributing](./CONTRIBUTING.md)

## Benchmark Operations

- Use `.\bin\no-watermar.ps1 providers doctor` for the highest-signal local diagnosis view across interpreters, sidecar scripts, `.env`, runtime probes, and documented sidecar compatibility status.
- Use `benchmark.py aggregate` to summarize repeated runs by dataset and provider pair.
- Use `benchmark.py probe-providers` to verify whether configured sidecar interpreters can actually import their expected modules.
- Use `--mask-provider`, `--restore-provider`, `--run-after`, and `--run-before` to narrow aggregation windows.
- Use `benchmark.py evidence` to collapse repeated stable runs into one release-ready evidence summary with compare, aggregate, and trend links.
- Use [tools/benchmark/README.md](./tools/benchmark/README.md) for the release smoke wrapper and the repeated stable evidence capture wrapper.
- Use `python .\tools\benchmark\capture-disposable-evidence.py --clean` when you need the repo-native disposable evidence path that CI and local release preflight can reproduce without private inputs.
- Use `python .\tools\benchmark\build-review-bundle.py --report ... --label ... --output ...` to assemble a side-by-side human review bundle from benchmark reports, masks, overlays, restored images, and comparison artifacts, especially when multiple reports share the same provider name.

## Scan And Batch Planning

- Use `scan show` for a read-only discovery summary before writing any runtime artifacts.
- Use `scan run` to persist a versioned scan manifest under `runtime/scans/`.
- `scan run` refreshes `latest.json` for fast local reuse and agent-safe orchestration.
- `scan run` rejects scan output roots nested inside the selected input tree.
- Use `batch plan --scan-manifest .\runtime\scans\latest.json` to lock the plan to a persisted discovery result.
- Planned apply now reuses the item list stored in the plan instead of rescanning the input root.
- Use `batch plan` to persist an execution plan under `runtime/plans/`.
- `latest.json` is refreshed on each new plan for fast local reruns.
- Batch planning and direct apply now accept `--provider-profile`, `--mask-provider`, `--restore-provider`, and `--ocr-session-mode`.
- Use `batch apply --plan .\runtime\plans\latest.json` to replay the saved plan.
- Planned apply now prompts for confirmation unless you pass `--yes`.
- Use `--no-input --yes` for agent or CI execution with no interactive prompt.
- Direct `batch apply --input ...` remains available as a compatibility path.

## Batch Reports

- Each batch run now writes `summary.json`, `manifest.json`, and `results.jsonl` under `runtime/runs/<run-id>/`.
- `reports/report.json` and `runtime/runs/latest.json` are also refreshed for compatibility and quick lookup.
- Use `.\bin\no-watermar.ps1 batch report` to inspect the latest run summary.
- Use `.\bin\no-watermar.ps1 batch report --run-id <run-id>` or `--run-dir <path>` to inspect a specific run.
- Interrupted runs now persist `run_status = interrupted` and can be continued from the saved manifest and item-state files.
- Use `.\bin\no-watermar.ps1 batch resume` to continue the latest incomplete run without rescanning the input tree.
- `batch resume` no-ops cleanly when no pending items remain.

## Next Work

The current execution plan for follow-up work is tracked in [docs/plans/next-work-and-delivery-plan.md](./docs/plans/next-work-and-delivery-plan.md).

The CLI redesign for the next open-source iteration is tracked in [docs/plans/interactive-cli-redesign-plan.md](./docs/plans/interactive-cli-redesign-plan.md).

The immediate priorities are:

1. Finish Windows-first public packaging and release automation for the stable CLI path.
2. Harden the stable support matrix around `rule_based_roi` / `paddleocr` + `telea` / `corner_crop` / `lama`.
3. Capture and archive repeated stable baseline evidence for each release candidate instead of relying on ad-hoc compare and aggregate snapshots.
4. Keep experimental restore providers available for local evaluation without promoting them into the default release smoke path.
5. Expand Linux baseline validation for the lightweight CLI path without widening the supported model matrix prematurely.

## Development Principles

- Keep the core pipeline lightweight and deterministic by default
- Push heavyweight models behind provider boundaries
- Keep dataset-specific hints configurable rather than hardcoded
- Treat benchmarks as local workspace artifacts, not source assets
- Prefer local-first batch tooling over cloud-coupled workflows

## Authorized Use

Use the framework only for images and assets you are authorized to process.
