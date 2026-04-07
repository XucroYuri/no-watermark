# Development Guide

## Environment

Baseline development:

```powershell
python -m pip install -r .\requirements.txt
python -m unittest discover -s tests -v
```

## Project Conventions

- Keep defaults generic
- Avoid private sample references
- Prefer ASCII in code and config unless existing files already use Unicode
- Use provider abstractions for heavyweight integrations
- Keep benchmark artifacts local and disposable

## Local Commands

Batch baseline:

```powershell
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
.\bin\no-watermar.ps1 batch apply --input .\inputs
.\bin\no-watermar.ps1 batch apply --dataset-profile local_smoke --provider-profile ocr_telea --output .\runtime\runs
.\bin\no-watermar.ps1 config init --template default
.\bin\no-watermar.ps1 config show
.\bin\no-watermar.ps1 config validate
```

Benchmark setup:

```powershell
.\bin\no-watermar.ps1 providers doctor
.\bin\no-watermar.ps1 benchmark prepare --input .\inputs
.\bin\no-watermar.ps1 benchmark prepare --dataset-profile local_smoke --benchmark-root .\benchmarks
.\bin\no-watermar.ps1 providers probe
.\bin\no-watermar.ps1 benchmark run --dataset-profile local_smoke --provider-profile seed_telea --benchmark-root .\benchmarks
.\bin\no-watermar.ps1 benchmark run --dataset regular_corner_text --mask-provider seed_manifest --restore-provider telea --ocr-session-mode auto
.\bin\no-watermar.ps1 benchmark compare --baseline-report .\benchmarks\path\baseline.json --candidate-report .\benchmarks\path\candidate.json
.\bin\no-watermar.ps1 benchmark aggregate --dataset-profile local_smoke --provider-profile seed_telea --reports-root .\benchmarks\runs
.\bin\no-watermar.ps1 benchmark trends --dataset-profile local_smoke --baseline-provider-profile seed_telea --candidate-provider-profile ocr_telea --benchmark-root .\benchmarks
.\bin\no-watermar.ps1 benchmark aggregate --reports-root .\benchmarks\runs --dataset regular_corner_text --mask-provider seed_manifest --restore-provider telea
.\bin\no-watermar.ps1 benchmark trends --dataset regular_corner_text --baseline-mask-provider seed_manifest --baseline-restore-provider telea --candidate-mask-provider paddleocr --candidate-restore-provider telea
powershell -ExecutionPolicy Bypass -File .\tools\benchmark\run-release-smoke.ps1 -Limit 1
```

`benchmark trends` auto-resolves the latest matching compare and aggregate artifacts and writes a versioned JSON/Markdown snapshot plus `latest.json` / `latest.md` under `.\benchmarks\trends\`.
Dataset and provider profiles from `no-watermar.toml` can now drive the same scan, batch, and benchmark flows without repeating the same CLI argument sets.
For private real-local slices, point `--benchmark-root` at a dedicated ignored subdirectory so compare, aggregate, and trend artifacts stay isolated per dataset.
Provider profiles can now also carry `restore_prompt`, `restore_negative_prompt`, and `restore_options`, and those fields are persisted through batch plans, batch runs, and benchmark summaries.
`diffusers_inpaint` now consumes the same fields, so batch and benchmark provider profiles can carry both the text prompt contract and model load options such as `model_id`, `device`, `torch_dtype`, `steps`, and `guidance_scale`.
For local single-file checkpoints, the same profile can also carry `load_mode = "single_file"`, `pipeline_class`, and `original_config`.
For pretrained diffusers pipelines, the same profile can now also carry `pipeline_class = "FluxFillPipeline"` plus `enable_layerwise_casting`, `layerwise_storage_dtype`, and `layerwise_compute_dtype` for the current FP8-storage experiment.
That support is now intentionally frozen at the profile level on the current 16 GB local GPU, so keep `ocr_fluxfill_fp8` out of active benchmark runs until a validated smoke path or a larger-VRAM host is available.
`powerpaint_v2_1` reuses the same contract and expects `checkpoint_dir` plus any optional task settings in `restore_options`.
`powerpaint_v2_1` restore profiles can also set `restore_options.session_mode = "auto" | "persistent" | "oneshot"` to control whether the restore sidecar should be reused across the whole run.
`brushnet` now reuses the same contract and expects `brushnet_model_path`, plus an optional `base_model_path` and `source_dir` when the sidecar interpreter relies on a local BrushNet clone instead of an installed package.
The current FP8 attempt should prefer the `diffusers` sidecar because the local `diffusers` environment already exposes `enable_layerwise_casting()` and float8 dtypes, while the repository-local PowerPaint / BrushNet fork still sits on diffusers `0.27.0` without that API.
For repeated visual checks, `python .\tools\benchmark\build-review-bundle.py --report ... --label ... --output ...` now assembles a side-by-side local review bundle with the original inputs, seed artifacts, provider masks, overlays, restored images, and copied compare/trend artifacts. Use `--label` when you are comparing multiple runs from the same provider family, such as several `brushnet` tuning variants.

Legacy wrappers remain available during migration:

- `python .\run.py ...`
- `python .\benchmark.py ...`

Current scan and batch planning notes:

- `scan show` is read-only and returns the discovery summary directly.
- `scan run` writes versioned manifest JSON files plus `latest.json` under the scans root.
- `scan run` rejects scan output directories nested inside the selected input tree.
- `batch plan --scan-manifest ...` reuses the persisted scan item list instead of rescanning the input root.
- `batch plan` writes versioned plan JSON files plus `latest.json` under the plans root.
- `batch plan` and direct `batch apply` can now resolve provider contracts from `--provider-profile` or explicit provider flags.
- `batch apply --plan ...` reuses the saved input, output, and execution flags.
- `batch apply --plan ...` now also reuses the saved item list, so new files added after planning are not picked up accidentally.
- Batch runs now persist `summary.json`, `manifest.json`, and `results.jsonl` under each run directory.
- `batch report` reads the latest run by default and can also load a specific run id, run directory, or summary file.
- Interrupted runs are marked with `run_status = interrupted` and keep the partial result set in `results.jsonl`.
- `batch resume` loads the saved manifest, skips completed items, and only processes the remaining item set.
- `batch apply --plan ...` now prompts for confirmation by default.
- Use `--yes` to skip the confirmation prompt.
- Use `--no-input --yes` for agent-safe non-interactive execution.
- `batch apply --plan ...` rejects mixed direct execution flags to keep the contract unambiguous.

`benchmark.py run` reuses one OCR execution context across the whole run. This context is shared by the `paddleocr` mask provider and OCR residual scoring.

Supported OCR session modes:

- `auto`: try a persistent sidecar first, then fall back to one-shot execution
- `persistent`: require a persistent sidecar session
- `oneshot`: always launch a fresh sidecar process per OCR request

Supported `powerpaint_v2_1` restore session modes:

- `auto`: try a persistent restore sidecar first, then fall back to one-shot execution
- `persistent`: require a persistent restore sidecar session
- `oneshot`: always launch a fresh restore sidecar process per image

The current real-local 2-image validation pass confirms that the persistent PowerPaint path is working end to end, but the first image still absorbs most of the model warmup cost. On that slice, the second image restore time dropped materially while the 2-image mean restore latency still remained worse than the older one-shot average and far worse than `ocr_telea`.

`benchmark.py aggregate` can also filter by:

- `--mask-provider`
- `--restore-provider`
- `--run-after`
- `--run-before`

`benchmark trends` can auto-resolve sources from `.\benchmarks\comparisons` and `.\benchmarks\aggregations`, or it can be pointed at explicit `--comparison` / `--aggregation` JSON files.

Use `no-watermar providers doctor` first when provider availability is unclear. It combines current Python info, repo-local `.env`, sidecar script paths, runtime import probes, and the documented sidecar compatibility matrix in one summary.

Use `no-watermar providers probe` or `benchmark.py probe-providers` when you only need the raw runtime importability data without the broader diagnosis context.

## Sidecar Environments

When a provider is unavailable in the current environment, configure a sidecar interpreter:

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

The root CLI module entrypoint and the compatibility wrappers also auto-load the root `.env` file, so local sidecar paths can live there instead of being exported manually each shell session.

Setup details and bootstrap scripts are documented in [docs/setup/README.md](./setup/README.md).

## Testing Expectations

- Unit tests for deterministic logic
- Graceful-failure tests for unavailable providers
- Smoke tests for CLI entrypoints when feasible

## Release Discipline

- Update docs when behavior changes
- Update `CHANGELOG.md`
- Keep runtime and benchmark outputs out of commits
