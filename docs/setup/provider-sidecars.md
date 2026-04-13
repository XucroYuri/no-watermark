# Provider Sidecars

## Why Sidecars

The baseline repository stays lightweight on purpose. Heavy OCR, segmentation, and inpainting models should be isolated behind dedicated Python environments.

## Current Sidecar Slots

- `NO_WATERMAR_PADDLEOCR_PYTHON`
- `NO_WATERMAR_LAMA_PYTHON`
- `NO_WATERMAR_DIFFUSERS_PYTHON`
- `NO_WATERMAR_POWERPAINT_PYTHON`
- `NO_WATERMAR_BRUSHNET_PYTHON`

## Public Support Tiers

- Stable sidecars: `paddleocr`, `lama`
- Experimental sidecars: `diffusers_inpaint`, `powerpaint_v2_1`, `brushnet`

`providers doctor` now reports `support_tier`, validated platforms, and the recommended entrypoint for each provider slot so the public release surface stays explicit.

## Recommended Layout

```text
.venvs/
  paddleocr/
  lama/
  diffusers/
  powerpaint/
  brushnet/
tools/
  sidecars/
    paddleocr_mask.py
    lama_restore.py
    diffusers_restore.py
    powerpaint_restore.py
    brushnet_restore.py
```

## Bootstrap Commands

Create the default virtual environments:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\setup\bootstrap-sidecars.ps1
```

Validate interpreter discovery:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\setup\validate-sidecars.ps1
.\bin\no-watermar.ps1 providers doctor
```

When one provider needs a different base interpreter, override it explicitly during bootstrap:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\setup\bootstrap-sidecars.ps1 -LamaPythonCommand "C:\Path\To\Python312\python.exe"
powershell -ExecutionPolicy Bypass -File .\tools\setup\bootstrap-sidecars.ps1 -DiffusersPythonCommand "C:\Path\To\Python311\python.exe"
powershell -ExecutionPolicy Bypass -File .\tools\setup\bootstrap-sidecars.ps1 -PowerPaintPythonCommand "C:\Path\To\Python312\python.exe"
powershell -ExecutionPolicy Bypass -File .\tools\setup\bootstrap-sidecars.ps1 -BrushNetPythonCommand "C:\Path\To\Python39\python.exe"
```

## Environment Variables

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

The repository root `.env` file is loaded automatically by the root CLI module entrypoint, `benchmark.py`, and `run.py`. Relative `*_PYTHON` paths from `.env` are resolved against the repository root before provider discovery runs.

## Suggested Package Directions

- PaddleOCR sidecar: install `paddleocr`, then install the appropriate Paddle runtime for the target machine
- LaMa sidecar: install a `simple-lama-inpainting` compatible stack inside the dedicated `lama` environment
- Diffusers sidecar: install the matching `torch` build for the target machine first, then install `diffusers`, `transformers`, `accelerate`, and `safetensors`
- PowerPaint sidecar: install the matching `torch` build first, then install `diffusers`, `transformers`, `safetensors`, and either install the `powerpaint` package or point `NO_WATERMAR_POWERPAINT_SOURCE_DIR` at a local upstream clone
- BrushNet sidecar: install the matching `torch` build first, then install `transformers`, `accelerate`, image dependencies, and either install the upstream BrushNet repo in editable mode or point `NO_WATERMAR_BRUSHNET_SOURCE_DIR` at a local clone

Current validated local path:

- Python `3.12`
- package `simple-lama-inpainting`
- import target `simple_lama_inpainting`

## Compatibility Matrix

`providers doctor` now reports the configured interpreter version for each sidecar slot and compares it against this documented validated set.

| Provider | Env Var | Kind | Validated Python | Validated Package Path | Notes |
| --- | --- | --- | --- | --- | --- |
| `paddleocr` | `NO_WATERMAR_PADDLEOCR_PYTHON` | mask | `3.8` to `3.12` | `paddleocr` plus the matching Paddle runtime | Validate direct imports first because runtime compatibility still depends on machine, CUDA, and wheel selection. |
| `lama` | `NO_WATERMAR_LAMA_PYTHON` | restore | `3.12` | `simple-lama-inpainting` importing `simple_lama_inpainting` | This is the currently validated local path in this repository. Treat newer Python versions as unvalidated until re-tested. |
| `diffusers_inpaint` | `NO_WATERMAR_DIFFUSERS_PYTHON` | restore | `3.12` | `torch`, `diffusers`, `transformers`, `accelerate`, `safetensors` | The current validated local smoke path uses `from_single_file`, an explicit `original_config`, and the Hugging Face mirror env vars above. |
| `powerpaint_v2_1` | `NO_WATERMAR_POWERPAINT_PYTHON` | restore | `3.12` | `torch`, `diffusers`, `transformers`, `safetensors`, `powerpaint`, `mmengine` | The current validated local smoke path uses `torch 2.11.0+cu128`, `diffusers 0.27.0`, `transformers 4.41.2`, a local `JunhaoZhuang/PowerPaint-v2-1` checkpoint, and local-only model loading. |
| `brushnet` | `NO_WATERMAR_BRUSHNET_PYTHON` | restore | `3.12` | `torch`, `transformers`, `accelerate`, and the BrushNet-enabled diffusers fork | The current validated local smoke path reuses the repository-local PowerPaint Python `3.12` environment plus `NO_WATERMAR_BRUSHNET_SOURCE_DIR` pointing at a repo-local `.\models\brushnet-source` clone. |

## PaddleOCR Session Modes

The benchmark runner can use `paddleocr` in three modes:

- `auto`: prefer a persistent `stdin/stdout` JSONL sidecar, fall back to one-shot if startup fails
- `persistent`: require the long-lived sidecar and surface provider unavailability if startup fails
- `oneshot`: start a new sidecar process per OCR request

The persistent mode is the preferred benchmark path because the same OCR session is reused for:

- `paddleocr` mask generation
- OCR residual scoring after restoration

## PowerPaint Restore Session Modes

`powerpaint_v2_1` restore profiles can also set `restore_options.session_mode`:

- `auto`: prefer a persistent `stdin/stdout` JSONL restore sidecar, then fall back to one-shot if startup fails
- `persistent`: require the long-lived restore sidecar and surface provider unavailability if startup fails
- `oneshot`: start a new restore sidecar process per image

The current repository-local validation now covers a persistent `powerpaint_v2_1` restore session on a 2-image real-local slice. That removed the previous stdout protocol blocker and proved that later images can reuse the loaded model, but the current 2-image mean still trails both `ocr_telea` quality and the earlier one-shot PowerPaint average because the first-image warmup cost is still large.

This repository intentionally leaves exact provider package versions to the local environment owner because GPU, CUDA, and Python compatibility vary by machine.

## Compatibility Note

Provider wrapper compatibility is not uniform across Python versions.

- `paddleocr` can be bootstrapped independently from the core repository and should be validated with a direct import first.
- `simple-lama-inpainting` may require an older or more specific interpreter and dependency stack than the repository default environment.
- If the `lama` sidecar cannot be installed cleanly, create that venv with a compatible Python version and point `NO_WATERMAR_LAMA_PYTHON` at that interpreter.
- The current repository default shell Python may be too new for the `lama` stack. A dedicated `3.12` sidecar environment is the validated path in this workspace.
- If `providers doctor` marks a configured interpreter as `unvalidated`, recreate only that provider venv with one of the documented validated versions above.
- If the default Hugging Face domain is blocked or remapped locally, set `HF_ENDPOINT` to a reachable mirror and keep `HF_HUB_DISABLE_XET=1` in the local environment before validating `diffusers_inpaint`.
- If the PowerPaint package is not installed into the sidecar interpreter, set `NO_WATERMAR_POWERPAINT_SOURCE_DIR` to a local clone before validating `powerpaint_v2_1`.
- If the BrushNet fork is not installed into the sidecar interpreter, set `NO_WATERMAR_BRUSHNET_SOURCE_DIR` to a local BrushNet clone before validating `brushnet`.
- The current repository-local validated PowerPaint smoke path uses a source-dir style import path plus a local checkpoint under `.\models\PowerPaint-v2-1`.

## Validation Targets

After configuring the sidecar interpreters, validate at least:

```powershell
.\bin\no-watermar.ps1 providers doctor
python .\benchmark.py list-providers
python .\benchmark.py probe-providers
python .\benchmark.py prepare --input .\inputs
python .\benchmark.py run --dataset regular_corner_text --mask-provider paddleocr --restore-provider diffusers_inpaint --ocr-session-mode persistent
python .\benchmark.py run --dataset regular_corner_text --mask-provider paddleocr --restore-provider powerpaint_v2_1 --ocr-session-mode persistent
python .\benchmark.py run --dataset regular_corner_text --mask-provider paddleocr --restore-provider brushnet --ocr-session-mode persistent
```

The first public release path should treat only the `paddleocr` and `lama` validation targets above as release-blocking. The diffusion-backed commands remain opt-in local evaluation steps.

For the diffusion-backed runs above, also set either `NO_WATERMAR_DIFFUSERS_MODEL` or `restore_options.model_id` for `diffusers_inpaint`, set either `NO_WATERMAR_POWERPAINT_CHECKPOINT_DIR` or `restore_options.checkpoint_dir` for `powerpaint_v2_1`, and set either `NO_WATERMAR_BRUSHNET_MODEL` or `restore_options.brushnet_model_path` for `brushnet`. When validating the restore-session path, also set `restore_options.session_mode = "auto"` or `"persistent"` in the selected PowerPaint provider profile.

The doctor output now includes:

- the configured sidecar interpreter version, when it can be read
- the validated Python versions documented for each provider slot
- a compatibility status of `validated`, `unvalidated`, `unresolved`, or `unknown`
- for `diffusers_inpaint`, whether the restore sidecar interpreter exists even before a repository-local validated version has been recorded
- for `powerpaint_v2_1`, whether the restore sidecar interpreter is configured even before a repository-local validated matrix has been recorded
- for `brushnet`, whether the restore sidecar interpreter exists even before a repository-local validated matrix has been recorded

Then run a provider-backed benchmark with the configured environment variables.

```powershell
python .\benchmark.py run --dataset regular_corner_text --mask-provider paddleocr --restore-provider telea --ocr-session-mode auto
```

For a release-oriented local smoke pass that includes baseline, OCR-backed run, compare, and aggregate steps, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\benchmark\run-release-smoke.ps1 -Limit 1
```

The smoke script now checks `lama` availability from `probe-providers`. If the interpreter exists but the `simple_lama` module is missing, the script skips the model-backed restore smoke run and prints the concrete import failure.
