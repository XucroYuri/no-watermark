# Configuration

## Default Input and Output Paths

Default batch input:

- `./inputs`

Default batch outputs:

- `./runtime/runs`

Default benchmark workspace:

- `./benchmarks`

## Project Config File

Optional local config file name:

- `no-watermar.toml`

Discovery order:

1. `NO_WATERMAR_CONFIG`
2. `no-watermar.toml` searched from the current working directory upward

Current config-backed surface:

- named OCR watermark keyword presets
- dataset profiles shared by scan, batch, and benchmark flows
- provider profiles shared by batch and benchmark workflows
- restore prompt and option bundles for model-backed provider profiles

Example:

```toml
[watermark_keywords]
active_presets = ["brand", "stock_sites"]

[watermark_keywords.presets]
brand = ["brandname", "brand name", "@brand"]
stock_sites = ["example.com", "rights reserved"]

[profiles.datasets.local_smoke]
input = "./inputs"
recursive = false
limit = 2
benchmark_dataset = "regular_corner_text"

[profiles.providers.seed_telea]
mask_provider = "seed_manifest"
restore_provider = "telea"
ocr_session_mode = "auto"

[profiles.providers.ocr_telea]
mask_provider = "paddleocr"
restore_provider = "telea"
ocr_session_mode = "persistent"
restore_prompt = "remove the corner watermark and preserve the surrounding texture"
restore_negative_prompt = "smear, blur, melted text"

[profiles.providers.ocr_telea.restore_options]
steps = 30
guidance_scale = 6.5

[profiles.providers.ocr_corner_crop]
mask_provider = "paddleocr"
restore_provider = "corner_crop"
ocr_session_mode = "persistent"

[profiles.providers.ocr_corner_crop.restore_options]
edge_tolerance = 24

[profiles.providers.ocr_diffusers]
mask_provider = "paddleocr"
restore_provider = "diffusers_inpaint"
ocr_session_mode = "persistent"
restore_prompt = "remove the corner watermark and reconstruct the covered texture naturally"
restore_negative_prompt = "smear, blur, low detail, melted text"

[profiles.providers.ocr_diffusers.restore_options]
model_id = "runwayml/stable-diffusion-inpainting"
device = "cuda"
torch_dtype = "float16"
steps = 32
guidance_scale = 7.0

[profiles.providers.ocr_powerpaint_v21]
mask_provider = "paddleocr"
restore_provider = "powerpaint_v2_1"
ocr_session_mode = "persistent"
restore_prompt = "remove the corner watermark and reconstruct the covered texture naturally"
restore_negative_prompt = "text, logo, watermark, blur, smear, duplicated edges"

[profiles.providers.ocr_powerpaint_v21.restore_options]
checkpoint_dir = "D:/Models/PowerPaint-v2-1"
device = "cuda"
torch_dtype = "float16"
steps = 45
guidance_scale = 10.0

[profiles.providers.ocr_brushnet]
mask_provider = "paddleocr"
restore_provider = "brushnet"
ocr_session_mode = "persistent"
restore_prompt = "remove the corner watermark and reconstruct the covered texture naturally"
restore_negative_prompt = "text, logo, watermark, blur, smear, duplicated edges"

[profiles.providers.ocr_brushnet.restore_options]
brushnet_model_path = "D:/Models/BrushNet/segmentation_mask_brushnet_ckpt"
base_model_path = "D:/Models/BrushNet/realisticVisionV60B1_v51VAE"
source_dir = "D:/Code/BrushNet"
device = "cuda"
torch_dtype = "float16"
steps = 40
guidance_scale = 7.5
```

Behavior:

- preset names are matched case-insensitively
- dataset and provider profile names are matched case-insensitively
- whitespace inside tokens is ignored during OCR text matching
- unknown preset names fail fast with a clear error
- profile-relative paths are resolved relative to the selected `no-watermar.toml`
- provider profiles can now persist `restore_prompt`, `restore_negative_prompt`, and structured `restore_options`
- `corner_crop` is a local restore provider that crops the nearest image edge after watermark detection, so it offers a direct no-watermark output path without inpainting
- `diffusers_inpaint` expects a model id either in `restore_options.model_id` or `NO_WATERMAR_DIFFUSERS_MODEL`
- `powerpaint_v2_1` expects a local checkpoint dir either in `restore_options.checkpoint_dir` or `NO_WATERMAR_POWERPAINT_CHECKPOINT_DIR`
- `brushnet` expects a local BrushNet checkpoint either in `restore_options.brushnet_model_path` or `NO_WATERMAR_BRUSHNET_MODEL`

Use `./no-watermar.toml.example` as a starting point for local presets.

Bootstrap and inspect the local config with:

```powershell
.\bin\no-watermar.ps1 config init --template default
.\bin\no-watermar.ps1 config init --template brand-social --config .\configs\brand.toml
.\bin\no-watermar.ps1 config show
.\bin\no-watermar.ps1 config validate
.\bin\no-watermar.ps1 config validate --config .\no-watermar.toml
```

Built-in `config init` templates:

- `default`
- `brand-social`
- `stock-marketplaces`
- `mixed-corner-text`

Additional ready-made examples live under [docs/examples/config](./examples/config/README.md).

Compatibility wrappers still work for the same flow:

```powershell
python .\run.py config show
python .\run.py config validate
```

Use dataset and provider profiles in the CLI with:

```powershell
.\bin\no-watermar.ps1 scan show --dataset-profile local_smoke
.\bin\no-watermar.ps1 batch plan --dataset-profile local_smoke --output .\runtime\runs
.\bin\no-watermar.ps1 benchmark prepare --dataset-profile local_smoke --benchmark-root .\benchmarks
.\bin\no-watermar.ps1 benchmark run --dataset-profile local_smoke --provider-profile seed_telea
.\bin\no-watermar.ps1 benchmark aggregate --dataset-profile local_smoke --provider-profile seed_telea
.\bin\no-watermar.ps1 benchmark trends --dataset-profile local_smoke --baseline-provider-profile seed_telea --candidate-provider-profile ocr_telea
```

CLI rules:

- explicit CLI flags still override profile values
- `scan show`, `scan run`, `batch plan`, `batch apply`, `benchmark prepare`, `benchmark run`, `benchmark aggregate`, and `benchmark trends` can all reuse `profiles.datasets.<name>`
- provider profiles currently apply to `batch plan`, `batch apply`, `benchmark run`, `benchmark aggregate`, and `benchmark trends`
- `batch plan --scan-manifest ...` and `batch apply --plan ...` remain explicit input contracts and reject mixed direct profile/input flags
- batch and benchmark runs persist restore prompt and option fields so planned runs can be replayed consistently

## Environment Variables

### `NO_WATERMAR_CONFIG`

Optional explicit path to the local `no-watermar.toml` file.

When loaded from the repository-local `.env`, relative paths are resolved relative to that `.env` file.

### `NO_WATERMAR_WATERMARK_KEYWORD_PRESETS`

Comma-separated extra preset names to activate on top of `watermark_keywords.active_presets`.

Example:

```powershell
$env:NO_WATERMAR_WATERMARK_KEYWORD_PRESETS = "brand,stock_sites"
```

### `NO_WATERMAR_WATERMARK_KEYWORDS`

Comma-separated one-off extra keyword hints for OCR-based watermark filtering.

Example:

```powershell
$env:NO_WATERMAR_WATERMARK_KEYWORDS = "campaigntag,example.net,rightsreserved"
```

### `NO_WATERMAR_PADDLEOCR_PYTHON`

Path to a Python interpreter with PaddleOCR installed.

### `NO_WATERMAR_LAMA_PYTHON`

Path to a Python interpreter with a LaMa-compatible package installed.

### `NO_WATERMAR_DIFFUSERS_PYTHON`

Path to a Python interpreter with `torch`, `diffusers`, `transformers`, `accelerate`, and `safetensors` installed.

### `NO_WATERMAR_DIFFUSERS_MODEL`

Optional default diffusion inpainting model id or local model path for `diffusers_inpaint`.

### `NO_WATERMAR_POWERPAINT_PYTHON`

Path to a Python interpreter with `torch`, `diffusers`, `transformers`, `safetensors`, and the `powerpaint` package available.

### `NO_WATERMAR_POWERPAINT_CHECKPOINT_DIR`

Default local checkpoint directory for `powerpaint_v2_1`.

### `NO_WATERMAR_POWERPAINT_SOURCE_DIR`

Optional local clone of the upstream PowerPaint source tree. Use this when the sidecar environment points at a plain Python environment and the PowerPaint package is not installed into that interpreter.

### `NO_WATERMAR_BRUSHNET_PYTHON`

Path to a Python interpreter with the BrushNet-enabled diffusers fork importable, or pair it with `NO_WATERMAR_BRUSHNET_SOURCE_DIR`.

### `NO_WATERMAR_BRUSHNET_MODEL`

Default local BrushNet checkpoint directory for `brushnet`.

### `NO_WATERMAR_BRUSHNET_BASE_MODEL_PATH`

Optional default base-model folder or diffusers repo id for `brushnet`.

### `NO_WATERMAR_BRUSHNET_SOURCE_DIR`

Optional local clone of the upstream BrushNet repository. Use this when the sidecar interpreter does not have the BrushNet fork installed directly.

Recommended local values for provider sidecars:

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

## Provider Behavior

- Providers should not assume one fixed dataset layout
- Sidecar-backed providers must fail with a clear message when not configured
- Benchmark reports should record provider status even when execution is unavailable
- OCR-backed mask metadata now records the active keyword preset names and resolved config path
- `diffusers_inpaint` currently runs as a generic prompt-driven diffusion baseline and should be treated as experimental until it has local quality and latency evidence in this repository
- `diffusers_inpaint` can also be driven through `load_mode = "single_file"` plus `pipeline_class` and `original_config` when the model source is a `.ckpt` or `.safetensors` checkpoint instead of a diffusers repo
- `diffusers_inpaint` can now also be driven through pretrained pipeline classes such as `FluxFillPipeline`, and compatible transformers can use `enable_layerwise_casting`, `layerwise_storage_dtype`, and `layerwise_compute_dtype` for FP8-storage experiments
- the repo-local `ocr_fluxfill_fp8` profile is currently frozen on the active 16 GB local GPU until a validated smoke path or a larger-VRAM host exists, even though the underlying `FluxFillPipeline` support remains available in code
- `powerpaint_v2_1` is an experimental restore adapter for the official PowerPaint v2.1 object-removal pipeline and currently expects a local checkpoint directory with the upstream `PowerPaint_Brushnet` weights plus the base model folder
- `powerpaint_v2_1` also accepts `base_model_path`, `backbone_source`, `resize_longest_side`, `fitting_degree`, and `brushnet_conditioning_scale` inside `restore_options`
- the current validated local `powerpaint_v2_1` smoke path uses `local_files_only = true` plus a checkpoint directory downloaded from `JunhaoZhuang/PowerPaint-v2-1`
- `brushnet` is an experimental standalone restore adapter for the upstream BrushNet SD1.5 pipeline and currently expects a local BrushNet checkpoint directory plus a compatible SD1.5-style base model folder or repo id
- `brushnet` also accepts `source_dir`, `base_model_path`, `resize_longest_side`, `brushnet_conditioning_scale`, `guess_mode`, `control_guidance_start`, and `control_guidance_end` inside `restore_options`
- the current validated local `brushnet` smoke path reuses the repository-local PowerPaint Python `3.12` environment together with `NO_WATERMAR_BRUSHNET_SOURCE_DIR` pointing at a repo-local `.\models\brushnet-source` clone

See [docs/setup/provider-sidecars.md](./setup/provider-sidecars.md) for environment layout and setup flow.

When these variables are stored in the repository-local `.env`, they are loaded automatically by the root CLI module entrypoint as well as the legacy `run.py` and `benchmark.py` wrappers.
