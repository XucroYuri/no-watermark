# Windows Local Setup

## Goal

Prepare a clean Windows development workspace for the baseline pipeline and optional provider sidecars.

## Baseline Requirements

- Windows PowerShell
- A working Python installation for the core repository
- Local disk space for `inputs`, `runtime`, `benchmarks`, and optional model environments

## Baseline Bootstrap

From the repository root:

```powershell
python -m pip install -r .\requirements.txt
python -m unittest discover -s tests -v
python .\benchmark.py list-providers
python .\benchmark.py probe-providers
.\bin\no-watermar.ps1 providers doctor
```

## Recommended Workspace Conventions

- Keep private source images only under `.\inputs`
- Keep provider virtual environments under `.\.venvs`
- Keep local model checkpoints outside the Git working tree when possible
- Treat `runtime/` and `benchmarks/` as disposable outputs

## Sidecar Bootstrap

Create provider virtual environments with the helper script:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\setup\bootstrap-sidecars.ps1
```

If `lama` needs a different interpreter than the default shell Python, override only that slot:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\setup\bootstrap-sidecars.ps1 -LamaPythonCommand "C:\Path\To\Python312\python.exe"
```

If the diffusion restore sidecar should use a separate interpreter, override that slot explicitly:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\setup\bootstrap-sidecars.ps1 -DiffusersPythonCommand "C:\Path\To\Python311\python.exe"
```

If the PowerPaint restore sidecar should use its own interpreter, override that slot explicitly:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\setup\bootstrap-sidecars.ps1 -PowerPaintPythonCommand "C:\Path\To\Python312\python.exe"
```

Review environment status:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\setup\validate-sidecars.ps1
```

If the default shell Python is too new for the `lama` stack, create that sidecar with a separate interpreter. One validated path in this workspace used `uv` with Python `3.12`:

```powershell
uv venv --python "C:\Users\xuyu_\AppData\Roaming\uv\python\cpython-3.12.11-windows-x86_64-none\python.exe" .\.venvs\lama
uv pip install --python .\.venvs\lama\Scripts\python.exe simple-lama-inpainting
python .\benchmark.py probe-providers
.\bin\no-watermar.ps1 providers doctor
```

Use the doctor output to confirm that:

- `NO_WATERMAR_LAMA_PYTHON` resolves to the intended interpreter
- the configured interpreter version is `3.12`
- the sidecar compatibility status is `validated`

For `diffusers_inpaint`, the current goal is weaker: confirm that `NO_WATERMAR_DIFFUSERS_PYTHON` exists, imports `diffusers`, and then record the exact torch/runtime choice in local notes after the first real smoke run.
The current validated local smoke path in this workspace uses Python `3.12`, `torch 2.11.0+cu128`, `diffusers 0.37.1`, `HF_ENDPOINT=https://hf-mirror.com`, and a single-file Stable Diffusion inpainting checkpoint with an explicit `original_config`.

For `powerpaint_v2_1`, the current validated local smoke path uses Python `3.12`, `torch 2.11.0+cu128`, `diffusers 0.27.0`, `transformers 4.41.2`, `mmengine 0.10.7`, and a local `JunhaoZhuang/PowerPaint-v2-1` checkpoint under `.\models\PowerPaint-v2-1`.

## Notes

- The repository does not pin one global model stack for all hardware profiles.
- Keep provider package installation decisions isolated inside sidecar environments.
- If a provider stack depends on a specific CUDA or PyTorch combination, document that choice in local notes instead of changing the baseline repository requirements.
- If a model wrapper fails under the default Python version, recreate only that sidecar venv with a compatible interpreter rather than changing the whole repository environment.
- If a sidecar interpreter exists but a provider is still unavailable, use `providers doctor` first and `probe-providers` when you need the raw import probe output.
