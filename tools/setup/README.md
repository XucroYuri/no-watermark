# Setup Helpers

This directory contains local setup helpers for development and provider environments.

## Scripts

- `bootstrap-sidecars.ps1`: create the default provider virtual environments under `./.venvs`
- `validate-sidecars.ps1`: inspect expected sidecar interpreter paths and print environment export commands
- `..\benchmark\run-release-smoke.ps1`: run a release-oriented benchmark smoke pass once provider environments are configured

These scripts are conservative by design. They create predictable directory structure and validation output without forcing one provider package matrix on every machine.

`bootstrap-sidecars.ps1` separates the repository CLI interpreter from the sidecar interpreters:

- `-ConfigPythonCommand` controls the Python used for `no_watermar.cli config init`
- `-PythonCommand` controls the default Python used to create sidecar virtual environments

If the repo bootstrap should stay on the current editable-install environment but one sidecar needs a different interpreter, keep `-ConfigPythonCommand` unchanged and override only the relevant sidecar slot.

For the public stable setup path, prefer:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\setup\bootstrap-sidecars.ps1 -StableOnly -InstallPackages
powershell -ExecutionPolicy Bypass -File .\tools\setup\validate-sidecars.ps1 -StableOnly -RunDoctor
```

That path treats `paddleocr` as the release-blocking stable sidecar and `lama` as the optional stable model-backed restore path. When no repo-local `no-watermar.toml` exists yet, the stable bootstrap also initializes one from the built-in `stable-public` template so the standard `local_smoke`, `seed_telea`, `ocr_telea`, `lama_eval`, and `ocr_corner_crop` profiles are ready immediately.

Pass `-SkipConfigInit` if you want to manage `no-watermar.toml` yourself.

When only one provider needs a different interpreter, use the per-slot overrides:

- `-PaddlePythonCommand`
- `-LamaPythonCommand`
- `-DiffusersPythonCommand`
- `-PowerPaintPythonCommand`

`validate-sidecars.ps1 -RunProbe` will also invoke `python .\benchmark.py probe-providers` so you can confirm that each configured interpreter can import its expected provider module.

For the richer root-CLI diagnosis view, run `.\bin\no-watermar.ps1 providers doctor` after the sidecar paths are configured. The doctor output now includes the configured interpreter version, the documented compatibility status for each sidecar slot, and a `stable_setup` summary for the public stable bootstrap path.
