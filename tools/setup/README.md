# Setup Helpers

This directory contains local setup helpers for development and provider environments.

## Scripts

- `bootstrap-sidecars.ps1`: create the default provider virtual environments under `./.venvs`
- `validate-sidecars.ps1`: inspect expected sidecar interpreter paths and print environment export commands
- `..\benchmark\run-release-smoke.ps1`: run a release-oriented benchmark smoke pass once provider environments are configured

These scripts are conservative by design. They create predictable directory structure and validation output without forcing one provider package matrix on every machine.

`bootstrap-sidecars.ps1` uses the current `python` command by default. If a specific interpreter is required, pass `-PythonCommand` explicitly.

For the public stable setup path, prefer:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\setup\bootstrap-sidecars.ps1 -StableOnly -InstallPackages
powershell -ExecutionPolicy Bypass -File .\tools\setup\validate-sidecars.ps1 -StableOnly -RunDoctor
```

That path treats `paddleocr` as the release-blocking stable sidecar and `lama` as the optional stable model-backed restore path.

When only one provider needs a different interpreter, use the per-slot overrides:

- `-PaddlePythonCommand`
- `-LamaPythonCommand`
- `-DiffusersPythonCommand`
- `-PowerPaintPythonCommand`

`validate-sidecars.ps1 -RunProbe` will also invoke `python .\benchmark.py probe-providers` so you can confirm that each configured interpreter can import its expected provider module.

For the richer root-CLI diagnosis view, run `.\bin\no-watermar.ps1 providers doctor` after the sidecar paths are configured. The doctor output now includes the configured interpreter version, the documented compatibility status for each sidecar slot, and a `stable_setup` summary for the public stable bootstrap path.
