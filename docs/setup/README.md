# Setup Guide

This section documents how to prepare a local development workspace and optional provider sidecar environments.

## Documents

- [Windows Local Setup](./windows-local-setup.md)
- [Provider Sidecars](./provider-sidecars.md)
- [Workspace Layout](./workspace-layout.md)

## Recommended Flow

1. Create a baseline Python environment for the repository itself.
2. Keep input images under `./inputs`.
3. Create dedicated sidecar environments under `./.venvs` for heavyweight providers.
4. Put provider interpreter paths in the repo-local `.env` or export them in the current shell.
5. Bootstrap the stable public sidecars first with `powershell -ExecutionPolicy Bypass -File .\tools\setup\bootstrap-sidecars.ps1 -StableOnly -InstallPackages`.
6. Validate the stable path with `powershell -ExecutionPolicy Bypass -File .\tools\setup\validate-sidecars.ps1 -StableOnly -RunDoctor` and confirm `stable_setup.release_blocking_ready = true`.
7. Keep the public stable matrix focused on `paddleocr` as release-blocking and `lama` as the optional stable model-backed restore path; treat `diffusers_inpaint`, `powerpaint_v2_1`, and `brushnet` as experimental local-only evaluation paths.
8. Keep benchmark and runtime outputs local and disposable.
