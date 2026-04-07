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
5. Run `.\bin\no-watermar.ps1 providers doctor` before model-backed benchmarks and confirm the compatibility status for each configured sidecar.
6. Treat `diffusers_inpaint` as experimental until a real local smoke run records both the interpreter path and the exact torch/model combination.
7. Keep benchmark and runtime outputs local and disposable.
