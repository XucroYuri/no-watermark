# Agent Handoff

Last updated: `2026-04-14`

This document is the fast-resume entrypoint for the next agent session. It records what is already done on `main`, what still blocks the first live public release, and the exact next steps to continue safely.

## Current State

- Public CLI hardening, stable/disposable evidence automation, packaging, and tag-driven release workflows are merged into `main`.
- The repository version is `0.3.0`.
- CI on `main` is green for:
  - `ubuntu-latest` and `windows-latest`
  - Python `3.11` and `3.12`
  - disposable evidence generation
  - package build and `twine check`
- Workflow maintenance follow-ups are already merged:
  - PR `#1`: public CLI release hardening
  - PR `#2`: `0.3.0` release metadata preparation
  - PR `#3`: upgrade `actions/checkout` and `actions/setup-python` to Node 24-compatible majors
  - PR `#4`: upgrade `actions/upload-artifact` / `actions/download-artifact` to current majors

## What Is Still Blocking The First Live Release

These are external-release blockers, not repository code gaps:

1. PyPI project setup is still incomplete.
   - During validation on `2026-04-14`, both `https://pypi.org/project/no-watermar/` and `https://pypi.org/project/no-watermark/` returned `404`.
   - Re-check ownership and naming before tagging the first public release.
2. PyPI trusted publishing still needs to be enabled.
   - The release workflow already uses OIDC-based publishing.
   - The PyPI-side trusted publisher and any required GitHub environment approvals still need to be configured.
3. A fresh real stable evidence bundle still needs to be captured on a machine with the stable sidecars actually installed.
   - Disposable evidence is automated and green in CI.
   - Real release evidence still needs a configured `paddleocr + telea` machine, plus optional `lama`.

## Recommended Next Sequence

Execute these steps in order:

1. Reconfirm the release namespace.
   - Decide whether the first public package name is `no-watermar` exactly.
   - Verify the chosen name in PyPI before tagging.
2. Configure live publishing.
   - Create the PyPI project if it does not exist yet.
   - Add the GitHub repository/workflow as a trusted publisher in PyPI.
   - Confirm the GitHub `pypi` environment is usable for the release workflow.
3. Run local release preflight from a clean worktree on `main`.
   - Run the release helper and confirm disposable evidence + packaging still pass.
4. Run the stable public matrix on a prepared machine.
   - Bootstrap sidecars.
   - Run stable smoke.
   - Capture a repeated-run stable evidence bundle.
5. Cut the first release tag.
   - Create annotated tag `v0.3.0`.
   - Push the tag.
   - Confirm `.github/workflows/release.yml` creates the GitHub Release and publishes to PyPI.

## Commands To Resume Quickly

Recommended Windows-first flow from a clean worktree:

```powershell
git fetch origin main
git worktree add .worktrees/release-resume -b release/resume origin/main
Set-Location .worktrees/release-resume
python -m venv .venv
$env:PATH = "$PWD\.venv\Scripts;$env:PATH"
python -m pip install --upgrade pip
python -m pip install -e .[dev]
python -m unittest discover -s tests -v
python -m no_watermar.cli --help
powershell -ExecutionPolicy Bypass -File .\tools\releases\build-release.ps1 -CleanDist
```

If using the repo venv on Windows, keep the venv `Scripts` directory at the front of `PATH` before running tests. Some subprocess-based tests call `python` directly and expect the repo environment to win command resolution.

Stable-sidecar validation path:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\setup\bootstrap-sidecars.ps1 -StableOnly -InstallPackages -RunDoctor
powershell -ExecutionPolicy Bypass -File .\tools\setup\validate-sidecars.ps1 -StableOnly -RunDoctor
powershell -ExecutionPolicy Bypass -File .\tools\benchmark\run-release-smoke.ps1 -Limit 1
powershell -ExecutionPolicy Bypass -File .\tools\benchmark\capture-stable-baseline.ps1 -Repetitions 3
```

## Files That Matter Most

- [docs/releases/release-checklist.md](./release-checklist.md)
- [docs/releases/release-process.md](./release-process.md)
- [docs/releases/public-release-readiness.md](./public-release-readiness.md)
- [tools/releases/README.md](../../tools/releases/README.md)
- [tools/benchmark/README.md](../../tools/benchmark/README.md)
- [.github/workflows/ci.yml](../../.github/workflows/ci.yml)
- [.github/workflows/release.yml](../../.github/workflows/release.yml)

## Expected Release Evidence

Before the first live release, the following should exist together:

- local preflight output under `.\runtime\release-preflight\disposable-evidence\`
- packaged disposable evidence zip under `.\runtime\release-preflight\disposable-evidence\package\`
- stable evidence summary under `.\benchmarks\evidence\latest.json`
- stable evidence markdown under `.\benchmarks\evidence\latest.md`

## If The Next Session Starts Cold

Use this triage order:

1. Open this file.
2. Open [release-checklist.md](./release-checklist.md).
3. Check whether PyPI project/trusted publishing has already been configured outside the repo.
4. If yes, run the release preflight and stable evidence capture.
5. If no, do not tag yet; finish PyPI/trusted-publishing setup first.
