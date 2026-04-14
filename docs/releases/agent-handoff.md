# Agent Handoff

Last updated: `2026-04-14`

This document is the fast-resume entrypoint for the next agent session. It records what is already done on `main`, what still blocks the first live public release, and the exact next steps to continue safely.

## Current State

- Public CLI hardening, stable/disposable evidence automation, packaging, and tag-driven release workflows are merged into `main`.
- The repository version is `0.3.0`.
- The GitHub-side `pypi` environment already exists, and the `Release` workflow dry-run on `main` completed successfully on `2026-04-14`.
- CI on `main` is green for:
  - `ubuntu-latest` and `windows-latest`
  - Python `3.11` and `3.12`
  - disposable evidence generation
  - package build and `twine check`
- The stable sidecar bootstrap path was revalidated locally on `2026-04-14`:
  - `validate-sidecars.ps1 -StableOnly -RunProbe -RunDoctor` now reaches `stable_setup.status = ready` without requiring a separate manual export step when the default `.venvs` interpreters exist
  - `run-release-smoke.ps1` passed on the public synthetic fixture with `paddleocr + telea` and `seed_manifest + lama`
  - `capture-stable-baseline.ps1 -Repetitions 3 -RequireLama` produced a ready-state stable evidence bundle on the public synthetic fixture
  - the packaged stable evidence archive is `.\runtime\public-stable-matrix\package\stable-public-evidence-20260414-181705-220310.zip`
- Workflow maintenance follow-ups are already merged:
  - PR `#1`: public CLI release hardening
  - PR `#2`: `0.3.0` release metadata preparation
  - PR `#3`: upgrade `actions/checkout` and `actions/setup-python` to Node 24-compatible majors
  - PR `#4`: upgrade `actions/upload-artifact` / `actions/download-artifact` to current majors
- GitHub release-side setup has advanced:
  - repository environment `pypi` now exists
  - manual `Release` workflow dry run on `main` succeeded on `2026-04-14`
  - dry-run reference: GitHub Actions run `24390507894`

## What Is Still Blocking The First Live Release

These are still the live-release blockers:

1. PyPI project setup is still incomplete.
   - During validation on `2026-04-14`, both `https://pypi.org/project/no-watermar/` and `https://pypi.org/project/no-watermark/` returned `404`.
   - Re-check ownership and naming before tagging the first public release.
2. PyPI trusted publishing still needs to be enabled.
   - The release workflow already uses OIDC-based publishing.
   - The GitHub-side `pypi` environment already exists.
   - The remaining work is on the PyPI side: create the trusted publisher entry for this repository/workflow.
   - Browser validation on `2026-04-14` in this environment reached `https://pypi.org/account/login/?next=%2Fmanage%2Faccount%2F`, so the current Chrome session is not logged into PyPI.

The repository no longer has an unresolved stable-evidence code blocker. Re-run the public synthetic fixture evidence path only if the sidecar stack or release candidate changes again before tagging.

## Recommended Next Sequence

Execute these steps in order:

1. Reconfirm the release namespace.
   - Decide whether the first public package name is `no-watermar` exactly.
   - Verify the chosen name in PyPI before tagging.
2. Configure live publishing.
   - Create the PyPI project if it does not exist yet.
   - Add the GitHub repository/workflow as a trusted publisher in PyPI.
   - GitHub-side environment setup is already done; do not repeat it unless the repository is recreated.
   - Use the exact PyPI pending-publisher values listed below; do not guess workflow identifiers.
3. Run local release preflight from a clean worktree on `main`.
   - Run the release helper and confirm disposable evidence + packaging still pass.
4. Refresh the public stable evidence only if the release candidate changed after `2026-04-14`.
   - Bootstrap sidecars.
   - Run stable smoke on the public synthetic fixture.
   - Capture and package a repeated-run stable evidence bundle again when a fresher timestamp is required.
5. Cut the first release tag.
   - Create annotated tag `v0.3.0`.
   - Push the tag.
   - Confirm `.github/workflows/release.yml` creates the GitHub Release and publishes to PyPI.

## Already Verified Outside The Repo

- `Release` workflow `workflow_dispatch` on `main` succeeded without publishing:
  - run id: `24390507894`
  - successful steps included tests, disposable evidence capture, evidence packaging, build, and `twine check`
- Repository environment `pypi` exists in GitHub and is ready for the release workflow to reference
- The current browser session in this environment is not logged into PyPI, so the remaining publisher setup cannot be completed here without an authenticated PyPI session

## Exact PyPI Trusted Publisher Values

When creating the pending publisher in PyPI, use:

- PyPI project name: `no-watermar`
- Owner: `XucroYuri`
- Repository name: `no-watermark`
- Workflow name: `release.yml`
- Environment name: `pypi`

The matching GitHub workflow is [`.github/workflows/release.yml`](../../.github/workflows/release.yml), and the publish job already targets the `pypi` environment.

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

Public synthetic fixture path used for the latest stable evidence bundle:

```powershell
python -c "from pathlib import Path; from no_watermar.disposable_benchmark_fixture import create_disposable_benchmark_fixture; create_disposable_benchmark_fixture(Path(r'.\runtime\public-stable-matrix\inputs'))"
$env:NO_WATERMAR_PADDLEOCR_PYTHON = (Resolve-Path .\.venvs\paddleocr\Scripts\python.exe).Path
$env:NO_WATERMAR_LAMA_PYTHON = (Resolve-Path .\.venvs\lama\Scripts\python.exe).Path
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = "True"
powershell -ExecutionPolicy Bypass -File .\tools\benchmark\capture-stable-baseline.ps1 -InputRoot .\runtime\public-stable-matrix\inputs -BenchmarkRoot .\runtime\public-stable-matrix\benchmarks-release -Limit 2 -Repetitions 3 -RequireLama
python .\tools\releases\package-evidence.py --evidence-root .\runtime\public-stable-matrix\benchmarks-release\evidence --output-dir .\runtime\public-stable-matrix\package --bundle-name stable-public-evidence
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
- stable evidence summary under `.\runtime\public-stable-matrix\benchmarks-release\evidence\latest.json`
- stable evidence markdown under `.\runtime\public-stable-matrix\benchmarks-release\evidence\latest.md`
- packaged stable evidence zip under `.\runtime\public-stable-matrix\package\stable-public-evidence-20260414-181705-220310.zip`

## If The Next Session Starts Cold

Use this triage order:

1. Open this file.
2. Open [release-checklist.md](./release-checklist.md).
3. Check whether PyPI project/trusted publishing has already been configured outside the repo.
4. If yes, run the release preflight and decide whether the `2026-04-14` public stable evidence bundle is still fresh enough for the release candidate.
5. If no, do not tag yet; finish PyPI/trusted-publishing setup first.
