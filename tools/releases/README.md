# Release Helpers

This directory contains local release build helpers that mirror the public packaging and release workflow.

## Scripts

- `build-release.ps1`: install the editable package with release tooling, run the CLI smoke check, run tests, capture disposable stable evidence, package that evidence as a zip bundle, build wheel and sdist artifacts, and run `twine check`
- `package-evidence.py`: package an evidence directory into a release-friendly zip bundle with `latest.json`, `latest.md`, and linked benchmark artifacts

## Example

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\releases\build-release.ps1 -CleanDist
```

This helper is the local counterpart to:

- the `package` job in `.github/workflows/ci.yml`
- the disposable-evidence automation in `.github/workflows/ci.yml`
- the tag-driven release workflow in `.github/workflows/release.yml`

Use it before cutting a release commit or when validating packaging changes locally. Pass `-SkipDisposableEvidence` only when you intentionally want to skip the repo-native synthetic evidence bundle.

After the helper runs, the disposable release-proof zip lives under `.\runtime\release-preflight\disposable-evidence\package\`.
