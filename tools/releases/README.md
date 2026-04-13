# Release Helpers

This directory contains local release build helpers that mirror the public packaging and release workflow.

## Scripts

- `build-release.ps1`: install the editable package with release tooling, run the CLI smoke check, run tests, build wheel and sdist artifacts, and run `twine check`

## Example

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\releases\build-release.ps1 -CleanDist
```

This helper is the local counterpart to:

- the `package` job in `.github/workflows/ci.yml`
- the tag-driven release workflow in `.github/workflows/release.yml`

Use it before cutting a release commit or when validating packaging changes locally.
