# Release Checklist

## Before Cut

- Confirm the target version number and release scope
- Review open items in [TODO.md](../../TODO.md) and release blockers
- Confirm there are no private assets, sample outputs, or local benchmark artifacts in the working tree
- Confirm public docs still describe the project generically

## Validation

- Run `python -m pip install -e .[dev]`
- Run `powershell -ExecutionPolicy Bypass -File .\tools\releases\build-release.ps1 -CleanDist`
- Confirm `.\runtime\release-preflight\disposable-evidence\benchmarks\evidence\latest.json` exists after the local preflight helper finishes
- Run `python -m no_watermar.cli --help`
- Run `python -m unittest discover -s tests -v`
- Run `python .\benchmark.py list-providers`
- Run `python .\benchmark.py probe-providers`
- Run `powershell -ExecutionPolicy Bypass -File .\tools\setup\validate-sidecars.ps1 -StableOnly -RunDoctor`
- Run at least one baseline smoke command against `.\inputs` or a local disposable test set
- Run `powershell -ExecutionPolicy Bypass -File .\tools\benchmark\run-release-smoke.ps1 -Limit 1`
- Run `powershell -ExecutionPolicy Bypass -File .\tools\benchmark\capture-stable-baseline.ps1 -Repetitions 3`
- Confirm stable providers report the expected `support_tier` and validated platforms through `providers list` or `providers doctor`
- Confirm `providers doctor` reports `stable_setup.release_blocking_ready = true`
- Confirm the disposable evidence bundle stays `status = ready` as a sidecar-free regression check
- Confirm `.\benchmarks\evidence\latest.json` reports `status = ready` or `status = release_blocking_ready`
- Save `.\benchmarks\evidence\latest.json` and `.\benchmarks\evidence\latest.md` as the release evidence bundle
- Confirm experimental providers are clearly marked and are not required for the default public smoke path
- Verify provider failures remain graceful when sidecar environments are absent
- If using aggregation windows for release review, capture `benchmark.py aggregate` output with provider filters
- If compare and aggregate outputs are part of the release evidence, confirm the evidence bundle links to the expected comparison and trend artifacts under `.\benchmarks\comparisons\` and `.\benchmarks\trends\`

## Docs And Metadata

- Update [CHANGELOG.md](../../CHANGELOG.md)
- Update release-facing docs if workflow, config, or compatibility changed
- Confirm [README.md](../../README.md) quick start is still valid
- Confirm release notes inputs are present for user-visible changes

## Tag Preparation

- Create a release commit with only intended source and doc changes
- Create an annotated tag matching the target version
- Draft release notes using the changelog and merged pull request summaries
- Confirm `.github/workflows/release.yml` is ready to publish from the release tag and that the `pypi` environment is configured for trusted publishing

## Post Release

- Open or update the next milestone in [ROADMAP.md](../../ROADMAP.md)
- Refresh [TODO.md](../../TODO.md) for the next iteration
- Record any known release regressions or follow-ups as issues
