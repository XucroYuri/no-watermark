# Release Process

## 1. Freeze Scope

- Decide the target version
- Limit the release to reviewed changes
- Move unfinished work out of the release milestone

## 2. Validate

- Run the release checklist in [release-checklist.md](./release-checklist.md)
- Run `powershell -ExecutionPolicy Bypass -File .\tools\releases\build-release.ps1 -CleanDist` for the local packaging preflight
- Confirm documentation and changelog updates are complete
- Confirm no local-only artifacts are staged
- Keep the disposable evidence bundle from `.\runtime\release-preflight\disposable-evidence\benchmarks\evidence\` as the sidecar-free regression proof for the release candidate
- Save `.\benchmarks\evidence\latest.json` and `.\benchmarks\evidence\latest.md` from `capture-stable-baseline.ps1` as the release evidence bundle
- Save any additional filtered aggregation summaries only when they materially change the release decision beyond the stable evidence bundle

## 3. Prepare The Commit

- Review `git status`
- Stage only intended project files
- Create a release commit with a version-oriented message such as `release: v0.3.0`

## 4. Tag The Release

- Create an annotated tag such as `v0.3.0`
- Draft release notes from [CHANGELOG.md](../../CHANGELOG.md) and merged changes
- Push the release tag so `.github/workflows/release.yml` can build artifacts, create the GitHub Release, and publish to PyPI through trusted publishing

## 5. Automated Publishing Notes

- `workflow_dispatch` on `.github/workflows/release.yml` is a dry-run build path for checking packaging and release assets without publishing
- Tag pushes matching `v*` run the full build, attach artifacts to a GitHub Release, and publish to PyPI
- The PyPI publish job expects repository-side trusted publishing configuration and the `pypi` GitHub environment to be enabled before the first live release

## 6. Post-Release Follow-Up

- Advance `TODO.md` and `ROADMAP.md`
- Open follow-up issues for deferred items
- Start the next development iteration on top of the tagged state
