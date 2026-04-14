# Public Release Readiness

## Target

Align the repository for its first broadly shareable public milestone, `0.3.0`.

## Required Conditions

- Generic project positioning with no private sample references
- Stable baseline CLI and benchmark scaffold
- Contributor-facing documentation complete enough for external onboarding
- Release checklist and versioning policy in place
- CI passing on the supported baseline Python versions and platforms
- Stable provider tiers clearly separated from experimental ones in the public CLI and docs

## Recommended Conditions

- At least one documented OCR-backed provider flow
- At least one documented model-backed restore flow
- Windows setup guidance for sidecar environments
- Example configuration patterns for common watermark layouts
- A redistributable disposable benchmark corpus for automation-only regression evidence

## Open Gaps

- Stable sidecar reproducibility is now validated on the Windows-first public fixture path, but Linux parity and broader cross-machine packaging still lag that path
- Benchmark reporting still needs stronger quality metrics beyond the current repeated-run evidence bundles
- The `no-watermar` PyPI project and pending trusted publisher still need to be created before the first live automated release
- Repository-side trusted publishing still needs to be enabled in GitHub and PyPI before the first live automated release

## Exit Definition

The project is ready for a public milestone when a new contributor can clone the repository, run the baseline pipeline locally, understand the provider model, and follow the documented release process without access to any private sample workspace.
