# Public Release Readiness

## Target

Prepare the repository for a first broadly shareable public milestone, tentatively `0.3.0`.

## Required Conditions

- Generic project positioning with no private sample references
- Stable baseline CLI and benchmark scaffold
- Contributor-facing documentation complete enough for external onboarding
- Release checklist and versioning policy in place
- CI passing on the supported baseline Python versions

## Recommended Conditions

- At least one documented OCR-backed provider flow
- At least one documented model-backed restore flow
- Windows setup guidance for sidecar environments
- Example configuration patterns for common watermark layouts

## Open Gaps

- Provider environment reproducibility is not yet packaged
- Benchmark reporting still needs stronger quality metrics
- Release smoke tests now cover baseline, OCR-backed, and `lama`-backed benchmark paths, but only for the currently validated local sidecar recipe
- Packaging and distribution artifacts are not yet automated

## Exit Definition

The project is ready for a public milestone when a new contributor can clone the repository, run the baseline pipeline locally, understand the provider model, and follow the documented release process without access to any private sample workspace.
