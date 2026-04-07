# Versioning Policy

## Scheme

`no-watermar` uses a `MAJOR.MINOR.PATCH` version format.

## Meaning

- `MAJOR`: incompatible public interface changes after `1.0.0`
- `MINOR`: new features, new providers, new workflows, or meaningful behavior changes
- `PATCH`: bug fixes, documentation corrections, and low-risk internal improvements

## Pre-1.0 Rule

The project is currently pre-`1.0.0`. While in this stage:

- Minor releases may still contain breaking changes
- Any breaking change must be called out explicitly in [CHANGELOG.md](../../CHANGELOG.md)
- Migration notes should be added when CLI flags, config keys, or provider contracts change

## Release Expectations

- Every released version must have a changelog entry
- Public behavior changes should be reflected in [README.md](../../README.md) or the relevant document under `docs/`
- Experimental provider integrations may ship behind clear documentation labels before they are considered stable
