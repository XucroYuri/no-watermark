# Workspace Layout

## Repository-Managed Paths

- `src/no_watermar/`: core package
- `tests/`: unit tests
- `docs/`: project documentation
- `tools/sidecars/`: provider sidecar entrypoints
- `tools/setup/`: setup and validation helpers

## Local-Only Paths

- `inputs/`: private input images
- `runtime/`: batch outputs
- `benchmarks/`: benchmark manifests and reports
- `.venvs/`: local provider environments
- `models/`, `checkpoints/`, `weights/`: local model assets

## Working Rule

Anything tied to one private dataset, one local benchmark run, or one machine-specific model environment should stay out of source control unless it is expressed as generic documentation or reusable code.
