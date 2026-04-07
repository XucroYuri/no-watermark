# Roadmap

## Phase 1: Generic Foundation

- Stabilize the local-first batch CLI
- Keep sample-independent defaults
- Expand configuration instead of embedding dataset assumptions
- Maintain a lightweight OpenCV baseline

## Phase 2: Benchmarking and Provider Evaluation

- Benchmark OCR-backed mask generation
- Benchmark segmentation-backed mask generation
- Benchmark LaMa, BrushNet, and PowerPaint providers
- Record latency, mask quality, and artifact metrics consistently

## Phase 3: Provider Productization

- Add reusable provider configs
- Support persistent sidecar processes for lower benchmark overhead
- Add benchmark report aggregation and comparison views
- Add reproducible benchmark manifests and per-run summaries
- Add release smoke wrappers around benchmark run, compare, and aggregate flows

## Phase 4: Interactive Review Workflow

- Add a local review UI
- Add manual mask editing
- Add retry routing for failed samples
- Add side-by-side comparison and approval flow

## Phase 5: Packaging and Release

- Publish reproducible local setup instructions
- Add release packaging and tagged versions
- Provide example configs and reference datasets without private assets
- Expand CI coverage for Windows and Linux
- Formalize release checklist, versioning policy, and release note inputs
