# Phase 04: Review Workflow

## Goal

Add a lightweight local review loop so failed, low-confidence, or visually risky outputs can be inspected and rerouted efficiently.

## Deliverables

- Review queue manifest
- Failed-sample routing rules
- Manual mask override support
- Side-by-side comparison output
- Approval and retry status tracking

## TODO Breakdown

- Define low-confidence heuristics for automatic queueing
- Persist per-image review state without touching source inputs
- Add a mask import path for edited masks
- Add before/after preview generation for review batches
- Add retry routing to alternate provider stacks

## Exit Criteria

- Review queue can be generated from runtime or benchmark outputs
- A reviewer can mark items as approved, retry, or rejected
- Retry jobs can reuse the existing benchmark/provider abstraction
