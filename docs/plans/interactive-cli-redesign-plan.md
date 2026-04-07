# Interactive CLI Redesign Plan

## Purpose

This plan defines the next CLI iteration for `no-watermar`.

The target is a more standard open-source command-line application that is:

- interactive for human operators
- scriptable for automation and agent use
- explicit about permissions and write boundaries
- organized around a root command with subcommands

This is a design and delivery plan. It does not assume the refactor is already implemented.

## Why This Iteration Exists

The repository already has working baseline and benchmark entrypoints, but the current surface is still closer to MVP tooling than to a polished open-source CLI.

Current gaps:

- the main runtime and benchmark flows are split across `run.py`, `benchmark.py`, and module-local parsers
- the CLI is functional but not yet organized as a standard root-command plus subcommand tree
- input and output decisions are still mostly argument-driven, with limited interactive guidance
- user confirmation is not first-class for write operations
- agent-friendly orchestration exists in practice, but not yet as a deliberately designed interface contract
- capability and permission boundaries are implicit instead of documented and enforced

## Design Goals

### 1. One Root CLI

The long-term public interface should converge on one root command:

```text
no-watermar <group> <subcommand> [options]
```

Examples:

```powershell
no-watermar scan run --input .\inputs
no-watermar batch plan --input .\inputs
no-watermar batch apply --plan .\runtime\plans\latest.json
no-watermar benchmark run --dataset regular_corner_text --mask-provider paddleocr --restore-provider telea
no-watermar providers probe
```

### 2. Human And Agent Modes

The same CLI should support both:

- human-driven interactive use with prompts and confirmation
- agent-driven scripted use with stable flags, machine-readable output, and no unexpected prompts

### 3. Clear Capability Boundaries

Every command should make its read/write scope obvious.

The CLI should separate:

- read-only inspection
- write-safe generation of new artifacts
- overwrite or destructive actions that require explicit confirmation

### 4. Stable Output Contracts

Each subcommand should be able to emit:

- readable console output for humans
- JSON summaries for automation and agents

### 5. Compatibility During Migration

Existing entrypoints should not break immediately.

`run.py` and `benchmark.py` should remain as compatibility shims for at least one transition phase while they delegate to the new root CLI.

## Proposed CLI Shape

## Root Command

```text
no-watermar
```

## Command Groups

### `scan`

Read-only or low-risk discovery commands.

Subcommands:

- `scan run`
- `scan show`

Responsibilities:

- inspect the input tree
- classify candidate images
- summarize what would be processed
- generate scan manifests

### `batch`

Main batch processing workflow.

Subcommands:

- `batch plan`
- `batch apply`
- `batch resume`
- `batch report`

Responsibilities:

- build an execution plan from inputs and config
- request confirmation before writes
- run masks, overlays, and restores
- resume or summarize prior runs

### `benchmark`

Benchmark workflow under one command group.

Subcommands:

- `benchmark prepare`
- `benchmark run`
- `benchmark compare`
- `benchmark aggregate`
- `benchmark probe`

Responsibilities:

- prepare benchmark datasets
- run provider combinations
- compare reports
- aggregate history
- probe provider runtimes

### `providers`

Provider and sidecar visibility.

Subcommands:

- `providers list`
- `providers probe`
- `providers doctor`

Responsibilities:

- list implemented and planned providers
- report runtime availability
- diagnose missing modules, interpreters, or environment mismatches

### `config`

Configuration management.

Subcommands:

- `config init`
- `config show`
- `config validate`

Responsibilities:

- bootstrap local config files
- show effective config
- validate paths and provider settings

### `review`

Future review and approval workflow.

Subcommands:

- `review list`
- `review show`
- `review approve`
- `review retry`

Responsibilities:

- inspect low-confidence or failed outputs
- approve, reject, or retry selected items

## Standard Entry Layout

## Target Repository Layout

```text
bin/
  no-watermar
  no-watermar.cmd
  no-watermar.ps1
src/no_watermar/cli/
  app.py
  context.py
  output.py
  confirm.py
  commands/
    scan.py
    batch.py
    benchmark.py
    providers.py
    config.py
    review.py
run.py
benchmark.py
```

## Entry Strategy

- `bin/` holds user-facing launcher wrappers for local checkout usage
- `pyproject.toml` continues to expose installable console scripts
- `run.py` becomes a thin compatibility wrapper for `no-watermar batch apply`
- `benchmark.py` becomes a thin compatibility wrapper for `no-watermar benchmark`

## Input Redesign

## Current Problem

Input selection is mostly flag-based and assumes the caller already knows the correct root, recursion mode, and downstream output intent.

## Target Model

The input side should be explicit and inspectable before mutation.

### Input Sources

Support these input forms:

- directory root
- prior scan manifest
- benchmark dataset id
- explicit file list in a manifest

### Input Commands

`scan run` and `batch plan` should be able to:

- show discovered item counts
- show include or exclude effects
- show category breakdown
- preview skipped paths

### Input Config

Promote stable input settings into config:

- default input root
- recursion behavior
- include glob list
- exclude glob list
- maximum item count
- provider defaults

### Input Validation

Before planning or applying work, validate:

- input path exists
- output path is not nested inside the selected input path in a risky way
- workspace-local generated directories are excluded from scans

## Output Redesign

## Current Problem

Outputs are useful but still oriented around generated artifacts rather than user-visible run contracts.

## Target Model

Every write-producing command should generate a stable run directory with machine-readable metadata.

### Output Structure

```text
runtime/
  plans/
    latest.json
    <plan-id>.json
  runs/
    <run-id>/
      summary.json
      manifest.json
      logs/
      scans/
      masks/
      overlays/
      restored/
      reports/
```

### Output Contract

Every write-producing operation should emit:

- a human-readable terminal summary
- a JSON summary path
- a stable run or plan id

### Output Modes

Support:

- default console mode
- `--json`
- `--quiet`

For agent orchestration, `--json` should always produce a stable summary schema.

## Confirmation Model

## Human Confirmation

Interactive commands should pause before a write-producing step and show:

- input root
- output root
- item count
- provider combination
- overwrite policy
- expected artifact categories

Example flow:

1. `no-watermar batch plan --input .\inputs`
2. review plan summary
3. confirm `yes` or `no`
4. execute or abort

## Agent-Friendly Confirmation

Agents should not be forced through TTY prompts.

Support these flags:

- `--yes` to accept the generated plan
- `--no-input` to disable prompts
- `--plan <path>` to apply a previously generated plan
- `--json` to return machine-readable summaries

Recommended agent flow:

1. `no-watermar batch plan --input .\inputs --json`
2. inspect returned plan id and summary
3. `no-watermar batch apply --plan <plan.json> --yes --no-input --json`

## Capability And Permission Boundaries

The CLI should make permission boundaries explicit.

## Capability Classes

### Read-Only

Commands:

- `scan show`
- `providers list`
- `providers probe`
- `benchmark compare`
- `benchmark aggregate`

Properties:

- no writes to source inputs
- may write only optional derived reports when explicitly asked

### Write-Safe

Commands:

- `scan run`
- `batch plan`
- `benchmark prepare`
- `benchmark run`

Properties:

- writes only under declared output roots
- never mutates source input files
- creates new artifacts by default

### Explicit Overwrite

Commands:

- `batch apply`
- `review retry`

Properties:

- may overwrite prior generated artifacts
- requires `--yes` or interactive confirmation
- should reject ambiguous output roots

## Hard Safety Rules

The redesigned CLI should enforce:

- never modify source input files in place
- never delete source inputs
- never write outside explicit output roots
- never trigger model downloads silently during normal batch execution
- never require network access for standard local runs unless the user explicitly opts in

## Agent Contract

To make agent orchestration reliable, commands should provide:

- deterministic exit codes
- `--json` summaries
- `--no-input` behavior with no hidden prompts
- stable subcommand names
- explicit output paths in results

Recommended JSON fields:

- `command`
- `mode`
- `status`
- `plan_id`
- `run_id`
- `input_root`
- `output_root`
- `item_count`
- `warnings`
- `report_paths`

## Delivery Phases

## Phase A: CLI Skeleton

Deliver:

- root command package under `src/no_watermar/cli/`
- grouped subcommands
- compatibility shims for `run.py` and `benchmark.py`

Acceptance:

- current baseline flows remain callable
- `no-watermar` root command exists

## Phase B: Plan And Confirm

Deliver:

- `batch plan`
- `batch apply`
- interactive confirmation helper
- `--yes`, `--no-input`, and `--json`

Acceptance:

- human mode can review before applying
- agent mode can execute without prompts

## Phase C: Input And Output Contracts

Deliver:

- plan artifacts under `runtime/plans/`
- standardized run summary schema
- improved output path validation

Acceptance:

- write-producing commands always emit plan or run metadata

## Phase D: Provider And Benchmark Alignment

Deliver:

- `providers list/probe/doctor`
- `benchmark` group folded fully under root CLI
- compatible machine-readable benchmark summaries

Acceptance:

- current benchmark features remain available under the root command

## Phase E: Documentation And Release Migration

Deliver:

- README refresh
- setup and development guide refresh
- deprecation notice for old entrypoints

Acceptance:

- a new contributor can use the root CLI without relying on legacy wrappers

## Testing Plan

The redesign should add tests for:

- subcommand parsing
- confirmation behavior
- `--yes` and `--no-input`
- output schema stability
- safety checks for invalid output roots
- compatibility wrapper behavior for `run.py` and `benchmark.py`

## Documentation Impact

When implementation starts, update:

- `README.md`
- `docs/ARCHITECTURE.md`
- `docs/DEVELOPMENT.md`
- `docs/setup/README.md`
- `docs/setup/windows-local-setup.md`
- `docs/PROGRESS.md`
- `TODO.md`
- `CHANGELOG.md`

## Immediate Follow-Up After This Plan

The first implementation slice should be:

1. introduce the root CLI package and `bin/` wrappers
2. move benchmark functionality under grouped subcommands
3. add `batch plan` and `batch apply` with confirmation
4. preserve `run.py` and `benchmark.py` as compatibility shims

That slice delivers the command architecture without forcing the whole processing engine to change at once.
