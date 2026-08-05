# no-watermar — AI Agent Onboarding

## What It Is
Professional local-first CLI for batch watermark removal, restoration benchmarking, and repeatable review workflows over image sets.

## Stack
- Python 3.12+ (Windows-first, Linux baseline)
- OpenCV, optional OCR (PaddleOCR), optional inpainting (LaMa, diffusers, PowerPaint, BrushNet)
- Package manager: pip, optional powershell launchers

## Quick Start
```powershell
python -m pip install -e .[dev]
.\bin\no-watermar.ps1 config init --template stable-public
.\bin\no-watermar.ps1 scan show --input .\inputs
.\bin\no-watermar.ps1 batch apply --input .\inputs --output .\runtime\runs
```

## Architecture
- `src/no_watermar/` — Core package
- `bin/` — PowerShell launcher wrappers
- Provider abstraction: OCR (rule_based_roi, paddleocr), Restore (telea, lama, corner_crop, diffusers_inpaint, powerpaint, brushnet)
- Two output modes: in-place repair, direct corner cropping
- Sidecar execution for heavyweight model providers

## Key Commands
```powershell
.\bin\no-watermar.ps1 providers doctor    # Diagnosis
.\bin\no-watermar.ps1 benchmark prepare   # Benchmark setup
.\bin\no-watermar.ps1 batch resume        # Resume interrupted run
```

## Development Principles
- Core pipeline: lightweight and deterministic by default
- Heavyweight models behind provider boundaries
- Dataset-specific hints configurable, not hardcoded
- Benchmarks are local workspace artifacts, not source assets
