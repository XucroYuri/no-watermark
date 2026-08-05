# no-watermar - Batch Watermark Removal CLI
<!-- MANUAL: -->
## Purpose
Professional local-first CLI for batch watermark removal, restoration benchmarking, and repeatable review workflows over image sets.

## Key Files
| File | Role |
|------|------|
| `README.md` | Full documentation and quick start |
| `src/no_watermar/` | Core Python package |
| `bin/no-watermar.ps1` | PowerShell launcher |
| `tools/sidecars/` | Model sidecar entrypoints |
| `tools/benchmark/` | Benchmark and release smoke scripts |
| `docs/` | Architecture, development guide, configuration |

## Subdirectories
| Dir | Purpose |
|-----|---------|
| `src/no_watermar/` | Core package: CLI, providers, batch, benchmark |
| `bin/` | PowerShell launcher wrappers |
| `tools/sidecars/` | PaddleOCR, LaMa, diffusers, PowerPaint, BrushNet sidecars |
| `tools/benchmark/` | Release smoke, evidence capture |
| `tools/setup/` | Bootstrap and validation scripts |
| `docs/` | Architecture, configuration, development guide |
| `tests/` | Unit tests |

## For AI Agents
- Windows-first, Linux baseline
- Provider abstraction: OCR (rule_based_roi, paddleocr), Restore (telea, lama, corner_crop)
- Two output modes: in-place repair, direct corner crop
- Sidecar execution for heavyweight providers
- Run: `.\bin\no-watermar.ps1 providers doctor` for diagnosis
- Stable smoke path: `python -m pip install -e .[dev]`

## Dependencies
- Python 3.12+, OpenCV, PaddleOCR (optional), LaMa (optional)
