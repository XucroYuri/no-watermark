# Benchmark Helpers

This directory contains local benchmark workflow helpers that sit above the Python CLI.

## Scripts

- `run-release-smoke.ps1`: execute a release-oriented smoke pass for the baseline, the release-blocking OCR-backed path, and optionally the stable `lama` restore path
- `capture-stable-baseline.ps1`: repeat the release smoke flow and build one JSON/Markdown evidence bundle for the stable provider matrix
- `capture-disposable-evidence.py`: generate a repo-native synthetic corpus and capture a repeated stable evidence bundle without private inputs or sidecars
- `build-review-bundle.py`: assemble a local side-by-side review bundle from benchmark reports, copied artifacts, and provider outputs

## Example

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\benchmark\run-release-smoke.ps1
```

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\benchmark\capture-stable-baseline.ps1 -Repetitions 3
```

```bash
python .\tools\benchmark\capture-disposable-evidence.py --clean
```

```powershell
python .\tools\benchmark\build-review-bundle.py `
  --report .\benchmarks\path\telea.json `
  --label telea `
  --report .\benchmarks\path\powerpaint.json `
  --label powerpaint_v2_1 `
  --report .\benchmarks\path\brushnet.json `
  --label brushnet_variant_a `
  --compare .\benchmarks\path\telea-vs-brushnet.json `
  --trend .\benchmarks\path\latest.md `
  --output .\benchmarks\review\slice2
```

The script will:

- run `python -m no_watermar.cli providers doctor`
- run `benchmark.py probe-providers`
- prepare the benchmark dataset unless `-SkipPrepare` is used
- run a baseline `seed_manifest + telea` smoke benchmark
- aggregate matching baseline runs
- require `paddleocr + telea` unless `-SkipOcr` is used
- optionally run `seed_manifest + lama` when the restore provider is available
- compare baseline and OCR-backed reports
- aggregate matching OCR-backed runs

After compare and aggregate outputs exist, capture a merged snapshot with:

```powershell
.\bin\no-watermar.ps1 benchmark trends --dataset regular_corner_text --baseline-mask-provider seed_manifest --baseline-restore-provider telea --candidate-mask-provider paddleocr --candidate-restore-provider telea
```

This writes JSON and Markdown summaries under `.\benchmarks\trends\`.

To capture repeated stable evidence in one pass, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\benchmark\capture-stable-baseline.ps1 -Repetitions 3
```

That wrapper:

- repeats `run-release-smoke.ps1` for the requested run count
- runs `python -m no_watermar.cli benchmark evidence`
- writes versioned and `latest.*` evidence files under `.\benchmarks\evidence\`
- treats `seed_manifest + telea` as the baseline, `paddleocr + telea` as the release-blocking candidate, and `seed_manifest + lama` as the optional stable path by default

Useful flags for `capture-stable-baseline.ps1`:

- `-Repetitions 3`
- `-MinimumRunCount 3`
- `-RequireLama`
- `-SkipSmoke`
- `-SkipOptionalEvidence`
- `-CandidateMaskProvider seed_manifest -CandidateRestoreProvider noop`
- `-OptionalMaskProvider seed_manifest -OptionalRestoreProvider corner_crop`

Use the provider override flags when you want to regenerate evidence from an existing benchmark root or exercise the wrapper against synthetic/local regression fixtures without requiring the default OCR-backed stable path.

For automation-only evidence that must stay redistributable and sidecar-free, use:

```bash
python .\tools\benchmark\capture-disposable-evidence.py --clean
```

That helper:

- writes a repo-native synthetic corpus under `.\runtime\disposable-evidence\inputs\`
- prepares both `regular_corner_text` and `cover_heavy` dataset manifests
- repeats lightweight `seed_manifest + telea`, `seed_manifest + noop`, and `seed_manifest + corner_crop` benchmark runs
- writes a ready-state evidence bundle under `.\runtime\disposable-evidence\benchmarks\evidence\`

Use this disposable path for CI and packaging preflight. Use `capture-stable-baseline.ps1` when you need the real stable provider matrix with `paddleocr` and optional `lama`.

`build-review-bundle.py` writes:

- `review.json` with provider summaries and copied artifact paths
- `README.md` with a quick human-review checklist
- `items/<item-id>/source/` for the original input plus seed artifacts
- `items/<item-id>/<provider>/` for each provider's mask, overlay, and restored image
- `artifacts/` copies of the selected reports, comparisons, and trends

Use `--label` whenever several reports share the same provider name so the review bundle can keep each variant distinct.

If you already keep local dataset and provider bundles in `no-watermar.toml`, the same flow can be driven with profiles:

```powershell
.\bin\no-watermar.ps1 benchmark prepare --dataset-profile local_smoke --benchmark-root .\benchmarks
.\bin\no-watermar.ps1 benchmark run --dataset-profile local_smoke --provider-profile seed_telea --benchmark-root .\benchmarks
.\bin\no-watermar.ps1 benchmark trends --dataset-profile local_smoke --baseline-provider-profile seed_telea --candidate-provider-profile ocr_telea --benchmark-root .\benchmarks
```

The same profile-driven flow now covers the first FP8-storage attempt as well:

```powershell
.\bin\no-watermar.ps1 benchmark run --dataset-profile local_smoke --provider-profile ocr_fluxfill_fp8 --benchmark-root .\benchmarks
```

For `powerpaint_v2_1`, keep the restore-side execution mode inside the provider profile under `restore_options.session_mode`. The current validated persistent path uses `session_mode = "auto"` so later images can reuse the long-lived restore sidecar without making the whole run fail if startup falls back to one-shot.

Useful flags:

- `-InputRoot .\inputs`
- `-BenchmarkRoot .\benchmarks`
- `-Dataset regular_corner_text`
- `-Limit 2`
- `-OcrSessionMode auto`
- `-SkipPrepare`
- `-SkipOcr`
- `-RequireLama`

When the release-blocking stable setup is incomplete, the script fails fast with the `stable_setup.blocking_issues` reported by `providers doctor`.

When `lama` is unavailable, the script surfaces the exact import probe failure instead of only reporting that the interpreter path exists. Add `-RequireLama` when the optional stable model-backed restore path must also pass.

When `lama` is available, the script also runs a model-backed `seed_manifest + lama` smoke benchmark and writes aggregation outputs for that provider pair.
