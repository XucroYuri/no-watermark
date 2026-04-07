# Benchmark Helpers

This directory contains local benchmark workflow helpers that sit above the Python CLI.

## Scripts

- `run-release-smoke.ps1`: execute a release-oriented smoke pass for baseline and, when available, OCR-backed benchmark runs
- `build-review-bundle.py`: assemble a local side-by-side review bundle from benchmark reports, copied artifacts, and provider outputs

## Example

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\benchmark\run-release-smoke.ps1
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

- run `benchmark.py probe-providers`
- prepare the benchmark dataset unless `-SkipPrepare` is used
- run a baseline `seed_manifest + telea` smoke benchmark
- aggregate matching baseline runs
- optionally run `paddleocr + telea` when the OCR provider is available
- optionally run `seed_manifest + lama` when the restore provider is available
- compare baseline and OCR-backed reports
- aggregate matching OCR-backed runs

After compare and aggregate outputs exist, capture a merged snapshot with:

```powershell
.\bin\no-watermar.ps1 benchmark trends --dataset regular_corner_text --baseline-mask-provider seed_manifest --baseline-restore-provider telea --candidate-mask-provider paddleocr --candidate-restore-provider telea
```

This writes JSON and Markdown summaries under `.\benchmarks\trends\`.

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

When `lama` is unavailable, the script surfaces the exact import probe failure instead of only reporting that the interpreter path exists.

When `lama` is available, the script also runs a model-backed `seed_manifest + lama` smoke benchmark and writes aggregation outputs for that provider pair.
