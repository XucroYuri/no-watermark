# TODO

## Now

- Prepare a `0.3.0` public release readiness pass
- Turn the current single-run real-local benchmark snapshot into a repeated-run baseline
- Review the new 10-image `ocr_diffusers` versus `ocr_telea` outputs side by side and reconcile human judgment with the current metrics
- Review the latest persistent 2-image `ocr_powerpaint_v21` versus `ocr_telea` outputs side by side and reconcile human judgment with the current metrics
- Use the new shortlist review bundle to review the current 2-image `brushnet` tuning variants against `ocr_telea` and the original `ocr_brushnet` baseline
- Reduce persistent `paddleocr` mask latency on real local slices
- Reduce `diffusers_inpaint` restore latency on real local slices without regressing repair detail
- Reduce `powerpaint_v2_1` first-image warmup and end-to-end latency now that the persistent restore session is in place
- Reduce OCR residual scoring latency on real local benchmark slices
- Decide whether any tuned `brushnet` variant is subjectively strong enough to justify a wider slice despite the current metric regressions
- Run a slightly larger persistent `ocr_powerpaint_v21` slice only if the human review still suggests it can beat `ocr_telea`
- Turn the current `lama`, `diffusers_inpaint`, `powerpaint_v2_1`, and `brushnet` sidecar recipes into documented reproducible setup paths
- Compare `seed_manifest + lama` against `seed_manifest + telea` on larger local sample slices
- Validate the documented sidecar compatibility matrix on at least one more local environment

## Next

- Integrate `EdgeSAM` as a mask provider
- Integrate `watermark-segmentation` as a mask provider
- Integrate `LaMa` with a persistent sidecar option
- Validate a second repository-local Python and checkpoint matrix for `brushnet`
- Validate a second repository-local Python and torch matrix for `diffusers_inpaint`
- Validate a second repository-local Python and checkpoint matrix for `powerpaint_v2_1`
- Add Windows-focused release smoke tests

## Frozen

- Keep the repo-local `ocr_fluxfill_fp8` profile frozen on the current 16 GB local GPU until a larger-VRAM host or a validated low-memory smoke recipe exists

## Later

- Add review UI for failed or low-confidence samples
- Add per-category routing and config presets for scan and batch flows
- Benchmark the composition-loss tradeoffs of `corner_crop` against repair-based restore providers on larger real-local slices
- Add automatic prompt generation for cover-heavy scenarios
- Expand persisted run state so interrupted provider steps can be resumed from finer-grained provider checkpoints

## Backlog

- ONNX export and inference experiments
- Cross-image patch retrieval for repeated backgrounds
- Better cover classification and heavy watermark routing
- Performance dashboards for provider comparisons
