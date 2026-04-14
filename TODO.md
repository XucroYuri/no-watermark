# TODO

## Now

- Cut the first `0.3.0` public release from the merged public-CLI baseline
- Finish Windows-first packaging for `pip install no-watermar`
- Keep the stable provider matrix release-blocking and the experimental matrix opt-in
- Expand CI and release smoke coverage for the stable CLI path on Windows and Linux
- Capture the first release-blocking repeated-run stable evidence bundle on the real `paddleocr + telea` path and archive it alongside the disposable automation bundle
- Create the `no-watermar` PyPI project and enable GitHub/PyPI trusted publishing so the tag-driven release workflow can go live

## Next

- Integrate `EdgeSAM` as a mask provider
- Integrate `watermark-segmentation` as a mask provider
- Promote one experimental restore provider only after it clears repeated benchmark plus human review gates
- Integrate `LaMa` with a persistent sidecar option
- Validate a second repository-local Python and checkpoint matrix for `brushnet`
- Validate a second repository-local Python and torch matrix for `diffusers_inpaint`
- Validate a second repository-local Python and checkpoint matrix for `powerpaint_v2_1`

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
