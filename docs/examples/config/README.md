# Config Examples

These example bundles provide starting points for common OCR watermark keyword presets.

Files:

- `brand-social.toml`: brand names, social handles, and studio tags
- `stock-marketplaces.toml`: stock-site domains and licensing phrases
- `mixed-corner-text.toml`: mixed marketplace plus generic rights text
- `benchmark-local-profiles.toml`: shared dataset profiles plus benchmark provider profiles for local smoke and evaluation runs, including an experimental `diffusers_inpaint` example

Copy one of these files into a local `no-watermar.toml` and adjust the tokens for your dataset.

For the benchmark profile example, you can drive the CLI with:

```powershell
.\bin\no-watermar.ps1 scan show --dataset-profile local_smoke
.\bin\no-watermar.ps1 batch plan --dataset-profile local_smoke --output .\runtime\runs
.\bin\no-watermar.ps1 benchmark prepare --dataset-profile local_smoke --benchmark-root .\benchmarks
.\bin\no-watermar.ps1 benchmark run --dataset-profile local_smoke --provider-profile seed_telea --benchmark-root .\benchmarks
.\bin\no-watermar.ps1 benchmark run --dataset-profile local_smoke --provider-profile ocr_diffusers --benchmark-root .\benchmarks
```

You can also generate the built-in versions directly:

```powershell
.\bin\no-watermar.ps1 config init --template default
.\bin\no-watermar.ps1 config init --template brand-social
.\bin\no-watermar.ps1 config init --template stock-marketplaces
.\bin\no-watermar.ps1 config init --template mixed-corner-text
```
