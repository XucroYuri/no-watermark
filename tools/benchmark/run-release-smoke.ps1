[CmdletBinding()]
param(
    [string]$InputRoot = ".\inputs",
    [string]$BenchmarkRoot = ".\benchmarks",
    [string]$Dataset = "regular_corner_text",
    [int]$Limit = 2,
    [string]$OcrSessionMode = "auto",
    [switch]$SkipPrepare,
    [switch]$SkipOcr,
    [switch]$RequireLama
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptDir "..\..")).Path
$benchmarkPy = Join-Path $repoRoot "benchmark.py"

function Invoke-CliJson {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $output = & python -m no_watermar.cli @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "no_watermar.cli failed: $($Arguments -join ' ')"
    }

    return $output | ConvertFrom-Json
}

function Invoke-BenchmarkJson {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $output = & python $benchmarkPy @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "benchmark.py failed: $($Arguments -join ' ')"
    }

    return $output | ConvertFrom-Json
}

Write-Host "Repository root: $repoRoot"
Write-Host "Benchmark root: $BenchmarkRoot"

$doctor = Invoke-CliJson -Arguments @("providers", "doctor")
$providers = Invoke-BenchmarkJson -Arguments @("probe-providers")

if (-not $SkipOcr) {
    $stableSetup = $doctor.stable_setup
    if (-not $stableSetup.release_blocking_ready) {
        $details = @(
            @($stableSetup.blocking_issues) | ForEach-Object {
                "$($_.provider_name) [$($_.issue_code)] $($_.detail)"
            }
        ) -join "; "
        throw "Release-blocking stable providers are not ready for smoke validation. $details"
    }
}

if (-not $SkipPrepare) {
    Write-Host "Preparing benchmark dataset..."
    $null = Invoke-BenchmarkJson -Arguments @(
        "prepare",
        "--input", $InputRoot,
        "--benchmark-root", $BenchmarkRoot,
        "--limit", "$Limit"
    )
}

Write-Host "Running baseline seed_manifest + telea smoke benchmark..."
$baseline = Invoke-BenchmarkJson -Arguments @(
    "run",
    "--benchmark-root", $BenchmarkRoot,
    "--dataset", $Dataset,
    "--mask-provider", "seed_manifest",
    "--restore-provider", "telea",
    "--limit", "$Limit",
    "--ocr-session-mode", $OcrSessionMode
)

$baselineReport = $baseline.dataset_summaries[0].report_json

Write-Host "Aggregating baseline runs for dataset/provider pair..."
$baselineAggregate = Invoke-BenchmarkJson -Arguments @(
    "aggregate",
    "--reports-root", (Join-Path $BenchmarkRoot "runs"),
    "--dataset", $Dataset,
    "--mask-provider", "seed_manifest",
    "--restore-provider", "telea"
)

$result = [ordered]@{
    baseline_run_id = $baseline.run_id
    baseline_report = $baselineReport
    baseline_aggregate_json = $baselineAggregate.output_json
    baseline_aggregate_csv = $baselineAggregate.output_csv
}

if (-not $SkipOcr) {
    $paddleProvider = $providers.mask_providers | Where-Object { $_.name -eq "paddleocr" } | Select-Object -First 1
    if ($paddleProvider -and $paddleProvider.runtime_available) {
        Write-Host "Running paddleocr + telea smoke benchmark..."
        $ocrRun = Invoke-BenchmarkJson -Arguments @(
            "run",
            "--benchmark-root", $BenchmarkRoot,
            "--dataset", $Dataset,
            "--mask-provider", "paddleocr",
            "--restore-provider", "telea",
            "--limit", "$Limit",
            "--ocr-session-mode", $OcrSessionMode
        )

        $ocrReport = $ocrRun.dataset_summaries[0].report_json
        Write-Host "Comparing baseline vs OCR-backed smoke benchmark..."
        $comparison = Invoke-BenchmarkJson -Arguments @(
            "compare",
            "--baseline-report", $baselineReport,
            "--candidate-report", $ocrReport
        )

        $ocrAggregate = Invoke-BenchmarkJson -Arguments @(
            "aggregate",
            "--reports-root", (Join-Path $BenchmarkRoot "runs"),
            "--dataset", $Dataset,
            "--mask-provider", "paddleocr",
            "--restore-provider", "telea"
        )

        $result["ocr_run_id"] = $ocrRun.run_id
        $result["ocr_report"] = $ocrReport
        $result["comparison_json"] = $comparison.output_json
        $result["comparison_csv"] = $comparison.output_csv
        $result["ocr_aggregate_json"] = $ocrAggregate.output_json
        $result["ocr_aggregate_csv"] = $ocrAggregate.output_csv
    }
    else {
        throw "paddleocr runtime is unavailable; stable release smoke requires the OCR-backed path unless -SkipOcr is set."
    }
}

$lamaProvider = $providers.restore_providers | Where-Object { $_.name -eq "lama" } | Select-Object -First 1
if ($lamaProvider -and $lamaProvider.runtime_available) {
    Write-Host "Running model-backed seed_manifest + lama smoke benchmark..."
    $lamaRun = Invoke-BenchmarkJson -Arguments @(
        "run",
        "--benchmark-root", $BenchmarkRoot,
        "--dataset", $Dataset,
        "--mask-provider", "seed_manifest",
        "--restore-provider", "lama",
        "--limit", "$Limit",
        "--ocr-session-mode", $OcrSessionMode
    )

    $lamaReport = $lamaRun.dataset_summaries[0].report_json
    $lamaAggregate = Invoke-BenchmarkJson -Arguments @(
        "aggregate",
        "--reports-root", (Join-Path $BenchmarkRoot "runs"),
        "--dataset", $Dataset,
        "--mask-provider", "seed_manifest",
        "--restore-provider", "lama"
    )

    $result["lama_run_id"] = $lamaRun.run_id
    $result["lama_report"] = $lamaReport
    $result["lama_aggregate_json"] = $lamaAggregate.output_json
    $result["lama_aggregate_csv"] = $lamaAggregate.output_csv
}
elseif ($lamaProvider) {
    if ($RequireLama) {
        throw "lama runtime is unavailable; -RequireLama makes the stable model-backed restore path mandatory. $($lamaProvider.runtime_note)"
    }
    Write-Warning "lama runtime is unavailable; skipping model-backed smoke run. $($lamaProvider.runtime_note)"
}

$result | ConvertTo-Json -Depth 8
