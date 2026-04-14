[CmdletBinding()]
param(
    [string]$InputRoot = ".\inputs",
    [string]$BenchmarkRoot = ".\benchmarks",
    [string]$Dataset = "regular_corner_text",
    [int]$Limit = 2,
    [int]$Repetitions = 3,
    [string]$OcrSessionMode = "auto",
    [int]$MinimumRunCount = 0,
    [string]$BaselineMaskProvider = "seed_manifest",
    [string]$BaselineRestoreProvider = "telea",
    [string]$CandidateMaskProvider = "paddleocr",
    [string]$CandidateRestoreProvider = "telea",
    [string]$OptionalMaskProvider = "seed_manifest",
    [string]$OptionalRestoreProvider = "lama",
    [switch]$SkipPrepare,
    [switch]$SkipSmoke,
    [switch]$RequireLama,
    [switch]$SkipOptionalEvidence
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptDir "..\..")).Path
$smokeScript = Join-Path $scriptDir "run-release-smoke.ps1"
$requiredRunCount = if ($MinimumRunCount -gt 0) { $MinimumRunCount } else { $Repetitions }

if (-not $SkipSmoke) {
    for ($index = 0; $index -lt $Repetitions; $index++) {
        Write-Host "Stable smoke iteration $($index + 1) / $Repetitions"
        if ($SkipPrepare -or $index -gt 0) {
            if ($RequireLama) {
                & $smokeScript -InputRoot $InputRoot -BenchmarkRoot $BenchmarkRoot -Dataset $Dataset -Limit $Limit -OcrSessionMode $OcrSessionMode -SkipPrepare -RequireLama
            }
            else {
                & $smokeScript -InputRoot $InputRoot -BenchmarkRoot $BenchmarkRoot -Dataset $Dataset -Limit $Limit -OcrSessionMode $OcrSessionMode -SkipPrepare
            }
        }
        elseif ($RequireLama) {
            & $smokeScript -InputRoot $InputRoot -BenchmarkRoot $BenchmarkRoot -Dataset $Dataset -Limit $Limit -OcrSessionMode $OcrSessionMode -RequireLama
        }
        else {
            & $smokeScript -InputRoot $InputRoot -BenchmarkRoot $BenchmarkRoot -Dataset $Dataset -Limit $Limit -OcrSessionMode $OcrSessionMode
        }
        if ($LASTEXITCODE -ne 0) {
            throw "run-release-smoke.ps1 failed during iteration $($index + 1)."
        }
    }
}

$evidenceArgs = @(
    "-m", "no_watermar.cli",
    "benchmark", "evidence",
    "--benchmark-root", $BenchmarkRoot,
    "--dataset", $Dataset,
    "--minimum-run-count", "$requiredRunCount",
    "--baseline-mask-provider", $BaselineMaskProvider,
    "--baseline-restore-provider", $BaselineRestoreProvider,
    "--candidate-mask-provider", $CandidateMaskProvider,
    "--candidate-restore-provider", $CandidateRestoreProvider
)
if ($SkipOptionalEvidence) {
    $evidenceArgs += "--skip-optional"
}
else {
    $evidenceArgs += @(
        "--optional-mask-provider", $OptionalMaskProvider,
        "--optional-restore-provider", $OptionalRestoreProvider
    )
}

$evidenceOutput = & python @evidenceArgs
if ($LASTEXITCODE -ne 0) {
    throw "benchmark evidence failed."
}

$evidence = $evidenceOutput | ConvertFrom-Json
Write-Output "EVIDENCE status=$($evidence.status)"
Write-Output "EVIDENCE release_blocking_ready=$($evidence.release_blocking.ready)"
Write-Output "EVIDENCE optional_status=$($evidence.optional_stable.status)"
Write-Output "OUTPUT json=$($evidence.output_json)"
Write-Output "OUTPUT markdown=$($evidence.output_markdown)"
$evidence | ConvertTo-Json -Depth 8
