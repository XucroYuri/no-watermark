param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,
    [switch]$RunProbe,
    [switch]$RunDoctor,
    [switch]$StableOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRootPath = (Resolve-Path $ProjectRoot).Path
$targets = @(
    @{
        Name = "paddleocr"
        EnvVar = "NO_WATERMAR_PADDLEOCR_PYTHON"
        DefaultPath = Join-Path $projectRootPath ".venvs\paddleocr\Scripts\python.exe"
        Stable = $true
    },
    @{
        Name = "lama"
        EnvVar = "NO_WATERMAR_LAMA_PYTHON"
        DefaultPath = Join-Path $projectRootPath ".venvs\lama\Scripts\python.exe"
        Stable = $true
    },
    @{
        Name = "diffusers"
        EnvVar = "NO_WATERMAR_DIFFUSERS_PYTHON"
        DefaultPath = Join-Path $projectRootPath ".venvs\diffusers\Scripts\python.exe"
        Stable = $false
    },
    @{
        Name = "powerpaint"
        EnvVar = "NO_WATERMAR_POWERPAINT_PYTHON"
        DefaultPath = Join-Path $projectRootPath ".venvs\powerpaint\Scripts\python.exe"
        Stable = $false
    },
    @{
        Name = "brushnet"
        EnvVar = "NO_WATERMAR_BRUSHNET_PYTHON"
        DefaultPath = Join-Path $projectRootPath ".venvs\brushnet\Scripts\python.exe"
        Stable = $false
    }
)

if ($StableOnly) {
    $targets = @($targets | Where-Object { $_.Stable })
}

foreach ($target in $targets) {
    $configured = [Environment]::GetEnvironmentVariable($target.EnvVar, "Process")
    if ([string]::IsNullOrWhiteSpace($configured)) {
        $configured = $target.DefaultPath
    }

    $exists = Test-Path -LiteralPath $configured
    $status = if ($exists) { "FOUND" } else { "MISSING" }
    Write-Output "$status $($target.Name) $configured"
    Write-Output "EXPORT `$env:$($target.EnvVar) = `"$configured`""
}

if ($RunProbe) {
    $benchmarkEntry = Join-Path $projectRootPath "benchmark.py"
    if (Test-Path -LiteralPath $benchmarkEntry) {
        Write-Output "PROBE python $benchmarkEntry probe-providers"
        & python $benchmarkEntry probe-providers
    }
}

if ($RunDoctor) {
    Write-Output "DOCTOR python -m no_watermar.cli providers doctor"
    $doctorOutput = & python -m no_watermar.cli providers doctor
    if ($LASTEXITCODE -ne 0) {
        throw "providers doctor failed"
    }

    $doctor = $doctorOutput | ConvertFrom-Json
    if ($StableOnly -and $doctor.stable_setup) {
        Write-Output "STABLE status=$($doctor.stable_setup.status)"
        Write-Output "STABLE release_blocking_ready=$($doctor.stable_setup.release_blocking_ready)"
        Write-Output "STABLE optional_ready=$($doctor.stable_setup.optional_ready)"
        foreach ($issue in @($doctor.stable_setup.blocking_issues)) {
            Write-Output "BLOCKER $($issue.provider_name) [$($issue.issue_code)] $($issue.detail)"
        }
        foreach ($issue in @($doctor.stable_setup.optional_issues)) {
            Write-Output "OPTIONAL $($issue.provider_name) [$($issue.issue_code)] $($issue.detail)"
        }
        foreach ($command in @($doctor.stable_setup.recommended_commands)) {
            Write-Output "NEXT $command"
        }
    }
    else {
        Write-Output $doctorOutput
    }
}
