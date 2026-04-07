param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,
    [switch]$RunProbe
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRootPath = (Resolve-Path $ProjectRoot).Path
$targets = @(
    @{
        Name = "paddleocr"
        EnvVar = "NO_WATERMAR_PADDLEOCR_PYTHON"
        DefaultPath = Join-Path $projectRootPath ".venvs\paddleocr\Scripts\python.exe"
    },
    @{
        Name = "lama"
        EnvVar = "NO_WATERMAR_LAMA_PYTHON"
        DefaultPath = Join-Path $projectRootPath ".venvs\lama\Scripts\python.exe"
    },
    @{
        Name = "diffusers"
        EnvVar = "NO_WATERMAR_DIFFUSERS_PYTHON"
        DefaultPath = Join-Path $projectRootPath ".venvs\diffusers\Scripts\python.exe"
    },
    @{
        Name = "powerpaint"
        EnvVar = "NO_WATERMAR_POWERPAINT_PYTHON"
        DefaultPath = Join-Path $projectRootPath ".venvs\powerpaint\Scripts\python.exe"
    },
    @{
        Name = "brushnet"
        EnvVar = "NO_WATERMAR_BRUSHNET_PYTHON"
        DefaultPath = Join-Path $projectRootPath ".venvs\brushnet\Scripts\python.exe"
    }
)

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
