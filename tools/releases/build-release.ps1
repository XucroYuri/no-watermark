param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,
    [switch]$SkipInstall,
    [switch]$SkipTests,
    [switch]$CleanDist
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [string]$Label,
        [scriptblock]$Action
    )

    Write-Output "RUN  $Label"
    & $Action
}

$projectRootPath = (Resolve-Path $ProjectRoot).Path
Push-Location $projectRootPath

try {
    if ($CleanDist) {
        foreach ($path in @("dist", "build")) {
            $resolved = Join-Path $projectRootPath $path
            if (Test-Path -LiteralPath $resolved) {
                Invoke-Step "Remove $resolved" { Remove-Item -LiteralPath $resolved -Recurse -Force }
            }
        }
    }

    if (-not $SkipInstall) {
        Invoke-Step "Upgrade pip" { python -m pip install --upgrade pip }
        Invoke-Step "Install editable package with dev extras" { python -m pip install -e .[dev] }
    }

    Invoke-Step "Check CLI entrypoint" { python -m no_watermar.cli --help }

    if (-not $SkipTests) {
        Invoke-Step "Run tests" { python -m unittest discover -s tests -v }
    }

    Invoke-Step "Build distribution artifacts" { python -m build }
    Invoke-Step "Check distribution metadata" { python -m twine check dist/* }
}
finally {
    Pop-Location
}
