param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,
    [string]$PythonCommand = "python",
    [string]$ConfigPythonCommand = "python",
    [string]$PaddlePythonCommand = "",
    [string]$LamaPythonCommand = "",
    [string]$DiffusersPythonCommand = "",
    [string]$PowerPaintPythonCommand = "",
    [string]$BrushNetPythonCommand = "",
    [switch]$StableOnly,
    [switch]$SkipConfigInit,
    [switch]$InstallPackages,
    [switch]$PrintOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [string]$Label,
        [scriptblock]$Action
    )

    if ($PrintOnly) {
        Write-Output "PLAN $Label"
        return
    }

    Write-Output "RUN  $Label"
    $global:LASTEXITCODE = 0
    & $Action
    if (-not $?) {
        throw "Step failed: $Label"
    }
    if ($global:LASTEXITCODE -ne 0) {
        throw "Step failed with exit code $global:LASTEXITCODE: $Label"
    }
}

$projectRootPath = (Resolve-Path $ProjectRoot).Path
$venvRoot = Join-Path $projectRootPath ".venvs"
$configPath = Join-Path $projectRootPath "no-watermar.toml"
$targets = @(
    @{
        Name = "paddleocr"
        Path = Join-Path $venvRoot "paddleocr"
        EnvVar = "NO_WATERMAR_PADDLEOCR_PYTHON"
        PythonCommand = if ([string]::IsNullOrWhiteSpace($PaddlePythonCommand)) { $PythonCommand } else { $PaddlePythonCommand }
        Stable = $true
        Packages = @("paddleocr")
        PostInstallNotes = @(
            "Install the matching Paddle runtime for this machine after paddleocr is present."
        )
    },
    @{
        Name = "lama"
        Path = Join-Path $venvRoot "lama"
        EnvVar = "NO_WATERMAR_LAMA_PYTHON"
        PythonCommand = if ([string]::IsNullOrWhiteSpace($LamaPythonCommand)) { $PythonCommand } else { $LamaPythonCommand }
        Stable = $true
        Packages = @("simple-lama-inpainting")
        PostInstallNotes = @(
            "Keep lama on a validated Python 3.12 sidecar when the default shell interpreter is newer."
        )
    },
    @{
        Name = "diffusers"
        Path = Join-Path $venvRoot "diffusers"
        EnvVar = "NO_WATERMAR_DIFFUSERS_PYTHON"
        PythonCommand = if ([string]::IsNullOrWhiteSpace($DiffusersPythonCommand)) { $PythonCommand } else { $DiffusersPythonCommand }
        Stable = $false
        Packages = @()
        PostInstallNotes = @()
    },
    @{
        Name = "powerpaint"
        Path = Join-Path $venvRoot "powerpaint"
        EnvVar = "NO_WATERMAR_POWERPAINT_PYTHON"
        PythonCommand = if ([string]::IsNullOrWhiteSpace($PowerPaintPythonCommand)) { $PythonCommand } else { $PowerPaintPythonCommand }
        Stable = $false
        Packages = @()
        PostInstallNotes = @()
    },
    @{
        Name = "brushnet"
        Path = Join-Path $venvRoot "brushnet"
        EnvVar = "NO_WATERMAR_BRUSHNET_PYTHON"
        PythonCommand = if ([string]::IsNullOrWhiteSpace($BrushNetPythonCommand)) { $PythonCommand } else { $BrushNetPythonCommand }
        Stable = $false
        Packages = @()
        PostInstallNotes = @()
    }
)

if ($StableOnly) {
    $targets = @($targets | Where-Object { $_.Stable })
}

if (-not (Test-Path -LiteralPath $venvRoot)) {
    Invoke-Step "Create $venvRoot" { New-Item -ItemType Directory -Path $venvRoot | Out-Null }
}

if ($StableOnly -and -not $SkipConfigInit) {
    if (Test-Path -LiteralPath $configPath) {
        Write-Output "SKIP stable-public config already exists at $configPath"
    }
    else {
        Invoke-Step "Initialize stable-public config at $configPath" {
            & $ConfigPythonCommand -m no_watermar.cli config init --template stable-public --config $configPath
        }
    }
}

foreach ($target in $targets) {
    $venvPath = $target.Path
    $targetPythonCommand = $target.PythonCommand
    if (-not (Test-Path -LiteralPath $venvPath)) {
        Invoke-Step "Create venv $($target.Name)" { & $targetPythonCommand -m venv $venvPath }
    } else {
        Write-Output "SKIP $venvPath already exists"
    }

    $pythonPath = Join-Path $venvPath "Scripts\python.exe"
    Write-Output "ENV  $($target.EnvVar)=$pythonPath"
    Write-Output "BASE $($target.Name) via $targetPythonCommand"

    if ($InstallPackages) {
        $packages = @($target.Packages)
        if ($packages.Count -gt 0) {
            Invoke-Step "Upgrade pip in $($target.Name)" { & $pythonPath -m pip install --upgrade pip }
            Invoke-Step "Install packages for $($target.Name)" { & $pythonPath -m pip install @packages }
            foreach ($note in @($target.PostInstallNotes)) {
                Write-Output "NOTE $($target.Name): $note"
            }
        }
        else {
            Write-Output "SKIP package bootstrap remains manual for $($target.Name)"
        }
    }
}

Write-Output ""
if ($StableOnly) {
    if (-not $InstallPackages) {
        Write-Output "NEXT Install the stable sidecar packages:"
        Write-Output "NEXT   .\\.venvs\\paddleocr\\Scripts\\python.exe -m pip install --upgrade pip"
        Write-Output "NEXT   .\\.venvs\\paddleocr\\Scripts\\python.exe -m pip install paddleocr"
        Write-Output "NEXT   Install the matching Paddle runtime for your machine inside .\\.venvs\\paddleocr"
        Write-Output "NEXT   .\\.venvs\\lama\\Scripts\\python.exe -m pip install --upgrade pip"
        Write-Output "NEXT   .\\.venvs\\lama\\Scripts\\python.exe -m pip install simple-lama-inpainting"
    }
    Write-Output "NEXT Validate the stable public path:"
    Write-Output "NEXT   powershell -ExecutionPolicy Bypass -File .\\tools\\setup\\validate-sidecars.ps1 -StableOnly -RunDoctor"
    Write-Output "NEXT   powershell -ExecutionPolicy Bypass -File .\\tools\\benchmark\\run-release-smoke.ps1 -Limit 1"
    if ($SkipConfigInit) {
        Write-Output "NEXT Initialize the local stable config when needed:"
        Write-Output "NEXT   python -m no_watermar.cli config init --template stable-public"
    }
    else {
        Write-Output "NEXT The stable-public config template now seeds local_smoke, seed_telea, ocr_telea, lama_eval, and ocr_corner_crop profiles when no local config exists."
    }
}
else {
    Write-Output "NEXT Install provider packages into each venv as needed."
    Write-Output "NEXT Example:"
    Write-Output "NEXT   .\\.venvs\\paddleocr\\Scripts\\python.exe -m pip install --upgrade pip"
    Write-Output "NEXT   .\\.venvs\\paddleocr\\Scripts\\python.exe -m pip install paddleocr"
    Write-Output "NEXT   .\\.venvs\\lama\\Scripts\\python.exe -m pip install --upgrade pip"
    Write-Output "NEXT   .\\.venvs\\lama\\Scripts\\python.exe -m pip install simple-lama-inpainting"
    Write-Output "NEXT   .\\.venvs\\diffusers\\Scripts\\python.exe -m pip install --upgrade pip"
    Write-Output "NEXT   Install the matching torch build for your machine inside .\\.venvs\\diffusers first"
    Write-Output "NEXT   .\\.venvs\\diffusers\\Scripts\\python.exe -m pip install diffusers transformers accelerate safetensors"
    Write-Output "NEXT   .\\.venvs\\powerpaint\\Scripts\\python.exe -m pip install --upgrade pip"
    Write-Output "NEXT   Install the matching torch build for your machine inside .\\.venvs\\powerpaint first"
    Write-Output "NEXT   .\\.venvs\\powerpaint\\Scripts\\python.exe -m pip install diffusers transformers safetensors"
    Write-Output "NEXT   Install PowerPaint as a package in .\\.venvs\\powerpaint or set NO_WATERMAR_POWERPAINT_SOURCE_DIR to a local clone"
    Write-Output "NEXT   .\\.venvs\\brushnet\\Scripts\\python.exe -m pip install --upgrade pip"
    Write-Output "NEXT   Install the matching torch build for your machine inside .\\.venvs\\brushnet first"
    Write-Output "NEXT   .\\.venvs\\brushnet\\Scripts\\python.exe -m pip install transformers accelerate opencv-python pillow"
    Write-Output "NEXT   Install the BrushNet upstream repo in editable mode inside .\\.venvs\\brushnet, or set NO_WATERMAR_BRUSHNET_SOURCE_DIR to a local clone"
}
Write-Output "NEXT Override one provider interpreter when needed:"
Write-Output "NEXT   powershell -ExecutionPolicy Bypass -File .\\tools\\setup\\bootstrap-sidecars.ps1 -ConfigPythonCommand `"C:\\Path\\To\\Repo\\python.exe`""
Write-Output "NEXT   powershell -ExecutionPolicy Bypass -File .\\tools\\setup\\bootstrap-sidecars.ps1 -LamaPythonCommand `"C:\\Path\\To\\Python312\\python.exe`""
Write-Output "NEXT   powershell -ExecutionPolicy Bypass -File .\\tools\\setup\\bootstrap-sidecars.ps1 -DiffusersPythonCommand `"C:\\Path\\To\\Python311\\python.exe`""
Write-Output "NEXT   powershell -ExecutionPolicy Bypass -File .\\tools\\setup\\bootstrap-sidecars.ps1 -PowerPaintPythonCommand `"C:\\Path\\To\\Python312\\python.exe`""
Write-Output "NEXT   powershell -ExecutionPolicy Bypass -File .\\tools\\setup\\bootstrap-sidecars.ps1 -BrushNetPythonCommand `"C:\\Path\\To\\Python39\\python.exe`""
