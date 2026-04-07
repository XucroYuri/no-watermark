$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptDir "..")).Path

if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$repoRoot\src;$env:PYTHONPATH"
}
else {
    $env:PYTHONPATH = "$repoRoot\src"
}

python -m no_watermar.cli @args
