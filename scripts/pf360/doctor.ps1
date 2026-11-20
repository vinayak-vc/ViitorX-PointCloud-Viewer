<#
.SYNOPSIS
  Runs the pf360 environment diagnostic (python -m pf360 doctor).
  Read-only: reports what is installed and how to fix what is missing.
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\pf360\doctor.ps1
  powershell -ExecutionPolicy Bypass -File scripts\pf360\doctor.ps1 -Json
#>
param([switch]$Json, [switch]$Quick, [string]$Python = "")
$worker = Resolve-Path (Join-Path $PSScriptRoot "..\..\tools\pf360")
if (-not $Python) {
    $envPy = Join-Path $env:LOCALAPPDATA "ViitorX\PointForge\engines\pf360-env\Scripts\python.exe"
    if (Test-Path $envPy) { $Python = $envPy }
    elseif (Get-Command py -ErrorAction SilentlyContinue) { $Python = "py" }
    else { $Python = "python" }
}
$args2 = @("-m", "pf360", "doctor")
if ($Json) { $args2 += "--json" }
if ($Quick) { $args2 += "--quick" }
Push-Location $worker
try { & $Python @args2; exit $LASTEXITCODE } finally { Pop-Location }
