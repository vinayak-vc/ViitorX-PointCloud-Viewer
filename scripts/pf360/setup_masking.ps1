<#
.SYNOPSIS
  OPTIONAL add-on for PointForge's 360 Reconstruction Studio: installs the
  automatic masking model (people / vehicles / sky) into the existing pf360
  Nerfstudio environment.

.DESCRIPTION
  * Requires the environment created by setup_environment.ps1.
  * Installs the Python packages listed in tools\pf360\compat.json > masking
    (transformers + its small dependencies) into that venv only, with torch
    pinned so pip cannot replace the CUDA build.
  * Downloads the segmentation model (UperNet + ConvNeXt-Base trained on
    ADE20K, MIT licence, safetensors weights only - no pickle files) from
    Hugging Face into %LOCALAPPDATA%\ViitorX\PointForge\engines\models\.
    Roughly 0.5 GB in total; asks first (-Yes skips the prompt).
  * Nothing global changes. Jobs never download models: masking is simply
    unavailable until this script has run.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\pf360\setup_masking.ps1
#>
[CmdletBinding()]
param(
    [string]$EnvDir = "",
    [switch]$Yes
)
$ErrorActionPreference = "Stop"

$repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$compat = Get-Content (Join-Path $repo "tools\pf360\compat.json") -Raw | ConvertFrom-Json
$mk = $compat.masking
if (-not $EnvDir) { $EnvDir = Join-Path $env:LOCALAPPDATA "ViitorX\PointForge\engines\pf360-env" }
$envPy = Join-Path $EnvDir "Scripts\python.exe"
if (-not (Test-Path $envPy)) { throw "pf360 environment not found at $EnvDir - run setup_environment.ps1 first." }
$modelDir = Join-Path $env:LOCALAPPDATA ("ViitorX\PointForge\engines\models\" + $mk.modelDirName)

Write-Host ""
Write-Host "PointForge 360 masking add-on"
Write-Host "  Environment : $EnvDir"
Write-Host "  Packages    : $($mk.packages -join ', ')"
Write-Host "  Model       : $($mk.model) ($($mk.licence) licence) -> $modelDir"
Write-Host "  Download    : about $($mk.approxDownloadMB) MB"
Write-Host ""
if (-not $Yes) {
    $a = Read-Host "Install the masking packages and download the model? [y/N]"
    if ($a -notmatch '^[yY]') { Write-Host "Aborted - nothing was changed."; exit 1 }
}

$torchVer = & $envPy -c "import torch; print(torch.__version__)"
if ($LASTEXITCODE -ne 0) { throw "torch is not importable in $EnvDir" }
$constraint = Join-Path $env:TEMP "pf360-masking-constraints.txt"
Set-Content -Path $constraint -Value "torch==$torchVer" -Encoding ascii
& $envPy -m pip install @($mk.packages) -c $constraint
if ($LASTEXITCODE -ne 0) { throw "Installing the masking packages failed" }
Remove-Item $constraint -ErrorAction SilentlyContinue

New-Item -ItemType Directory -Force -Path $modelDir | Out-Null
$files = ($mk.files -join ",")
& $envPy -c @"
import sys
from huggingface_hub import hf_hub_download
repo, rev, out = sys.argv[1], sys.argv[2], sys.argv[3]
for f in sys.argv[4].split(','):
    p = hf_hub_download(repo_id=repo, filename=f, revision=rev, local_dir=out)
    print('downloaded', p)
"@ $mk.model $mk.revision $modelDir $files
if ($LASTEXITCODE -ne 0) { throw "Model download failed" }

Write-Host ""
Write-Host "Checking that the model loads..."
$env:PYTHONPATH = Join-Path $repo "tools\pf360"
& $envPy -m pf360.ns_helpers.mask --self-test --model $modelDir
if ($LASTEXITCODE -ne 0) { throw "Masking self-test failed" }
Write-Host "Masking is available. Enable it in the 360 wizard (Processing step)."
