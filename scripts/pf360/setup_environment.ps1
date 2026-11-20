<#
.SYNOPSIS
  Creates an ISOLATED Python environment with Nerfstudio for PointForge's
  360 Reconstruction Studio (pf360 worker).

.DESCRIPTION
  * Installs into %LOCALAPPDATA%\ViitorX\PointForge\engines\pf360-env (a venv)
    unless -EnvDir is given. Nothing global is modified: no system Python
    packages, no PATH changes, no CUDA toolkit / driver installation.
  * Downloads roughly 6-10 GB (PyTorch CUDA wheels + Nerfstudio + deps) and
    asks for confirmation first (-Yes skips the prompt).
  * Versions come from tools\pf360\compat.json. They are the combination
    documented by Nerfstudio 1.1.x; see that file's "verified" flag - this
    script does not claim a combination works until a real run has passed.
  * COLMAP is NOT installed here. Use PointForge's photogrammetry engine setup
    (Convert > photos, which installs the official COLMAP release) or install
    COLMAP yourself and set its path in Preferences > 360 Reconstruction.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\pf360\setup_environment.ps1
  powershell -ExecutionPolicy Bypass -File scripts\pf360\setup_environment.ps1 -Python "C:\Python310\python.exe" -Yes
#>
[CmdletBinding()]
param(
    [string]$Python = "",
    [string]$EnvDir = "",
    [switch]$Yes
)
$ErrorActionPreference = "Stop"

$repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$compat = Get-Content (Join-Path $repo "tools\pf360\compat.json") -Raw | ConvertFrom-Json
$target = $compat.recommended
if (-not $EnvDir) { $EnvDir = Join-Path $env:LOCALAPPDATA "ViitorX\PointForge\engines\pf360-env" }

function Find-Python {
    if ($Python) { return $Python }
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        foreach ($v in $target.pythonPreferred) {
            $exe = & py "-$v" -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $exe) { return $exe.Trim() }
        }
    }
    $p = Get-Command python -ErrorAction SilentlyContinue
    if ($p -and $p.Source -notlike "*WindowsApps*") { return $p.Source }
    return $null
}

$basePy = Find-Python
if (-not $basePy) { throw "No Python found. Install Python $($target.pythonPreferred -join ' / ') from python.org, then re-run (or pass -Python)." }
$ver = & $basePy -c "import sys; print('%d.%d' % sys.version_info[:2])"
if ($target.pythonSupported -notcontains $ver) {
    throw "Python $ver at $basePy is outside the supported range ($($target.pythonSupported -join ', ')). Pass -Python <path to a supported python.exe>."
}

$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $smi) { Write-Warning "nvidia-smi not found: Gaussian training needs an NVIDIA CUDA GPU + driver. Continuing installs the software only." }
else { & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader }

Write-Host ""
Write-Host "PointForge 360 reconstruction environment"
Write-Host "  Base Python : $basePy ($ver)"
Write-Host "  Target env  : $EnvDir"
Write-Host "  PyTorch     : $($target.torch) ($($target.torchIndexUrl))"
Write-Host "  Nerfstudio  : $($target.nerfstudio)"
Write-Host "  Download    : about $($target.approxDownloadGB) GB, disk ~$($target.approxDiskGB) GB"
Write-Host ""
if (-not $Yes) {
    $a = Read-Host "Create the environment and download these packages? [y/N]"
    if ($a -notmatch '^[yY]') { Write-Host "Aborted - nothing was changed."; exit 1 }
}

# Some Nerfstudio dependencies (e.g. fpsample) ship no Windows wheel for every
# Python version and gsplat JIT-compiles CUDA kernels, so an MSVC compiler must
# be on PATH. If `cl` is missing, import a Visual Studio developer environment
# for this process only (nothing global changes). CUDA 11.8's nvcc rejects
# MSVC >= 19.40, so prefer a 14.3x toolset when several are installed.
. (Join-Path $PSScriptRoot "msvc_env.ps1")
if (-not (Import-PF360MsvcEnv)) {
    Write-Warning "No usable MSVC C++ toolset found. Packages without wheels and gsplat's CUDA JIT will fail. Install 'Desktop development with C++' (MSVC v143 14.3x) in Visual Studio."
}

$drive = (Split-Path -Qualifier $EnvDir)
$free = (Get-PSDrive ($drive.TrimEnd(':'))).Free
if ($free -lt ($target.approxDiskGB * 1GB)) { throw "Not enough free space on $drive ($([math]::Round($free/1GB,1)) GB free)." }

if (-not (Test-Path (Join-Path $EnvDir "Scripts\python.exe"))) {
    New-Item -ItemType Directory -Force -Path (Split-Path $EnvDir) | Out-Null
    & $basePy -m venv $EnvDir
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
}
$envPy = Join-Path $EnvDir "Scripts\python.exe"
# torch 2.1.x does `from pkg_resources import packaging`, which setuptools >= 70
# no longer provides (ModuleNotFoundError / ImportError on gsplat's CUDA JIT).
& $envPy -m pip install --upgrade pip wheel $target.setuptools
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
& $envPy -m pip install $target.torch $target.torchvision --index-url $target.torchIndexUrl
if ($LASTEXITCODE -ne 0) { throw "PyTorch install failed" }
& $envPy -m pip install $target.nerfstudio
if ($LASTEXITCODE -ne 0) { throw "Nerfstudio install failed" }

# Prefer gsplat's PREBUILT CUDA wheel over the PyPI source package. The source
# package JIT-compiles its kernels on first training run, and on Windows that
# fails with CUDA 11.8 (CUB's `small` member vs. the `#define small char` in
# rpcndr.h) besides needing ~4 GB RAM per nvcc job. Non-fatal: the JIT path stays
# as a fallback.
& $envPy -m pip install --no-deps --force-reinstall $target.gsplatWheelSpec --index-url $target.gsplatWheelIndex
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Prebuilt gsplat wheel not installed (index unreachable?). gsplat will try to JIT-compile on first training run."
} else {
    & $envPy -c "from gsplat.cuda._backend import _C; import sys; sys.exit(0 if _C is not None else 1)"
    if ($LASTEXITCODE -ne 0) { Write-Warning "gsplat CUDA kernels did not load; see docs\360-reconstruction.md > Troubleshooting." }
}

Write-Host ""
Write-Host "Installed. Running the pf360 environment diagnostic..."
$env:PF360_NS_PYTHON = $envPy
Push-Location (Join-Path $repo "tools\pf360")
try { & $envPy -m pf360 doctor } finally { Pop-Location }
Write-Host ""
Write-Host "Note: gsplat compiles its CUDA kernels on first training run; on Windows that needs the"
Write-Host "Visual Studio C++ build tools and a CUDA toolkit matching PyTorch ($($target.cudaToolkitForGsplatJit))."
Write-Host "If training fails at startup, see docs\360-reconstruction.md > Troubleshooting."
