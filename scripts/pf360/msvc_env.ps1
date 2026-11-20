<#
.SYNOPSIS
  Dot-source helper: Import-PF360MsvcEnv puts an MSVC x64 developer environment
  into the CURRENT PowerShell process (PATH/INCLUDE/LIB...). Nothing persistent
  or global is changed.

  CUDA 11.8 (the toolkit matching torch 2.1.2+cu118) refuses MSVC _MSC_VER >= 1940,
  i.e. toolsets 14.40+. When several toolsets are installed the newest 14.3x one is
  selected; otherwise the default toolset is used and a warning is printed.
#>
function Import-PF360MsvcEnv {
    param([switch]$Force)
    if (-not $Force -and (Get-Command cl.exe -ErrorAction SilentlyContinue)) { return $true }

    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    $roots = @()
    if (Test-Path $vswhere) {
        $roots = & $vswhere -all -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    }
    if (-not $roots) {
        $roots = Get-ChildItem "$env:ProgramFiles\Microsoft Visual Studio", "${env:ProgramFiles(x86)}\Microsoft Visual Studio" -Directory -ErrorAction SilentlyContinue |
                 ForEach-Object { Get-ChildItem $_.FullName -Directory } | ForEach-Object { $_.FullName }
    }
    foreach ($root in $roots) {
        $vcvars = Join-Path $root "VC\Auxiliary\Build\vcvars64.bat"
        if (-not (Test-Path $vcvars)) { continue }
        $msvcDir = Join-Path $root "VC\Tools\MSVC"
        $toolsets = Get-ChildItem $msvcDir -Directory -ErrorAction SilentlyContinue | Sort-Object { [version]$_.Name } -Descending
        $pick = $toolsets | Where-Object { ([version]$_.Name).Minor -lt 40 } | Select-Object -First 1
        $verArg = ""
        if ($pick) {
            $v = [version]$pick.Name
            $verArg = "-vcvars_ver=$($v.Major).$($v.Minor)"
        } else {
            Write-Warning "Only MSVC 14.40+ found under $root; CUDA 11.8 nvcc may reject it (gsplat JIT)."
        }
        # Run vcvars in cmd and capture the resulting environment.
        $out = & cmd.exe /d /s /c "`"$vcvars`" $verArg >nul 2>&1 && set"
        if ($LASTEXITCODE -ne 0 -or -not $out) { continue }
        foreach ($line in $out) {
            $i = $line.IndexOf('=')
            if ($i -gt 0) { Set-Item -Path ("Env:" + $line.Substring(0, $i)) -Value $line.Substring($i + 1) }
        }
        if (Get-Command cl.exe -ErrorAction SilentlyContinue) {
            Write-Host "MSVC environment: $root $verArg"
            return $true
        }
    }
    return $false
}
