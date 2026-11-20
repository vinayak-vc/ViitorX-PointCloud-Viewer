<#
.SYNOPSIS
  CI-style gate: exits 0 only when the pf360 environment is READY.
  Thin wrapper over doctor.ps1 (kept as a separate name for automation).
#>
param([string]$Python = "")
& (Join-Path $PSScriptRoot "doctor.ps1") -Python $Python
exit $LASTEXITCODE
