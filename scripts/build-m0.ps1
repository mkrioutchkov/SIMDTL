# Builds and runs the M0 foundation programs with MSVC (no CMake required).
# Mirrors the VsDevCmd + cl pattern. CMake remains the source of truth for the
# full build/test/CI matrix; this script is for quick local M0 verification.
$ErrorActionPreference = 'Stop'

$root    = Split-Path -Parent $PSScriptRoot
$inc     = Join-Path $root 'include'
$virInc  = Join-Path $root 'third_party/vir-simd/include'
$out     = Join-Path $root 'build'
$vsDevCmd = 'C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat'

if (-not (Test-Path -LiteralPath $vsDevCmd)) { throw "VsDevCmd not found: $vsDevCmd" }
if (-not (Test-Path -LiteralPath $out)) { New-Item -ItemType Directory -Path $out | Out-Null }

# Our headers at /W4; vir-simd is treated as an external header (warnings off) so
# the dependency's diagnostics don't pollute our build.
$flags = '/nologo /std:c++20 /EHsc /O2 /arch:AVX2 /W4 /external:W0'
$incs  = "/I`"$inc`" /external:I`"$virInc`""

foreach ($tool in @('m0_cpu_probe', 'm0_smoke')) {
    $src = Join-Path $root "tools/$tool.cpp"
    $exe = Join-Path $out "$tool.exe"
    Write-Host "=== building $tool ===" -ForegroundColor Cyan
    $cmd = "call `"$vsDevCmd`" -arch=x64 >nul && cl $flags `"$src`" $incs /Fo`"$($out -replace '\\','/')/`" /Fe`"$($exe -replace '\\','/')`""
    cmd.exe /d /c $cmd
    if ($LASTEXITCODE -ne 0) { throw "Build failed: $tool ($LASTEXITCODE)" }
    Write-Host "=== running $tool ===" -ForegroundColor Green
    & $exe
    if ($LASTEXITCODE -ne 0) { throw "$tool exited $LASTEXITCODE" }
}
Write-Host "M0 OK" -ForegroundColor Green
