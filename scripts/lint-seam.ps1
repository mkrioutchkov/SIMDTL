# Seam lint: nothing above the L0 backend layer may name the backend directly.
# The point of include/simdtl/backend/ is that swapping vir-simd for native C++26
# <simd> is a one-directory edit. Fail if a library header outside backend/ spells
# a backend name (vir::, stdx::, native_simd, the alignment tags).
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$scan = Join-Path $root 'include/simdtl'
$forbidden = @('vir::', 'stdx::', 'std::experimental', 'native_simd', 'element_aligned', 'vector_aligned')
$violations = New-Object System.Collections.Generic.List[string]

$files = Get-ChildItem -LiteralPath $scan -Recurse -Filter *.hpp |
    Where-Object { $_.FullName -notmatch '[\\/]backend[\\/]' }

foreach ($f in $files) {
    $n = 0
    foreach ($line in Get-Content -LiteralPath $f.FullName) {
        $n++
        $code = $line                       # strip // line comments before checking
        $idx = $code.IndexOf('//')
        if ($idx -ge 0) { $code = $code.Substring(0, $idx) }
        foreach ($needle in $forbidden) {
            if ($code.Contains($needle)) {
                $violations.Add(("{0}:{1}: {2}" -f $f.Name, $n, $line.Trim()))
            }
        }
    }
}

if ($violations.Count -gt 0) {
    Write-Host "SEAM LINT FAILED - backend names leaked outside backend/:"
    foreach ($v in $violations) { Write-Host "  $v" }
    exit 1
}
Write-Host "Seam lint OK - no backend names outside backend/."
