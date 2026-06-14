# Configure, build, and run the test suite (+ benchmarks) via the CMake/Ninja that
# ship inside Visual Studio. Mirrors what CI does. Runs the seam lint first.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot

$vsDevCmd = 'C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat'
$vsExt = 'C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake'
$cmake = "$vsExt\CMake\bin\cmake.exe"
$ninja = "$vsExt\Ninja\ninja.exe"
$ctest = "$vsExt\CMake\bin\ctest.exe"
$bld = Join-Path $root 'build/test'

& "$root/scripts/lint-seam.ps1"
if ($LASTEXITCODE) { throw "seam lint failed" }

$opts = "-DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM=`"$ninja`" -DCMAKE_CXX_COMPILER=cl " +
        "-DSIMDTL_BUILD_TESTS=ON -DSIMDTL_FAST_KERNELS=ON -DSIMDTL_BUILD_BENCHMARKS=ON"
cmd.exe /d /c "call `"$vsDevCmd`" -arch=x64 >nul && `"$cmake`" -G Ninja $opts -S `"$root`" -B `"$bld`""
if ($LASTEXITCODE) { throw "configure failed" }
cmd.exe /d /c "call `"$vsDevCmd`" -arch=x64 >nul && `"$cmake`" --build `"$bld`""
if ($LASTEXITCODE) { throw "build failed" }
& $ctest --test-dir $bld --output-on-failure
if ($LASTEXITCODE) { throw "tests failed" }
Write-Host "Tests OK. Run benchmarks with: .\build\test\benchmarks\bench_count.exe" -ForegroundColor Green
