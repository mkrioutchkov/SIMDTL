# M0 — Foundation de-risk findings

Goal of M0: prove the riskiest unknowns *before* building breadth.
Toolchain measured: **MSVC 14.51.36231 (Visual Studio 18 Community), `cl /std:c++20 /EHsc /O2 /arch:AVX2`, x64.**

## 1. vir-simd compiles & runs on MSVC 14.51 / C++20 ✅
`#include <vir/simd.h>` builds clean. Default namespace `vir::stdx`. Under
`/arch:AVX2`, `vir::stdx::native_simd<float>::size() == 8` (it maps "native" to
`fixed_size<8>`). Vendored at tag **v0.4.4** under `third_party/vir-simd/`.

## 2. The runtime CPU probe works ✅
`simdtl::platform::detect_cpu_features()` on this laptop reports:
`sse2 sse42 popcnt avx avx2 os_avx` = true, `avx512f avx512bw os_avx512` = false →
**best tier = `avx2`**. The mandatory `XGETBV` OS-state gate is in place, so AVX/
AVX-512 are never reported usable unless the OS saves YMM/ZMM. This is the
structural fix for the old library's "AVX hardcoded → SIGILL on non-AVX" bug.

## 3. ⚠️ Decisive finding: vir-simd's `fixed_size<N>` does NOT auto-vectorize on MSVC
A `simd<float, fixed_size<8>>` accumulate loop (`acc += v`) compiled at
`/O2 /arch:AVX2` lowers to **8 scalar `vaddss` instructions — zero `vaddps ymm`**:

```asm
$LL4@bench_redu:
    vaddss  xmm0, xmm0, DWORD PTR [rcx+r8*4]       ; lane 0
    vaddss  xmm1, xmm1, DWORD PTR [rcx+r8*4+4]     ; lane 1
    ...                                            ; 8 scalar adds, one per lane
```

(Reproduce: `tools/m0_smoke.cpp`, then inspect with `cl /FAs`.)

### Consequence for the architecture
The portable `std::simd` layer, on **MSVC/x86**, buys us **correctness, a clean
portable vocabulary, and the path that *is* fast on GCC/Clang** (where a native
register ABI exists) — **but not SIMD speed on this machine.** Therefore:

- **All guaranteed SIMD wins on x86 must come from SIMDTL's own intrinsic kernels**
  behind the L1 runtime-dispatch layer — not only the compaction/reverse/string
  kernels, but the hot "portable" algorithms (`count`/`find`/`reduce`) will want
  x86 intrinsic kernel paths too.
- The `std::simd` layer remains the correctness baseline + the GCC/Clang/ARM/WASM
  fast path + the future when MSVC ships a native ABI.
- Re-measure per kernel; don't assume the portable layer carries throughput on MSVC.

This *strengthens* the project thesis: SIMDTL's value is the kernels `std::simd`
can't express, and on MSVC that now explicitly includes native-width fast paths.
