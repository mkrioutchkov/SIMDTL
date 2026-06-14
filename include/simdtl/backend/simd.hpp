#pragma once
// ── L0 seam: the ONLY place the std::simd backend is named ────────────────────
// Everything above this file uses `simdtl::stdx::…` and the stable wrappers in
// names.hpp — never `vir::`, `std::experimental::`, or raw `std::` simd names.
// Swapping backends (vir-simd today → native C++26 <simd> tomorrow) is editing
// THIS file plus names.hpp, with zero algorithm changes. A CI grep-lint enforces
// that `vir::`/`stdx::` never appear outside include/simdtl/backend/.
//
// We do NOT include <experimental/simd> directly: it merely needs to *exist* for
// __has_include to be true (libc++ ships a stub on macOS/Emscripten that does NOT
// define std::experimental::parallelism_v2), which would break those platforms.
// vir-simd does the correct detection internally — it uses libstdc++'s real
// <experimental/simd> when present (keeping the native ABI on GCC) and its own
// portable fallback otherwise. So: native C++26 <simd> if available, else vir-simd.

#if defined(__cpp_lib_simd) && !defined(SIMDTL_FORCE_VIR_SIMD)
#  include <simd>
namespace simdtl { namespace stdx = std; }            // C++26: std::simd lives in std
#  define SIMDTL_SIMD_BACKEND "std (C++26)"
#else
#  include <vir/simd.h>
namespace simdtl { namespace stdx = vir::stdx; }
#  define SIMDTL_SIMD_BACKEND "vir-simd"
#endif
