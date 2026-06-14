#pragma once
// ── L0 seam: the ONLY place the std::simd backend is named ────────────────────
// Everything above this file uses `simdtl::stdx::…` and the stable wrappers in
// names.hpp — never `vir::`, `std::experimental::`, or raw `std::` simd names.
// Swapping backends (vir-simd today → native C++26 <simd> tomorrow) is editing
// THIS file plus names.hpp, with zero algorithm changes. A CI grep-lint enforces
// that `vir::`/`stdx::` never appear outside include/simdtl/backend/.
//
// Selection priority: native C++26 <simd> → experimental TS → vir-simd backport.
// Define SIMDTL_FORCE_VIR_SIMD to pin the vir-simd backport (useful for matching
// MSVC behaviour across compilers in CI).

#if defined(SIMDTL_FORCE_VIR_SIMD)
#  include <vir/simd.h>
namespace simdtl { namespace stdx = vir::stdx; }
#  define SIMDTL_SIMD_BACKEND "vir-simd (forced)"
#elif defined(__cpp_lib_simd)
#  include <simd>
namespace simdtl { namespace stdx = std; }            // C++26: std::simd lives in std
#  define SIMDTL_SIMD_BACKEND "std (C++26)"
#elif __has_include(<experimental/simd>)
#  include <experimental/simd>
namespace simdtl { namespace stdx = std::experimental::parallelism_v2; }
#  define SIMDTL_SIMD_BACKEND "std::experimental (Parallelism TS)"
#else
#  include <vir/simd.h>
namespace simdtl { namespace stdx = vir::stdx; }
#  define SIMDTL_SIMD_BACKEND "vir-simd"
#endif
