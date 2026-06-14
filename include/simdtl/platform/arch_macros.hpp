#pragma once
// ── Architecture / compiler detection ────────────────────────────────────────
// Gate x86 SIMD code on the ARCHITECTURE, never on __SSE4_2__/__AVX2__: MSVC does
// not define those feature macros, and the SSE4.2/AVX intrinsics are available in
// its headers regardless of /arch. Instruction-set *availability* is decided at
// RUNTIME via cpu.hpp (CPUID), not at compile time.

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#  define SIMDTL_ARCH_X86 1
#else
#  define SIMDTL_ARCH_X86 0
#endif

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__)
#  define SIMDTL_ARCH_ARM 1
#else
#  define SIMDTL_ARCH_ARM 0
#endif

#if defined(_MSC_VER)
#  define SIMDTL_COMPILER_MSVC 1
#else
#  define SIMDTL_COMPILER_MSVC 0
#endif
