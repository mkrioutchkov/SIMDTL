#pragma once
// ── L1: runtime CPU feature detection (CPUID + XGETBV) ───────────────────────
// This is the fix for the old SIMDTL's headline bug: it hardcoded AVX as the
// default and SIGILL'd on non-AVX CPUs. Here every wide kernel is gated on a
// runtime probe, and AVX/AVX-512 are only ever reported as usable after XGETBV
// confirms the OS actually saves the YMM/ZMM register state.
#include "arch_macros.hpp"
#include <cstdint>

#if SIMDTL_ARCH_X86
#  if SIMDTL_COMPILER_MSVC
#    include <intrin.h>      // __cpuid, __cpuidex
#    include <immintrin.h>   // _xgetbv
#  else
#    include <cpuid.h>       // __cpuid_count, __get_cpuid_max
#  endif
#endif

namespace simdtl::platform
{
    // Ordered capability tiers. Scoped enums support the relational operators,
    // so `best_isa() >= isa_level::avx2` is well-formed.
    enum class isa_level : int
    {
        scalar = 0,
        sse2   = 1,
        sse42  = 2,
        avx2   = 3,
        avx512 = 4,
    };

    struct cpu_features
    {
        bool sse2      = false;
        bool sse42     = false;
        bool popcnt    = false;
        bool avx       = false;
        bool avx2      = false;
        bool avx512f   = false;
        bool avx512bw  = false;
        bool os_avx    = false;   // OS saves XMM+YMM (XCR0 bits 1,2)
        bool os_avx512 = false;   // OS saves opmask+ZMM hi+ZMM (XCR0 bits 5,6,7)
    };

    namespace detail
    {
#if SIMDTL_ARCH_X86
        inline void cpuid(std::uint32_t leaf, std::uint32_t subleaf, std::uint32_t out[4]) noexcept
        {
#  if SIMDTL_COMPILER_MSVC
            int regs[4];
            __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
            out[0] = static_cast<std::uint32_t>(regs[0]);
            out[1] = static_cast<std::uint32_t>(regs[1]);
            out[2] = static_cast<std::uint32_t>(regs[2]);
            out[3] = static_cast<std::uint32_t>(regs[3]);
#  else
            unsigned int a = 0, b = 0, c = 0, d = 0;
            __cpuid_count(leaf, subleaf, a, b, c, d);
            out[0] = a; out[1] = b; out[2] = c; out[3] = d;
#  endif
        }

        // Read extended control register 0 (XCR0). Only call when OSXSAVE is set.
        inline std::uint64_t xgetbv0() noexcept
        {
#  if SIMDTL_COMPILER_MSVC
            return _xgetbv(0);
#  else
            std::uint32_t eax = 0, edx = 0;
            __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
            return (static_cast<std::uint64_t>(edx) << 32) | eax;
#  endif
        }
#endif // SIMDTL_ARCH_X86
    } // namespace detail

    inline cpu_features detect_cpu_features() noexcept
    {
        cpu_features f;
#if SIMDTL_ARCH_X86
        std::uint32_t r[4] = {0, 0, 0, 0};
        detail::cpuid(0, 0, r);
        const std::uint32_t max_leaf = r[0];

        bool osxsave = false;
        if (max_leaf >= 1)
        {
            detail::cpuid(1, 0, r);
            const std::uint32_t ecx = r[2];
            const std::uint32_t edx = r[3];
            f.sse2   = (edx >> 26) & 1u;
            f.popcnt = (ecx >> 23) & 1u;
            f.sse42  = (ecx >> 20) & 1u;
            f.avx    = (ecx >> 28) & 1u;
            osxsave  = (ecx >> 27) & 1u;
        }

        if (osxsave)
        {
            const std::uint64_t xcr0 = detail::xgetbv0();
            f.os_avx    = (xcr0 & 0x6u) == 0x6u;     // XMM(1) + YMM(2)
            f.os_avx512 = (xcr0 & 0xE6u) == 0xE6u;   // + opmask(5)+ZMMhi(6)+ZMM(7)
        }

        if (max_leaf >= 7)
        {
            detail::cpuid(7, 0, r);
            const std::uint32_t ebx = r[1];
            f.avx2     = (ebx >> 5) & 1u;
            f.avx512f  = (ebx >> 16) & 1u;
            f.avx512bw = (ebx >> 30) & 1u;
        }

        // An instruction set is only USABLE if the OS preserves its registers.
        if (!f.os_avx)    { f.avx = f.avx2 = false; }
        if (!f.os_avx512) { f.avx512f = f.avx512bw = false; }
#endif // SIMDTL_ARCH_X86
        return f;
    }

    inline isa_level detect_isa_level() noexcept
    {
        const cpu_features f = detect_cpu_features();
        if (f.avx512f && f.avx512bw) return isa_level::avx512;
        if (f.avx2)                  return isa_level::avx2;
        if (f.sse42)                 return isa_level::sse42;
        if (f.sse2)                  return isa_level::sse2;
        return isa_level::scalar;
    }

    // Detected once, cached. Dispatch tables (M1) resolve against this.
    inline isa_level best_isa() noexcept
    {
        static const isa_level level = detect_isa_level();
        return level;
    }

    inline const char* isa_name(isa_level l) noexcept
    {
        switch (l)
        {
            case isa_level::avx512: return "avx512";
            case isa_level::avx2:   return "avx2";
            case isa_level::sse42:  return "sse42";
            case isa_level::sse2:   return "sse2";
            default:                return "scalar";
        }
    }
} // namespace simdtl::platform
