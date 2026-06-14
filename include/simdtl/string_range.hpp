#pragma once
// ── L4 (x86): SSE4.2 string-range ops ─────────────────────────────────────────
// PCMPISTRM (_mm_cmpistrm) range-compares 16 bytes against [lo,hi] pairs in one
// instruction — something std::simd does not expose at all. Ported from the old
// range_comparisons.h, but now: gated on a RUNTIME CPUID SSE4.2 check (NOT the
// __SSE4_2__ macro, which MSVC never defines), with a portable scalar fallback so
// the API works everywhere (ARM/WASM/older x86).
//
// NULL CAVEAT: PCMPISTRM uses *implicit* string length, i.e. it stops at the first
// 0 byte in a 16-byte chunk. The SSE4.2 fast path is therefore for NUL-free text
// (normal string data); an embedded 0 in the DATA ends that chunk's scan early.
//
// The implicit length also constrains the RANGES operand: it must be 1..8 [lo,hi]
// pairs with no 0 boundary byte (a 0 would truncate it). The public API guards on
// this and silently falls back to the (fully general) scalar path when a request
// can't be encoded faithfully, so callers always get correct results regardless of
// bounds/pair count; only the speed differs.
#include "platform/arch_macros.hpp"
#include "platform/cpu.hpp"

#include <cstddef>

#if SIMDTL_ARCH_X86
#  include <nmmintrin.h>   // SSE4.2: _mm_cmpistrm
#  if defined(_MSC_VER)
#    include <intrin.h>    // __popcnt
#  endif
#endif

namespace simdtl
{
    namespace detail
    {
        inline std::size_t count_in_range_scalar(const char* s, std::size_t n, char lo, char hi) noexcept
        {
            std::size_t c = 0;
            for (std::size_t i = 0; i < n; ++i)
                if (s[i] >= lo && s[i] <= hi) ++c;
            return c;
        }

        // pairs = [lo0,hi0, lo1,hi1, ...]; npairs ranges. Flip the 0x20 case bit on
        // any char falling in any range.
        inline void convert_case_scalar(char* s, std::size_t n, const char* pairs, int npairs) noexcept
        {
            for (std::size_t i = 0; i < n; ++i)
            {
                const char c = s[i];
                for (int p = 0; p < npairs; ++p)
                    if (c >= pairs[2 * p] && c <= pairs[2 * p + 1]) { s[i] = static_cast<char>(c ^ 0x20); break; }
            }
        }

#if SIMDTL_ARCH_X86
        inline bool have_sse42() noexcept
        {
            static const bool v = platform::detect_cpu_features().sse42;
            return v;
        }

        // PCMPISTRM uses an IMPLICIT-LENGTH ranges operand: it is read as a
        // NUL-terminated string of at most 8 [lo,hi] pairs (16 bytes). So the
        // SSE4.2 path is only faithful when there are 1..8 pairs AND no boundary
        // byte is 0 (a 0 truncates the operand). Otherwise we must use scalar.
        inline bool ranges_ok_for_sse42(const char* pairs, int npairs) noexcept
        {
            if (npairs < 1 || npairs > 8) return false;
            for (int j = 0; j < npairs * 2; ++j)
                if (pairs[j] == 0) return false;
            return true;
        }

        inline unsigned popcnt(unsigned m) noexcept
        {
#  if defined(_MSC_VER)
            return __popcnt(m);
#  else
            return static_cast<unsigned>(__builtin_popcount(m));
#  endif
        }

        inline std::size_t count_in_range_sse42(const char* s, std::size_t n, char lo, char hi) noexcept
        {
            // Implicit length on operand1 makes {lo,hi,0,...} exactly one [lo,hi] pair.
            alignas(16) char ranges[16] = {lo, hi};
            const __m128i vr = _mm_load_si128(reinterpret_cast<const __m128i*>(ranges));
            std::size_t c = 0, i = 0;
            for (; i + 16 <= n; i += 16)
            {
                const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s + i));
                const __m128i m = _mm_cmpistrm(vr, v, _SIDD_SBYTE_OPS | _SIDD_CMP_RANGES | _SIDD_BIT_MASK);
                const unsigned bits = static_cast<unsigned>(_mm_cvtsi128_si32(m)) & 0xFFFFu;
                c += popcnt(bits);
            }
            for (; i < n; ++i)
                if (s[i] >= lo && s[i] <= hi) ++c;
            return c;
        }

        inline void convert_case_sse42(char* s, std::size_t n, const char* pairs, int npairs) noexcept
        {
            alignas(16) char ranges[16] = {0};
            for (int j = 0; j < npairs * 2 && j < 16; ++j) ranges[j] = pairs[j];
            const __m128i vr  = _mm_load_si128(reinterpret_cast<const __m128i*>(ranges));
            const __m128i bit = _mm_set1_epi8(0x20);
            std::size_t i = 0;
            for (; i + 16 <= n; i += 16)
            {
                __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s + i));
                // UNIT_MASK -> 0xFF per matching byte; AND 0x20, XOR into the data.
                __m128i m = _mm_cmpistrm(vr, v, _SIDD_SBYTE_OPS | _SIDD_CMP_RANGES | _SIDD_UNIT_MASK);
                m = _mm_and_si128(m, bit);
                v = _mm_xor_si128(v, m);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(s + i), v);
            }
            convert_case_scalar(s + i, n - i, pairs, npairs);
        }
#endif // SIMDTL_ARCH_X86
    } // namespace detail

    // Count chars c with lo <= c <= hi.
    inline std::size_t count_in_range(const char* s, std::size_t n, char lo, char hi) noexcept
    {
#if SIMDTL_ARCH_X86
        // A 0 boundary would truncate PCMPISTRM's implicit-length ranges operand
        // (-> 0 matches), so only take the SSE4.2 path when both bounds are non-zero.
        if (detail::have_sse42() && lo != 0 && hi != 0)
            return detail::count_in_range_sse42(s, n, lo, hi);
#endif
        return detail::count_in_range_scalar(s, n, lo, hi);
    }

    // Flip the 0x20 case bit on chars within any [lo,hi] pair.
    inline void convert_case(char* s, std::size_t n, const char* pairs, int npairs) noexcept
    {
#if SIMDTL_ARCH_X86
        // Use SSE4.2 only when PCMPISTRM can faithfully encode the ranges (1..8
        // pairs, no 0 boundary); otherwise the scalar path handles it correctly.
        if (detail::have_sse42() && detail::ranges_ok_for_sse42(pairs, npairs))
        {
            detail::convert_case_sse42(s, n, pairs, npairs);
            return;
        }
#endif
        detail::convert_case_scalar(s, n, pairs, npairs);
    }

    inline void to_lower(char* s, std::size_t n) noexcept { const char p[] = {'A', 'Z'};            convert_case(s, n, p, 1); }
    inline void to_upper(char* s, std::size_t n) noexcept { const char p[] = {'a', 'z'};            convert_case(s, n, p, 1); }
    inline void flip_case(char* s, std::size_t n) noexcept { const char p[] = {'A', 'Z', 'a', 'z'}; convert_case(s, n, p, 2); }
} // namespace simdtl
