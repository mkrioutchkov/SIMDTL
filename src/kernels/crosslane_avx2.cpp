// ── Opt-in AVX2 kernels for the cross-lane gap ops ────────────────────────────
// Compiled as its own /arch:AVX2 TU; self-registers remove_i32 and reverse_i32
// into the dispatch slots (only "sticks" if the CPU supports AVX2).
//   remove_i32  : keep lanes != value, pack via a 256-entry vpermd LUT, advance by
//                 popcount. In-place is safe because the write cursor k never gets
//                 ahead of the read cursor i (k <= i  =>  k+8 <= i+8).
//   reverse_i32 : reverse 8-lane blocks from both ends (vpermd) + scalar middle.
#include "simdtl/platform/dispatch.hpp"

#include <immintrin.h>
#if defined(_MSC_VER)
#  include <intrin.h>   // __popcnt
#endif
#include <cstddef>
#include <cstdint>

namespace
{
    // perm_lut[m][t] = index of the t-th set bit of m (rest zero-filled).
    alignas(32) std::uint32_t perm_lut[256][8];

    void build_lut() noexcept
    {
        for (int m = 0; m < 256; ++m)
        {
            int t = 0;
            for (int b = 0; b < 8; ++b)
                if (m & (1 << b)) perm_lut[m][t++] = static_cast<std::uint32_t>(b);
            for (; t < 8; ++t) perm_lut[m][t] = 0;
        }
    }

    inline unsigned popcnt8(unsigned m) noexcept
    {
#if defined(_MSC_VER)
        return __popcnt(m);
#else
        return static_cast<unsigned>(__builtin_popcount(m));
#endif
    }

    std::size_t remove_i32_avx2(std::int32_t* a, std::size_t n, std::int32_t value) noexcept
    {
        const __m256i needle = _mm256_set1_epi32(value);
        std::size_t i = 0, k = 0;
        for (; i + 8 <= n; i += 8)
        {
            const __m256i v   = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
            const __m256i eq  = _mm256_cmpeq_epi32(v, needle);
            const unsigned rm = static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(eq)));
            const unsigned keep = (~rm) & 0xFFu;                 // lanes to KEEP (!= value)
            const __m256i idx  = _mm256_load_si256(reinterpret_cast<const __m256i*>(perm_lut[keep]));
            const __m256i pack = _mm256_permutevar8x32_epi32(v, idx);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(a + k), pack);
            k += popcnt8(keep);
        }
        for (; i < n; ++i)
            if (a[i] != value) a[k++] = a[i];
        return k;
    }

    void reverse_i32_avx2(std::int32_t* a, std::size_t n) noexcept
    {
        const __m256i rev = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
        std::size_t lo = 0, hi = n;
        while (lo + 16 <= hi)   // two non-overlapping 8-blocks remain
        {
            const __m256i L = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + lo));
            const __m256i R = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + hi - 8));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(a + lo),     _mm256_permutevar8x32_epi32(R, rev));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(a + hi - 8), _mm256_permutevar8x32_epi32(L, rev));
            lo += 8;
            hi -= 8;
        }
        while (lo < hi)         // scalar middle (< 16 elements)
        {
            const std::int32_t t = a[lo];
            a[lo] = a[hi - 1];
            a[hi - 1] = t;
            ++lo;
            --hi;
        }
    }

    struct registrar
    {
        registrar() noexcept
        {
            build_lut();
            using namespace simdtl::platform;
            register_remove_i32(isa_level::avx2, &remove_i32_avx2);
            register_reverse_i32(isa_level::avx2, &reverse_i32_avx2);
        }
    };
    const registrar g_registrar{};
} // namespace
