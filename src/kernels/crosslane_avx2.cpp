// ── Opt-in AVX2 cross-lane kernels (self-registering; only stick if CPU has AVX2) ─
//   remove_i32 : keep lanes != value, pack via a 256-entry vpermd LUT.
//   remove_i16 : 8 shorts/iter, pshufb left-pack via a 256-entry word LUT.
//   remove_i8  : 16 bytes/iter, two 8-byte pshufb left-packs via a 256-entry byte LUT.
//   reverse_i32: reverse 8-lane blocks from both ends (vpermd) + scalar middle.
// In-place compaction is safe because the write cursor k never overtakes the read
// cursor i (k <= i  =>  every store stays within [.., i+chunk)).
#include "simdtl/platform/dispatch.hpp"

#include <immintrin.h>
#if defined(_MSC_VER)
#  include <intrin.h>   // __popcnt
#endif
#include <cstddef>
#include <cstdint>

namespace
{
    inline unsigned popcnt(unsigned m) noexcept
    {
#if defined(_MSC_VER)
        return __popcnt(m);
#else
        return static_cast<unsigned>(__builtin_popcount(m));
#endif
    }

    // perm_lut[m][t] = index of the t-th set bit of m (rest zero) -> vpermd control.
    alignas(32) std::uint32_t perm_lut[256][8];
    // byte_lut[m]  : pshufb control compacting the kept bytes of an 8-bit group (low 8).
    alignas(16) std::uint8_t  byte_lut[256][16];
    // short_lut[m] : pshufb control compacting the kept shorts of an 8-short group.
    alignas(16) std::uint8_t  short_lut[256][16];

    void build_luts() noexcept
    {
        for (int m = 0; m < 256; ++m)
        {
            int t = 0;
            for (int b = 0; b < 8; ++b)
                if (m & (1 << b)) perm_lut[m][t++] = static_cast<std::uint32_t>(b);
            for (; t < 8; ++t) perm_lut[m][t] = 0;

            int kb = 0;
            for (int b = 0; b < 8; ++b)
                if (m & (1 << b)) byte_lut[m][kb++] = static_cast<std::uint8_t>(b);
            for (; kb < 16; ++kb) byte_lut[m][kb] = 0x80;          // pshufb 0x80 -> zero

            int ks = 0;
            for (int b = 0; b < 8; ++b)
                if (m & (1 << b)) { short_lut[m][ks++] = static_cast<std::uint8_t>(2 * b);
                                    short_lut[m][ks++] = static_cast<std::uint8_t>(2 * b + 1); }
            for (; ks < 16; ++ks) short_lut[m][ks] = 0x80;
        }
    }

    std::size_t remove_i8_avx2(std::int8_t* a, std::size_t n, std::int8_t value) noexcept
    {
        const __m128i needle = _mm_set1_epi8(value);
        std::size_t i = 0, k = 0;
        for (; i + 16 <= n; i += 16)
        {
            const __m128i v  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
            const unsigned rm = static_cast<unsigned>(_mm_movemask_epi8(_mm_cmpeq_epi8(v, needle)));
            const unsigned keep = (~rm) & 0xFFFFu;
            const unsigned lo = keep & 0xFFu, hi = (keep >> 8) & 0xFFu;
            const __m128i clo = _mm_shuffle_epi8(v, _mm_load_si128(reinterpret_cast<const __m128i*>(byte_lut[lo])));
            const __m128i chi = _mm_shuffle_epi8(_mm_srli_si128(v, 8), _mm_load_si128(reinterpret_cast<const __m128i*>(byte_lut[hi])));
            _mm_storel_epi64(reinterpret_cast<__m128i*>(a + k), clo); k += popcnt(lo);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(a + k), chi); k += popcnt(hi);
        }
        for (; i < n; ++i) if (a[i] != value) a[k++] = a[i];
        return k;
    }

    std::size_t remove_i16_avx2(std::int16_t* a, std::size_t n, std::int16_t value) noexcept
    {
        const __m128i needle = _mm_set1_epi16(value);
        std::size_t i = 0, k = 0;
        for (; i + 8 <= n; i += 8)
        {
            const __m128i v  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
            const __m128i eq = _mm_cmpeq_epi16(v, needle);
            // pack 8 shorts -> 8 bytes (0xFF/0x00), then one bit per short.
            const unsigned rm = static_cast<unsigned>(_mm_movemask_epi8(_mm_packs_epi16(eq, eq))) & 0xFFu;
            const unsigned keep = (~rm) & 0xFFu;
            const __m128i c = _mm_shuffle_epi8(v, _mm_load_si128(reinterpret_cast<const __m128i*>(short_lut[keep])));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(a + k), c);   // up to 8 shorts; only `keep` stick
            k += popcnt(keep);
        }
        for (; i < n; ++i) if (a[i] != value) a[k++] = a[i];
        return k;
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
            const unsigned keep = (~rm) & 0xFFu;
            const __m256i idx  = _mm256_load_si256(reinterpret_cast<const __m256i*>(perm_lut[keep]));
            const __m256i pack = _mm256_permutevar8x32_epi32(v, idx);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(a + k), pack);
            k += popcnt(keep);
        }
        for (; i < n; ++i) if (a[i] != value) a[k++] = a[i];
        return k;
    }

    void reverse_i32_avx2(std::int32_t* a, std::size_t n) noexcept
    {
        const __m256i rev = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
        std::size_t lo = 0, hi = n;
        while (lo + 16 <= hi)
        {
            const __m256i L = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + lo));
            const __m256i R = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + hi - 8));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(a + lo),     _mm256_permutevar8x32_epi32(R, rev));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(a + hi - 8), _mm256_permutevar8x32_epi32(L, rev));
            lo += 8;
            hi -= 8;
        }
        while (lo < hi)
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
            build_luts();
            using namespace simdtl::platform;
            register_remove_i8 (isa_level::avx2, &remove_i8_avx2);
            register_remove_i16(isa_level::avx2, &remove_i16_avx2);
            register_remove_i32(isa_level::avx2, &remove_i32_avx2);
            register_reverse_i32(isa_level::avx2, &reverse_i32_avx2);
        }
    };
    const registrar g_registrar{};
} // namespace
