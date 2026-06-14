// ── Opt-in AVX2 kernels for count<int8 / int16 / int32> ───────────────────────
// Compiled as its own /arch:AVX2 TU; self-registers into the dispatch slots (only
// "sticks" if the CPU supports AVX2). Pattern: compare a whole register, collapse
// the per-lane results to a bitmask, popcount it.
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

    std::size_t count_i8_avx2(const std::int8_t* p, std::size_t n, std::int8_t value) noexcept
    {
        const __m256i needle = _mm256_set1_epi8(value);
        std::size_t total = 0, i = 0;
        for (; i + 32 <= n; i += 32)   // 32 bytes / iter
        {
            const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + i));
            total += popcnt(static_cast<unsigned>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, needle))));
        }
        for (; i < n; ++i) total += (p[i] == value) ? std::size_t{1} : std::size_t{0};
        return total;
    }

    std::size_t count_i16_avx2(const std::int16_t* p, std::size_t n, std::int16_t value) noexcept
    {
        const __m256i needle = _mm256_set1_epi16(value);
        std::size_t total = 0, i = 0;
        for (; i + 16 <= n; i += 16)   // 16 shorts / iter
        {
            const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + i));
            // movemask is byte-granular -> 2 bits per matching 16-bit lane -> /2.
            total += popcnt(static_cast<unsigned>(_mm256_movemask_epi8(_mm256_cmpeq_epi16(v, needle)))) / 2;
        }
        for (; i < n; ++i) total += (p[i] == value) ? std::size_t{1} : std::size_t{0};
        return total;
    }

    std::size_t count_i32_avx2(const std::int32_t* p, std::size_t n, std::int32_t value) noexcept
    {
        const __m256i needle = _mm256_set1_epi32(value);
        std::size_t total = 0, i = 0;
        for (; i + 8 <= n; i += 8)     // 8 ints / iter
        {
            const __m256i v  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + i));
            const __m256i eq = _mm256_cmpeq_epi32(v, needle);
            total += popcnt(static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(eq))));
        }
        for (; i < n; ++i) total += (p[i] == value) ? std::size_t{1} : std::size_t{0};
        return total;
    }

    struct registrar
    {
        registrar() noexcept
        {
            using namespace simdtl::platform;
            register_count_i8 (isa_level::avx2, &count_i8_avx2);
            register_count_i16(isa_level::avx2, &count_i16_avx2);
            register_count_i32(isa_level::avx2, &count_i32_avx2);
        }
    };
    const registrar g_registrar{};
} // namespace
