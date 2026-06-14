// ── Opt-in AVX2 kernel for count<int32> ───────────────────────────────────────
// Compiled as its OWN translation unit with /arch:AVX2 (MSVC) or -mavx2 (GCC/Clang).
// Self-registers into the dispatch slot at static-init time; the registration only
// "sticks" if the running CPU supports AVX2 (checked inside register_count_i32).
// This proves the multi-TU, dispatch-without-target_clones mechanism end-to-end.
#include "simdtl/platform/dispatch.hpp"

#include <immintrin.h>
#if defined(_MSC_VER)
#  include <intrin.h>   // __popcnt
#endif
#include <cstddef>
#include <cstdint>

namespace
{
    std::size_t count_i32_avx2(const std::int32_t* p, std::size_t n, std::int32_t value) noexcept
    {
        const __m256i needle = _mm256_set1_epi32(value);
        std::size_t total = 0;
        std::size_t i = 0;
        for (; i + 8 <= n; i += 8)
        {
            const __m256i v  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + i));
            const __m256i eq = _mm256_cmpeq_epi32(v, needle);
            // 8 packed int32 → 8-bit mask via movemask_ps; popcount = match count.
            const unsigned m = static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(eq)));
#if defined(_MSC_VER)
            total += static_cast<std::size_t>(__popcnt(m));
#else
            total += static_cast<std::size_t>(__builtin_popcount(m));
#endif
        }
        for (; i < n; ++i)
            total += (p[i] == value) ? std::size_t{1} : std::size_t{0};
        return total;
    }

    struct registrar
    {
        registrar() noexcept
        {
            simdtl::platform::register_count_i32(
                simdtl::platform::isa_level::avx2, &count_i32_avx2);
        }
    };
    const registrar g_registrar{};
} // namespace
