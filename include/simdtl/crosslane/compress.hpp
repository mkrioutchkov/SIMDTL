#pragma once
// ── L3: stream compaction primitive (the marquee gap in std::simd) ────────────
// std::simd offers only position-preserving blend (where()); it cannot pack the
// selected lanes into a contiguous prefix. compress_store does exactly that.
//
// This portable version extracts lanes and writes exactly `count` elements (so it
// never overruns dst — dst needs room only for the kept elements). The fast AVX2
// path lives in src/kernels/crosslane_avx2.cpp and is reached, for concrete ops
// like remove(value), through the runtime dispatch slot.
#include "../backend/names.hpp"
#include <cstddef>

namespace simdtl
{
    template <class T>
    std::size_t compress_store(T* dst, native<T> v, native_mask<T> keep) noexcept
    {
        constexpr std::size_t W = native<T>::size();
        std::size_t k = 0;
        for (std::size_t j = 0; j < W; ++j)
            if (keep[j])
                dst[k++] = v[j];
        return k;
    }
} // namespace simdtl
