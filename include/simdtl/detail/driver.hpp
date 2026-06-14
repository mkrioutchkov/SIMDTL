#pragma once
// ── L2: the tail-handling driver (the spine under every algorithm) ────────────
// Runs `vec_step` over floor(n/W)*W elements in native-width chunks, then
// `scalar_step` over the remainder. Uses unaligned (element_aligned) loads — no
// head-alignment peel; on modern uarch the unaligned penalty is negligible and a
// head peel buys nothing. Modernizes the old hand-rolled detail::process().
#include "../backend/names.hpp"
#include <cstddef>

namespace simdtl::detail
{
    template <class T, class VecStep, class ScalarStep>
    void for_each_chunk(const T* first, std::size_t n, VecStep vec_step, ScalarStep scalar_step)
    {
        using V = native<T>;
        constexpr std::size_t W = V::size();

        std::size_t i = 0;
        for (; i + W <= n; i += W)
        {
            V v(first + i, elem_aligned);
            vec_step(v);
        }
        for (; i < n; ++i)
            scalar_step(first[i]);
    }
} // namespace simdtl::detail
