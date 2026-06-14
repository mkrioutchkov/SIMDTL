#pragma once
// ── L4: replace / replace_if (in place) ───────────────────────────────────────
// where(mask, v) = new is a hardware blend — correct for ALL element types incl.
// float, with no strict-aliasing hazard. This OBSOLETES the old library's XOR
// trick (compare → AND replacer^replacee → XOR into data) and force_xor().
#include "../backend/names.hpp"
#include <cstddef>

namespace simdtl
{
    template <class T>
    void replace(T* first, std::size_t n, T old_value, T new_value) noexcept
    {
        using V = native<T>;
        constexpr std::size_t W = V::size();
        std::size_t i = 0;
        for (; i + W <= n; i += W)
        {
            V v(first + i, elem_aligned);
            where(v == V(old_value), v) = V(new_value);
            v.copy_to(first + i, elem_aligned);
        }
        for (; i < n; ++i)
            if (first[i] == old_value) first[i] = new_value;
    }

    template <class T, class Pred>
    void replace_if(T* first, std::size_t n, Pred pred, T new_value) noexcept
    {
        using V = native<T>;
        constexpr std::size_t W = V::size();
        std::size_t i = 0;
        for (; i + W <= n; i += W)
        {
            V v(first + i, elem_aligned);
            where(pred(v), v) = V(new_value);
            v.copy_to(first + i, elem_aligned);
        }
        for (; i < n; ++i)
            if (pred(first[i])) first[i] = new_value;
    }
} // namespace simdtl
