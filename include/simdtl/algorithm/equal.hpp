#pragma once
// ── L4: equal / mismatch ──────────────────────────────────────────────────────
// equal short-circuits on the first chunk that isn't all-equal; mismatch locates
// the first differing index via any_of()-gated find_first().
#include "../backend/names.hpp"
#include <cstddef>
#include <utility>

namespace simdtl
{
    template <class T>
    bool equal(const T* a, const T* b, std::size_t n) noexcept
    {
        using V = native<T>;
        constexpr std::size_t W = V::size();
        std::size_t i = 0;
        for (; i + W <= n; i += W)
            if (!all_of(V(a + i, elem_aligned) == V(b + i, elem_aligned)))
                return false;
        for (; i < n; ++i)
            if (!(a[i] == b[i]))
                return false;
        return true;
    }

    template <class T>
    std::pair<const T*, const T*> mismatch(const T* a, const T* b, std::size_t n) noexcept
    {
        using V = native<T>;
        constexpr std::size_t W = V::size();
        std::size_t i = 0;
        for (; i + W <= n; i += W)
        {
            const auto m = (V(a + i, elem_aligned) != V(b + i, elem_aligned));
            if (any_of(m))
            {
                const std::size_t k = i + static_cast<std::size_t>(find_first(m));
                return {a + k, b + k};
            }
        }
        for (; i < n; ++i)
            if (!(a[i] == b[i]))
                return {a + i, b + i};
        return {a + n, b + n};
    }

    template <class C>
    bool equal(const C& a, const C& b)
    {
        return a.size() == b.size() && equal(a.data(), b.data(), a.size());
    }
} // namespace simdtl
