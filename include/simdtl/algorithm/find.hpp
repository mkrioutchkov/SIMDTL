#pragma once
// ── L4: find / find_if — early-exit scan ──────────────────────────────────────
// any_of() GATES find_first() because find_first is UB on an all-false mask.
// Returns a pointer to the first match, or first+n if none (std::find semantics).
#include "../backend/names.hpp"
#include <cstddef>

namespace simdtl
{
    template <class T>
    const T* find(const T* first, std::size_t n, T value) noexcept
    {
        using V = native<T>;
        constexpr std::size_t W = V::size();
        std::size_t i = 0;
        for (; i + W <= n; i += W)
        {
            const auto m = (V(first + i, elem_aligned) == V(value));
            if (any_of(m))
                return first + i + static_cast<std::size_t>(find_first(m));
        }
        for (; i < n; ++i)
            if (first[i] == value)
                return first + i;
        return first + n;
    }

    template <class T, class Pred>
    const T* find_if(const T* first, std::size_t n, Pred pred) noexcept
    {
        using V = native<T>;
        constexpr std::size_t W = V::size();
        std::size_t i = 0;
        for (; i + W <= n; i += W)
        {
            const auto m = pred(V(first + i, elem_aligned));
            if (any_of(m))
                return first + i + static_cast<std::size_t>(find_first(m));
        }
        for (; i < n; ++i)
            if (pred(first[i]))
                return first + i;
        return first + n;
    }

    template <class C>
    const typename C::value_type* find(const C& c, typename C::value_type value)
    {
        return find(c.data(), c.size(), value);
    }
    template <class C, class Pred>
    const typename C::value_type* find_if(const C& c, Pred pred)
    {
        return find_if(c.data(), c.size(), pred);
    }
} // namespace simdtl
