#pragma once
// ── L4: reduce / accumulate (sum) ─────────────────────────────────────────────
// Vector accumulator across the loop, one horizontal sum at the end. Replaces the
// old float-only horizontal_sum. NOTE: this assumes associativity/commutativity —
// for floating point the result may differ bit-for-bit from a sequential
// std::accumulate because the summation order changes.
#include "../backend/names.hpp"
#include <cstddef>

namespace simdtl
{
    template <class T>
    T reduce(const T* first, std::size_t n, T init = T{}) noexcept
    {
        using V = native<T>;
        constexpr std::size_t W = V::size();
        std::size_t i = 0;
        T result = init;
        if (n >= W)
        {
            V acc(T{0});
            for (; i + W <= n; i += W)
                acc += V(first + i, elem_aligned);
            result += hsum(acc);
        }
        for (; i < n; ++i)
            result += first[i];
        return result;
    }

    // Alias matching the STL spelling (same associativity caveat as reduce).
    template <class T>
    T accumulate(const T* first, std::size_t n, T init = T{}) noexcept
    {
        return reduce(first, n, init);
    }

    template <class C>
    typename C::value_type reduce(const C& c, typename C::value_type init = {})
    {
        return reduce(c.data(), c.size(), init);
    }
} // namespace simdtl
