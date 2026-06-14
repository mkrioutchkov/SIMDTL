#pragma once
// ── L4: min / max / minmax (value) and min_element / max_element (pointer) ─────
// Carry a vector accumulator across the loop, then ONE horizontal fold. Generic
// over element type (fixes the old float-only horizontal_sum). Precondition n>0
// for the value forms; *_element return first+n on empty (std::*_element style).
#include "../backend/names.hpp"
#include "find.hpp"
#include <cstddef>
#include <utility>

namespace simdtl
{
    template <class T>
    T min_value(const T* first, std::size_t n) noexcept
    {
        using V = native<T>;
        constexpr std::size_t W = V::size();
        if (n == 0) return T{};
        std::size_t i = 0;
        T result;
        if (n >= W)
        {
            V acc(first, elem_aligned);
            for (i = W; i + W <= n; i += W)
                acc = elem_min(acc, V(first + i, elem_aligned));
            result = hmin(acc);
        }
        else { result = first[0]; i = 1; }
        for (; i < n; ++i)
            if (first[i] < result) result = first[i];
        return result;
    }

    template <class T>
    T max_value(const T* first, std::size_t n) noexcept
    {
        using V = native<T>;
        constexpr std::size_t W = V::size();
        if (n == 0) return T{};
        std::size_t i = 0;
        T result;
        if (n >= W)
        {
            V acc(first, elem_aligned);
            for (i = W; i + W <= n; i += W)
                acc = elem_max(acc, V(first + i, elem_aligned));
            result = hmax(acc);
        }
        else { result = first[0]; i = 1; }
        for (; i < n; ++i)
            if (result < first[i]) result = first[i];
        return result;
    }

    template <class T>
    std::pair<T, T> minmax_value(const T* first, std::size_t n) noexcept
    {
        using V = native<T>;
        constexpr std::size_t W = V::size();
        if (n == 0) return {T{}, T{}};
        std::size_t i = 0;
        T lo, hi;
        if (n >= W)
        {
            V vlo(first, elem_aligned), vhi = vlo;
            for (i = W; i + W <= n; i += W)
            {
                const V v(first + i, elem_aligned);
                vlo = elem_min(vlo, v);
                vhi = elem_max(vhi, v);
            }
            lo = hmin(vlo); hi = hmax(vhi);
        }
        else { lo = hi = first[0]; i = 1; }
        for (; i < n; ++i)
        {
            if (first[i] < lo) lo = first[i];
            if (hi < first[i]) hi = first[i];
        }
        return {lo, hi};
    }

    // Pointer to the FIRST minimum/maximum (std::min_element / std::max_element).
    template <class T>
    const T* min_element(const T* first, std::size_t n) noexcept
    {
        if (n == 0) return first;
        return find(first, n, min_value(first, n));
    }
    template <class T>
    const T* max_element(const T* first, std::size_t n) noexcept
    {
        if (n == 0) return first;
        return find(first, n, max_value(first, n));
    }

    template <class C> auto min_value(const C& c)  { return min_value(c.data(), c.size()); }
    template <class C> auto max_value(const C& c)  { return max_value(c.data(), c.size()); }
    template <class C> auto minmax_value(const C& c){ return minmax_value(c.data(), c.size()); }
} // namespace simdtl
