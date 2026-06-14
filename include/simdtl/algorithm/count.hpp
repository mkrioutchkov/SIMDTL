#pragma once
// ── L4: count / count_if — the MVP vertical slice and the template others copy ──
// Portable path: drive native-width chunks, build a mask, accumulate lane_count.
// (Note: lane_count == popcount(mask) returns the LANE count directly — the old
// library's movemask+popcnt-then-divide-by-sizeof correction is gone.)
// Fast path: for std::int32_t, route through the runtime dispatch slot if a kernel
// is installed; otherwise the portable path runs everywhere.
#include "../backend/names.hpp"
#include "../detail/driver.hpp"
#include "../platform/dispatch.hpp"
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace simdtl
{
    namespace detail
    {
        template <class T>
        std::size_t count_portable(const T* first, std::size_t n, T value) noexcept
        {
            using V = native<T>;
            std::size_t total = 0;
            for_each_chunk<T>(
                first, n,
                [&](V v) { total += static_cast<std::size_t>(lane_count(v == V(value))); },
                [&](T x) { total += (x == value) ? std::size_t{1} : std::size_t{0}; });
            return total;
        }
    } // namespace detail

    // count(range) — number of elements equal to `value`.
    template <class T>
    std::size_t count(const T* first, std::size_t n, T value) noexcept
    {
        if constexpr (std::is_same_v<T, std::int32_t>)
            if (auto fn = platform::count_i32_slot())
                return fn(first, n, value);
        return detail::count_portable<T>(first, n, value);
    }

    // count(container) — any contiguous container exposing data()/size().
    template <class C>
    std::size_t count(const C& c, typename C::value_type value)
    {
        return count(c.data(), c.size(), value);
    }

    // count_if(range, pred) — `pred` is ELEMENTAL: callable on both simd<T> (→ mask)
    // and T (→ bool), e.g. `[](auto x){ return x > decltype(x)(5); }`.
    template <class T, class Pred>
    std::size_t count_if(const T* first, std::size_t n, Pred pred) noexcept
    {
        using V = native<T>;
        std::size_t total = 0;
        detail::for_each_chunk<T>(
            first, n,
            [&](V v) { total += static_cast<std::size_t>(lane_count(pred(v))); },
            [&](T x) { total += pred(x) ? std::size_t{1} : std::size_t{0}; });
        return total;
    }

    template <class C, class Pred>
    std::size_t count_if(const C& c, Pred pred)
    {
        return count_if(c.data(), c.size(), pred);
    }
} // namespace simdtl
