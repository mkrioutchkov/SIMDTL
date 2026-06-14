#pragma once
// ── L3/L4: reverse (any element size) ─────────────────────────────────────────
// std::simd has no portable cross-lane permute, so reversal is SIMDTL's to own.
// Portable two-pointer swap works for ANY element size (generalizing the old
// library, which handled only 1- and 2-byte elements). For int32 a dispatched
// AVX2 kernel (vpermd block-reverse from both ends) takes over when available.
#include "../backend/names.hpp"
#include "../platform/dispatch.hpp"
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace simdtl
{
    template <class T>
    void reverse(T* first, std::size_t n) noexcept
    {
        if constexpr (std::is_same_v<T, std::int32_t>)
            if (auto fn = platform::reverse_i32_slot())
            {
                fn(first, n);
                return;
            }

        if (n < 2) return;
        std::size_t i = 0, j = n - 1;
        while (i < j)
        {
            const T t = first[i];
            first[i] = first[j];
            first[j] = t;
            ++i;
            --j;
        }
    }

    template <class C>
    void reverse(C& c) { reverse(c.data(), c.size()); }
} // namespace simdtl
