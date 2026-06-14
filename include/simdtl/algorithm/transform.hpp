#pragma once
// ── L4: transform (unary & binary) ────────────────────────────────────────────
// `op` is ELEMENTAL: callable on both simd<T> (→ simd<T>) and T (→ T). A generic
// lambda works for both, e.g. unary `[](auto x){ return x * x; }`, binary
// `[](auto a, auto b){ return a + b; }`. (Trivial maps auto-vectorize anyway; the
// value of transform is fused/masked elemental ops expressed once.)
#include "../backend/names.hpp"
#include <cstddef>

namespace simdtl
{
    template <class T, class Op>
    void transform(const T* first, std::size_t n, T* out, Op op) noexcept
    {
        using V = native<T>;
        constexpr std::size_t W = V::size();
        std::size_t i = 0;
        for (; i + W <= n; i += W)
        {
            V r = op(V(first + i, elem_aligned));
            r.copy_to(out + i, elem_aligned);
        }
        for (; i < n; ++i)
            out[i] = op(first[i]);
    }

    template <class T, class Op>
    void transform(const T* a, const T* b, std::size_t n, T* out, Op op) noexcept
    {
        using V = native<T>;
        constexpr std::size_t W = V::size();
        std::size_t i = 0;
        for (; i + W <= n; i += W)
        {
            V r = op(V(a + i, elem_aligned), V(b + i, elem_aligned));
            r.copy_to(out + i, elem_aligned);
        }
        for (; i < n; ++i)
            out[i] = op(a[i], b[i]);
    }
} // namespace simdtl
