#pragma once
// ── L4: copy_if / remove_if / remove (built on stream compaction) ─────────────
// The generic predicate forms are portable (std::simd mask + compress_store). The
// concrete remove(value) routes through a dispatched AVX2 compaction kernel for
// int32 (LUT + vpermd), falling back to the portable path everywhere else. This is
// the algorithm family std::simd cannot express on its own.
#include "../backend/names.hpp"
#include "../crosslane/compress.hpp"
#include "../platform/dispatch.hpp"
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace simdtl
{
    // copy_if: write kept elements to `out` (capacity >= count); return count written.
    template <class T, class Pred>
    std::size_t copy_if(const T* first, std::size_t n, T* out, Pred pred) noexcept
    {
        using V = native<T>;
        constexpr std::size_t W = V::size();
        std::size_t i = 0, k = 0;
        for (; i + W <= n; i += W)
        {
            V v(first + i, elem_aligned);
            k += compress_store(out + k, v, pred(v));
        }
        for (; i < n; ++i)
            if (pred(first[i])) out[k++] = first[i];
        return k;
    }

    // remove_if: in-place compaction keeping !pred; returns the new logical length.
    template <class T, class Pred>
    std::size_t remove_if(T* first, std::size_t n, Pred pred) noexcept
    {
        using V = native<T>;
        constexpr std::size_t W = V::size();
        std::size_t i = 0, k = 0;
        for (; i + W <= n; i += W)
        {
            V v(first + i, elem_aligned);
            k += compress_store(first + k, v, !pred(v));   // k <= i, so the store stays behind the read
        }
        for (; i < n; ++i)
            if (!pred(first[i])) first[k++] = first[i];
        return k;
    }

    // remove(value): in-place, drops every element == value; returns new length.
    template <class T>
    std::size_t remove(T* first, std::size_t n, T value) noexcept
    {
        if constexpr (std::is_same_v<T, std::int32_t>)
            if (auto fn = platform::remove_i32_slot())
                return fn(first, n, value);

        using V = native<T>;
        constexpr std::size_t W = V::size();
        std::size_t i = 0, k = 0;
        for (; i + W <= n; i += W)
        {
            V v(first + i, elem_aligned);
            k += compress_store(first + k, v, v != V(value));
        }
        for (; i < n; ++i)
            if (first[i] != value) first[k++] = first[i];
        return k;
    }

    // unique: drop consecutive duplicates in place; return the new logical length
    // (std::unique semantics). Sequential by nature; compares each element to the
    // ORIGINAL predecessor (kept in `prev`) so in-place writes don't corrupt it.
    template <class T>
    std::size_t unique(T* first, std::size_t n) noexcept
    {
        if (n == 0) return 0;
        std::size_t k = 1;
        T prev = first[0];
        for (std::size_t i = 1; i < n; ++i)
        {
            const T cur = first[i];
            if (!(cur == prev)) first[k++] = cur;
            prev = cur;
        }
        return k;
    }

    // partition: rearrange so all pred-true elements come first; return the
    // partition point (count of true). STABLE (preserves relative order within
    // each side) via two SIMD compaction passes through a scratch buffer.
    template <class T, class Pred>
    std::size_t partition(T* first, std::size_t n, Pred pred) noexcept
    {
        std::vector<T> tmp(n);
        const std::size_t k = copy_if(first, n, tmp.data(), pred);            // trues, in order
        copy_if(first, n, tmp.data() + k, [&](auto x) { return !pred(x); });  // falses, in order
        for (std::size_t i = 0; i < n; ++i) first[i] = tmp[i];
        return k;
    }

    template <class C, class Pred>
    std::size_t remove_if(C& c, Pred pred) { return remove_if(c.data(), c.size(), pred); }
} // namespace simdtl
