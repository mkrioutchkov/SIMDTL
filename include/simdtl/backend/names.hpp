#pragma once
// ── L0 seam: stable SIMDTL names over the backend's spelling ──────────────────
// The Parallelism TS (what vir-simd implements) and C++26 <simd> use DIFFERENT
// free-function names for the same operations. We funnel every use through these
// stable wrappers so the rename is absorbed in ONE place. The C++26 column is the
// P1928R15 rename map; flip the branch when a toolchain ships native <simd>.
//
//   stable name        TS / vir (today)     C++26 <simd> (tomorrow)
//   ----------------   ------------------   -----------------------
//   lane_count(mask)   popcount             reduce_count
//   find_first(mask)   find_first_set       reduce_min_index
//   find_last(mask)    find_last_set        reduce_max_index
//   hmin(v)            hmin                 reduce_min
//   hmax(v)            hmax                 reduce_max
//   hsum(v)            reduce               reduce
//   any_of/all_of/none_of                   (unchanged)
//   elem_aligned       element_aligned      simd_flag_default
//   vec_aligned        vector_aligned       simd_flag_aligned
#include "simd.hpp"
#include <cstddef>

namespace simdtl
{
    // Stable type aliases so nothing above this layer ever spells the backend
    // (`stdx::`) directly — a CI lint enforces it, making the C++26 swap one file.
    template <class T> using native = stdx::native_simd<T>;
    template <class T> using native_mask = typename stdx::native_simd<T>::mask_type;
    template <class T, std::size_t N>
    using fixed = stdx::simd<T, stdx::simd_abi::fixed_size<N>>;

    // Alignment flags used at every load/store site. Default to element_aligned
    // (always safe for arbitrary container memory); vec_aligned is the fast,
    // alignment-required form.
    inline constexpr auto elem_aligned = stdx::element_aligned;
    inline constexpr auto vec_aligned  = stdx::vector_aligned;

    // Mask reductions ---------------------------------------------------------
    template <class Mask> int  lane_count(const Mask& m) noexcept { return stdx::popcount(m); }
    template <class Mask> bool any_of    (const Mask& m) noexcept { return stdx::any_of(m); }
    template <class Mask> bool all_of    (const Mask& m) noexcept { return stdx::all_of(m); }
    template <class Mask> bool none_of   (const Mask& m) noexcept { return stdx::none_of(m); }

    // UB if no lane is set — callers MUST gate with any_of() first.
    template <class Mask> int find_first(const Mask& m) noexcept { return stdx::find_first_set(m); }
    template <class Mask> int find_last (const Mask& m) noexcept { return stdx::find_last_set(m); }

    // Horizontal value reductions (generic over element type — this is the fix
    // for the old horizontal_sum() that was hardcoded to float).
    template <class V> auto hsum(const V& v) noexcept { return stdx::reduce(v); }
    template <class V> auto hmin(const V& v) noexcept { return stdx::hmin(v); }
    template <class V> auto hmax(const V& v) noexcept { return stdx::hmax(v); }
} // namespace simdtl
