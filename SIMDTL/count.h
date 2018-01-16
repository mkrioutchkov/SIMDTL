#pragma once
#include "simd.h"

namespace simd
{
    template<typename simd_type, typename T>
    static auto count(const T* arr, const size_t size, const T& find)
    {
        const auto scalar = [&](const T* arr, size_t size)
        {
            return std::count(arr, arr + size, find);
        };

        auto find_broadcast = detail::broadcast<typename simd_type::intergral_t>(find);

        size_t vectorized_count = 0;
        const auto vectorized = [&](const T* arr)
        {
            auto arr_mem = reinterpret_cast<const typename simd_type::intergral_t*>(arr);
            const auto v_mask = detail::compare_equality<T>(*arr_mem, find_broadcast);
            const auto mask = detail::move_mask(v_mask);
            vectorized_count += _mm_popcnt_u64(mask);
        };

        auto result = detail::process<simd_type>(arr, size, scalar, vectorized);
        vectorized_count /= sizeof(T);
        return result.before_alignment_result + result.after_alignment_result + vectorized_count;
    }

    template< typename TSIMD = AVX, typename T, typename U>
    static auto count(T& contiguous_container, const U& find)
    {
        return detail::do_contiguous(count<TSIMD, U>, contiguous_container, find);
    }
}
