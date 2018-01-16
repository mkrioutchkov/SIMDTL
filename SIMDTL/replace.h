#pragma once
#include "simd.h"

namespace simd
{
    template<typename simd_type, typename T>
    static void replace(T* arr, const size_t size, const T& replacee, const T& replacer)
    {
        static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>, "Replace not supported for this type");

        const auto scalar = [&](T* arr, size_t size)
        {
            std::replace(arr, arr + size, replacee, replacer);
            return int(); // avoid compile error
        };

        auto replacee_broadcast = detail::broadcast<typename simd_type::intergral_t>(replacee);
        auto replacer_broadcast = detail::broadcast<typename simd_type::intergral_t>(detail::force_xor(replacer, replacee));

        const auto vectorized = [&](T* arr)
        {
            auto arr_mem = reinterpret_cast<typename simd_type::intergral_t*>(arr);
            auto mask = detail::compare_equality<T>(*arr_mem, replacee_broadcast);
            mask = detail::and_si128(mask, replacer_broadcast);
            *arr_mem = detail::xor_si128(*arr_mem, mask);
        };

        detail::process<simd_type>(arr, size, scalar, vectorized);
    }

    template< typename TSIMD = AVX, typename T, typename U>
    static void replace(T& contiguous_container, const U& replacee, const U& replacer)
    {
        detail::do_contiguous(replace<TSIMD, U>, contiguous_container, replacee, replacer);
    }
}
