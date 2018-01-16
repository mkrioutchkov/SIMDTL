#pragma once
#include "simd.h"

namespace simd
{
    template<typename T, typename memory_t>
    static void add(T* arr, size_t size, memory_t v_value)
    {
        // On a good compiler, there is no reason to use this function
        const auto scalar_value = *reinterpret_cast<T*>(&v_value);
        const auto scalar = [&](T* arr, size_t size)
        {
            std::for_each(arr, arr + size, [&](T& value) { value += scalar_value; });
            return int();
        };

        const auto vectorized = [&](T* arr)
        {
            auto arr_mem = reinterpret_cast<memory_t*>(arr);
            *arr_mem = detail::add<T>(*arr_mem, v_value);
        };

        detail::process<detail::simd_t<memory_t>>(arr, size, scalar, vectorized);
    }

    template< typename TSIMD = AVX, typename T, typename U>
    static void add(T& contiguous_container, const U& v_value)
    {
        detail::do_contiguous(add<TSIMD, U>, contiguous_container, v_value);
    }
}
