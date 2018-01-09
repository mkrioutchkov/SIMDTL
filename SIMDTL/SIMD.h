#pragma once
#include "simd_instructions.h"
#include <algorithm>
#include <intrin.h>
#include <vector>
#include <numeric>
#include <iterator>

#define WRAP_IN_CLOSURE(template_func) [](auto&&... args) { return template_func(std::forward<decltype(args)>(args)...); };
#define WRAP_IN_CLOSURE_EXTENDED(before, template_func, after) [&](auto&&... args) { before; template_func(std::forward<decltype(args)>(args)...); after; };

namespace simd
{
    namespace detail
    {
        template<typename T>
        struct simd_t_specialize 
        { 
            simd_t_specialize() { static_assert(false, "Invalid argument - doesn't map to any supported simd_t"); }
        };

        template<>
        struct simd_t_specialize<__m128> 
        {
            typedef __m128i intergral_t;
        };

        template<>
        struct simd_t_specialize<__m256>
        {
            typedef __m256i intergral_t;
        };

        template<typename T>
        struct simd_t : simd_t_specialize<T>
        {
            typedef T basic_t;
            static constexpr size_t bytes_in_register = sizeof(basic_t);
            template<typename U>
            static constexpr size_t number_in_register = bytes_in_register / sizeof(U);
        };

        template<typename T, typename TOut = T, typename basic_t>
        static auto horizontal_sum(const basic_t& data)
        {
            // these are usually done at the end of tight loops, so performance isn't really important. Something generic is better.
            auto& values = reinterpret_cast<const float(&)[simd_t<basic_t>::number_in_register<float>]>(data);
            return std::accumulate(std::begin(values), std::end(values), TOut{});
        }

        template<typename T, typename basic_t>
        static void broadcast(basic_t& value, const T& fill_with)
        {
            // like horizontal_sum, performance here is unlikely to be important as this is usually BEFORE entry into a tight loop
            auto& set_broadcast = reinterpret_cast<T(&)[simd_t<basic_t>::number_in_register<T>]>(value);
            std::fill(std::begin(set_broadcast), std::end(set_broadcast), fill_with);
        }

        template<typename T>
        static std::enable_if_t<std::is_integral_v<T>, T> force_xor(const T& lhs, const T& rhs)
        {
            return T(lhs ^ rhs);
        }

        inline uint32_t force_xor(float lhs, float rhs)
        {
            return *reinterpret_cast<uint32_t*>(&lhs) ^ *reinterpret_cast<uint32_t*>(&rhs);
        }

        inline uint64_t force_xor(double lhs, double rhs)
        {
            return *reinterpret_cast<uint64_t*>(&lhs) ^ *reinterpret_cast<uint64_t*>(&rhs);
        }

        template<typename simd_t, typename T, typename ScalarFunc, typename VectorizedFunc>
        static auto process(T* arr, size_t size, const ScalarFunc& scalar_func, VectorizedFunc& vectorized_func)
        {
            typedef decltype(scalar_func(arr, size)) ReturnValue;
            struct scalar_result_t
            {
                ReturnValue before_alignment_result;
                ReturnValue after_alignment_result;
                T* alignment_starts_at;
                size_t remaining_elements;
            };

            constexpr auto number_of_times_to_unroll_loop = 1; // haven't really found any benefits for > 1. keeping it here as an idea.
            constexpr auto align_to = simd_t::bytes_in_register * number_of_times_to_unroll_loop;
            constexpr auto alignToMask = align_to - 1;
            const auto addressStart = reinterpret_cast<size_t>(arr);
            const auto elementsNeededToAlign = (align_to - (addressStart & alignToMask)) % align_to / sizeof(T);
            const auto addressEnd = addressStart + size * sizeof(T);
            const auto elementsAtEnd = (addressEnd & alignToMask) / sizeof(T);
           
            auto scalar_result = scalar_result_t
            {
                scalar_func(arr, elementsNeededToAlign),
                scalar_func(arr + size - elementsAtEnd, elementsAtEnd),
                arr + elementsNeededToAlign,
                size - elementsNeededToAlign - elementsAtEnd
            };

            // todo: can use loop unrolling here in the future
            for (size_t i = 0; i < scalar_result.remaining_elements; i += simd_t::number_in_register<T> * number_of_times_to_unroll_loop)
                vectorized_func(scalar_result.alignment_starts_at + i);
            
            return std::move(scalar_result);
        }
    }

    typedef detail::simd_t<__m256> AVX;
    typedef detail::simd_t<__m128> SSE;

    template<typename F, typename T, typename... Args>
    static auto do_contiguous(const F& func, T& container, Args&&... args)
    {
        return func(container.data(), container.size(), std::forward<Args>(args)...);
    }

    template<typename simd_type, typename T>
    static void replace(T* arr, const size_t size, const T& replacee, const T& replacer)
    {
        static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>, "Replace not supported for this type");

        const auto scalar = [&](T* arr, size_t size)
        {
            std::replace(arr, arr + size, replacee, replacer);
            return int(); // avoid compile error
        };

        typename simd_type::intergral_t replacee_broadcast;
        detail::broadcast(replacee_broadcast, replacee);
        typename simd_type::intergral_t replacer_broadcast;
        detail::broadcast(replacer_broadcast, detail::force_xor(replacer, replacee));

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
        do_contiguous(replace<TSIMD, U>, contiguous_container, replacee, replacer);
    }

    template<typename simd_type, typename T>
    static auto count(const T* arr, const size_t size, const T& find)
    {
        const auto scalar = [&](const T* arr, size_t size)
        {
            return std::count(arr, arr + size, find);
        };

        typename simd_type::intergral_t find_broadcast;
        detail::broadcast(find_broadcast, find);

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
        return do_contiguous(count<TSIMD, U>, contiguous_container, find);
    }

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
        do_contiguous(add<TSIMD, U>, contiguous_container, v_value);
    }
}
