#pragma once
#include "instructions.h"
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

        template<typename basic_t, typename T>
        static basic_t broadcast(const T& fill_with)
        {
            basic_t value;
            auto& set_broadcast = reinterpret_cast<T(&)[simd_t<basic_t>::number_in_register<T>]>(value);
            std::fill(std::begin(set_broadcast), std::end(set_broadcast), fill_with);
            return value;
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
        static auto process(T* arr, size_t size, const ScalarFunc& scalar_func, VectorizedFunc&& vectorized_func)
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
            
            return scalar_result;
        }

        template<typename F, typename T, typename... Args>
        static auto do_contiguous(const F& func, T& container, Args&&... args)
        {
            return func(container.data(), container.size(), std::forward<Args>(args)...);
        }
    }

    typedef detail::simd_t<__m256> AVX;
    typedef detail::simd_t<__m128> SSE;
}
