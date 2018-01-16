#pragma once
#include "simd.h"
#include "make_integer_sequence.h"

namespace simd
{
    template<typename simd_type>
    struct reverse_constant;

    template<>
    struct reverse_constant<SSE>
    {
        template<typename T>
        static constexpr std::enable_if_t<sizeof(T) == 1, __m128i> value()
        {
            return mdk::construct_with_reverse_sequence<__m128i, SSE::number_in_register<T>>();
        }

        template<typename T>
        static constexpr std::enable_if_t<sizeof(T) == 2, __m128i> value()
        {
            return { 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1 };
        }
    };

    template<>
    struct reverse_constant<AVX>
    {
        template<typename T>
        static constexpr std::enable_if_t<sizeof(T) == 1, __m256i> value()
        {
            return mdk::construct_with_reverse_sequence<__m256i, AVX::number_in_register<T>>();
        }

        template<typename T>
        static constexpr std::enable_if_t<sizeof(T) == 2, __m256i> value()
        {
            return { 30, 31, 28, 29, 26, 27, 24, 25, 22, 23, 20, 21, 18, 19, 16, 17, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1 };
        }
    };

    template<template <typename, typename...> class Container, typename T, typename... Args, typename simd_type = SSE>
    static void reverse(Container<T, Args...>& container)
    {
        const size_t size = container.size();
        const size_t half_way = size / 2 / SSE::number_in_register<T>;

        for (size_t i = 0; i < half_way; ++i)
        {
            auto& lhs = *reinterpret_cast<__m128i*>(&container[i * SSE::number_in_register<T>]);
            auto& rhs = *reinterpret_cast<__m128i*>(&container[size - SSE::number_in_register<T> - SSE::number_in_register<T> * i]);
            auto tmp = detail::shuffle_bytes(lhs, reverse_constant<simd_type>::value<T>()); 
            lhs = detail::shuffle_bytes(rhs, reverse_constant<simd_type>::value<T>());
            rhs = tmp;
        }

        std::reverse(container.begin() + half_way * SSE::number_in_register<T>, container.end() - half_way * SSE::number_in_register<T>);
    }
}
