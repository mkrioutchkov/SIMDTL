#pragma once
#include "simd.h"
#include "preserve_constness.h"
// Everything to do with SSE4 _mm_cmpistrm instruction. In here we assume SSE only

namespace simd
{
    namespace detail
    {
        namespace range_comparison
        {
            template<int mask, typename T, typename GF, typename SF, typename VF>
            static auto iterate(T* arr, const size_t size, const GF& get_src_operand, const SF& process_scalar, const VF& process_vectorized)
            {
                static_assert(std::is_same_v<__m128i, decltype(get_src_operand())>, "get_src_operand must return __m128i");

                return detail::process<SSE>(arr, size, process_scalar, [&](auto* arr)
                {
                    auto& value = *reinterpret_cast<mdk::preserve_constness_t<T, __m128i*>>(arr);
                    auto v_mask = _mm_cmpistrm(get_src_operand(), value, mask);
                    process_vectorized(value, v_mask);
                });
            }
        }
    }

    template<int mask = _SIDD_UNIT_MASK, typename T, typename SF, typename VF>
    static auto transform_in_range(T* arr, const size_t size, const T(&ranges)[sizeof(__m128i) / sizeof(T)], const SF& process_scalar, const VF& process_vectorized)
    {
        const auto get_src_operand = [&]() { return *reinterpret_cast<const __m128i*>(ranges); };
        return detail::range_comparison::iterate<_SIDD_CMP_RANGES | mask>(arr, size, get_src_operand, process_scalar, process_vectorized);
    }

    namespace specific
    {
        namespace range_comparison
        {
            namespace string
            {     
                inline auto convert_case(char* arr, size_t size, const char(&ranges)[sizeof(__m128i)])
                {
                    const auto process_scalar = [&](char* arr, size_t size) 
                    { 
                        std::for_each(arr, arr + size, [&](char& c) 
                        {
                            if (c >= ranges[0] && c <= ranges[1])
                                c ^= 0x20;
                        });
                        return int();
                    };
                    const static auto process_vectorized = [](__m128i& chars, __m128i mask) 
                    { 
                        mask = _mm_and_si128(mask, detail::broadcast<__m128i>((char)0x20));
                        chars = _mm_xor_si128(chars, mask);
                        return int();
                    };

                    transform_in_range(arr, size, ranges, process_scalar, process_vectorized);
                }

                inline auto to_lower(char* arr, size_t size)
                {
                    convert_case(arr, size, { 'A', 'Z' });
                }
            }
        }
    }
}