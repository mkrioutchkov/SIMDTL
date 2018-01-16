#pragma once
#include "simd.h"
#include "preserve_constness.h"
// Everything to do with SSE4 _mm_cmpistrm instruction. In here we assume SSE only

namespace simd
{
    namespace detail
    {
        namespace string
        {
            namespace range_comparison
            {
                template<int mask, typename T, typename GF, typename SF, typename VF>
                static auto iterate(T* arr, const size_t size, const GF& get_src_operand, const SF& process_scalar, const VF& process_vectorized)
                {
                    static_assert(std::is_same_v<__m128i, decltype(get_src_operand())>, "get_src_operand must return __m128i");

                    return detail::process<SSE>(arr, size, process_scalar, [&](auto* arr)
                    {
                        auto& value = *reinterpret_cast<mdk::preserve_constness_t<T, __m128i>*>(arr);
                        auto v_mask = _mm_cmpistrm(get_src_operand(), value, mask);
                        process_vectorized(value, v_mask);
                    });
                }

                template<typename> struct Get_Ops {};
                template<> struct Get_Ops<char> { static constexpr auto value = _SIDD_SBYTE_OPS; };
                template<> struct Get_Ops<uint8_t> { static constexpr auto value = _SIDD_UBYTE_OPS; };
                template<> struct Get_Ops<int16_t> { static constexpr auto value = _SIDD_SWORD_OPS; };
                template<> struct Get_Ops<uint16_t> { static constexpr auto value = _SIDD_UWORD_OPS; };

                template<int mask = _SIDD_UNIT_MASK, typename T, typename SF, typename VF>
                static auto iterate_in_range(T* arr, const size_t size, const T(&ranges)[sizeof(__m128i) / sizeof(T)], const SF& process_scalar, const VF& process_vectorized)
                {
                    const auto get_src_operand = [&]() { return *reinterpret_cast<const __m128i*>(ranges); };
                    return iterate<_SIDD_CMP_RANGES | mask>(arr, size, get_src_operand, process_scalar, process_vectorized);
                }
            }
        }
    }

    namespace specific
    {
        namespace string
        {
            namespace range_comparison
            {     
                template<typename T>
                static auto count_in_range(const T* arr, const size_t size, const T(&ranges)[sizeof(__m128i) / sizeof(T)])
                {
                    const auto process_scalar = [&](const T* arr, const size_t size)
                    {
                        return std::count_if(arr, arr + size, [&](auto& c)
                        {
                            for (size_t i = 0; i < sizeof(__m128i) / sizeof(T); i += 2)
                                if (c >= ranges[i] && c <= ranges[i + 1])
                                    return true;
                            return false;
                        });
                    };

                    size_t vectorized_count = 0;
                    const static auto process_vectorized = [&](const __m128i&, __m128i v_mask)
                    {
                        const auto mask = *reinterpret_cast<uint32_t*>(&v_mask);
                        vectorized_count += _mm_popcnt_u64(mask);
                        return int();
                    };

                    constexpr auto mask = _SIDD_BIT_MASK | detail::string::range_comparison::Get_Ops<T>::value;
                    auto result = detail::string::range_comparison::iterate_in_range<mask, const T>(arr, size, ranges, process_scalar, process_vectorized);
                    return result.before_alignment_result + result.after_alignment_result + vectorized_count;
                }

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

                    detail::string::range_comparison::iterate_in_range(arr, size, ranges, process_scalar, process_vectorized);
                }

                inline auto to_lower(char* arr, size_t size)
                {
                    convert_case(arr, size, { 'A', 'Z' });
                }

                inline auto to_upper(char* arr, size_t size)
                {
                    convert_case(arr, size, { 'a', 'z' });
                }

                inline auto flip_case(char* arr, size_t size)
                {
                    convert_case(arr, size, { 'A', 'Z', 'a', 'z' });
                }
            }
        }
    }
}