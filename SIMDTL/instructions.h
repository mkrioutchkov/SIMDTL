#pragma once
#include <intrin.h>
#include <type_traits>

namespace simd
{
    namespace detail
    {
        template<typename T>
        inline std::enable_if_t<sizeof(T) == 1, __m128i> compare_equality(__m128i lhs, __m128i rhs) { return _mm_cmpeq_epi8(lhs, rhs); }
        template<typename T>
        inline std::enable_if_t<sizeof(T) == 1, __m256i> compare_equality(__m256i lhs, __m256i rhs) { return _mm256_cmpeq_epi8(lhs, rhs); }
        template<typename T>
        inline std::enable_if_t<sizeof(T) == 2, __m128i> compare_equality(__m128i lhs, __m128i rhs) { return _mm_cmpeq_epi16(lhs, rhs); }
        template<typename T>
        inline std::enable_if_t<sizeof(T) == 2, __m256i> compare_equality(__m256i lhs, __m256i rhs) { return _mm256_cmpeq_epi16(lhs, rhs); }
        template<typename T>
        inline std::enable_if_t<sizeof(T) == 4, __m128i> compare_equality(__m128i lhs, __m128i rhs) { return _mm_cmpeq_epi32(lhs, rhs); }
        template<typename T>
        inline std::enable_if_t<sizeof(T) == 4, __m256i> compare_equality(__m256i lhs, __m256i rhs) { return _mm256_cmpeq_epi32(lhs, rhs); }
        template<typename T>
        inline std::enable_if_t<sizeof(T) == 8, __m128i> compare_equality(__m128i lhs, __m128i rhs) { return _mm_cmpeq_epi64(lhs, rhs); }
        template<typename T>
        inline std::enable_if_t<sizeof(T) == 8, __m256i> compare_equality(__m256i lhs, __m256i rhs) { return _mm256_cmpeq_epi64(lhs, rhs); }

        inline auto shuffle_bytes(__m128i lhs, __m128i rhs) { return _mm_shuffle_epi8(lhs, rhs); }
        inline auto shuffle_bytes(__m256i lhs, __m256i rhs) { return _mm256_shuffle_epi8(lhs, rhs); }

        template<typename T>
        inline std::enable_if_t<std::is_same_v<T, float>, __m128> add(__m128 lhs, __m128 rhs) { return _mm_add_ps(lhs, rhs); }
        template<typename T>
        inline std::enable_if_t<std::is_same_v<T, float>, __m256> add(__m256 lhs, __m256 rhs) { return _mm256_add_ps(lhs, rhs); }
        template<typename T>
        inline std::enable_if_t<std::is_same_v<T, double>, __m128> add(__m128 lhs, __m128 rhs) { return _mm_add_pd(lhs, rhs); }
        template<typename T>
        inline std::enable_if_t<std::is_same_v<T, double>, __m256> add(__m256 lhs, __m256 rhs) { return _mm256_add_pd(lhs, rhs); }

        inline auto move_mask(__m128i mask) { return _mm_movemask_epi8(mask); }
        inline auto move_mask(__m256i mask) { return _mm256_movemask_epi8(mask); }

        inline auto and_si128(__m128i lhs, __m128i rhs) { return _mm_and_si128(lhs, rhs); }
        inline auto and_si128(__m256i lhs, __m256i rhs) { return _mm256_and_si256(lhs, rhs); }

        inline auto xor_si128(__m128i lhs, __m128i rhs) { return _mm_xor_si128(lhs, rhs); }
        inline auto xor_si128(__m256i lhs, __m256i rhs) { return _mm256_xor_si256(lhs, rhs); }
    }
}