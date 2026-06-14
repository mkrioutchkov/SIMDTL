#pragma once
#include <cstddef>
#include <random>
#include <vector>

namespace simdtl_test
{
    // Whole-number values (in [lo,hi]) cast to T, so equality-based algorithms are
    // meaningful even for floating-point T. Fixed seed → reproducible in CI.
    template <class T>
    std::vector<T> make_values(std::size_t n, long long lo, long long hi, unsigned seed)
    {
        std::vector<T> v(n);
        std::mt19937 gen(seed);
        std::uniform_int_distribution<long long> dist(lo, hi);
        for (auto& x : v)
            x = static_cast<T>(dist(gen));
        return v;
    }
}
