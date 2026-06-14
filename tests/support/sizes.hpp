#pragma once
#include <cstddef>
#include <array>

namespace simdtl_test
{
    // Edge-case lengths that stress vector-width boundaries and tails:
    // empty, sub-vector, around 4/8/16/32-lane widths (W-1, W, W+1), and larger.
    inline constexpr std::array<std::size_t, 22> kEdgeSizes = {
        0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 256, 1000, 1024};
}
