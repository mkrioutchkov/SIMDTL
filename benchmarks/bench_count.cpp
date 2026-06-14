#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench/nanobench.h>

#include <simdtl/simdtl.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

int main()
{
    std::mt19937 gen(12345);
    std::uniform_int_distribution<int> dist(0, 3);
    std::vector<std::int32_t> data(std::size_t{1} << 20);
    for (auto& x : data) x = dist(gen);
    const std::int32_t needle = 2;

    ankerl::nanobench::Bench bench;
    bench.title("count<int32> over 2^20 elements").relative(true);

    bench.run("std::count", [&] {
        auto r = std::count(data.begin(), data.end(), needle);
        ankerl::nanobench::doNotOptimizeAway(r);
    });
    bench.run("simdtl::count (dispatched)", [&] {
        auto r = simdtl::count(data.data(), data.size(), needle);
        ankerl::nanobench::doNotOptimizeAway(r);
    });
    bench.run("simdtl::count_portable (std::simd)", [&] {
        auto r = simdtl::detail::count_portable<std::int32_t>(data.data(), data.size(), needle);
        ankerl::nanobench::doNotOptimizeAway(r);
    });
    return 0;
}
