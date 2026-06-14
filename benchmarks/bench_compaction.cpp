#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench/nanobench.h>

#include <simdtl/simdtl.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

int main()
{
    std::mt19937 gen(999);
    std::uniform_int_distribution<int> dist(0, 9);
    // Cache-resident size (~32 KB) so we measure the kernel, not DRAM bandwidth.
    // (At multi-MB sizes these ops are memory-bound and SIMD ~= scalar.)
    std::vector<std::int32_t> base(8192);
    for (auto& x : base) x = dist(gen);
    std::vector<std::int32_t> work(base.size());
    const std::int32_t value = 5;
    const std::size_t bytes = base.size() * sizeof(std::int32_t);

    // Each run restores `work` from `base` via memcpy (equal overhead for both
    // contenders), then times the in-place op.
    // remove() mutates+shrinks, so it needs a per-iteration restore. The memcpy is
    // equal overhead for both contenders but is large relative to the scan, so treat
    // these numbers as a floor on the speedup, not the isolated kernel cost.
    {
        ankerl::nanobench::Bench b;
        b.title("remove(==5), 8192 int32 / 32 KB cache-resident (incl. memcpy restore)").relative(true).minEpochIterations(20);
        b.run("std::remove", [&] {
            std::memcpy(work.data(), base.data(), bytes);
            auto e = std::remove(work.begin(), work.end(), value);
            ankerl::nanobench::doNotOptimizeAway(e);
        });
        b.run("simdtl::remove (dispatched AVX2)", [&] {
            std::memcpy(work.data(), base.data(), bytes);
            auto k = simdtl::remove(work.data(), work.size(), value);
            ankerl::nanobench::doNotOptimizeAway(k);
        });
    }
    // reverse is its own inverse, so we can run it repeatedly with NO restore —
    // this isolates the actual reverse cost (no memcpy in the timed region).
    {
        std::memcpy(work.data(), base.data(), bytes);
        ankerl::nanobench::Bench b;
        b.title("reverse, 8192 int32 / 32 KB cache-resident (restore-free)").relative(true).minEpochIterations(50);
        b.run("std::reverse", [&] {
            std::reverse(work.begin(), work.end());
            ankerl::nanobench::doNotOptimizeAway(work[0]);
        });
        b.run("simdtl::reverse (dispatched AVX2)", [&] {
            simdtl::reverse(work.data(), work.size());
            ankerl::nanobench::doNotOptimizeAway(work[0]);
        });
    }
    return 0;
}
