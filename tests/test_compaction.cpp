#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <simdtl/simdtl.hpp>
#include "support/sizes.hpp"
#include "support/differential.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

using simdtl_test::kEdgeSizes;
using simdtl_test::make_values;

TEST_CASE("copy_if matches std::copy_if")
{
    for (std::size_t n : kEdgeSizes)
    {
        auto data = make_values<int>(n, 0, 9, 101u + (unsigned)n);
        auto pred_s = [](int x) { return (x & 1) == 0; };           // even
        auto pred_e = [](auto x) { using X = decltype(x); return (x & X(1)) == X(0); };

        std::vector<int> expect;
        std::copy_if(data.begin(), data.end(), std::back_inserter(expect), pred_s);

        std::vector<int> got(n);
        const std::size_t k = simdtl::copy_if(data.data(), n, got.data(), pred_e);
        got.resize(k);
        CHECK(got == expect);
    }
}

TEST_CASE("remove_if matches std::remove_if")
{
    for (std::size_t n : kEdgeSizes)
    {
        auto data = make_values<int>(n, 0, 9, 202u + (unsigned)n);
        auto e = data;

        auto pred_s = [](int x) { return x > 6; };
        auto pred_e = [](auto x) { using X = decltype(x); return x > X(6); };

        auto se = std::remove_if(e.begin(), e.end(), pred_s);
        e.erase(se, e.end());

        const std::size_t k = simdtl::remove_if(data.data(), n, pred_e);
        data.resize(k);
        CHECK(data == e);
    }
}

TEST_CASE("remove(value) matches std::remove (dispatched int32 + portable)")
{
    for (std::size_t n : kEdgeSizes)
    {
        for (int value : {0, 5, 9})
        {
            auto data = make_values<std::int32_t>(n, 0, 9, 303u + (unsigned)n);
            auto e = data;
            auto se = std::remove(e.begin(), e.end(), value);
            e.erase(se, e.end());

            const std::size_t k = simdtl::remove(data.data(), n, value);
            data.resize(k);
            CHECK(data == e);
        }
    }
}

template <class T>
static void check_reverse()
{
    for (std::size_t n : kEdgeSizes)
    {
        auto data = make_values<T>(n, -50, 50, 404u + (unsigned)n);
        auto e = data;
        std::reverse(e.begin(), e.end());
        simdtl::reverse(data.data(), n);
        CHECK(data == e);
    }
}

TEST_CASE("reverse matches std::reverse (dispatched int32 + portable any-size)")
{
    check_reverse<std::int32_t>();   // AVX2 block-reverse kernel
    check_reverse<std::int16_t>();   // portable
    check_reverse<std::int64_t>();
    check_reverse<float>();
}

TEST_CASE("unique matches std::unique")
{
    for (std::size_t n : kEdgeSizes)
    {
        // small value range -> plenty of consecutive duplicates
        auto data = make_values<int>(n, 0, 2, 505u + (unsigned)n);
        auto e = data;
        e.erase(std::unique(e.begin(), e.end()), e.end());
        data.resize(simdtl::unique(data.data(), n));
        CHECK(data == e);
    }
}

TEST_CASE("partition matches std::stable_partition (stable + partition point)")
{
    for (std::size_t n : kEdgeSizes)
    {
        auto data = make_values<int>(n, 0, 9, 606u + (unsigned)n);
        auto e = data;
        auto sp = std::stable_partition(e.begin(), e.end(), [](int x) { return x < 5; });
        const std::size_t point = static_cast<std::size_t>(sp - e.begin());

        const std::size_t k = simdtl::partition(data.data(), n, [](auto x) { using X = decltype(x); return x < X(5); });
        CHECK(k == point);
        CHECK(data == e);          // stable -> exact element-wise match
    }
}

TEST_CASE("M3 kernels installed when the CPU supports AVX2")
{
    using namespace simdtl::platform;
#ifdef SIMDTL_HAVE_FAST_KERNELS
    if (best_isa() >= isa_level::avx2)
    {
        CHECK(remove_i32_slot() != nullptr);
        CHECK(reverse_i32_slot() != nullptr);
    }
#endif
}
