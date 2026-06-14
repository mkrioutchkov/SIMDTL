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

// Differential: simdtl::count must equal std::count for every edge size & value.
template <class T>
static void check_count_matches_std()
{
    for (std::size_t n : kEdgeSizes)
    {
        const std::vector<T> data = make_values<T>(n, 0, 3, /*seed*/ 1234u + static_cast<unsigned>(n));
        for (T value : {T(0), T(1), T(2), T(3), T(9)})
        {
            const std::size_t expected =
                static_cast<std::size_t>(std::count(data.begin(), data.end(), value));

            CHECK(simdtl::count(data.data(), n, value) == expected);   // range overload
            CHECK(simdtl::count(data, value) == expected);             // container overload
            CHECK(simdtl::detail::count_portable<T>(data.data(), n, value) == expected); // portable path
        }
    }
}

TEST_CASE("count matches std::count across element types")
{
    check_count_matches_std<std::int32_t>();   // dispatched AVX2 kernel (+ portable)
    check_count_matches_std<std::int16_t>();   // dispatched AVX2 kernel
    check_count_matches_std<std::int8_t>();    // dispatched AVX2 kernel
    check_count_matches_std<std::int64_t>();
    check_count_matches_std<std::uint8_t>();
    check_count_matches_std<float>();
    check_count_matches_std<double>();
}

TEST_CASE("count_if matches std::count_if")
{
    auto gt1_elemental = [](auto x) { using X = decltype(x); return x > X(1); };
    auto gt1_scalar    = [](int x)  { return x > 1; };

    for (std::size_t n : kEdgeSizes)
    {
        const std::vector<int> data = make_values<int>(n, 0, 3, 77u + static_cast<unsigned>(n));
        const std::size_t expected =
            static_cast<std::size_t>(std::count_if(data.begin(), data.end(), gt1_scalar));
        CHECK(simdtl::count_if(data.data(), n, gt1_elemental) == expected);
        CHECK(simdtl::count_if(data, gt1_elemental) == expected);
    }
}

TEST_CASE("runtime dispatch installs the AVX2 kernel when the CPU supports it")
{
    using namespace simdtl::platform;
#ifdef SIMDTL_HAVE_FAST_KERNELS
    if (best_isa() >= isa_level::avx2)
    {
        CHECK(count_i32_slot() != nullptr);
        CHECK(count_i32_installed_level() == isa_level::avx2);
        CHECK(count_i16_slot() != nullptr);
        CHECK(count_i8_slot()  != nullptr);
    }
#else
    CHECK(count_i32_slot() == nullptr);   // header-only: portable path only
#endif
}
