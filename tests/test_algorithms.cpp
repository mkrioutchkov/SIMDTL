#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <simdtl/simdtl.hpp>
#include "support/sizes.hpp"
#include "support/differential.hpp"

#include <algorithm>
#include <numeric>
#include <cstddef>
#include <cstdint>
#include <vector>

using simdtl_test::kEdgeSizes;
using simdtl_test::make_values;

TEST_CASE("find / find_if match std::find / std::find_if")
{
    for (std::size_t n : kEdgeSizes)
    {
        auto data = make_values<int>(n, 0, 9, 11u + (unsigned)n);
        for (int value : {0, 5, 9, 42})
        {
            auto std_it  = std::find(data.begin(), data.end(), value);
            auto std_off = static_cast<std::size_t>(std_it - data.begin());
            CHECK(static_cast<std::size_t>(simdtl::find(data.data(), n, value) - data.data()) == std_off);
        }
        auto pred_s = [](int x) { return x > 6; };
        auto pred_e = [](auto x) { using X = decltype(x); return x > X(6); };
        auto std_off = static_cast<std::size_t>(std::find_if(data.begin(), data.end(), pred_s) - data.begin());
        CHECK(static_cast<std::size_t>(simdtl::find_if(data.data(), n, pred_e) - data.data()) == std_off);
    }
}

template <class T>
static void check_minmax()
{
    for (std::size_t n : kEdgeSizes)
    {
        if (n == 0) continue;
        auto data = make_values<T>(n, -20, 50, 22u + (unsigned)n);
        const T smin = *std::min_element(data.begin(), data.end());
        const T smax = *std::max_element(data.begin(), data.end());
        CHECK(simdtl::min_value(data.data(), n) == smin);
        CHECK(simdtl::max_value(data.data(), n) == smax);
        auto mm = simdtl::minmax_value(data.data(), n);
        CHECK(mm.first == smin);
        CHECK(mm.second == smax);
        // first-occurrence pointer semantics
        auto so = static_cast<std::size_t>(std::min_element(data.begin(), data.end()) - data.begin());
        CHECK(static_cast<std::size_t>(simdtl::min_element(data.data(), n) - data.data()) == so);
    }
}

TEST_CASE("min/max/minmax + min_element match the STL")
{
    check_minmax<std::int32_t>();
    check_minmax<std::int16_t>();
    check_minmax<float>();
    check_minmax<double>();
}

TEST_CASE("reduce matches std::accumulate (integers exact, float approx)")
{
    for (std::size_t n : kEdgeSizes)
    {
        auto idata = make_values<std::int64_t>(n, -5, 5, 33u + (unsigned)n);
        const std::int64_t s = std::accumulate(idata.begin(), idata.end(), std::int64_t{0});
        CHECK(simdtl::reduce(idata.data(), n, std::int64_t{0}) == s);

        auto fdata = make_values<float>(n, 0, 10, 34u + (unsigned)n);
        const float fs = std::accumulate(fdata.begin(), fdata.end(), 0.0f);
        CHECK(simdtl::reduce(fdata.data(), n, 0.0f) == doctest::Approx(fs).epsilon(0.001));
    }
}

TEST_CASE("equal / mismatch match the STL")
{
    for (std::size_t n : kEdgeSizes)
    {
        auto a = make_values<int>(n, 0, 4, 44u + (unsigned)n);
        auto b = a;                              // identical
        CHECK(simdtl::equal(a.data(), b.data(), n) == true);
        if (n > 0)
        {
            b[n / 2] ^= 0x7;                     // introduce one difference
            const bool std_eq = std::equal(a.begin(), a.end(), b.begin());
            CHECK(simdtl::equal(a.data(), b.data(), n) == std_eq);
            auto sm = std::mismatch(a.begin(), a.end(), b.begin());
            auto off = static_cast<std::size_t>(sm.first - a.begin());
            auto got = simdtl::mismatch(a.data(), b.data(), n);
            CHECK(static_cast<std::size_t>(got.first - a.data()) == off);
        }
    }
}

TEST_CASE("transform (unary & binary) matches std::transform")
{
    for (std::size_t n : kEdgeSizes)
    {
        auto a = make_values<int>(n, -7, 7, 55u + (unsigned)n);
        auto b = make_values<int>(n, -7, 7, 56u + (unsigned)n);

        std::vector<int> expect(n), got(n);
        std::transform(a.begin(), a.end(), expect.begin(), [](int x) { return x * x; });
        simdtl::transform(a.data(), n, got.data(), [](auto x) { return x * x; });
        CHECK(got == expect);

        std::transform(a.begin(), a.end(), b.begin(), expect.begin(), [](int x, int y) { return x + y; });
        simdtl::transform(a.data(), b.data(), n, got.data(), [](auto x, auto y) { return x + y; });
        CHECK(got == expect);
    }
}

TEST_CASE("replace / replace_if match the STL (incl. float)")
{
    for (std::size_t n : kEdgeSizes)
    {
        auto base = make_values<int>(n, 0, 3, 66u + (unsigned)n);

        auto a = base, e = base;
        simdtl::replace(a.data(), n, 2, 99);
        std::replace(e.begin(), e.end(), 2, 99);
        CHECK(a == e);

        auto c = base, f = base;
        simdtl::replace_if(c.data(), n, [](auto x) { using X = decltype(x); return x < X(1); }, -1);
        std::replace_if(f.begin(), f.end(), [](int x) { return x < 1; }, -1);
        CHECK(c == f);

        // float path proves where()=value works where the old XOR trick could not
        auto fb = make_values<float>(n, 0, 3, 67u + (unsigned)n);
        auto fa = fb, ff = fb;
        simdtl::replace(fa.data(), n, 2.0f, 99.0f);
        std::replace(ff.begin(), ff.end(), 2.0f, 99.0f);
        CHECK(fa == ff);
    }
}
