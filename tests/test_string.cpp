#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <simdtl/simdtl.hpp>
#include "support/sizes.hpp"

#include <cctype>
#include <cstddef>
#include <random>
#include <string>
#include <utility>
#include <vector>

using simdtl_test::kEdgeSizes;

// NUL-free printable text (the SSE4.2 path's contract); fixed seed for CI.
static std::string make_text(std::size_t n, unsigned seed)
{
    std::string s(n, ' ');
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> d(33, 126);   // printable ASCII, no NUL
    for (auto& c : s) c = static_cast<char>(d(gen));
    return s;
}

TEST_CASE("count_in_range matches an independent scalar reference")
{
    for (std::size_t n : kEdgeSizes)
    {
        const std::string s = make_text(n, 7u + (unsigned)n);
        // a few representative ranges incl. boundaries and empty
        const std::pair<char, char> ranges[] = {{'A', 'Z'}, {'a', 'z'}, {'0', '9'}, {'!', '~'}, {'z', 'a'}};
        for (auto [lo, hi] : ranges)
        {
            std::size_t expect = 0;
            for (char c : s) if (c >= lo && c <= hi) ++expect;
            CHECK(simdtl::count_in_range(s.data(), n, lo, hi) == expect);
        }
    }
}

TEST_CASE("to_lower / to_upper match std::tolower / std::toupper")
{
    for (std::size_t n : kEdgeSizes)
    {
        const std::string base = make_text(n, 88u + (unsigned)n);

        std::string lo = base, lo_ref = base;
        simdtl::to_lower(lo.data(), n);
        for (auto& c : lo_ref) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        CHECK(lo == lo_ref);

        std::string up = base, up_ref = base;
        simdtl::to_upper(up.data(), n);
        for (auto& c : up_ref) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        CHECK(up == up_ref);
    }
}

// --- Regression tests for the adversarial-review findings (PCMPISTRM operand
//     limits: zero boundary truncates, >8 pairs overflow). Each would FAIL on an
//     SSE4.2 CPU before the guard was added; now the public API falls back to scalar.

TEST_CASE("count_in_range is correct for zero-boundary ranges (regression)")
{
    for (std::size_t n : kEdgeSizes)
    {
        std::vector<char> data(n);
        std::mt19937 gen(123u + (unsigned)n);
        std::uniform_int_distribution<int> d(0, 30);          // includes 0 and small bytes
        for (auto& c : data) c = static_cast<char>(d(gen));
        for (auto [lo, hi] : {std::pair<char, char>{0, 20}, {0, 0}})
        {
            std::size_t expect = 0;
            for (char c : data) if (c >= lo && c <= hi) ++expect;
            CHECK(simdtl::count_in_range(data.data(), n, lo, hi) == expect);
        }
    }
}

TEST_CASE("convert_case is correct for >8 pairs and zero-boundary pairs (regression)")
{
    auto ref = [](std::vector<char> v, const char* pairs, int np) {
        for (auto& c : v)
            for (int p = 0; p < np; ++p)
                if (c >= pairs[2 * p] && c <= pairs[2 * p + 1]) { c = static_cast<char>(c ^ 0x20); break; }
        return v;
    };
    for (std::size_t n : kEdgeSizes)
    {
        // Deterministic data cycling A..Z so pairs 8-9 (Q-R) are actually present.
        std::vector<char> data(n);
        for (std::size_t i = 0; i < n; ++i) data[i] = static_cast<char>('A' + (i % 26));

        // 9 pairs (npairs > 8) -> must fall back to scalar and stay correct.
        const char pairs9[] = {'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R'};
        auto got9 = data; simdtl::convert_case(got9.data(), n, pairs9, 9);
        CHECK(got9 == ref(data, pairs9, 9));

        // Zero-boundary single pair {0,20} on binary data.
        std::vector<char> bin(n);
        std::mt19937 gen(321u + (unsigned)n);
        std::uniform_int_distribution<int> d(0, 30);
        for (auto& c : bin) c = static_cast<char>(d(gen));
        const char p0[] = {0, 20};
        auto gotz = bin; simdtl::convert_case(gotz.data(), n, p0, 1);
        CHECK(gotz == ref(bin, p0, 1));
    }
}

TEST_CASE("flip_case toggles letters only")
{
    for (std::size_t n : kEdgeSizes)
    {
        const std::string base = make_text(n, 99u + (unsigned)n);
        std::string fc = base, ref = base;
        simdtl::flip_case(fc.data(), n);
        for (auto& c : ref)
        {
            if (std::isupper(static_cast<unsigned char>(c))) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            else if (std::islower(static_cast<unsigned char>(c))) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        }
        CHECK(fc == ref);
    }
}
