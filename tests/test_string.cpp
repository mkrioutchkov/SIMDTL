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
