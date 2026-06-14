// SIMDTL showcase — the operations std::simd does NOT provide.
//
// std::simd gives you the vector TYPE and lane-wise math. It has: no range
// algorithms, no stream compaction (only position-preserving where()), no
// cross-lane permute, no string instructions, and no runtime dispatch. This
// program demonstrates each of those gaps being filled, and verifies every result
// against the STL.
#include <simdtl/simdtl.hpp>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

template <class T>
static void show(const char* label, const std::vector<T>& v)
{
    std::cout << label;
    for (auto x : v) std::cout << x << ' ';
    std::cout << '\n';
}

int main()
{
    using namespace simdtl;

    std::cout << "=== SIMDTL showcase: what std::simd canNOT do ===\n";
    std::cout << "backend=" << SIMDTL_SIMD_BACKEND
              << "  ISA=" << platform::isa_name(platform::best_isa())
              << "  int lanes=" << native<int>::size() << "\n\n";

    std::vector<int> data = {5, 3, 8, 8, 2, 8, 1, 9, 8, 4, 7, 8, 0, 6, 8, 3};
    show("input: ", data);
    std::cout << '\n';

    // [1] RANGE ALGORITHMS — std::simd is only the vector type; you hand-write the
    //     loop. SIMDTL gives whole-container algorithms in one call.
    std::cout << "[1] Algorithms over a container (std::simd has none):\n";
    std::cout << "    count(8)     = " << count(data, 8) << '\n';
    std::cout << "    count_if(>5) = " << count_if(data, [](auto x) { using X = decltype(x); return x > X(5); }) << '\n';
    std::cout << "    reduce(sum)  = " << reduce(data.data(), data.size()) << '\n';
    auto mm = minmax_value(data.data(), data.size());
    std::cout << "    minmax       = [" << mm.first << ", " << mm.second << "]\n";
    std::cout << "    find(9) @idx = " << (find(data.data(), data.size(), 9) - data.data()) << "\n\n";

    // [2] STREAM COMPACTION — std::simd's where() can only blend lanes IN PLACE
    //     (positions preserved); it cannot pack matches into a contiguous prefix.
    std::cout << "[2] Stream compaction (std::simd's where() can't pack lanes):\n";
    std::vector<int> evens(data.size());
    evens.resize(copy_if(data.data(), data.size(), evens.data(),
                         [](auto x) { using X = decltype(x); return (x & X(1)) == X(0); }));
    show("    copy_if(even): ", evens);
    std::vector<int> no8 = data;
    no8.resize(remove(no8.data(), no8.size(), 8));              // dispatched AVX2 kernel for int32
    show("    remove(8)    : ", no8);
    std::vector<int> part = data;
    const std::size_t pp = partition(part.data(), part.size(),
                                     [](auto x) { using X = decltype(x); return x < X(5); });
    show("    partition<5 : ", part);
    std::cout << "      (partition point = " << pp << ")\n";
    std::vector<int> uniq = data;
    std::sort(uniq.begin(), uniq.end());
    uniq.resize(unique(uniq.data(), uniq.size()));
    show("    sort+unique  : ", uniq);
    std::cout << '\n';

    // [3] CROSS-LANE REVERSE — std::simd (TS) has no permute at all.
    std::cout << "[3] In-register reverse, any element size (std::simd has no permute):\n";
    std::vector<int> rev = data;
    reverse(rev.data(), rev.size());                           // dispatched AVX2 block-reverse
    show("    reverse<int>  : ", rev);
    std::vector<std::int16_t> sh = {1, 2, 3, 4, 5, 6, 7};
    reverse(sh.data(), sh.size());                            // portable, any element size
    show("    reverse<short>: ", sh);
    std::cout << '\n';

    // [4] SSE4.2 STRING-RANGE OPS — no std::simd analog whatsoever.
    std::cout << "[4] SSE4.2 string ops (PCMPISTRM; nothing like it in std::simd):\n";
    std::string s = "SIMDTL Makes Vectors Fun! 2026";
    std::string up = s, lo = s, fl = s;
    to_upper(up.data(), up.size());
    to_lower(lo.data(), lo.size());
    flip_case(fl.data(), fl.size());
    std::cout << "    text     : " << s << '\n';
    std::cout << "    to_upper : " << up << '\n';
    std::cout << "    to_lower : " << lo << '\n';
    std::cout << "    flip_case: " << fl << '\n';
    std::cout << "    letters  : " << (count_in_range(s.data(), s.size(), 'A', 'Z')
                                       + count_in_range(s.data(), s.size(), 'a', 'z')) << '\n';
    std::cout << "    digits   : " << count_in_range(s.data(), s.size(), '0', '9') << "\n\n";

    // [5] RUNTIME DISPATCH — std::simd bakes ONE ABI per TU; no runtime selection.
    std::cout << "[5] Runtime CPU dispatch (std::simd is fixed-ABI per TU):\n";
    std::cout << "    detected ISA       : " << platform::isa_name(platform::best_isa()) << '\n';
#ifdef SIMDTL_HAVE_FAST_KERNELS
    std::cout << "    remove<int32> path : " << (platform::remove_i32_slot()  ? "AVX2 (dispatched)" : "portable") << '\n';
    std::cout << "    reverse<int32> path: " << (platform::reverse_i32_slot() ? "AVX2 (dispatched)" : "portable") << '\n';
#else
    std::cout << "    (header-only build; -DSIMDTL_FAST_KERNELS=ON adds intrinsic kernels)\n";
#endif

    // Verify everything against the STL.
    bool ok = count(data, 8) == static_cast<std::size_t>(std::count(data.begin(), data.end(), 8))
           && reduce(data.data(), data.size()) == std::accumulate(data.begin(), data.end(), 0);
    { auto e = data; std::reverse(e.begin(), e.end()); ok = ok && rev == e; }
    { auto e = data; e.erase(std::remove(e.begin(), e.end(), 8), e.end()); ok = ok && no8 == e; }
    std::cout << "\n" << (ok ? "OK: every result verified against the STL." : "MISMATCH!") << '\n';
    return ok ? 0 : 1;
}
