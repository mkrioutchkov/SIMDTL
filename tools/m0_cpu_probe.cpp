// M0 CPU-probe + include-tree smoke test.
// Compiles the full simdtl umbrella header, prints detected CPU features, exercises
// the L0 seam (a simd compare through the stable names), and self-checks that the
// detected capability tier is internally consistent. Returns non-zero on failure.
#include <simdtl/simdtl.hpp>
#include <cstdio>

int main()
{
    using namespace simdtl;
    const platform::cpu_features f = platform::detect_cpu_features();
    const platform::isa_level    lvl = platform::best_isa();

    std::printf("backend       : %s\n", SIMDTL_SIMD_BACKEND);
    std::printf("native<int>   : %zu lanes\n", stdx::native_simd<int>::size());
    std::printf("features      : sse2=%d sse42=%d popcnt=%d avx=%d avx2=%d "
                "avx512f=%d avx512bw=%d os_avx=%d os_avx512=%d\n",
                f.sse2, f.sse42, f.popcnt, f.avx, f.avx2,
                f.avx512f, f.avx512bw, f.os_avx, f.os_avx512);
    std::printf("best isa tier : %s\n", platform::isa_name(lvl));

    // Exercise the seam through the stable wrapper names (proves names.hpp links).
    using V = stdx::native_simd<int>;
    V a([](auto i) { return static_cast<int>(i); });
    auto m = (a == V(3));
    std::printf("seam check    : lane_count(a==3)=%d any=%d all=%d\n",
                lane_count(m), static_cast<int>(any_of(m)), static_cast<int>(all_of(m)));

    // Self-consistency: feature implication + OS-state gating.
    int rc = 0;
    auto fail = [&](const char* why) { std::printf("FAIL: %s\n", why); rc = 1; };
    if (lvl >= platform::isa_level::avx2  && !(f.avx && f.sse2)) fail("avx2 tier without avx/sse2");
    if (lvl >= platform::isa_level::sse42 && !f.sse2)           fail("sse42 tier without sse2");
    if (f.avx    && !f.os_avx)    fail("avx usable but OS does not save YMM");
    if (f.avx512f && !f.os_avx512) fail("avx512 usable but OS does not save ZMM");
    if (lane_count(m) != 1)        fail("exactly one lane should equal 3");

    std::printf("selfcheck     : %s\n", rc == 0 ? "OK" : "FAIL");
    return rc;
}
