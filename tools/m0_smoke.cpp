// M0 foundation smoke test — confirms vir-simd compiles on MSVC /std:c++20
// and probes whether fixed_size<N> reduce auto-vectorizes under /O2 /arch:AVX2.
#include <vir/simd.h>
#include <cstdio>
#include <cstddef>

namespace stdx = vir::stdx;

// Keep this in its own function so we can find it in /FA assembly output.
float bench_reduce(const float* p, std::size_t n) noexcept
{
    using V = stdx::simd<float, stdx::simd_abi::fixed_size<8>>;
    constexpr std::size_t W = V::size();
    V acc{0.0f};
    std::size_t i = 0;
    for (; i + W <= n; i += W)
    {
        V v(p + i, stdx::element_aligned);
        acc += v;
    }
    float s = stdx::reduce(acc);
    for (; i < n; ++i) s += p[i];
    return s;
}

int main()
{
    std::printf("vir-simd version: 0x%06x\n", VIR_SIMD_VERSION);
    std::printf("native_simd<float>::size() = %zu\n",
                stdx::native_simd<float>::size());
    std::printf("fixed_size<8> simd<float>::size() = %zu\n",
                stdx::simd<float, stdx::simd_abi::fixed_size<8>>::size());

    static float data[1024];
    for (int i = 0; i < 1024; ++i) data[i] = float(i % 7);
    std::printf("bench_reduce sum = %f\n", bench_reduce(data, 1024));
    return 0;
}
