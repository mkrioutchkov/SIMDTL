# Showcase: what SIMDTL does that `std::simd` can't

`std::simd` gives you the vector *type* and lane-wise math — and nothing else. No
range algorithms, no stream compaction, no cross-lane permute, no string
instructions, no runtime dispatch. SIMDTL is exactly those gaps.

Run it yourself:
```
cmake -B build -DSIMDTL_BUILD_EXAMPLES=ON -DSIMDTL_FAST_KERNELS=ON
cmake --build build
./build/examples/showcase
```

Actual output (MSVC, AVX2 laptop — note the showcase TU is built at *baseline*, so
`int lanes=4`, while the dispatched kernels still run AVX2):

```
=== SIMDTL showcase: what std::simd canNOT do ===
backend=vir-simd  ISA=avx2  int lanes=4
input: 5 3 8 8 2 8 1 9 8 4 7 8 0 6 8 3

[1] Algorithms over a container (std::simd has none):
    count(8)     = 6
    count_if(>5) = 9
    reduce(sum)  = 88
    minmax       = [0, 9]
    find(9) @idx = 7

[2] Stream compaction (std::simd's where() can't pack lanes):
    copy_if(even): 8 8 2 8 8 4 8 0 6 8
    remove(8)    : 5 3 2 1 9 4 7 0 6 3
    partition<5 : 3 2 1 4 0 3 5 8 8 8 9 8 7 8 6 8   (partition point = 6)
    sort+unique  : 0 1 2 3 4 5 6 7 8 9

[3] In-register reverse, any element size (std::simd has no permute):
    reverse<int>  : 3 8 6 0 8 7 4 8 9 1 8 2 8 8 3 5
    reverse<short>: 7 6 5 4 3 2 1

[4] SSE4.2 string ops (PCMPISTRM; nothing like it in std::simd):
    text     : SIMDTL Makes Vectors Fun! 2026
    to_upper : SIMDTL MAKES VECTORS FUN! 2026
    to_lower : simdtl makes vectors fun! 2026
    flip_case: simdtl mAKES vECTORS fUN! 2026
    letters  : 21    digits : 4

[5] Runtime CPU dispatch (std::simd is fixed-ABI per TU):
    detected ISA       : avx2
    remove<int32> path : AVX2 (dispatched)
    reverse<int32> path: AVX2 (dispatched)

OK: every result verified against the STL.
```

---

## The contrasts, in code

### Range algorithms — `std::simd` makes you write the loop
```cpp
// Raw std::simd: you hand-roll the whole thing, every time.
namespace stdx = std::experimental;             // (or vir::stdx today)
std::size_t count_eq(const int* p, std::size_t n, int value) {
    using V = stdx::native_simd<int>;
    constexpr std::size_t W = V::size();
    const V needle(value);
    std::size_t total = 0, i = 0;
    for (; i + W <= n; i += W)
        total += stdx::popcount(V(p + i, stdx::element_aligned) == needle);
    for (; i < n; ++i) total += (p[i] == value);
    return total;
}

// SIMDTL:
simdtl::count(p, n, value);
```

### Stream compaction — `std::simd` *fundamentally* can't
`std::simd` offers only `where(mask, v) = x`, a **position-preserving** blend. It
overwrites matching lanes in place; there is no operation to gather the kept lanes
into a contiguous prefix.
```cpp
// std::simd: can mask/blend, CANNOT pack —
stdx::where(v == needle, v) = replacement;      // same length, same positions

// SIMDTL: packs the kept elements (this is the marquee feature) —
std::size_t kept = simdtl::copy_if(p, n, out, pred);   // out = matches, contiguous
n = simdtl::remove(data, n, value);                    // drop a value, AVX2 vpermd kernel
n = simdtl::partition(data, n, pred);                  // stable; returns the split point
n = simdtl::unique(data, n);
```

### Cross-lane permute / reverse — not in the `std::simd` TS
```cpp
// std::simd (TS): no permute exists. SIMDTL:
simdtl::reverse(data, n);            // any element size; AVX2 vpermd kernel for int32
```

### SSE4.2 string instructions — no `std::simd` analog at all
```cpp
// PCMPISTRM range-compare in one instruction; std::simd has nothing like it.
simdtl::to_upper(s.data(), s.size());
std::size_t digits = simdtl::count_in_range(s.data(), s.size(), '0', '9');
```

### Runtime dispatch — `std::simd` bakes one ABI per TU
`std::simd` picks its ABI at compile time from your `/arch`/`-march`. SIMDTL detects
the CPU at runtime (CPUID + XGETBV) and selects the best kernel, with a scalar floor
so it never executes an unsupported instruction. The showcase line
`remove<int32> path : AVX2 (dispatched)` is that selection happening at startup.

See [USAGE.md](USAGE.md) for the per-operation dispatch table and the compile-time
width rules.
