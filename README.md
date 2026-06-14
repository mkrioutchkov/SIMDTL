# SIMDTL

**SIMD Template Library** — the SIMD operations `std::simd` deliberately leaves out.

`std::simd` (C++26, Parallelism TS) gives you the *vector type* and lane-wise math.
It does **not** give you range/container **algorithms**, **stream compaction**
(compress/expand → `copy_if`/`remove_if`/`partition`), arbitrary **shuffle/permute**
(in-register reverse), **gather/scatter**, the **SSE4.2 string-range** instructions,
or **runtime CPU dispatch**. SIMDTL is exactly that gap — STL-shaped, header-only,
cross-platform — built *on top of* the `std::simd` programming model.

> Status: **M0 (foundation) complete.** The backend seam, the runtime CPU probe,
> and the build/packaging scaffold compile and run on MSVC 14.51 / C++20.
> See [PLAN.md](PLAN.md) for the full design and roadmap.

## What it does that `std::simd` can't

`std::simd` is only the vector type — no algorithms, no compaction, no permute, no
string ops, no runtime dispatch. SIMDTL fills exactly those gaps:

- **Range algorithms in one call** — `count`, `find`, `min/max/minmax`, `reduce`,
  `equal/mismatch`, `transform`, `replace` (with `std::simd` you hand-write the loop).
- **Stream compaction** — `copy_if` / `remove` / `remove_if` / `partition` / `unique`.
  `std::simd`'s `where()` only blends lanes *in place*; it cannot pack matches into a
  contiguous prefix. This is the marquee feature.
- **Cross-lane reverse**, any element size (the `std::simd` TS has no permute).
- **SSE4.2 string-range ops** — `count_in_range`, `to_lower/upper/flip_case` via
  `PCMPISTRM` (no `std::simd` analog).
- **Runtime CPU dispatch** (CPUID + XGETBV) — `std::simd` bakes one ABI per TU.

See the runnable **[examples/showcase.cpp](examples/showcase.cpp)** and its annotated
output + side-by-side code contrasts in **[docs/SHOWCASE.md](docs/SHOWCASE.md)**.

## How it's layered

| Layer | What it is | Provided by |
|------|------------|-------------|
| **L0 seam** | `simdtl::stdx` + stable op names | `std::simd` model via the [`vir-simd`](https://github.com/mattkretz/vir-simd) backport (one-line swap to native `<simd>` later) |
| **L1 platform** | CPUID + XGETBV runtime ISA detection / dispatch | SIMDTL |
| **L2 driver** | `for_each_chunk` tail-handling loop spine | SIMDTL |
| **L3 cross-lane** | `compress_store`, `reverse_inplace`, horizontal reductions | SIMDTL (the value-add `std::simd` can't express) |
| **L4 algorithms** | `count` `find` `min/max` `reduce` `equal` `transform` `replace` `copy_if` … + SSE4.2 string ops | SIMDTL |

## Usage

```cpp
#include <simdtl/simdtl.hpp>

std::vector<int> v = {3,1,2,2,5,2};
std::size_t twos = simdtl::count(v, 2);                                   // 3
std::size_t gt2  = simdtl::count_if(v, [](auto x){ using X = decltype(x); return x > X(2); });
v.resize(simdtl::remove(v.data(), v.size(), 2));                          // drop all 2s (AVX2 kernel)
simdtl::reverse(v.data(), v.size());
```

Full examples for every algorithm, the **elemental-predicate** pattern
(`[](auto x){ return x > decltype(x)(4); }`), and how **compile-time width vs.
runtime dispatch** works are in **[docs/USAGE.md](docs/USAGE.md)**.

## Building

Requires Visual Studio 2026 (MSVC 14.5x) with C++20. `vir-simd` is vendored under
`third_party/` — no download needed.

**Quick (M0 tools, no CMake):**
```powershell
.\scripts\build-m0.ps1
```

**CMake (source of truth):**
```powershell
cmake -G Ninja -DCMAKE_CXX_COMPILER=cl -DSIMDTL_BUILD_M0=ON -S . -B build/cmake
cmake --build build/cmake
```

As a dependency, the public target is `simdtl::simdtl` (header-only INTERFACE).

## Dependencies & license

- **vir-simd** (LGPL-3.0), vendored in `third_party/vir-simd/`, kept strictly
  behind the L0 seam so it can be removed once a toolchain ships native `std::simd`.
- The legacy MSVC-only prototype lives in [`SIMDTL/`](SIMDTL/) for reference; the
  modern library lives under [`include/simdtl/`](include/simdtl/).
