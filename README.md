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

## How it's layered

| Layer | What it is | Provided by |
|------|------------|-------------|
| **L0 seam** | `simdtl::stdx` + stable op names | `std::simd` model via the [`vir-simd`](https://github.com/mattkretz/vir-simd) backport (one-line swap to native `<simd>` later) |
| **L1 platform** | CPUID + XGETBV runtime ISA detection / dispatch | SIMDTL |
| **L2 driver** | `for_each_chunk` tail-handling loop spine | SIMDTL |
| **L3 cross-lane** | `compress_store`, `reverse_inplace`, horizontal reductions | SIMDTL (the value-add `std::simd` can't express) |
| **L4 algorithms** | `count` `find` `min/max` `reduce` `equal` `transform` `replace` `copy_if` … + SSE4.2 string ops | SIMDTL |

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
