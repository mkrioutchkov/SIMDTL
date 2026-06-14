# M1 — `count` vertical slice: results

The MVP slice drives ONE algorithm (`count`) through every layer + every tool, so
the riskiest mechanisms are proven before breadth.

## What's proven
- **L2 driver** (`for_each_chunk`) + **L4 `count`/`count_if`** (portable std::simd path)
  match `std::count`/`std::count_if` across all edge sizes and element types
  (int8/16/32/64, float, double) — `tests/test_count.cpp`, run via doctest/ctest.
- **L1 runtime dispatch works without `target_clones`** on MSVC: the test/bench TUs
  are compiled at baseline (no `/arch:AVX2`); the AVX2 `count<int32>` kernel is a
  separate `/arch:AVX2` TU that self-registers via CPUID and is reached only through
  the cached function pointer. Verified the slot is installed at `avx2` on this CPU.
- **Seam lint** (`scripts/lint-seam.ps1`) passes: no `vir::`/`stdx::`/backend names
  leak outside `include/simdtl/backend/`, so the C++26 `<simd>` swap stays one file.

## Benchmark (nanobench, Release, 2^20 int32, MSVC 14.51, AVX2 laptop)

| variant | ns/op | relative to `std::count` |
|---|---:|---:|
| `std::count` | 287,775 | 1.00× |
| **`simdtl::count` (dispatched AVX2 kernel)** | **83,210** | **3.46× faster** |
| `simdtl::count_portable` (std::simd / vir fixed_size) | 318,933 | 0.90× (slower) |

This is the M0 finding made concrete: on MSVC the **portable `std::simd` path is no
faster than scalar** (the `fixed_size<N>` fallback doesn't vectorize), while the
**hand-written dispatched intrinsic kernel is ~3.5× faster**. The project's value
lives in the dispatched kernels — confirmed, not assumed.

## Carried-forward note
Kernels are linked as direct sources here, so the self-registering object is never
dropped. Packaging them as a *static library* will need `/WHOLEARCHIVE` (MSVC) or
`--whole-archive` — that's the open linkage question to settle when the kernel set grows.
