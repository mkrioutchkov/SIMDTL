# SIMDTL — modern rewrite plan

## Thesis

**`std::simd` is the vector layer. SIMDTL is everything `std::simd` deliberately left out.**

| `std::simd` **gives** you (we reimplement none of it) | `std::simd` **does NOT** give you (this is SIMDTL) |
|---|---|
| `simd<T>` / `simd_mask<T>` types | range/container **algorithms** (it's just the type — you write the loop) |
| element-wise math, comparisons → mask | **stream compaction** (compress/expand → `copy_if`/`remove_if`/`partition`) |
| broadcast/splat, load/store + alignment flags | arbitrary **shuffle/permute by index** (so: in-register reverse) |
| `reduce` / `hmin` / `hmax` | **gather/scatter** (its generator ctor is compile-time only) |
| `where()` masked assign, `any/all/none`, `popcount`, `find_first_set` | **SSE4.2 string-range** instructions (`_mm_cmpistrm`) |
| `min/max/clamp`, math fns, `split/concat/cast` | **runtime CPU dispatch** (it bakes one ABI per TU at compile time) |

The right column *is* the library.

## Foundation & toolchain reality

- Target dev machine: **MSVC 14.51 (VS 2026), `/std:c++20`, x64.** The MSVC STL ships
  **neither `<simd>` nor `<experimental/simd>`** → native `std::simd` is unavailable.
- We adopt the `std::simd` *programming model* via the header-only **`vir-simd`**
  backport (vendored, `vir::stdx`), behind a one-file seam so swapping to native
  `<simd>` later is a one-line change with **zero algorithm edits**.
- **Measured in M0** (see [docs/M0_FINDINGS.md](docs/M0_FINDINGS.md)): on MSVC,
  vir-simd's `fixed_size<N>` does **not** auto-vectorize (8× scalar `vaddss`, no
  `vaddps ymm`). So the portable layer buys correctness + portability + the GCC/Clang
  fast path; **guaranteed x86 SIMD speed comes from SIMDTL's own intrinsic kernels**
  behind runtime dispatch. This is the path we chose intentionally.

## Architecture

```
include/simdtl/
  backend/simd.hpp      L0  the ONE swap point: vir::stdx today → std tomorrow
  backend/names.hpp     L0  stable op names + TS↔C++26 rename map
  platform/arch_macros.hpp  L1  SIMDTL_ARCH_X86 (arch-based, never __SSE4_2__)
  platform/cpu.hpp      L1  CPUID + mandatory XGETBV probe → best_isa()
  platform/dispatch.hpp L1  detect-once cached fn-ptr table              [M1]
  detail/driver.hpp     L2  for_each_chunk: W=size() body + scalar tail  [M1]
  crosslane/*.hpp       L3  compress_store, reverse_inplace, horizontal  [M1/M3]
  algorithm/*.hpp       L4  count, find, minmax, reduce, equal, transform, replace, copy_if  [M1/M2/M3]
  string_range.hpp      L4  SSE4.2 count_in_range / to_lower / to_upper / flip_case  [M4]
src/kernels/            opt-in per-/arch intrinsic kernels (SIMDTL_FAST_KERNELS)
tests/ (doctest)  benchmarks/ (nanobench)  .github/workflows/ci.yml  CMakeLists.txt
```

Every algorithm is ~10 lines riding the shared `for_each_chunk` driver. Adding one =
fill 3 slots (predicate, accumulator, finalizer); a copy-this template + `ADD_ALGORITHM.md`.

## Algorithm set

### Cross-lane primitives (the marquee — `std::simd` can't express these)
| Primitive | Technique | Priority |
|---|---|---|
| `compress_store` | AVX-512 compress-to-**register** + storeu + popcount-advance (never `compressstoreu` — 40× slower on Zen4); AVX2 movemask → 256-entry permute-LUT → `vpermd`; scalar fallback | **P1** |
| `reverse_inplace` (any element size) | SSSE3 `pshufb`+const; AVX2 `permute4x64`+in-lane pshufb; NEON `vrev` — **generalizes the old 1/2-byte-only reverse** | P0 |
| `hsum`/`hmin`/`hmax`, `lane_count` | thin over `reduce`/`hmin`/`hmax`/`popcount` — **fixes the float-only `horizontal_sum` bug** | P0 |
| `gather`/`scatter` | `_mm*_i32gather` + scalar fallback (facade slot reserved) | P2 |

### Algorithms (public surface)
| Algorithm | `std::simd` does | SIMDTL adds | Priority |
|---|---|---|---|
| `count` / `count_if` | `==`, `popcount(mask)` | range loop; lane-accurate (**deletes ÷sizeof hack**) | **P0** (MVP slice) |
| `find` / `find_if` | `any_of`, `find_first_set` | early-exit loop + UB guard | P0 (new) |
| `min`/`max`/`minmax` (value & element) | element `min/max`, `hmin/hmax` | vector accumulator + index tracking | P0 (new) |
| `reduce` / `accumulate` | `reduce(v,op)` | chunked accumulator; **replaces `horizontal_sum`** (docs FP reorder) | P0 |
| `equal` / `mismatch` | `==`, `all_of` | two-range driver + first-diff index | P0 (new) |
| `transform` (unary/binary) | arithmetic/math ops | load/store driver + `vectorize()` adapter; **subsumes `add.h`** | P0 |
| `replace` / `replace_if` | `where(mask,v)=val` | in-place driver — **deletes the XOR trick + `force_xor`** | P0 |
| `copy_if`/`remove_if`/`partition` | only position-preserving blend | built on `compress_store` — **the headline feature** | P1 |
| `reverse` | — | wraps `reverse_inplace` | P0 |

### x86 string module (P0)
`count_in_range`, `to_lower`/`to_upper`/`flip_case` — ported from `range_comparisons.h`,
runtime-gated on CPUID SSE4.2 (not `__SSE4_2__`), portable scalar fallback.

### Deleted (obsoleted by `std::simd`)
`broadcast` (→ splat ctor) · `horizontal_sum` (→ generic `reduce`) · `force_xor` +
the replace-XOR trick (→ `where()=value`, float-safe) · `compare_equality` (→ `==`) ·
`move_mask` (→ `popcount`/`find_first_set`) · `add(ps/pd)` + `and/xor_si128`
(→ operators; raw ones survive only inside the SSE4.2 string kernel) ·
`make_integer_sequence.h` (→ `std::make_integer_sequence`) · `static_assert(false)`
(→ `if constexpr`) · AVX-hardcoded default (→ runtime dispatch) · `scoped_timer.h`
(→ nanobench) · UTF-16 `tests.cpp` (→ doctest differential suite). `preserve_constness.h`
+ `do_contiguous` survive, modernized.

## Runtime dispatch (the chosen model)

Detect-once, cache a function pointer per primitive, scalar floor always registered →
**the library never SIGILLs.** Detection: MSVC `__cpuid`/`__cpuidex` + mandatory
`_xgetbv` OS-state check; GCC/Clang `__builtin_cpu_supports` + `target_clones`.
Because MSVC has no per-function `/arch`, each ISA kernel set is its own `.cpp`
compiled with its own `/arch` (baseline dispatcher TU + `kernels_avx2.cpp` `/arch:AVX2`
+ `kernels_avx512.cpp` `/arch:AVX512`), exposing extern entries the baseline dispatcher
calls through the cached pointer — AVX never leaks into a non-AVX path. Header-only stays
the default; the wide kernels are an opt-in compiled component (`SIMDTL_FAST_KERNELS`).
AVX-512 compaction MUST use compress-to-register, never the memory form (Zen4 gotcha).

## Migration seam

One directory (`include/simdtl/backend/`) absorbs both the backend swap and the
TS↔C++26 rename. `simd.hpp` feature-detects native `<simd>` → experimental TS →
vir-simd. `names.hpp` maps stable names (`lane_count`/`hmin`/`hmax`/`find_first`/…)
onto the backend spelling (`popcount`/`hmin`/… today; `reduce_count`/`reduce_min`/… in
C++26). Nothing above L0 names `vir::`/`stdx::` directly — a CI grep-lint enforces it,
so the swap is provably one directory.

## Milestones

- **M0 — Foundation de-risk ✅ (done)** — repo scaffold, CMake INTERFACE target,
  vendored vir-simd, L0 seam, L1 CPU probe (CPUID+XGETBV), auto-vec smoke benchmark.
  Both M0 tools build & run on MSVC; findings recorded.
- **M1 — `count` vertical slice ✅ (done)** — `for_each_chunk` driver, `dispatch.hpp`
  fn-ptr table + an AVX2 `count<int32>` kernel through it (multi-TU dispatch proven),
  doctest differential harness (all edge sizes × types), nanobench, GitHub Actions
  matrix (MSVC/GCC/Clang/AppleClang/`ubuntu-24.04-arm` NEON/Emscripten), seam lint.
  Results in [docs/M1_RESULTS.md](docs/M1_RESULTS.md): dispatched kernel **3.46×**
  `std::count`; portable path 0.90× (the M0 finding, confirmed).
- **M2 — Stamp the template ✅ (done)** — find/find_if, min/max/minmax (value +
  `min_element`/`max_element`), reduce/accumulate, equal/mismatch, transform
  (unary/binary), replace/replace_if (`where()=value`, float-safe). All portable
  std::simd paths, each differential-tested vs the STL across the edge-size matrix
  and int8/16/32/64 + float/double (`tests/test_algorithms.cpp`). Fast intrinsic
  kernels for these remain M3.
- **M3 — Cross-lane marquee ✅ (done)** — `compress_store` primitive + `copy_if` /
  `remove_if` (generic, portable) + `remove(value)` and `reverse` with dispatched
  AVX2 kernels (LUT+`vpermd` compaction; block-reverse). All differential-tested vs
  the STL (`tests/test_compaction.cpp`). Results in [docs/M3_RESULTS.md](docs/M3_RESULTS.md):
  cache-resident `remove` ~1.14× `std::remove`; `reverse` is memory-bandwidth-bound
  (parity) — correctness is the deliverable, `count` remains the perf headline.
  AVX-512 `vpcompress` path + `partition`/`unique` remain follow-ups.
- **M4 — String-range port + polish** — SSE4.2 module + scalar fallback; delete-pass;
  install/export; benchmark report.
- **M5 (stretch)** — gather/scatter; NEON fast paths; native `<simd>` swap rehearsal.

## Decisions made
- **Runtime CPUID dispatch** (not AVX2-baseline) — robust on any CPU; header-only by
  default with opt-in compiled kernels.
- **vir-simd LGPL-3.0 accepted** (personal/public project) — contained behind the seam.
- **Public repo** — free GitHub `ubuntu-24.04-arm` runner gives a native NEON CI job.

## Open questions (carried forward)
- Native C++26 `<simd>` final identifiers — re-verify `names.hpp`'s std26 branch when a
  toolchain ships `<simd>` (corroborated via cppreference mirrors; live pages 403'd).
- The opt-in fn-ptr kernel linkage across direct build / FetchContent / installed
  package — the subtlest mechanism; front-loaded into M1 to surface bugs early.
- AVX2 full-256-bit reverse/compaction for 8/16-bit elements needs `permute4x64` +
  in-lane `pshufb`; unit-test per element size in M3.
- `copy_if`/`remove_if` API shape (variable output count breaks the pure template) —
  decide when designing the L3 compress facade.
