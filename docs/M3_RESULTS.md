# M3 — cross-lane marquee: results

The ops `std::simd` genuinely cannot express: stream compaction and cross-lane
reverse. std::simd offers only position-preserving `where()`; packing selected
lanes into a contiguous prefix is SIMDTL's to own.

## What's proven
- **`compress_store`** primitive (portable lane-extract; never overruns `dst`).
- **`copy_if` / `remove_if`** (generic predicate, portable) and **`remove(value)`**
  (dispatched AVX2 `int32` kernel: `cmpeq` → 256-entry `vpermd` LUT → `storeu` +
  popcount-advance; in-place safe because the write cursor never passes the read).
- **`reverse`** — portable two-pointer for ANY element size (generalizing the old
  library's 1/2-byte-only limit) + a dispatched AVX2 `int32` block-reverse kernel.
- All differential-tested vs the STL across the edge-size matrix and multiple element
  types (`tests/test_compaction.cpp`); the AVX2 kernels are confirmed installed at the
  `avx2` tier. Seam lint clean.

## Benchmark (nanobench, Release, MSVC 14.51, AVX2 laptop)

**Cache-resident (8192 int32 / 32 KB)** — measures the kernel, not DRAM:

| op | std | simdtl (AVX2) | relative |
|---|---:|---:|---:|
| `remove(==5)` (incl. memcpy restore) | 1,999 ns | 1,760 ns | **1.14× faster** |
| `reverse` (restore-free) | 345 ns | 346 ns | ~1.0× (parity) |

**Multi-MB (2^20 int32 / 4 MB)** — both ops become **memory-bandwidth-bound**, so
SIMD ≈ scalar (`reverse` ~144 µs both; `remove` memcpy-dominated). Expected: these
ops are dominated by data movement, not compute, once they spill cache.

## Reading the numbers honestly
- **`count` (M1) remains the perf headline** at **3.46×** — it's read-only, so compute
  (compare + popcount) dominates and SIMD wins big.
- **`remove`** gives a real but modest win where it's compute-bound (branchless
  compaction vs std's per-element branch); the gain shrinks as size grows and
  bandwidth dominates.
- **`reverse`** is pure data movement — SIMD matches but can't beat memory bandwidth.
  The deliverable here is *correctness for any element size* plus the dispatched path,
  not a speedup.
- Bigger wins would come from the **AVX-512 `vpcompress`** path (compress-to-register)
  and from fusing compaction into producer loops so the data is already hot — both
  future work.
