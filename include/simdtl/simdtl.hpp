#pragma once
// ── SIMDTL umbrella include ───────────────────────────────────────────────────
// SIMDTL = the SIMD operations std::simd deliberately leaves out: STL-style range
// algorithms, cross-lane / stream-compaction primitives, the SSE4.2 string-range
// ops, and runtime ISA dispatch — built ON TOP of the std::simd programming model
// (supplied today by the header-only vir-simd backport).
//
// M0 foundation: backend seam + CPU-feature probe.
// M1: tail driver, runtime dispatch table, count/count_if.
#include "backend/simd.hpp"
#include "backend/names.hpp"
#include "platform/arch_macros.hpp"
#include "platform/cpu.hpp"
#include "platform/dispatch.hpp"
#include "detail/driver.hpp"
#include "algorithm/count.hpp"

// Future milestones (kept here as the public surface map):
// #include "algorithm/find.hpp"       // M2: find / find_if
// #include "algorithm/minmax.hpp"     // M2: min/max/minmax
// #include "algorithm/reduce.hpp"     // M2: reduce / accumulate
// #include "algorithm/equal.hpp"      // M2: equal / mismatch
// #include "algorithm/transform.hpp"  // M2: transform (+ vectorize adapter)
// #include "algorithm/replace.hpp"    // M2: replace / replace_if (where()=value)
// #include "crosslane/compress.hpp"   // M3: stream compaction
// #include "crosslane/reverse.hpp"    // M3: any-size reverse
// #include "algorithm/copy_if.hpp"    // M3: copy_if / remove_if / partition
// #include "string_range.hpp"         // M4: SSE4.2 case/range ops
