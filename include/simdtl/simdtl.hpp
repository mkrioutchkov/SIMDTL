#pragma once
// ── SIMDTL umbrella include ───────────────────────────────────────────────────
// SIMDTL = the SIMD operations std::simd deliberately leaves out: STL-style range
// algorithms, cross-lane / stream-compaction primitives, the SSE4.2 string-range
// ops, and runtime ISA dispatch — built ON TOP of the std::simd programming model
// (supplied today by the header-only vir-simd backport).
//
// M0 (foundation) exposes the backend seam and the CPU-feature probe.
// M1+ add: detail/driver.hpp, crosslane/*, algorithm/* (count, find, reduce, …).
#include "backend/simd.hpp"
#include "backend/names.hpp"
#include "platform/arch_macros.hpp"
#include "platform/cpu.hpp"

// Future milestones (kept here as the public surface map):
// #include "detail/driver.hpp"        // M1: for_each_chunk tail driver
// #include "platform/dispatch.hpp"    // M1: runtime fn-ptr table
// #include "crosslane/horizontal.hpp" // M1: hsum/hmin/hmax + lane_count
// #include "algorithm/count.hpp"      // M1: count / count_if (the template)
// #include "crosslane/compress.hpp"   // M3: stream compaction
// #include "crosslane/reverse.hpp"    // M3: any-size reverse
// #include "string_range.hpp"         // M4: SSE4.2 case/range ops
