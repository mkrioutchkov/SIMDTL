# Adversarial correctness review (post-M4)

A multi-agent review swept the kernels/algorithms (5 area reviewers → 2 independent
skeptics per finding). Four areas — **compaction, reverse, scan-algorithms, and
dispatch/seam/build — came back clean.** All confirmed findings were in the SSE4.2
string module, with one root cause.

## Root cause
`PCMPISTRM` (`_mm_cmpistrm`) reads its **ranges operand with implicit length** — as a
NUL-terminated string of at most 8 `[lo,hi]` pairs (16 bytes). So:

| # | Bug | Trigger | Symptom (SSE4.2 path) |
|---|-----|---------|------------------------|
| 1 | `count_in_range` zero low bound | `lo == 0` | operand truncates → always returns 0 |
| 2 | `convert_case` too many pairs | `npairs > 8` | extra pairs silently ignored in 16-byte chunks |
| 3 | `convert_case` zero boundary | a pair byte `== 0` | first 16 bytes left untransformed |

These were silent and **CPU-dependent** (the scalar fallback was already correct), and
the test suite missed them by only using printable-ASCII data and ranges.

## Fix
Gate the SSE4.2 fast path to inputs `PCMPISTRM` can encode faithfully; otherwise fall
back to the (fully general) scalar path — so the public API is correct for *all* inputs,
only the speed differs:
- `count_in_range`: SSE4.2 only when `lo != 0 && hi != 0`.
- `convert_case`: SSE4.2 only when `1 <= npairs <= 8` and no boundary byte is 0
  (`detail::ranges_ok_for_sse42`).

Regression tests added in `tests/test_string.cpp` (zero-boundary range, `>8` pairs,
zero-boundary pair) — each compared to an independent reference; each would fail on an
SSE4.2 CPU before the guard.
