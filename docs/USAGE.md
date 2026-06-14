# Using SIMDTL

Header-only. Point your include path at `include/` and the vendored backport at
`third_party/vir-simd/include/`, then:

```cpp
#include <simdtl/simdtl.hpp>
```

All algorithms live in namespace `simdtl` and take either a raw `(pointer, size)`
range or, where noted, a contiguous container.

---

## Examples

### count / count_if
```cpp
std::vector<int> v = {3,1,2,2,5,2};
std::size_t twos  = simdtl::count(v, 2);                       // 3   (container overload)
std::size_t big   = simdtl::count(v.data(), v.size(), 5);      // 1   (range overload)
std::size_t gt2   = simdtl::count_if(v, [](auto x){ using X = decltype(x); return x > X(2); }); // 2
```

### find / find_if  (returns a pointer, or first+n if not found)
```cpp
const int* p = simdtl::find(v.data(), v.size(), 5);
bool found   = (p != v.data() + v.size());
const int* q = simdtl::find_if(v.data(), v.size(),
                               [](auto x){ using X = decltype(x); return x > X(4); });
```

### min / max / minmax  (value + first-occurrence element pointer)
```cpp
int lo = simdtl::min_value(v.data(), v.size());
int hi = simdtl::max_value(v.data(), v.size());
auto [a, b]     = simdtl::minmax_value(v.data(), v.size());
const int* pmin = simdtl::min_element(v.data(), v.size());     // like std::min_element
```

### reduce / accumulate  (sum; associative — FP order may differ from std)
```cpp
int sum = simdtl::reduce(v.data(), v.size());                  // init defaults to 0
int s2  = simdtl::reduce(v.data(), v.size(), 100);             // with an initial value
```

### equal / mismatch
```cpp
bool same = simdtl::equal(a.data(), b.data(), n);
auto [pa, pb] = simdtl::mismatch(a.data(), b.data(), n);       // first differing position
```

### transform  (unary or binary; the op is elemental — see below)
```cpp
simdtl::transform(in.data(), n, out.data(), [](auto x){ return x * x; });          // unary
simdtl::transform(a.data(), b.data(), n, out.data(), [](auto x, auto y){ return x + y; }); // binary
```

### replace / replace_if  (in place; correct for floats — uses where()=value)
```cpp
simdtl::replace(v.data(), v.size(), 2, -1);                    // every 2 -> -1
simdtl::replace_if(v.data(), v.size(),
                   [](auto x){ using X = decltype(x); return x < X(0); }, 0);       // negatives -> 0
```

### copy_if / remove_if / remove / partition / unique  (stream compaction)
```cpp
std::vector<int> out(v.size());
std::size_t k = simdtl::copy_if(v.data(), v.size(), out.data(),
                                [](auto x){ using X = decltype(x); return (x & X(1)) == X(0); });
out.resize(k);                                                 // kept the evens

v.resize(simdtl::remove(v.data(), v.size(), 2));               // drop all 2s (dispatched AVX2 for int32)
v.resize(simdtl::remove_if(v.data(), v.size(),
                           [](auto x){ using X = decltype(x); return x > X(3); }));
v.resize(simdtl::unique(v.data(), v.size()));                  // drop consecutive duplicates

std::size_t point = simdtl::partition(v.data(), v.size(),      // stable; trues first
                                      [](auto x){ using X = decltype(x); return x < X(5); });
```

### reverse  (any element size; dispatched AVX2 for int32)
```cpp
simdtl::reverse(v.data(), v.size());
```

### string ops (x86 SSE4.2 fast path + portable scalar fallback)
```cpp
std::string s = "Hello, World 123";
simdtl::to_upper(s.data(), s.size());                          // "HELLO, WORLD 123"
simdtl::to_lower(s.data(), s.size());
simdtl::flip_case(s.data(), s.size());
std::size_t digits = simdtl::count_in_range(s.data(), s.size(), '0', '9');
```

---

## Elemental predicates / operators

Every `*_if` / `transform` takes an **elemental** callable — a generic lambda that
works on BOTH a whole SIMD vector and a single scalar:

```cpp
[](auto x){ using X = decltype(x); return x > X(4); }
```

Internally the driver calls it twice with different argument types:

| call site | `x` is | `X(4)` is | result |
|---|---|---|---|
| vector body | `simd<T>` (a whole register) | a broadcast (4 in every lane) | `simd_mask<T>` |
| scalar tail | `T` (one element) | `T(4)` | `bool` |

That's why you write `X(4)` (via `decltype(x)`) rather than a bare `x > 4`: it
constructs the right type in each instantiation. The mask path is reduced with
`lane_count` (popcount of true lanes); the scalar path is a plain `bool`.

Same idea for `transform`: `[](auto x){ return x * x; }` compiles once as
`simd<T> -> simd<T>` and once as `T -> T`.

---

## How width and dispatch actually work

**Vector width is a COMPILE-TIME decision, not a runtime one.** The portable
algorithms use `simdtl::native<T>` (= `std::simd`'s native ABI), whose width is
baked in by the flags the translation unit is built with — `std::simd` selects one
ABI per TU:

| build flags | `native<int>::size()` |
|---|---|
| `/arch:AVX2` (MSVC) or `-mavx2` (GCC/Clang) | 8 (256-bit) |
| x64 baseline (SSE2) | 4 (128-bit) |
| ARM | NEON width |

So `count`, `find`, `reduce`, etc. adapt to the **build target**, not the running
CPU. To get AVX out of them, compile with AVX.

**Runtime CPUID dispatch exists only for the hand-written intrinsic kernels.**
Those live in separate per-`/arch` TUs and are picked at first use after CPUID +
XGETBV confirm support (with a scalar/portable floor so it never SIGILLs):

| operation | runtime-dispatched fast path | otherwise |
|---|---|---|
| `count` (int8 / int16 / int32) | AVX2 `cmpeq`+`movemask`+`popcnt` kernel | portable `std::simd` |
| `remove` (int8 / int16 / int32) | AVX2 compaction kernel (`pshufb`/`vpermd` left-pack) | portable |
| `reverse` (int32) | AVX2 block-reverse kernel | portable (any element size) |
| string ops (`char`) | SSE4.2 `cmpistrm` | portable scalar |
| everything else | — | portable `std::simd` |

> Perf note: on MSVC at `/arch:AVX2` the compiler auto-vectorizes simple `std::` loops
> (count/replace) and even byte/word `std::remove`, so the kernels mostly *tie* there;
> the clear win is `int32` compaction (`std::remove<int32>` stays scalar) and any platform
> whose compiler doesn't auto-vectorize. The kernels also rescue narrow-type performance
> from the portable fallback (which is ~80× slower for `int8` on MSVC).

**MSVC caveat:** vir-simd's `fixed_size<N>` fallback does not emit packed AVX on
MSVC (it lowers to scalar ops). So on MSVC the portable layer is correctness +
portability; the *speed* comes from the dispatched kernels above. On GCC/Clang with
a native ABI, the portable layer is real SIMD. See [M0_FINDINGS.md](M0_FINDINGS.md).

---

## Building against SIMDTL (CMake)

```cmake
add_subdirectory(SIMDTL)            # or FetchContent
target_link_libraries(myapp PRIVATE simdtl::simdtl)
# Optional, for the dispatched intrinsic kernels:
#   -DSIMDTL_FAST_KERNELS=ON   (compiles src/kernels/*.cpp per-/arch)
```

Build your consuming TU with `/arch:AVX2` (MSVC) or `-mavx2` (GCC/Clang) to give the
portable layer a wide target. The dispatched kernels are compiled at their own arch
regardless and selected at runtime.
