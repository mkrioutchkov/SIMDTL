#pragma once
// ── L1: runtime dispatch table ────────────────────────────────────────────────
// A slot per dispatchable kernel, holding the best fn-ptr for the running CPU.
// Default slot = nullptr → callers fall back to the portable std::simd path, so
// the library is fully usable header-only (zero kernel TUs). When an opt-in
// per-/arch kernel TU is linked, its static initializer calls register_*() and
// upgrades the slot — but only if the CPU actually supports that ISA. The
// baseline-built dispatcher only ever calls through the cached pointer, so wide
// instructions never execute on a CPU that lacks them.
//
// NOTE (open question carried in PLAN.md): a self-registering kernel object is
// kept only when its TU is linked directly into the binary. Packaging the kernels
// in a *static library* requires /WHOLEARCHIVE (MSVC) or --whole-archive so the
// registrar is not dropped. M1 links kernel sources directly to side-step this.
#include "cpu.hpp"
#include <cstddef>
#include <cstdint>

namespace simdtl::platform
{
    using count_i32_fn = std::size_t (*)(const std::int32_t*, std::size_t, std::int32_t) noexcept;

    inline count_i32_fn& count_i32_slot() noexcept
    {
        static count_i32_fn fn = nullptr;
        return fn;
    }

    inline isa_level& count_i32_installed_level() noexcept
    {
        static isa_level lvl = isa_level::scalar;
        return lvl;
    }

    // Called from per-/arch kernel TUs at static-init time.
    inline void register_count_i32(isa_level lvl, count_i32_fn fn) noexcept
    {
        if (best_isa() >= lvl && (count_i32_slot() == nullptr || lvl > count_i32_installed_level()))
        {
            count_i32_slot() = fn;
            count_i32_installed_level() = lvl;
        }
    }

    // --- remove(value) compaction: returns new logical length (std::remove) ---
    using remove_i32_fn = std::size_t (*)(std::int32_t*, std::size_t, std::int32_t) noexcept;
    inline remove_i32_fn& remove_i32_slot() noexcept { static remove_i32_fn fn = nullptr; return fn; }
    inline isa_level& remove_i32_installed_level() noexcept { static isa_level lvl = isa_level::scalar; return lvl; }
    inline void register_remove_i32(isa_level lvl, remove_i32_fn fn) noexcept
    {
        if (best_isa() >= lvl && (remove_i32_slot() == nullptr || lvl > remove_i32_installed_level()))
        {
            remove_i32_slot() = fn;
            remove_i32_installed_level() = lvl;
        }
    }

    // --- reverse in place ---
    using reverse_i32_fn = void (*)(std::int32_t*, std::size_t) noexcept;
    inline reverse_i32_fn& reverse_i32_slot() noexcept { static reverse_i32_fn fn = nullptr; return fn; }
    inline isa_level& reverse_i32_installed_level() noexcept { static isa_level lvl = isa_level::scalar; return lvl; }
    inline void register_reverse_i32(isa_level lvl, reverse_i32_fn fn) noexcept
    {
        if (best_isa() >= lvl && (reverse_i32_slot() == nullptr || lvl > reverse_i32_installed_level()))
        {
            reverse_i32_slot() = fn;
            reverse_i32_installed_level() = lvl;
        }
    }

    // --- count(value) for 1- and 2-byte ints (AVX2 cmpeq + movemask + popcount) ---
    using count_i8_fn  = std::size_t (*)(const std::int8_t*,  std::size_t, std::int8_t)  noexcept;
    using count_i16_fn = std::size_t (*)(const std::int16_t*, std::size_t, std::int16_t) noexcept;
    inline count_i8_fn&  count_i8_slot()  noexcept { static count_i8_fn  fn = nullptr; return fn; }
    inline count_i16_fn& count_i16_slot() noexcept { static count_i16_fn fn = nullptr; return fn; }
    inline isa_level& count_i8_lvl()  noexcept { static isa_level l = isa_level::scalar; return l; }
    inline isa_level& count_i16_lvl() noexcept { static isa_level l = isa_level::scalar; return l; }
    inline void register_count_i8(isa_level lvl, count_i8_fn fn) noexcept
    { if (best_isa() >= lvl && (count_i8_slot() == nullptr || lvl > count_i8_lvl())) { count_i8_slot() = fn; count_i8_lvl() = lvl; } }
    inline void register_count_i16(isa_level lvl, count_i16_fn fn) noexcept
    { if (best_isa() >= lvl && (count_i16_slot() == nullptr || lvl > count_i16_lvl())) { count_i16_slot() = fn; count_i16_lvl() = lvl; } }

    // --- remove(value) compaction for 1- and 2-byte ints (pshufb left-pack) ---
    using remove_i8_fn  = std::size_t (*)(std::int8_t*,  std::size_t, std::int8_t)  noexcept;
    using remove_i16_fn = std::size_t (*)(std::int16_t*, std::size_t, std::int16_t) noexcept;
    inline remove_i8_fn&  remove_i8_slot()  noexcept { static remove_i8_fn  fn = nullptr; return fn; }
    inline remove_i16_fn& remove_i16_slot() noexcept { static remove_i16_fn fn = nullptr; return fn; }
    inline isa_level& remove_i8_lvl()  noexcept { static isa_level l = isa_level::scalar; return l; }
    inline isa_level& remove_i16_lvl() noexcept { static isa_level l = isa_level::scalar; return l; }
    inline void register_remove_i8(isa_level lvl, remove_i8_fn fn) noexcept
    { if (best_isa() >= lvl && (remove_i8_slot() == nullptr || lvl > remove_i8_lvl())) { remove_i8_slot() = fn; remove_i8_lvl() = lvl; } }
    inline void register_remove_i16(isa_level lvl, remove_i16_fn fn) noexcept
    { if (best_isa() >= lvl && (remove_i16_slot() == nullptr || lvl > remove_i16_lvl())) { remove_i16_slot() = fn; remove_i16_lvl() = lvl; } }
} // namespace simdtl::platform
