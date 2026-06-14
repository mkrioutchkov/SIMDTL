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
} // namespace simdtl::platform
