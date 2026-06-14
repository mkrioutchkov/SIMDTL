cmake_minimum_required(VERSION 3.21)

project(simdtl
    VERSION 0.1.0
    DESCRIPTION "SIMD operations std::simd leaves out: range algorithms, cross-lane/compaction, SSE4.2 string ops, runtime dispatch"
    LANGUAGES CXX)

# ── Options ───────────────────────────────────────────────────────────────────
option(SIMDTL_BUILD_TESTS      "Build the doctest differential test suite" OFF)
option(SIMDTL_BUILD_BENCHMARKS "Build the nanobench STL-vs-SIMD benchmarks" OFF)
option(SIMDTL_BUILD_EXAMPLES   "Build the runnable examples (showcase)" OFF)
option(SIMDTL_BUILD_M0         "Build the M0 foundation tools (cpu probe + smoke)" OFF)
option(SIMDTL_FAST_KERNELS     "Compile the opt-in per-/arch intrinsic kernels (M1+)" OFF)
set(SIMDTL_VIR_SIMD_PROVIDER "vendored" CACHE STRING "How to obtain vir-simd: vendored | fetch | system")
set_property(CACHE SIMDTL_VIR_SIMD_PROVIDER PROPERTY STRINGS vendored fetch system)

# ── Dependency: vir-simd (the std::simd programming model on MSVC C++20) ───────
# Exposed as an imported/interface target carrying its include dir as SYSTEM so
# its diagnostics never pollute our /W4 build. Kept strictly behind the L0 seam.
if(SIMDTL_VIR_SIMD_PROVIDER STREQUAL "system")
    find_package(vir-simd CONFIG REQUIRED)
    add_library(simdtl_vir INTERFACE)
    target_link_libraries(simdtl_vir INTERFACE vir-simd::vir-simd)
elseif(SIMDTL_VIR_SIMD_PROVIDER STREQUAL "fetch")
    include(FetchContent)
    FetchContent_Declare(vir-simd
        GIT_REPOSITORY https://github.com/mattkretz/vir-simd.git
        GIT_TAG        v0.4.4)
    FetchContent_MakeAvailable(vir-simd)
    add_library(simdtl_vir INTERFACE)
    target_link_libraries(simdtl_vir INTERFACE vir-simd)
else() # vendored (default): third_party/vir-simd/include
    add_library(simdtl_vir INTERFACE)
    target_include_directories(simdtl_vir SYSTEM INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/vir-simd/include>
        $<INSTALL_INTERFACE:include/simdtl/third_party>)
endif()

# ── The library (header-only INTERFACE target) ────────────────────────────────
add_library(simdtl INTERFACE)
add_library(simdtl::simdtl ALIAS simdtl)
target_compile_features(simdtl INTERFACE cxx_std_20)
target_include_directories(simdtl INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(simdtl INTERFACE simdtl_vir)

# Helper: apply the widest baseline ISA so the auto-vectorizer/codegen has a real
# target. (Runtime dispatch + per-/arch kernels arrive with SIMDTL_FAST_KERNELS.)
function(simdtl_apply_baseline tgt)
    if(MSVC)
        target_compile_options(${tgt} PRIVATE /arch:AVX2 /W4 /external:W0 /EHsc)
    else()
        target_compile_options(${tgt} PRIVATE -mavx2 -mpopcnt -Wall -Wextra)
    endif()
endfunction()

# ── M0 foundation tools ───────────────────────────────────────────────────────
if(SIMDTL_BUILD_M0)
    foreach(tool m0_cpu_probe m0_smoke)
        add_executable(${tool} tools/${tool}.cpp)
        target_link_libraries(${tool} PRIVATE simdtl::simdtl)
        simdtl_apply_baseline(${tool})
    endforeach()
endif()

if(SIMDTL_BUILD_TESTS)
    enable_testing()
    add_library(doctest INTERFACE)
    target_include_directories(doctest SYSTEM INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party)   # <doctest/doctest.h>
    add_subdirectory(tests)          # M1: doctest differential suite
endif()
if(SIMDTL_BUILD_BENCHMARKS)
    add_library(nanobench INTERFACE)
    target_include_directories(nanobench SYSTEM INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party)   # <nanobench/nanobench.h>
    add_subdirectory(benchmarks)     # M1: nanobench harness
endif()
if(SIMDTL_BUILD_EXAMPLES)
    add_subdirectory(examples)       # runnable showcase
endif()

# ── Install / export ──────────────────────────────────────────────────────────
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
install(TARGETS simdtl simdtl_vir EXPORT simdtl-targets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
if(SIMDTL_VIR_SIMD_PROVIDER STREQUAL "vendored")
    install(DIRECTORY third_party/vir-simd/include/
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/simdtl/third_party)
endif()
install(EXPORT simdtl-targets NAMESPACE simdtl:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/simdtl)
configure_package_config_file(
    cmake/simdtl-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/simdtl-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/simdtl)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/simdtl-config-version.cmake
    COMPATIBILITY SameMajorVersion)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/simdtl-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/simdtl-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/simdtl)
