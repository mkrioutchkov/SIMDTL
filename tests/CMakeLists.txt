# M1 test suite (doctest). The test TU is compiled at BASELINE (no /arch:AVX2) so
# that the AVX2 path is reached only through runtime dispatch — proving the kernel
# is selected via CPUID, not baked into the whole binary.
add_executable(test_count test_count.cpp)
target_link_libraries(test_count PRIVATE simdtl::simdtl doctest)

if(MSVC)
    target_compile_options(test_count PRIVATE /EHsc /W4 /external:W0)
else()
    target_compile_options(test_count PRIVATE -Wall -Wextra)
endif()

# Opt-in fast kernels: add the per-/arch kernel SOURCE directly (not as a static
# lib, so the self-registering object is never dropped) and compile only that file
# with /arch:AVX2.
if(SIMDTL_FAST_KERNELS)
    target_sources(test_count PRIVATE ${PROJECT_SOURCE_DIR}/src/kernels/count_avx2.cpp)
    target_compile_definitions(test_count PRIVATE SIMDTL_HAVE_FAST_KERNELS)
    if(MSVC)
        set_source_files_properties(${PROJECT_SOURCE_DIR}/src/kernels/count_avx2.cpp
            PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    else()
        set_source_files_properties(${PROJECT_SOURCE_DIR}/src/kernels/count_avx2.cpp
            PROPERTIES COMPILE_OPTIONS "-mavx2;-mpopcnt")
    endif()
endif()

add_test(NAME test_count COMMAND test_count)
