# Tests are compiled at BASELINE (no /arch:AVX2); the AVX2 kernels are added as
# separate sources compiled with /arch:AVX2, so the wide paths are reached only via
# runtime dispatch (proving the kernel is selected by CPUID, not baked into the TU).

# Add a per-/arch kernel SOURCE directly to a target (not a static lib, so the
# self-registering object is never dropped) and compile just that file with AVX2.
function(simdtl_add_avx2_kernel target src)
    if(SIMDTL_FAST_KERNELS)
        target_sources(${target} PRIVATE ${PROJECT_SOURCE_DIR}/src/kernels/${src})
        target_compile_definitions(${target} PRIVATE SIMDTL_HAVE_FAST_KERNELS)
        if(MSVC)
            set_source_files_properties(${PROJECT_SOURCE_DIR}/src/kernels/${src}
                PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
        else()
            set_source_files_properties(${PROJECT_SOURCE_DIR}/src/kernels/${src}
                PROPERTIES COMPILE_OPTIONS "-mavx2;-mpopcnt")
        endif()
    endif()
endfunction()

function(simdtl_add_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE simdtl::simdtl doctest)
    if(MSVC)
        target_compile_options(${name} PRIVATE /EHsc /W4 /external:W0)
    else()
        target_compile_options(${name} PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME ${name} COMMAND ${name})
endfunction()

simdtl_add_test(test_count)
simdtl_add_avx2_kernel(test_count count_avx2.cpp)

simdtl_add_test(test_algorithms)   # M2 portable paths only

simdtl_add_test(test_compaction)   # M3 cross-lane
simdtl_add_avx2_kernel(test_compaction crosslane_avx2.cpp)
