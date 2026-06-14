# M1/M3 benchmarks (nanobench). Like the tests, the main TU is baseline and the
# AVX2 kernels are separately-compiled sources, so the dispatched path is exercised.

function(simdtl_bench_avx2_kernel target src)
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

function(simdtl_add_bench name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE simdtl::simdtl nanobench)
    if(MSVC)
        target_compile_options(${name} PRIVATE /EHsc /external:W0)
    endif()
endfunction()

simdtl_add_bench(bench_count)
simdtl_bench_avx2_kernel(bench_count count_avx2.cpp)

simdtl_add_bench(bench_compaction)
simdtl_bench_avx2_kernel(bench_compaction crosslane_avx2.cpp)
