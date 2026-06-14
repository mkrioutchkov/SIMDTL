# M1 benchmark (nanobench). Like the tests, the main TU is baseline and the AVX2
# kernel is a separately-compiled source, so the dispatched path is exercised.
add_executable(bench_count bench_count.cpp)
target_link_libraries(bench_count PRIVATE simdtl::simdtl nanobench)

# Optimization comes from the build type (configure with -DCMAKE_BUILD_TYPE=Release);
# don't hardcode /O2 here or it clashes with Debug's /RTC1.
if(MSVC)
    target_compile_options(bench_count PRIVATE /EHsc /external:W0)
endif()

if(SIMDTL_FAST_KERNELS)
    target_sources(bench_count PRIVATE ${PROJECT_SOURCE_DIR}/src/kernels/count_avx2.cpp)
    target_compile_definitions(bench_count PRIVATE SIMDTL_HAVE_FAST_KERNELS)
    if(MSVC)
        set_source_files_properties(${PROJECT_SOURCE_DIR}/src/kernels/count_avx2.cpp
            PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    else()
        set_source_files_properties(${PROJECT_SOURCE_DIR}/src/kernels/count_avx2.cpp
            PROPERTIES COMPILE_OPTIONS "-mavx2;-mpopcnt")
    endif()
endif()
