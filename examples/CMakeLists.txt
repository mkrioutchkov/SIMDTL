# Runnable examples. Built at baseline; the AVX2 kernels are added as separate
# /arch:AVX2 sources so the dispatched paths are exercised at runtime.
add_executable(showcase showcase.cpp)
target_link_libraries(showcase PRIVATE simdtl::simdtl)
if(MSVC)
    target_compile_options(showcase PRIVATE /EHsc /external:W0)
endif()

if(SIMDTL_FAST_KERNELS)
    target_compile_definitions(showcase PRIVATE SIMDTL_HAVE_FAST_KERNELS)
    foreach(k count_avx2.cpp crosslane_avx2.cpp)
        target_sources(showcase PRIVATE ${PROJECT_SOURCE_DIR}/src/kernels/${k})
        if(MSVC)
            set_source_files_properties(${PROJECT_SOURCE_DIR}/src/kernels/${k}
                PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
        else()
            set_source_files_properties(${PROJECT_SOURCE_DIR}/src/kernels/${k}
                PROPERTIES COMPILE_OPTIONS "-mavx2;-mpopcnt")
        endif()
    endforeach()
endif()
