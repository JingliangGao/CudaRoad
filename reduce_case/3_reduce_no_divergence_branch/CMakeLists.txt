
# add executable target
add_executable(reduce_no_divergence_branch reduce_no_divergence_branch.cu   )

# link libraries
target_link_libraries(reduce_no_divergence_branch PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

# set compile options
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_no_divergence_branch PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G >)
endif()