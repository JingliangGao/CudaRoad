
# add executable target
add_executable(reduce_unroll_last_dim reduce_unroll_last_dim.cu)

# link libraries
target_link_libraries(reduce_unroll_last_dim PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

# set compile options
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_unroll_last_dim PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G >)
endif()