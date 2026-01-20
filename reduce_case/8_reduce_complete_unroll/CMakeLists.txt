
# add executable target
add_executable(reduce_complete_unroll reduce_complete_unroll.cu)

# link libraries
target_link_libraries(reduce_complete_unroll PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

# set compile options
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_complete_unroll PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G >)
endif()
