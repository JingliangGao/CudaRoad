
# add executable target
add_executable(reduce_global_memory reduce_global_memory.cu   )

# link libraries
target_link_libraries(reduce_global_memory PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

# set compile options
# if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_global_memory PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G >)
# endif()