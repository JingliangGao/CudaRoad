
# add executable target
add_executable(sgemm_shared_memory_sliding_window sgemm_shared_memory_sliding_window.cu)

# link libraries
target_link_libraries(sgemm_shared_memory_sliding_window PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

# set compile options
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(sgemm_shared_memory_sliding_window PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G >)
endif()