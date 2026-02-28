
# add executable target
add_executable(multi_cuda_stream multi_cuda_stream.cu)

# link libraries
target_link_libraries(multi_cuda_stream PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

# set compile options
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(multi_cuda_stream PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G >)
endif()