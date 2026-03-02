
# add executable target
add_executable(capture_cuda_graph capture_cuda_graph.cu)

# link libraries
target_link_libraries(capture_cuda_graph PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

# set architecture
set_target_properties(capture_cuda_graph PROPERTIES
    CUDA_ARCHITECTURES 70
)

# set compile options
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(capture_cuda_graph PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G >)
endif()