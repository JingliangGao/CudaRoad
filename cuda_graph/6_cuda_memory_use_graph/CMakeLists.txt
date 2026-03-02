
# add executable target
add_executable(cuda_memory_use_graph cuda_memory_use_graph.cu)

# link libraries
target_link_libraries(cuda_memory_use_graph PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

# # set architecture
# set_target_properties(cuda_memory_use_graph PROPERTIES
#     CUDA_ARCHITECTURES 70
# )

# set compile options
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(cuda_memory_use_graph PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G >)
endif()