
# add executable target
add_executable(measure_cuda_event measure_cuda_event.cu)

# link libraries
target_link_libraries(measure_cuda_event PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

# set architecture
set_target_properties(measure_cuda_event PROPERTIES
    CUDA_ARCHITECTURES 70
)

# set compile options
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(measure_cuda_event PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G >)
endif()