
# add executable target
add_executable(print_string print_string.cu)

# link libraries
target_link_libraries(print_string PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

# set compile options
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(print_string PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G >)
endif()