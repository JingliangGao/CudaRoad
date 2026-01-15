
# add executable target
add_executable(reduce_no_bank_conflict reduce_no_bank_conflict.cu   )

# link libraries
target_link_libraries(reduce_no_bank_conflict PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

# set compile options
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_no_bank_conflict PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G >)
endif()