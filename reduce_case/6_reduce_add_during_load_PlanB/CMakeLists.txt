
# add executable target
add_executable(reduce_add_during_load_PlanB reduce_add_during_load_PlanB.cu)

# link libraries
target_link_libraries(reduce_add_during_load_PlanB PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

# set compile options
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_add_during_load_PlanB PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G >)
endif()