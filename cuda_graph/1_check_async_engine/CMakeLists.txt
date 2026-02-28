
# add executable target
add_executable(check_async_engine check_async_engine.cu)

# link libraries
target_link_libraries(check_async_engine PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

# set compile options
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(check_async_engine PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G >)
endif()