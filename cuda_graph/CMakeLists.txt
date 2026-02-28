# set cmake minimum required version
cmake_minimum_required(VERSION 3.20.0)

# set project name and version
project(cuda_practice VERSION 0.1.0 LANGUAGES CXX C CUDA)

# find packages
find_package(CUDAToolkit)

# add subdirectories
add_subdirectory(1_check_async_engine)
add_subdirectory(2_multi_cuda_stream)
add_subdirectory(3_measure_cuda_event)

                 