# set cmake minimum required version
cmake_minimum_required(VERSION 3.20.0)

# set project name and version
project(cuda_practice VERSION 0.1.0 LANGUAGES CXX C CUDA)

# find packages
find_package(CUDAToolkit)

# add subdirectories
add_subdirectory(0_prepare_env)
add_subdirectory(1_reduce_global_memory)
add_subdirectory(2_reduce_shared_memory)
add_subdirectory(3_reduce_no_divergence_branch)
add_subdirectory(4_reduce_no_bank_conflict)
add_subdirectory(5_reduce_add_during_load_PlanA)
add_subdirectory(6_reduce_add_during_load_PlanB)
add_subdirectory(7_reduce_unroll_last_dim)
add_subdirectory(8_reduce_complete_unroll)