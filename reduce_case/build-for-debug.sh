#!/bin/bash

CASE_DIR=$(pwd)

# refresh build dir
echo ">> [INFO] Refresh 'build' dir ..."
if [ -d build ]; then
    rm -rf  build
fi
mkdir build

# build case
echo ">> [INFO] Build cases ..."
cd ${CASE_DIR}/build
cmake ../
make -j

# run case
echo ">> [INFO] Run 'print_string' case ..."
cd ${CASE_DIR}/
./build/0_prepare_env/print_string > ./build/0_prepare_env/print_string.log 2>&1

echo ">> [INFO] Run 'reduce_global_memory' case ..."
./build/1_reduce_global_memory/reduce_global_memory > ./build/1_reduce_global_memory/reduce_global_memory.log 2>&1

echo ">> [INFO] Run 'reduce_shared_memory' case ..."
./build/2_reduce_shared_memory/reduce_shared_memory > ./build/2_reduce_shared_memory/reduce_shared_memory.log 2>&1

echo ">> [INFO] Run 'reduce_no_divergence_branch' case ..."
./build/3_reduce_no_divergence_branch/reduce_no_divergence_branch > ./build/3_reduce_no_divergence_branch/reduce_no_divergence_branch.log 2>&1

echo ">> [INFO] Run cases success!"

