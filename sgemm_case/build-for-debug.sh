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
echo ">> [INFO] Run 'sgemm_global_memory' case ..."
cd ${CASE_DIR}/
./build/1_sgemm_global_memory/sgemm_global_memory > ./build/1_sgemm_global_memory/1_sgemm_global_memory.log 2>&1

# run case
echo ">> [INFO] Run 'sgemm_shared_memory' case ..."
cd ${CASE_DIR}/
./build/2_sgemm_shared_memory/sgemm_shared_memory > ./build/2_sgemm_shared_memory/2_sgemm_shared_memory.log 2>&1

# run case
echo ">> [INFO] Run 'sgemm_shared_memory_sliding_window' case ..."
cd ${CASE_DIR}/
./build/3_sgemm_shared_memory_sliding_window/sgemm_shared_memory_sliding_window > ./build/3_sgemm_shared_memory_sliding_window/3_sgemm_shared_memory_sliding_window.log 2>&1