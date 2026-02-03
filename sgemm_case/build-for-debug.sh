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
echo ">> [INFO] Run 'sgemm_v0_global_memory' case ..."
cd ${CASE_DIR}/
./build/1_sgemm_global_memory/sgemm_v0_global_memory > ./build/1_sgemm_global_memory/1_sgemm_global_memory.log 2>&1