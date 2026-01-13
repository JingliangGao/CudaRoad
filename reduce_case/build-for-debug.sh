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
./build/0_prepare_env/print_string

echo ">> [INFO] Run 'reduce_v0_global_memory' case ..."
./build/1_reduce_v0_global_memory/reduce_v0_global_memory

echo ">> [INFO] Run cases success!"

