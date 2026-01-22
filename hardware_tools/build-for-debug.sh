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
cd ${CASE_DIR}
echo ">> [INFO] Run 'reduce_global_memory' case ..."
./build/1_reduce_global_memory/reduce_global_memory > ./build/1_reduce_global_memory/1_reduce_global_memory.log 2>&1

# collect log files
# cd ${CASE_DIR}/
# find ./build -type f -name "*.log" -exec cp {} doc/log/ \;

echo ">> [INFO] All done!"

