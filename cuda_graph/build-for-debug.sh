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
echo ">> [INFO] Run cases 'check_async_engine' ..."
./build/1_check_async_engine/check_async_engine
echo ">> [INFO] Run cases 'multi_cuda_stream' ..."
./build/2_multi_cuda_stream/multi_cuda_stream
echo ">> [INFO] Run cases 'measure_cuda_event' ..."
./build/3_measure_cuda_event/measure_cuda_event