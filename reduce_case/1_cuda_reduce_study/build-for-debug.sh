#!/bin/bash

CASE_DIR=$(pwd)

# refresh build dir
if [ -d build ]; then
    rm -rf  build
fi
mkdir build

# build case
cd ${CASE_DIR}
cmake ../
make -j
