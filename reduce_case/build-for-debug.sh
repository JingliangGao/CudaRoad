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
./build/0_prepare_env/print_string > ./build/0_prepare_env/0_print_string.log 2>&1

echo ">> [INFO] Run 'reduce_global_memory' case ..."
./build/1_reduce_global_memory/reduce_global_memory > ./build/1_reduce_global_memory/1_reduce_global_memory.log 2>&1

echo ">> [INFO] Run 'reduce_shared_memory' case ..."
./build/2_reduce_shared_memory/reduce_shared_memory > ./build/2_reduce_shared_memory/2_reduce_shared_memory.log 2>&1

echo ">> [INFO] Run 'reduce_no_divergence_branch' case ..."
./build/3_reduce_no_divergence_branch/reduce_no_divergence_branch > ./build/3_reduce_no_divergence_branch/3_reduce_no_divergence_branch.log 2>&1

echo ">> [INFO] Run 'reduce_no_bank_conflict' case ..."
./build/4_reduce_no_bank_conflict/reduce_no_bank_conflict > ./build/4_reduce_no_bank_conflict/4_reduce_no_bank_conflict.log 2>&1

echo ">> [INFO] Run 'reduce_add_during_load(Plan A)' case ..."
./build/5_reduce_add_during_load_PlanA/reduce_add_during_load_PlanA > ./build/5_reduce_add_during_load_PlanA/5_reduce_add_during_load_PlanA.log 2>&1

echo ">> [INFO] Run 'reduce_add_during_load(Plan B)'  case ..."
./build/6_reduce_add_during_load_PlanB/reduce_add_during_load_PlanB > ./build/6_reduce_add_during_load_PlanB/6_reduce_add_during_load_PlanB.log 2>&1

echo ">> [INFO] Run 'reduce_unroll_last_dim'  case ..."
./build/7_reduce_unroll_last_dim/reduce_unroll_last_dim > ./build/7_reduce_unroll_last_dim/7_reduce_unroll_last_dim.log 2>&1

echo ">> [INFO] Run 'reduce_complete_unroll'  case ..."
./build/8_reduce_complete_unroll/reduce_complete_unroll > ./build/8_reduce_complete_unroll/8_reduce_complete_unroll.log 2>&1

echo ">> [INFO] Run cases success!"

# collect log files
# cd ${CASE_DIR}/
# find ./build -type f -name "*.log" -exec cp {} doc/log/ \;

echo ">> [INFO] All done!"

