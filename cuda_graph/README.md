# CUDA Graph

## Note
| case                           |  information                                 |
| ------------------------------ | -------------------------------------------- |
| 1_check_async_engine           | 检查设备中async_engine的数量                 |
| 2_multi_cuda_stream            | 2个stream, 3个kernel的执行过程               |
| 3_measure_cuda_event           | 2个stream, 3个kernel的执行过程               |



# Build && Run
```bash
chmod +x build-for-debug.sh && ./build-for-debug.sh 
```