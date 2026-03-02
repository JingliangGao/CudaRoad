# CUDA Graph

## Note
| case                           |  information                                 |
| ------------------------------ | -------------------------------------------- |
| 1_check_async_engine           | 检查设备中async_engine的数量                 |
| 2_multi_cuda_stream            | 2个stream, 3个kernel的执行过程               |
| 3_measure_cuda_event           | 2个stream, 3个kernel的执行过程               |
| 4_capture_cuda_graph           | cuda_graph的使用方法                         |
| 5_cuda_memory_no_graph         | 不使用cuda graph时的内存池的显存变化         |
| 6_cuda_memory_use_graph        | 使用cuda graph时的内存池的显存变化           |


# Build && Run
```bash
chmod +x build-for-debug.sh && ./build-for-debug.sh 
```