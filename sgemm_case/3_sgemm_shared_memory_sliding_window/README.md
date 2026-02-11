# SGEMM using shared memory
利用共享内存计算SGEMM，分块滑动相乘后累加和

## Description
对于矩阵A[M, K]、矩阵B[K, N]，求解矩阵相乘结果，即矩阵C[M, N]。算法的示意图，如下图所示：
![image](image/shared_memory_sliding_window.png)

算法的核心思想：将灰色分块处的数据迁移至Shared_Memory, 相乘后累加，从而加速load和计算的时间。      
    