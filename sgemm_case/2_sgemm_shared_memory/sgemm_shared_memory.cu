#include <cstdio>
#include <iostream>
#include <chrono>

#define A(i, j) a[(i) * n + (j)]
#define B(i, j) b[(i) * n + (j)]

/* func: initialize matrix with random numbers */
void random_matrix(int m, int n, float *a) {
  int i, j;

  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
#if 1
      A(i, j) = 2.0 * (float)drand48() - 1.0;
#else
      A(i, j) = (j - i) % 3;
#endif
}

/* func: compare 2 matrices */
float compare_matrices(int m, int n, float *a, float *b) {

  int i, j;
  float max_diff = 0.0, diff;
  int printed = 0;

  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      diff = abs(A(i, j) - B(i, j));
      max_diff = (diff > max_diff ? diff : max_diff);
      if (0 == printed)
        if (max_diff > 0.5f || max_diff < -0.5f) {
          fprintf(stdout, "error: i %d  j %d diff %f  got %f  expect %f \n", i,
                  j, max_diff, A(i, j), B(i, j));
          printed = 1;
        }
    }
  }

  return max_diff;
}

/* func: sgemm in host */
void sgemm_cpu(float *A_ptr, float *B_ptr, float *C_ptr, const int M, const int N, const int K) {
      for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                  float sum = 0.0f;
                  for (int k = 0; k < K; k++) {
                        sum += A_ptr[m*K + k] * B_ptr[k*N + n];
                  }
                  C_ptr[m * N + n] = sum;
            }
      }
}

/* func: sgemm in device */
template <unsigned int BLOCK_SIZE, unsigned int TILE_SIZE>
__global__ void sgemm_cuda(float *A_ptr, float *B_ptr, float *C_ptr, const int M, const int N, const int K) {

      const int x = blockIdx.x * blockDim.x + threadIdx.x;
      const int y = blockIdx.y * blockDim.y + threadIdx.y;
      float *A_ptr_start = A_ptr + blockIdx.y * blockDim.y * K;
      float *B_ptr_start = B_ptr + blockIdx.x * blockDim.x;

      /* copy data : global memory -> shared memory */
      __shared__ float A_shared[BLOCK_SIZE][TILE_SIZE];
      __shared__ float B_shared[TILE_SIZE][BLOCK_SIZE];
      for(int s = 0; s < K; s+=blockDim.x) {
            A_shared[threadIdx.y][threadIdx.x+ s] = A_ptr_start[threadIdx.x + s + threadIdx.y * K];
            B_shared[threadIdx.y + s][threadIdx.x] = B_ptr_start[(threadIdx.y + s) * N + threadIdx.x];
      }
      __syncthreads();

      float sum = 0.0f;
      for (int k = 0; k < K; k++) {           
            sum += A_shared[threadIdx.y][k] * B_shared[k][threadIdx.x];
      }

      C_ptr[x + y * N] = sum;
}


int main() {
    constexpr int m = 256;
    constexpr int n = 256;
    constexpr int k = 256;
    constexpr int BLOCK_SIZE = 16;

    const size_t mem_size_A = sizeof(float) * m * k;
    const size_t mem_size_B = sizeof(float) * k * n;
    const size_t mem_size_C = sizeof(float) * m * n;    
    std::cout << "*************************** Shape Information ***************************" << std::endl;
    std::cout << "Host matrix_A shape  : " << m << " x " << k << std::endl;
    std::cout << "Host matrix_B shape  : " << k << " x " << n << std::endl;
    std::cout << "Host matrix_C shape  : " << m << " x " << n << std::endl;
    std::cout << "*************************************************************************" << std::endl;

    /* malloc memory in host */
    float *matrix_A_host = (float *)malloc(mem_size_A);
    float *matrix_B_host = (float *)malloc(mem_size_B);
    float *matrix_C_host_gpu = (float *)malloc(mem_size_C);
    float *matrix_C_host_cpu = (float *)malloc(mem_size_C);

    /* malloc memory in device */
    float *matrix_A_device;
    float *matrix_B_device;
    float *matrix_C_device;
    cudaMalloc((void **)&matrix_A_device, mem_size_A);
    cudaMalloc((void **)&matrix_B_device, mem_size_B);
    cudaMalloc((void **)&matrix_C_device, mem_size_C);

    /* get shared memory information in device */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    size_t shared_bytes = (BLOCK_SIZE * k + k * BLOCK_SIZE) * sizeof(float);


    std::cout << "*************************** Memory Allocation ***************************" << std::endl;
    std::cout << "Host matrix_A memory  : " << sizeof(float) * mem_size_A / (1024.0f * 1024.0f) << " MB" << std::endl;
    std::cout << "Host matrix_B memory  : " << sizeof(float) * mem_size_B / (1024.0f * 1024.0f) << " MB" << std::endl;
    std::cout << "Host matrix_C memory  : " << sizeof(float) * mem_size_C / (1024.0f * 1024.0f) << " MB" << std::endl;
    std::cout << "Device matrix_A memory: " << sizeof(float) * mem_size_A / (1024.0f * 1024.0f) << " MB" << std::endl;
    std::cout << "Device matrix_B memory: " << sizeof(float) * mem_size_B / (1024.0f * 1024.0f) << " MB" << std::endl;
    std::cout << "Device matrix_C memory: " << sizeof(float) * mem_size_C / (1024.0f * 1024.0f) << " MB" << std::endl;
    std::cout << "Tile memory: " << shared_bytes / 1024.0f << " KB" << std::endl;
    std::cout << "Shared memory per block: " << prop.sharedMemPerBlock / 1024.0f << " KB" << std::endl;
    std::cout << "Shared memory per SM: " << prop.sharedMemPerMultiprocessor / 1024.0f << " KB" << std::endl;
    std::cout << "*************************************************************************" << std::endl;

    /* initialize matrix in host */
    random_matrix(m, k, matrix_A_host);
    random_matrix(k, n, matrix_B_host);
    memset(matrix_C_host_gpu, 0, mem_size_C);
    memset(matrix_C_host_cpu, 0, mem_size_C);

    /* calculate sgemm in host */
    auto t_host_start = std::chrono::high_resolution_clock::now();
    sgemm_cpu(matrix_A_host, matrix_B_host, matrix_C_host_cpu, m, n, k);
    auto t_host_end = std::chrono::high_resolution_clock::now();

    /* calculate sgemm in device */
    auto t_device_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(matrix_A_device, matrix_A_host, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(matrix_B_device, matrix_B_host, mem_size_B, cudaMemcpyHostToDevice);
    auto t_device_H2D = std::chrono::high_resolution_clock::now();
    dim3 dimGrid((m + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    std::cout << "*************************** Dim Information ***************************" << std::endl;
    std::cout << "dimGrid: " << dimGrid.x << " x " << dimGrid.y << std::endl;
    std::cout << "dimBlock: " << dimBlock.x << " x " << dimBlock.y << std::endl;
    std::cout << "***********************************************************************" << std::endl;
    sgemm_cuda<BLOCK_SIZE, k><<<dimGrid, dimBlock>>>(matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);
    auto t_device_kernel = std::chrono::high_resolution_clock::now();
    cudaMemcpy(matrix_C_host_gpu, matrix_C_device, mem_size_C, cudaMemcpyDeviceToHost);
    auto t_device_D2H = std::chrono::high_resolution_clock::now();
    std::cout << "*************************** Time Consume ***************************" << std::endl;
    std::cout << "Host cost time: " << std::chrono::duration_cast<std::chrono::microseconds>(t_host_end - t_host_start).count() << " us" << std::endl;
    std::cout << "Device cost time: " << std::chrono::duration_cast<std::chrono::microseconds>(t_device_D2H - t_device_start).count() << " us " 
                                      << " (" << std::chrono::duration_cast<std::chrono::microseconds>(t_device_H2D - t_device_start).count() 
                                      << " + " << std::chrono::duration_cast<std::chrono::microseconds>(t_device_kernel - t_device_H2D).count() 
                                      << " + " << std::chrono::duration_cast<std::chrono::microseconds>(t_device_D2H - t_device_kernel).count()
                                      << ")" <<std::endl;
    std::cout << "*************************************************************************" << std::endl;

    /* compare result */
    float diff = compare_matrices(m, n, matrix_C_host_cpu, matrix_C_host_gpu);
    if (diff > 0.5f || diff < -0.5f) {
        printf(">> [ERROR] SGEMM v0: verification failed! max diff = %f\n", diff);
    } else {
        printf(">> [INFO] SGEMM v0: verification success! max diff = %f\n", diff);
    }
        
    printf(">> [INFO] all finished!\n");

    return 0;
}