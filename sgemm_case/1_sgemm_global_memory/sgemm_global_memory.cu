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
__global__ void sgemm_cuda(float *A_ptr, float *B_ptr, float *C_ptr, const int M, const int N, const int K) {
      /* Mat_C position */
      const int x = blockIdx.x * blockDim.x + threadIdx.x;
      const int y = blockIdx.y * blockDim.y + threadIdx.y;

      /* Mat_A, Mat_B block start pointer */
      float *A_ptr_start = A_ptr + blockIdx.y * blockDim.y * K;
      float *B_ptr_start = B_ptr + blockIdx.x * blockDim.x;

      float sum = 0.0f;    /* temporary store sum result */
      for (int k = 0; k < K; k++) {   /* traverse K elements */        
            sum += A_ptr_start[threadIdx.y * K + k] * B_ptr_start[k * N + threadIdx.x];
      }
      C_ptr[x + y * N] = sum;
}


int main() {
    int m = 512;
    int n = 512;
    int k = 512;

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

    std::cout << "*************************** Memory Allocation ***************************" << std::endl;
    std::cout << "Host matrix_A size  : " << sizeof(float) * mem_size_A / (1024.0f * 1024.0f) << " MB" << std::endl;
    std::cout << "Host matrix_B size  : " << sizeof(float) * mem_size_B / (1024.0f * 1024.0f) << " MB" << std::endl;
    std::cout << "Host matrix_C size  : " << sizeof(float) * mem_size_C / (1024.0f * 1024.0f) << " MB" << std::endl;
    std::cout << "Device matrix_A size: " << sizeof(float) * mem_size_A / (1024.0f * 1024.0f) << " MB" << std::endl;
    std::cout << "Device matrix_B size: " << sizeof(float) * mem_size_B / (1024.0f * 1024.0f) << " MB" << std::endl;
    std::cout << "Device matrix_C size: " << sizeof(float) * mem_size_C / (1024.0f * 1024.0f) << " MB" << std::endl;
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
    constexpr int BLOCK_SIZE = 16;
    dim3 dimGrid((m + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    sgemm_cuda<<<dimGrid, dimBlock>>>(matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);
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