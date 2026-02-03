#include <cstdio>

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
void sgemm_cpu( float *A_ptr, float *B_ptr, float *C_ptr, const int M, const int N, const int K) {
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


int main() {
    int m = 2048;
    int n = 2048;
    int k = 2048;

    const size_t mem_size_A = sizeof(float) * m * k;
    const size_t mem_size_B = sizeof(float) * k * n;
    const size_t mem_size_C = sizeof(float) * m * n;    

    /* malloc memory */
    float *matrix_A_host = (float *)malloc(mem_size_A);
    float *matrix_B_host = (float *)malloc(mem_size_B);
    float *matrix_C_host_gpu = (float *)malloc(mem_size_C);
    float *matrix_C_host_cpu = (float *)malloc(mem_size_C);

    /* initialize matrix in host */
    random_matrix(m, k, matrix_A_host);
    random_matrix(k, n, matrix_B_host);
    memset(matrix_C_host_gpu, 0, mem_size_C);
    memset(matrix_C_host_cpu, 0, mem_size_C);

    /* malloc memory in device */
    float *matrix_A_device;
    float *matrix_B_device;
    float *matrix_C_device;
    cudaMalloc((void **)&matrix_A_device, mem_size_A);
    cudaMalloc((void **)&matrix_B_device, mem_size_B);
    cudaMalloc((void **)&matrix_C_device, mem_size_C);
    cudaMemcpy(matrix_A_device, matrix_A_host, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(matrix_B_device, matrix_B_host, mem_size_B, cudaMemcpyHostToDevice);

    /* calculate sgemm in host */
    sgemm_cpu(matrix_A_host, matrix_B_host, matrix_C_host_cpu, m, n, k);


    /* calculate sgemm in device */
//     constexpr int BLOCK_SIZE = 16;
//     dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
//     dim3 dimGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
//     sgemm_cuda<<<dimGrid, dimBlock>>>(matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);
//     cudaMemcpy(matrix_C_host_gpu, matrix_C_device, mem_size_C, cudaMemcpyDeviceToHost);

    float diff = compare_matrices(m, n, matrix_C_host_cpu, matrix_C_host_gpu);
    if (diff > 1e-6 || diff < -1e-6) {
        printf("SGEMM v0: verification failed! max diff = %f\n", diff);
    } else {
        printf("SGEMM v0: verification success! max diff = %f\n", diff);
    }
        

    printf("SGEMM v0: using global memory only\n");

    return 0;
}