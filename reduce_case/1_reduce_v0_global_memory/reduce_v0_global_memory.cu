#include <cstdio>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#define THREADS_PER_BLOCK 256

int main()
{
    const int N = 32 * 1024 * 1024;

    float *h_input = (float *)malloc(sizeof(float) * N);  /* malloc host memory : 32*4 Mb = 128 Mb */
    float *d_input;
    cudaMalloc((void **)&d_input, sizeof(float) * N);     /* malloc device memory ：32*4 Mb = 128 Mb */

    int block_num = N / THREADS_PER_BLOCK;  /* number of blocks : 32*1024*1024 / 256 = 131072 */
    float *h_output = (float *)malloc(sizeof(float) * block_num);   /* malloc host memory for output : 131072 * 4 bytes = 524288 bytes = 512 KB */
    float *d_output;
    cudaMalloc((void **)&d_output, sizeof(float) * block_num);      /* malloc device memory for output : 131072 * 4 bytes = 524288 bytes = 512 KB */
    std::cout << "**** Memory allocation *****" << std::endl;
    std::cout << "Host input size  : " << sizeof(float) * N / (1024.0f * 1024.0f) << " MB" << std::endl;
    std::cout << "Device input size: " << sizeof(float) * N / (1024.0f * 1024.0f) << " MB" << std::endl;
    std::cout << "Host output size : " << sizeof(float) * block_num / 1024.0f << " KB" << std::endl;    
    std::cout << "Device output size: " << sizeof(float) * block_num / 1024.0f << " KB" << std::endl;
    std::cout << "****************************" << std::endl;

    /* initialize input data */
    for (int i = 0; i < N; i++)
    {
        h_input[i] = 1.0f;   /* input data all set to 1.0f */
    }

    return 0;
}