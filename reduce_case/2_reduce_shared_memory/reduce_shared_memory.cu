#include <cstdio>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <chrono>
#define THREADS_PER_BLOCK 256


/* 'reduce' kernel function */
__global__ void reduce(float *d_input, float *d_output) {

    __shared__ float shared_data[THREADS_PER_BLOCK];   /* create array in shared memory */
    int index = blockDim.x * blockIdx.x + threadIdx.x; /* pointer to global position */
    shared_data[threadIdx.x] = d_input[index];         /* load data : global memory -> shared memory */
    __syncthreads();    /* ensure all threads have loaded data */

    for (int i = 1; i < blockDim.x; i *= 2) {
        if (threadIdx.x % (i * 2) == 0) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        d_output[blockIdx.x] = shared_data[0];   /* store result to global memory */
    }

}



/* check function */
bool check(float *out, float *res, int n) {
    for(int i = 0; i < n; i++) {
        if(abs(out[i] - res[i]) > 0.005) {
            std::cout << "Mismatch at index " << i << ": " << out[i] << " vs " << res[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main()
{
    const int N = 32 * 1024 * 1024; /* number of elements in the array */

    float *h_input = (float *)malloc(sizeof(float) * N);  /* malloc host memory */
    float *d_input;
    cudaMalloc((void **)&d_input, sizeof(float) * N);     /* malloc device memory */

    int block_num = N / THREADS_PER_BLOCK;  /* number of blocks */
    float *h_output = (float *)malloc(sizeof(float) * block_num);   /* malloc host memory for output */
    float *d_output;
    cudaMalloc((void **)&d_output, sizeof(float) * block_num);      /* malloc device memory for output*/
    float *h_result = (float *)malloc(sizeof(float) * block_num);   /* malloc host memory for storage result */
    std::cout << "***** Memory allocation *****" << std::endl;
    std::cout << "Host input size  : " << sizeof(float) * N / (1024.0f * 1024.0f) << " MB" << std::endl;
    std::cout << "Device input size: " << sizeof(float) * N / (1024.0f * 1024.0f) << " MB" << std::endl;
    std::cout << "Host output size : " << sizeof(float) * block_num / 1024.0f << " KB" << std::endl;    
    std::cout << "Device output size: " << sizeof(float) * block_num / 1024.0f << " KB" << std::endl;
    std::cout << "*****************************" << std::endl;

    /* initialize input data */
    for (int i = 0; i < N; i++)
    {
        h_input[i] = 2.0 * (float)drand48() - 1.0;  /* random number between -1.0 and 1.0 */
    }

    /* CPU reduction */
    auto t_host_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < block_num; i++)
    {
        float cur_result = 0.0;
        for (int j =0; j < THREADS_PER_BLOCK; j ++) {
            cur_result += h_input[i*THREADS_PER_BLOCK + j];
        }
        h_result[i] = cur_result;
    }
    auto t_host_end = std::chrono::high_resolution_clock::now();
    std::cout << ">> [INFO] CPU reduction done " << std::endl;

    auto t_device_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_input, h_input, sizeof(float)*N, cudaMemcpyHostToDevice);   /* copy input data from host to device */
    auto t_h2d_end = std::chrono::high_resolution_clock::now();
    dim3 Grid(block_num, 1, 1);
    dim3 Block(THREADS_PER_BLOCK, 1, 1);
    reduce<<<Grid, Block>>>(d_input, d_output);   /* launch kernel */
    auto t_kernel_end = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_output, d_output, block_num*sizeof(float), cudaMemcpyDeviceToHost);
    auto t_device_end = std::chrono::high_resolution_clock::now();
    std::cout << ">> [INFO] GPU reduction done " << std::endl;

    /* compare result */
    if(check(h_output, h_result, block_num)) {
        printf(">> [INFO] The result is right. \n"); 
    }else{
        printf(">> [ERROR] The result is wrong! \n");
    }

    std::cout << "***** Time Consume *****" << std::endl;
    std::cout << "Host CPU time: " << std::chrono::duration_cast<std::chrono::microseconds>(t_host_end - t_host_start).count() << " microseconds" << std::endl;
    std::cout << "Device H2D time: " << std::chrono::duration_cast<std::chrono::microseconds>(t_h2d_end - t_device_start).count() << " microseconds" << std::endl;
    std::cout << "Device kernel time: " << std::chrono::duration_cast<std::chrono::microseconds>(t_kernel_end - t_h2d_end).count() << " microseconds" << std::endl;
    std::cout << "Device D2H time: " << std::chrono::duration_cast<std::chrono::microseconds>(t_device_end - t_kernel_end).count() << " microseconds" << std::endl;
    std::cout << "************************" << std::endl;

    /* free memory */
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_result);

    return 0;
}