#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>

#include <cuda_runtime.h>

#include "vec_add_gpu.h"


int main(int argc, char* argv[]) {
    int num_to_compute = atoi(argv[1]);
    printf("num to compute %d\n", num_to_compute);
    float *a_d, *b_d, *c_d;
    float *a_h, *b_h, *c_h;
    /***
     * memory alloc
    */
    int size = num_to_compute * sizeof(float);
    a_h = (float*)malloc(size);
    b_h = (float*)malloc(size);
    c_h = (float*)malloc(size);
    init_data(a_h, num_to_compute);
    init_data(b_h, num_to_compute);
    memset(c_h, 0, num_to_compute);

    cudaMalloc(&a_d, size);
    cudaMalloc(&b_d, size);
    cudaMalloc(&c_d, size);
    init_data(a_h, num_to_compute);
    init_data(b_h, num_to_compute);
    memset(c_h, 0, num_to_compute);

    cudaMemcpy(a_d, a_h, num_to_compute, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, num_to_compute, cudaMemcpyHostToDevice);

    int thread_of_block = 256;
    int block_of_grid = (num_to_compute + thread_of_block - 1) / thread_of_block;
    struct timeval start, end;
    gettimeofday(&start, NULL);
    vec_add<<<block_of_grid, thread_of_block>>>(a_d, b_d, c_d, num_to_compute);
    cudaMemcpy(c_h, c_d, num_to_compute, cudaMemcpyDeviceToHost);
    gettimeofday(&end, NULL);

    long timeuse_u = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    printf("func time elapse %f s\n", timeuse_u / 1000000.0);

}

__global__ void vec_add(const float* a_d, const float* b_d, float* c_d, const int num_to_compute) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < num_to_compute) {
        c_d[i] = a_d[i] + b_d[i];
    }
}

void init_data(float* p, int num) {
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < num; i++) {
        p[i] = (float)(rand()&0xffff) / 1000.0f;
    }
}