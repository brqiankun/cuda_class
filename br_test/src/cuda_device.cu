#include <cuda_runtime.h>
#include <stdio.h>
#include <unistd.h>

#define N 10000

__device__ int dev_count = 0;

__device__ int my_push_back(void) {
  atomicAdd(&dev_count, 1);   // 排序 ?
// dev_count++;
  return 0;
}

__global__ void test() {
    // my_push_back();
    size_t i = 0;
    while (i < 1000000000000000) i++;
    my_push_back();
    // printf("dev_count: %ld\n", i);
}

int main() {
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    int host_count = 999;
    test<<<1, 5, 0, stream1>>>();
    // cudaStreamSynchronize(stream1);
    cudaMemcpyFromSymbolAsync((void*)&host_count, dev_count, sizeof(int), 0, cudaMemcpyDeviceToHost, stream1);
    // cudaStreamSynchronize(stream1);
    printf("host_count: %d\n", host_count);
    // cudaDeviceSynchronize();
    // cudaStreamDestroy(stream1);
    return 0;
}