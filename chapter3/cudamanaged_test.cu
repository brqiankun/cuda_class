#include <iostream>
#include <cuda_runtime.h>


#define checkCudaErrors(status)                                   \
{                                                                 \
  if (status != 0)                                                \
  {                                                               \
    std::cout << "Cuda failure: " << cudaGetErrorString(status)   \
              << " at line " << __LINE__                          \
              << " in file " << __FILE__                          \
              << " error status: " << status                      \
              << std::endl;                                       \
              abort();                                            \
    }                                                             \
}

void Getinfo(void)
{
    cudaDeviceProp prop;

    int count = 0;
    cudaGetDeviceCount(&count);
    printf("\nGPU has cuda devices: %d\n", count);
    for (int i = 0; i < count; ++i) {
        cudaGetDeviceProperties(&prop, i);
        printf("----device id: %d info----\n", i);
        printf("  GPU : %s \n", prop.name);
        printf("  Capbility: %d.%d\n", prop.major, prop.minor);
        printf("  Global memory: %luMB\n", prop.totalGlobalMem >> 20);
        printf("  Const memory: %luKB\n", prop.totalConstMem  >> 10);
        printf("  SM in a block: %luKB\n", prop.sharedMemPerBlock >> 10);
        printf("  warp size: %d\n", prop.warpSize);
        printf("  threads in a block: %d\n", prop.maxThreadsPerBlock);
        printf("  block dim: (%d,%d,%d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  grid dim: (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }
    printf("\n");
}

__global__ void test_kernel(float* data_p, int cnt) {
    int idx = threadIdx.x;
    *(data_p + idx) = idx + 1;
    if (idx < cnt) {
        printf("thread_%d: %f\n", idx, *(data_p + idx));
    }
    int dummy;
    for (int i = 0; i < 10000; i++) {
        dummy++;
    }
} 

int main() {
    Getinfo();
    float* d_p;
    checkCudaErrors(cudaMallocManaged(&d_p, sizeof(float)*10));
    checkCudaErrors(cudaMemset(d_p, 0, sizeof(float)*10));
    test_kernel<<<1, 15>>>(d_p, 10);
    checkCudaErrors(cudaDeviceSynchronize());
    for (int i = 0; i < 10; i++) {
        printf("from cpu: d_p[%d]: %f\n", i, d_p[i]);
    }


    return 0;
}