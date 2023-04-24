#include <cuda_runtime.h>
#include <stdio.h>

__global__ void malloc_host_test(float* dev_cout_) {
    atomicAdd(dev_cout_, 1.0);
    printf("GPU: dev_cout_: %p\n", dev_cout_);
    printf("GPU: dev_cout_: %f\n", *dev_cout_);
}

int main() {
    float* host_count = NULL;
    float* dev_count = NULL;
    // Allocates size bytes of host memory that is page-locked and accessible to the device. 
    cudaMallocHost((void**)&host_count, sizeof(float));
    // cudaDeviceSynchronize();
    memset(host_count, 0, sizeof(float));
    malloc_host_test<<<1, 5>>>(host_count);  //unified virtual address
    cudaHostGetDevicePointer(&dev_count, host_count, 0);
    malloc_host_test<<<1, 5>>>(dev_count);
    cudaDeviceSynchronize();
    (*host_count) ++;
    printf("CPU: host_cout : %f\n", *host_count);
    printf("CPU: host_cout : %p\n", host_count);
    (*dev_count) ++;
    printf("CPU: dev_cout : %f\n", *dev_count);
    printf("CPU: dev_cout : %p\n", dev_count);
    return 0;

}