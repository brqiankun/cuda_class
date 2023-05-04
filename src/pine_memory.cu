#include <cuda_runtime.h>
#include <stdio.h>
#include "freshman.h"

void sumArrays(float* a, float* b, float* res, const int size) {
    for(int i=0;i<size;i+=4) {
        res[i] = a[i]+b[i];
        res[i+1] = a[i+1] + b[i+1];
        res[i+2] = a[i+2] + b[i+2];
        res[i+3] = a[i+3] + b[i+3];
    }
}

__global__ void sumArrayGPU(float* a, float* b, float* res) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    res[i] = a[i] + b[i];
}

int main(int argc, char* argv[]) {
    int dev = 0;
    cudaSetDevice(dev);

    int power = 10;
    if(argc > 2)
        power = atoi(argv[1]);
    
    int nElem=1<<power;
    printf("vector size: %d\n", nElem);
    int nByte = nElem*sizeof(float);
    // float* a_h = (float*) malloc(nByte);
    // float* b_h = (float*) malloc(nByte);
    float* res_h = (float*) malloc(nByte);
    float* res_from_gpu_h = (float*) malloc(nByte);
    memset(res_h, 0, nByte);
    memset(res_from_gpu_h, 0, nByte);

    float *a_h, *b_h;
    float* a_d, *b_d, *res_d;
    //pinned memory allocate
    CHECK(cudaHostAlloc((float**)&a_h, nByte, cudaHostAllocMapped));
    CHECK(cudaHostAlloc((float**)&b_h, nByte, cudaHostAllocMapped));
    CHECK(cudaMalloc((float**)&res_d, nByte));

    initialData(a_h, nElem);
    initialData(b_h, nElem);

    CHECK(cudaHostGetDevicePointer((void**)&a_d, (void*)a_h, 0));
    CHECK(cudaHostGetDevicePointer((void**)&b_d, (void*)b_h, 0));
    // CHECK(cudaMemcpy(a_d, a_h, nByte, cudaMemcpyHostToDevice));
    // CHECK(cudaMemcpy(b_d, b_h, nByte, cudaMemcpyHostToDevice));
    printf("%p  %p  %p  %p\n",a_h, a_d, b_h, b_d);

    dim3 block(1024);
    dim3 grid(nElem/block.x);
    sumArrayGPU<<<grid, block>>>(a_d, b_d, res_d);
    printf("Execution configutation <<<%d, %d>>>\n", grid.x, block.x);

    CHECK(cudaMemcpy(res_from_gpu_h, res_d, nByte, cudaMemcpyDeviceToHost));
    sumArrays(a_h, b_h, res_h, nElem);

    checkResult(res_h, res_from_gpu_h, nElem);
    cudaFreeHost(a_h);
    cudaFreeHost(b_h);
    
    cudaFree(res_d);
    free(res_h);
    free(res_from_gpu_h);

    return 0;

}