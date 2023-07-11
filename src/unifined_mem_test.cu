#include <cuda_runtime.h>
#include <stdio.h>

// memcpyDtoH version
__global__ void AplusB(int *ret, int a, int b) {
    ret[threadIdx.x] = a + b + threadIdx.x;
}
int main() {
    int *ret;
    cudaMalloc(&ret, 1000 * sizeof(int));
    AplusB<<< 1, 1000 >>>(ret, 10, 100);
    int *host_ret = (int *)malloc(1000 * sizeof(int));
    cudaMemcpy(host_ret, ret, 1000 * sizeof(int), cudaMemcpyDeviceToHost);
    // the synchronous cudaMemcpy() routine is used both to synchronize the kernel (that is, to wait for it to finish running), 
    // and to transfer the data to the host
    for(int i = 0; i < 1000; i++)
        printf("%d: A+B = %d\n", i, host_ret[i]);
    free(host_ret);
    cudaFree(ret);
    return 0;
}


// // 
// __global__ void AplusB(int *ret, int a, int b) {
//     ret[threadIdx.x] = a + b + threadIdx.x;
// }
// int main() {
//     int *ret;
//     cudaMallocManaged(&ret, 1000 * sizeof(int));
//     AplusB<<< 1, 1000 >>>(ret, 10, 100);
//     cudaDeviceSynchronize();
//     for(int i = 0; i < 1000; i++)
//         printf("%d: A+B = %d\n", i, ret[i]);
//     cudaFree(ret);
//     return 0;
// }


// __device__ __managed__ int ret[1000];
// __global__ void AplusB(int a, int b) {
//     ret[threadIdx.x] = a + b + threadIdx.x;
// }
// int main() {
//     AplusB<<< 1, 1000 >>>(10, 100);
//     cudaDeviceSynchronize();
//     for(int i = 0; i < 1000; i++)
//         printf("%d: A+B = %d\n", i, ret[i]);
//     return 0;
// }