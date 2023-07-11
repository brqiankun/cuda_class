#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hello() {
    int j = 0;
  for (int i = 0; i < 1e6; i++) {
    j++;
  }
  printf("GPU: HELLO\n");
}

int main() {
    printf("CPU: HELLO\n"); 
    hello<<<1, 10>>>();
    int* d_p = nullptr;
    cudaMalloc(&d_p, 10);
    hello<<<1, 10>>>();
    cudaDeviceSynchronize();
    return 0;
}
