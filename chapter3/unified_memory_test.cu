#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#define SWITCH 1

#if SWITCH == 0 || SWITCH == 1
__global__ void AplusB(int* ret, int a, int b) {
  ret[threadIdx.x] = a + b + threadIdx.x;
}
#endif

#if SWITCH == 0
// both host- and device-side storage for the return values is required
int main() {
  auto begin = std::chrono::high_resolution_clock::now();
  int* ret;
  cudaMalloc(&ret, 1000 * sizeof(int));   // deivice return values
  AplusB<<<1, 1000>>>(ret, 10, 100);
  int *host_ret = (int*)malloc(1000 * sizeof(int));  // host return values
  cudaMemcpy(host_ret, ret, 1000 * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 1000; i++) {
    printf("%d: A + B = %d\n", i, host_ret[i]);
  }
  free(host_ret);
  cudaFree(ret);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
  printf("switch 0 using %ld ms\n", duration_time.count());
  return 0;
}
#endif

#if SWITCH == 1
int main() {
  auto begin = std::chrono::high_resolution_clock::now();
  int* ret;
  cudaMallocManaged(&ret, 1000 * sizeof(int));
  AplusB<<<1, 1000>>>(ret, 10, 100);
  cudaDeviceSynchronize();
  for (int i = 0; i < 1000; i++) {
    printf("%d: A + B = %d\n", i, ret[i]);
  }
  cudaFree(ret);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
  printf("switch 1 using %ld ms\n", duration_time.count());
  return 0;
}
#endif

#if SWITCH == 2

__device__ __managed__ int ret[1000];
__global__ void AplusB(int a, int b) {
  ret[threadIdx.x] = a + b + threadIdx.x;
}

int main() {
  auto begin = std::chrono::high_resolution_clock::now();
  AplusB<<<1, 1000>>>(10, 1000);
  cudaDeviceSynchronize();
  for (int i = 0; i < 1000; i++) {
    printf("%d: A + B = %d\n", i, ret[i]);
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
  printf("switch 2 using %ld ms\n", duration_time.count());
  return 0;
}
#endif