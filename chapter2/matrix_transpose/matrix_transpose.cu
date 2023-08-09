#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define SWITCH 0

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file,
                               const int line) {
  cudaError_t err = cudaGetLastError();

  if (cudaSuccess != err) {
    fprintf(stderr,
            "%s(%i) : getLastCudaError() CUDA error :"
            " %s : (%d) %s.\n",
            file, line, errorMessage, static_cast<int>(err),
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

const int BLOCK_SIZE = 32;
const int M = 2000;
const int N = 1000;

__managed__ int matrix[M * N];
__managed__ int gpu_result[N * M];
__managed__ int cpu_result[N * M];

__global__ void GpuMatrixTranspose(int* input, int* output) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < M && y < N) {
    output[y * M + x] = input[x * N + y];
  }
}

#if SWITCH == 0

__global__ void GpuMatrixTransposeShared(int* input, int* output) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  __shared__ int sub_matrix[BLOCK_SIZE][BLOCK_SIZE + 1];    // 连续的32个数据放置在32个不同的bank中，将它们32个连续地址加1
  if (x < M && y < N) {
    int data_x = blockIdx.x * blockDim.x + threadIdx.y;
    int data_y = blockIdx.y * blockDim.y + threadIdx.x;
    if (data_x < M && data_y < N) {   // 有可能线程在范围内，但线程对称后访问的数据不在范围内
      sub_matrix[threadIdx.y][threadIdx.x] = input[data_x * N + data_y];
      __syncthreads();
      output[y * M + x] = sub_matrix[threadIdx.x][threadIdx.y]; // bank conflict 同一个warp(32个)的不同线程读取同一个bank中的数据时就会冲突
    }
  }
}

#elif SWITCH == 1

__global__ void GpuMatrixTransposeShared(int* input, int* output) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  __shared__ int sub_matrix[BLOCK_SIZE][BLOCK_SIZE];
  if (x < M && y < N) {
    sub_matrix[threadIdx.x][threadIdx.y] = input[x * N + y];
    __syncthreads();
    output[y * M + x] = sub_matrix[threadIdx.x][threadIdx.y];
  }
}

#endif

void CpuMatrixTranspose(int* input, int* output) {  // 也可以直接传入指针
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      output[j * M + i] = input[i * N + j];
    }
  }
}


bool CheckResult(int* res1, int* res2, int elem_num) {
  for (int i = 0; i < elem_num; i++) {
    if (abs(res2[i] - res1[i]) > 1.0e-10) {
      std::printf("(%d)res2[%d] != (%d)res2[%d];check Result faild\n", i, res2[i], i, res1[i]);
      return false;
    }
  }
  std::printf("check Result pass\n");
  return true;
}

int main() {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      matrix[i * N + j] = std::rand() % 1024;
    }
  }

  cudaEvent_t start_gpu1, stop_gpu1, start_gpu2, stop_gpu2, start_cpu, stop_cpu;
  cudaEventCreate(&start_gpu1);
  cudaEventCreate(&stop_gpu1);
  cudaEventCreate(&start_gpu2);
  cudaEventCreate(&stop_gpu2);
  cudaEventCreate(&start_cpu);
  cudaEventCreate(&stop_cpu);
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

  cudaEventRecord(start_cpu);
  CpuMatrixTranspose(matrix, cpu_result);
  cudaEventRecord(stop_cpu);
  cudaEventSynchronize(stop_cpu);

  cudaEventRecord(start_gpu1);
  // 循环调用kernel 20次来统计耗时
  for (int i = 0; i < 20; i++) {
    GpuMatrixTranspose<<<grid, block>>>(matrix, gpu_result);
    // cudaDeviceSynchronize();
  }
  cudaEventRecord(stop_gpu1);
  cudaEventSynchronize(stop_gpu1);
  CheckResult(gpu_result, cpu_result, M * N);
  getLastCudaError("GpuMatrixTranspose() kernel launch error");

  cudaEventRecord(start_gpu2);
  for (int i = 0; i < 20; i++) {
    GpuMatrixTransposeShared<<<grid, block>>>(matrix, gpu_result);
    // cudaDeviceSynchronize();
  }
  cudaEventRecord(stop_gpu2);
  cudaEventSynchronize(stop_gpu2);
  CheckResult(gpu_result, cpu_result, M * N);
  getLastCudaError("GpuMatrixTransposeShared() kernel launch error");


  float time_gpu1, time_gpu2, time_cpu;
  cudaEventElapsedTime(&time_gpu1, start_gpu1, stop_gpu1);
  time_gpu1 = time_gpu1 / 20;
  cudaEventElapsedTime(&time_gpu2, start_gpu2, stop_gpu2);
  time_gpu2 = time_gpu2 / 20;
  cudaEventElapsedTime(&time_cpu, start_cpu, stop_cpu);
  std::printf("cpu time: %f ms\ngpu1 time: %f ms\ngpu2 shared version time: %f ms\n", time_cpu, time_gpu1, time_gpu2);
  return 0;
}




