#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <functional>

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


#define N 100000000
#define BLOCK_SIZE 256
#define GRID_SIZE 32
#define topk 20

__managed__ int source[N];
__managed__ int gpu_result[topk];
int cpu_result[topk] = {0};
__managed__ int _1_pass_result[topk * GRID_SIZE];


__device__ __host__ void InsertValue(int* array, int k, int data) {
  for (int i = 0; i < k; i++) {
    if (array[i] == data) {   // 去重，如果待插入队列的data和队列中已有元素重复，则直接返回
      return;
    }
  }
  if (data < array[k - 1]) {  // 如果当前元素比队列中最后的元素小，则不需要插入，直接返回
    return;
  }
  for (int i = k - 2; i >= 0; i--) {
    if (data > array[i]) {  
      // 如果当前元素比倒数第二个元素开始的元素大，则当前元素可以被前一个元素覆盖
      // 类似于冒泡排序的逆过程，为当前较大的元素空出一个位置
      array[i + 1] = array[i];
    } else {  // data < array[i]
      if (array[i + 1] < data) {
        array[i + 1] = data;
      }
      return;
    }
  }
  // 退出循环未返回，表示当前元素可以放在队首
  array[0] = data;
}

__global__ void GpuTopK(int* input, int* output, int length, int k) {
  __shared__ int buffer[BLOCK_SIZE * topk];
  int top_array[topk];

  for (int i = 0; i < topk; i++) {
    top_array[i] = INT_MIN;
  }

  // grid_loop
  // 每个thread维护一个局部topk序列
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < length; idx += gridDim.x * blockDim.x) {
    InsertValue(top_array, topk, input[idx]);
  }
  // 一个block的线程把各自的topk序列放到shared mem中
  for (int i = 0; i < topk; i++) {
    buffer[threadIdx.x * topk + i] = top_array[i];
  }
  __syncthreads();

  for (int total_threads = BLOCK_SIZE / 2; total_threads >= 1; total_threads /= 2) {
    if (threadIdx.x < total_threads) {
      // 将线程threadIdx.x + total_threads 对应的topk序列也插入到threadIdx.x对应的topk序列中
      for (int i = 0; i < topk; i++) {
        InsertValue(top_array, topk, buffer[topk * (threadIdx.x + total_threads) + i]);
      }
    }
    __syncthreads();
    // 将经过两个线程所属的topk序列合并后，将结果放回threadIdx.x对应topk的shared mem中
    if (threadIdx.x < total_threads) {
      for (int i = 0; i < topk; i++) {
        buffer[topk * threadIdx.x + i] = top_array[i];
      }
    }
    __syncthreads();
  }
  // 在一个block所属的所有线程的topk序列放入到线程threadIdx.x == 0所属的topk中时
  // 将一个block的线程得到的topk序列放入全局内存中
  if (blockIdx.x * blockDim.x < length) {
    if (threadIdx.x == 0) {
      for (int i = 0; i < topk; i++) {
        output[topk * blockIdx.x + i] = buffer[i]; 
      }
    }
  }

}

void CpuTopK(int* input, int* output, int length, int k) {
  for (int i = 0; i < length; i++) {
    InsertValue(output, k ,input[i]);
  }
}


int main() {
  for (int i = 0; i < N; i++) {
    source[i] = std::rand();
  }
  cudaEvent_t cpu_start, cpu_stop, gpu_start, gpu_stop;
  cudaEventCreate(&cpu_start);
  cudaEventCreate(&cpu_stop);
  cudaEventCreate(&gpu_start);
  cudaEventCreate(&gpu_stop);

  cudaEventRecord(gpu_start);
  cudaEventSynchronize(gpu_start);
  for (int i = 0; i < 20; i++) {
    GpuTopK<<<GRID_SIZE, BLOCK_SIZE>>>(source, _1_pass_result, N, topk);
    GpuTopK<<<1, BLOCK_SIZE>>>(_1_pass_result, gpu_result, topk * GRID_SIZE, topk);
    cudaDeviceSynchronize();
  }
  getLastCudaError("kernel GpuTopK error\n");
  cudaEventRecord(gpu_stop);
  cudaEventSynchronize(gpu_stop);

  cudaEventRecord(cpu_start);
  cudaEventSynchronize(cpu_start);
  CpuTopK(source, cpu_result, N, topk);
  cudaEventRecord(cpu_stop);
  cudaEventSynchronize(cpu_stop);
  
  float gpu_time, cpu_time;
  cudaEventElapsedTime(&gpu_time, gpu_start, gpu_stop);
  gpu_time /= 20;
  cudaEventElapsedTime(&cpu_time, cpu_start, cpu_stop);

  bool flag = true;
  for (int i = 0; i < topk; i++) {
    std::printf("cpu_result[%d]: %d\ngpu_result[%d]: %d\n\n", i, cpu_result[i], i, gpu_result[i]);
    if (gpu_result[i] != cpu_result[i]) {
      flag = false;
      // break;
    }
  }
  if (flag == false) {
    std::printf("result error\n");
  } else {
    std::printf("result pass\n");
  }
  printf("gpu time: %f\ncpu time: %f\n", gpu_time, cpu_time);

  return 0;
}