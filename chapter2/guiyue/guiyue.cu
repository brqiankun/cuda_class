// s[N]: s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]
// step1: s[0] + s[4] -> s[0]; s[1] + s[5] -> s[1]; s[2] + s[6] -> s[2]; s[3] + s[7] -> s[3];
// step2: s[0] + s[2] -> s[0]; s[1] + s[3] -> s[1];
// step3: s[0] + s[1] -> s[0]

#include <iostream>
#include <cmath>
#include <cstdio>

const long long N = 100000000;
const int BLOCK_SIZE = 256;
const int GRID_SIZE = 32;

__managed__ int source[N];
__managed__ int gpu_result[1] = {0};
int cpu_result[1] = {0};

__global__ void SumGpu(int* input, int* output, long long count) {
  __shared__ int buffer[BLOCK_SIZE];
  // grid_loop 当线程数量远小于要处理的数据时也能处理
  // 每个线程处理多个数据  thread ID: blockDim.x * blockIdx.x + threadIdx.x + step * blockDim.x * gridDim.x   step = 0, 1, 2, 3, ...
  // thread 0: source[0, 8, 16, 24] sum -> shared memory
  // thread从全局整个grid的坐标开始，步长step为所有线程的数目
  int shared_tmp = 0;
  for (int idx = blockDim.x * blockIdx.x + threadIdx.x; idx < count; idx += gridDim.x * blockDim.x) {
    shared_tmp = shared_tmp + input[idx];
  }
  buffer[threadIdx.x] = shared_tmp;
  __syncthreads();
  int tmp = 0;
  for (int total_threads = BLOCK_SIZE / 2; total_threads >= 1; total_threads /= 2) {   // 规约计算，假设当前元素为n，则需要n / 2的线程来计算
    if (threadIdx.x < total_threads) {
      tmp = buffer[threadIdx.x] + buffer[threadIdx.x + total_threads];
    }
    __syncthreads();  // 产生分支的代码块一般不添加同步，因此不添加到上方的if块中
    if (threadIdx.x < total_threads) {
      buffer[threadIdx.x] = tmp;
    }
    __syncthreads();
  }
  // block_sum 放在 shared memory buffer[0]中
  // 将每个block的buffer[0]的值再累加到全局device memory中，可能存在写冲突，需要原子操作
  if (blockIdx.x * blockDim.x < count) {
    if (threadIdx.x == 0) {
      // output[0] = output[0] + buffer[0];  存在写冲突
      atomicAdd(output, buffer[0]);
    }
  }
}

int main() {
  std::printf("init input\n");
  for (int i = 0; i < N; i++) {
    source[i] = std::rand() % 10;
  }
  cudaEvent_t start_cpu, stop_cpu, start_gpu, stop_gpu;
  cudaEventCreate(&start_gpu);
  cudaEventCreate(&start_cpu);
  cudaEventCreate(&stop_cpu);
  cudaEventCreate(&stop_gpu);

  cudaEventRecord(start_gpu);
  cudaEventSynchronize(start_gpu);
  for (int i = 0; i < 20; i++) {
    gpu_result[0] = 0;
    SumGpu<<<GRID_SIZE, BLOCK_SIZE>>>(source, gpu_result, N);
    cudaDeviceSynchronize();
  }
  cudaEventRecord(stop_gpu);
  cudaEventSynchronize(stop_gpu);

  cpu_result[0] = 0;
  cudaEventRecord(start_cpu);
  for (int i = 0; i < N; i++) {
    cpu_result[0] = cpu_result[0] + source[i];
  }
  cudaEventRecord(stop_cpu);
  cudaEventSynchronize(stop_cpu);

  float cpu_time, gpu_time;
  cudaEventElapsedTime(&cpu_time, start_cpu, stop_cpu);
  cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);
  gpu_time = gpu_time / 20;
  std::printf("cpu_result: %d; cpu_time %.4f ms\n", cpu_result[0], cpu_time);
  std::printf("gpu_result: %d; gpu_time: %.4f ms\n", gpu_result[0], gpu_time);
  return 0;
}





