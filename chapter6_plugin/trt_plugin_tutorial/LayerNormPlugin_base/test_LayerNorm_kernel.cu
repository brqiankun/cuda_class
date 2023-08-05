#include "LayerNormPlugin.h"

#define N 512
__global__ void layerNormKernel(const float *pInput, float *pOutput);

int main() {
  float* input_d;
  float* output_d;
  float* input_h;
  float* output_h;
  input_h = reinterpret_cast<float*>(malloc(N * sizeof(float)));
  output_h = reinterpret_cast<float*>(malloc(N * sizeof(float)));
  for (int i = 0; i < N; i++) {
    input_h[i] = i;
  }
  cudaMalloc(reinterpret_cast<void**>(&input_d), N * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&output_d), N * sizeof(float));
  dim3 block(128);
  dim3 grid(N / block.x);
  cudaMemcpy(input_d, input_h, N * sizeof(float), cudaMemcpyHostToDevice);
  layerNormKernel<<<grid, block>>>(input_d, output_d);
  cudaMemcpy(output_h, output_d, N * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < N; i++) {
    printf("output_h[%d]: %f\n", i, output_h[i]);
  }
  free(input_h);
  free(output_h);
  cudaFree(input_d);
  cudaFree(output_d);
  return 0;
}