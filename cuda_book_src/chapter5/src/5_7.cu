#include<stdio.h>
#include<cuda_runtime.h>


__global__ void myhistogram256Kernel1_01(const unsigned char const* d_hist_data, 
                                         unsigned int* const d_bin_data) {
    //work out our thread id
    const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

    const unsigned int tid = idx + idy * blockDim.x * gridDim.x;

    const unsigned char value = d_hist_data[tid];

    atomicAdd(&(d_bin_data[value]), 1);

}

//each read is 4 bytes  32 * 4 = 128 bytes
__global__ void myhistogram256Kernel_02(const unsigned int const* d_hist_data,
                                        unsigned int* const d_bin_data) {
    const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

    const unsigned int tid = idx + idy * blockDim.x * gridDim.x;  

    //fetch the data value as 32 bit
    const unsigned int value_u32 = d_hist_data[tid];  //32 

    atomicAdd(&(d_bin_data[value_u32 & 0x000000FF]), 1);
    atomicAdd(&(d_bin_data[value_u32 & 0x0000FF00 >> 8]), 1);
    atomicAdd(&(d_bin_data[value_u32 & 0x00FF0000 >> 16]), 1);
    atomicAdd(&(d_bin_data[value_u32 & 0xFF000000 >> 24]), 1);
}

__shared__ unsigned int d_bin_data_shared[256];

__global__ void myhistogram256Kernel_03(const unsigned int const* d_hist_data, 
                                        unsigned int* const d_bin_data) {
    const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

    const unsigned int tid = idx + idy * blockDim.x * gridDim.x;

    //clear shared memory
    d_bin_data_shared[threadIdx.x] = 0;

    //fetch the data value as 32 bit
    const unsigned int value_u32 = d_hist_data[tid];

    __syncthreads();

    atomicAdd(&(d_bin_data_shared[value_u32 & 0x000000FF]), 1);
    atomicAdd(&(d_bin_data_shared[value_u32 & 0x0000FF00 >> 8]), 1);
    atomicAdd(&(d_bin_data_shared[value_u32 & 0x00FF0000 >> 16]), 1);
    atomicAdd(&(d_bin_data_shared[value_u32 & 0xFF000000 >> 24]), 1);

    atomicAdd(&(d_bin_data[threadIdx.x]), d_bin_data_shared[threadIdx.x]);

}

__global__ void myhistogram256Kernel_07(const unsigned int const* d_hist_data,
                                        unsigned int* const d_bin_data,
                                        unsigned int N) {
    const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

    const unsigned int tid = idx + idy * blockDim.x * gridDim.x;

    //clear the shared memory
    d_bin_data_shared[threadIdx.x] = 0;

    __syncthreads();    
                                    
}



int main() {
    
}



