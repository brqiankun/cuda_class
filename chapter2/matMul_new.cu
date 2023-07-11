#include <cuda_runtime.h>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <cassert>

const int M = 1024;
const int K = 1024;
const int N = 1024;
const int BLOCK_SIZE = 16;

void checkResult(float* hostRef, float* deviceRef, const int num_to_check) {
    double diff = 1.0e-6;
    for (size_t i = 0; i < num_to_check; i++) {
        if (std::abs(hostRef[i] - deviceRef[i]) > diff) {
            std::printf("result check faild\n");
            std::printf("%f(hostRef[%ld]) != %f(deviceRef[%ld]))", hostRef[i], i, deviceRef[i], i);
            return;
        }
    }
    printf("result check successfully\n");
}

void initial(float* array, const int size) {
    for (int i = 0; i < size; i++) {
        array[i] = (float)(std::rand() % 10 + 1);
    }
}

void printMatrix(float* array, int row, int col) {
    float* p = array;
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%10lf", p[j]);
        }
        p = p + col;
        printf("\n");
    }
}

void multiMatrixOnHost(float* array_A, float* array_B, float* array_C, int M_p, int K_p, int N_p) {
    for (int i = 0; i < M_p; i++) {
        for (int j = 0; j < N_p; j++) {
            float tmp_sum = 0;
            for (int k = 0; k < K_p; k++) {
                tmp_sum = tmp_sum + array_A[i * K_p + k] * array_B[k * N_p + j];
            }
            array_C[i * N_p + j] = tmp_sum;
        }
    }
}

__global__ void multiMatrixOnDevice(float* array_A, float* array_B, float* array_C, int M_p, int K_p, int N_p) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M_p && col < N_p) {
        float tmp_sum = 0;
        for (int k = 0; k < K_p; k++) {
            tmp_sum = tmp_sum + array_A[row * K_p + k] * array_B[k * N_p + col];
        }
        array_C[row * N_p + col] = tmp_sum; 
    } 
}

__global__ void multiMatricOnDeviceSharedMemory(float* array_A, float* array_B, float* array_C, int M_p, int K_p, int N_p) {
    __shared__ float array_A_sub[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float array_B_sub[BLOCK_SIZE][BLOCK_SIZE];
    assert(blockDim.x == BLOCK_SIZE);
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    float Csub = 0.0;
    for (int i = 0; i < ((K_p  + BLOCK_SIZE - 1) / BLOCK_SIZE); i++) {
        array_A_sub[threadIdx.x][threadIdx.y] = array_A[row * K_p + i * BLOCK_SIZE + threadIdx.y];
        array_B_sub[threadIdx.x][threadIdx.y] = array_B[(i * BLOCK_SIZE + threadIdx.x) * N_p + col];
        __syncthreads();   // 表示在一个线程块中的所有线程必须运行到此，之后向后执行

        for (int j = 0; j < BLOCK_SIZE; j++) {
            Csub = Csub + array_A_sub[threadIdx.x][j] * array_B_sub[j][threadIdx.y];
        }
        __syncthreads();
    }
    if (row < M_p && col < N_p) {
        array_C[row * K_p + col] = Csub;
    }
}


// __global__ void multiMatricOnDeviceSharedMemory(float* array_A, float* array_B, float* array_C, int M_p, int K_p, int N_p) {
//     __shared__ float array_A_sub[BLOCK_SIZE][BLOCK_SIZE];
//     __shared__ float array_B_sub[BLOCK_SIZE][BLOCK_SIZE];
//     int row = blockIdx.x * blockDim.x + threadIdx.x;
//     int col = blockIdx.y * blockDim.y + threadIdx.y;
//     for 
// }



int main() {
    int Axy = M * K;
    int Bxy = K * N;
    int Cxy = M * N;

    float *h_A, *h_B, *hostRef, *deviceRef;   // Ref表示host内存中存储的CPU计算结果和GPU计算结果
    h_A = (float*)std::malloc(Axy * sizeof(float));
    h_B = (float*)std::malloc(Bxy * sizeof(float));

    hostRef = (float*)malloc(Cxy * sizeof(float));
    deviceRef = (float*)malloc(Cxy * sizeof(float));
    std::memset(hostRef, 0, Cxy * sizeof(float));
    std::memset(deviceRef, 0, Cxy * sizeof(float));
    checkResult(hostRef, deviceRef, Cxy);

    initial(h_A, Axy);
    initial(h_B, Bxy);

    // CPU版本
    auto begin = std::chrono::high_resolution_clock::now();
    multiMatrixOnHost(h_A, h_B, hostRef, M, K, N);
    auto end = std::chrono::high_resolution_clock::now();
    auto running_time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
    std::printf("multiMarixOnHost running time : %ld us\n", running_time.count());

    // GPU 版本
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, Axy * sizeof(float));
    cudaMalloc(&d_B, Bxy * sizeof(float));
    cudaMalloc(&d_C, Cxy * sizeof(float));

    cudaMemcpy(d_A, h_A, Axy * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, Bxy * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dim_grid((M + dim_block.x - 1) / dim_block.x, (N + dim_block.y - 1) / dim_block.y);   // 上取整 共需矩阵C的元素个数的线程，每个线程计算一个元素
    begin = std::chrono::high_resolution_clock::now();
    multiMatrixOnDevice<<<dim_grid, dim_block>>>(d_A, d_B, d_C, M, K, N);
    cudaMemcpy(deviceRef, d_C, Cxy * sizeof(float), cudaMemcpyDeviceToHost);
    end = std::chrono::high_resolution_clock::now();
    running_time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
    std::printf("multiMatrixOnDevice running time: %ld us\n", running_time.count());
    checkResult(hostRef, deviceRef, Cxy);

    begin = std::chrono::high_resolution_clock::now();
    multiMatricOnDeviceSharedMemory<<<dim_grid, dim_block>>>(d_A, d_B, d_C, M, K, N);
    cudaMemcpy(deviceRef, d_C, Cxy * sizeof(float), cudaMemcpyDeviceToHost);
    end = std::chrono::high_resolution_clock::now();
    running_time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
    std::printf("multiMatricOnDeviceSharedMemory running time: %ld us\n", running_time.count());
    checkResult(hostRef, deviceRef, Cxy);


    return 0;

}