#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <assert.h>

// #include "dbg.h"

const int M = 1024;
const int K = 2048;
const int N = 1024;

const int BLOCK_SIZE = 16;

void initial(float* array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = (float)(rand() % 10 + 1);
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

// array_A [M_p, K_p]   array_B [K_p, N_p]
void multiMatrixOnHost(float* array_A, float* array_B, float* array_C, int M_p, int K_p, int N_p) {
    for (int i = 0; i < M_p; i++) {
        for (int j = 0; j < N_p; j ++) {  // 计算array_C 的第[i, j]元素
            float sum_tmp = 0;
            for (int k = 0; k < K_p; k++) {
                sum_tmp += array_A[i * K_p + k] * array_B[k * N_p + j];
            }
            array_C[i * N_p + j] = sum_tmp;
        }
    }
}

// 不使用共享内存的版本
__global__ void multiMatrixOnDevice(float* array_A, float* array_B, float* array_C, int M_p, int K_p, int N_p) {
    // 每个线程计算矩阵C中的一个元素
    int row = threadIdx.x + blockDim.x * blockIdx.x;  // row 行数
    int col = threadIdx.y + blockDim.y * blockIdx.y;  // col 列数
    // printf("col:%d, row:%d\n", col, row);
    // printf("N_p:%d, M_p:%d\n", N_p, M_p);
    if (col < N_p && row < M_p) {
        float tmp = 0;
        for (int k = 0; k < K_p; k++) {
            tmp += array_A[row * K_p + k] * array_B[k * N_p + col];
            // 计算一次乘加需要读array_A/B各一次，读取global memory需要几百个cycle，而计算只需几个cycle
        }
        array_C[row * N_p + col] = tmp;
        // if (col == 2)
        //     printf("col:%d, row:%d, val:%f\n", col, row, tmp);
    }
}

__global__ void matrixMultiShared1(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float subA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float subB[BLOCK_SIZE][BLOCK_SIZE];
    assert(blockDim.x == BLOCK_SIZE);

    int col = blockIdx.y * blockDim.y + threadIdx.y;  // 列号
    int row = blockIdx.x * blockDim.x + threadIdx.x;  // 行号

    float Csub = 0.0;
    // 通过for循环依次把numAcols/BLOCK_SIZE个子矩阵放入共享内存的subA, subB
    // 共享内存on-chip中的读约几十个cycle
    // 降低计算访问内存比
    // Loop over all the sub-matrices of A and B that are required to compute Csub
    // Multiply each pair of sub-matrices together and accumulate the results
    for (int i = 0; i < (K / BLOCK_SIZE); i++) {
        subA[threadIdx.x][threadIdx.y] = A[row * K + i * BLOCK_SIZE + threadIdx.y];
        subB[threadIdx.x][threadIdx.y] = B[(i * BLOCK_SIZE + threadIdx.x) * N + col];

        // synchronize to make sure the sub-matrieces are loaded before starting the computation
        __syncthreads();

        //计算每个元素
        for (int j = 0; j < BLOCK_SIZE; j++) {
            Csub = Csub + subA[threadIdx.x][j] * subB[j][threadIdx.y];
        }
        // synchronize to make sure that the preceding computation is done before starting the computation
        __syncthreads();
    }

    if (row < M && col < N) {
        C[N * row + col] = Csub;
    }
    
}

__global__ void matrixMultiShared2(float* A_p, float* B_p, float* C_p, int M, int N, int K) {
    __shared__ float subA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float subB[BLOCK_SIZE][BLOCK_SIZE];
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    float sum_tmp = 0;
    for (int i = 0; i < (K / BLOCK_SIZE); i++) {
        // 计算每个block前先将数据从global memory读入shared memory
        subA[threadIdx.x][threadIdx.y] = A_p[row * K + (i * BLOCK_SIZE + threadIdx.y)];
        subB[threadIdx.x][threadIdx.y] = B_p[(i * BLOCK_SIZE + threadIdx.x) * N + col];
        __syncthreads();
        
        for (int j = 0; j < BLOCK_SIZE; j++) {
            sum_tmp = sum_tmp + subA[threadIdx.x][j] * subB[j][threadIdx.y];
        }
        __syncthreads();
    }
    if (row < M && col < K) {
        C_p[row * N + col] = sum_tmp;
    }
}

void checkResult(float* hostRef, float* deviceRef, const int num_to_check) {
    double diff = 1.0E-6;
    for (size_t i = 0; i < num_to_check; i++) {
        if (abs(hostRef[i] - deviceRef[i]) > diff) {
            printf("result check faild\n");
            printf("%f(hostRef[%ld] != %f(deviceRef[%ld]))", hostRef[i], i, deviceRef[i], i);
            return;
        }
    }
    printf("result check successfully\n");
}



int main(int argc, char* argv[]) {
    clock_t start = 0, finish = 0;
    float time;
    int Axy = M * K;
    int Bxy = K * N;
    int Cxy = M * N;

    float *h_A, *h_B, *hostRef, *deviceRef;
    h_A = (float*)malloc(Axy * sizeof(float));
    h_B = (float*)malloc(Bxy * sizeof(float)); 

    hostRef = (float*)malloc(Cxy * sizeof(float));
    deviceRef = (float*)malloc(Cxy * sizeof(float));
    memset(hostRef, 0, Cxy * sizeof(float));
    memset(deviceRef, 0, Cxy * sizeof(float));
    checkResult(hostRef, deviceRef, Cxy);

    initial(h_A, Axy);
    //printMatrix(h_A, M, K);
    initial(h_B, Bxy);
    //printMatrix(h_B, K, N);

    start = clock();
    multiMatrixOnHost(h_A, h_B, hostRef, M, K, N);
    finish = clock();
    time = (float)(finish - start) / CLOCKS_PER_SEC;
    
    printf("\n");
	printf("------------------------------------------------------------------------------------\n");
	printf("Computing matrix product using multiplicateMatrixOnHost \n");
	printf("Matrix_hostRef: (%d x %d)  CPU运行时间为: %lfs\n", M, N, time);

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, Axy * sizeof(float));
    cudaMalloc((void**)&d_B, Bxy * sizeof(float));
    cudaMalloc((void**)&d_C, Cxy * sizeof(float));

    cudaMemcpy(d_A, h_A, Axy * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, Bxy * sizeof(float), cudaMemcpyHostToDevice);

    printf("\n\n");
	printf("------------------------------------------------------------------------------------\n");
	printf("Computing matrix product using multiplicateMatrixOnDevice \n");
    

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (M + dimBlock.y - 1) / dimBlock.y);
    printf("block: (%d  %d  %d)\n", dimBlock.x, dimBlock.y, dimBlock.z);
    printf("grid:  (%d  %d  %d)\n", dimGrid.x, dimGrid.y, dimGrid.z);

    cudaEvent_t gpustart, gpustop;
    // 未使用shared memory版本
    
    float elapsedTime = 0.0;
    cudaEventCreate(&gpustart);
    cudaEventCreate(&gpustop);
    cudaEventRecord(gpustart, 0);
    multiMatrixOnDevice<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, K, N);
    // cudaDeviceSynchronize();
    cudaEventRecord(gpustop, 0);
    cudaEventSynchronize(gpustop);
    cudaEventElapsedTime(&elapsedTime, gpustart, gpustop);
    cudaEventDestroy(gpustart);
    cudaEventDestroy(gpustop);

    cudaMemcpy(deviceRef, d_C, Cxy * sizeof(float), cudaMemcpyDeviceToHost);
    //printMatrix(deviceRef, M, N);
    checkResult(hostRef, deviceRef, Cxy);
    printf("Matrix_deviceRef: (%d×%d)  <<<(%d,%d),(%d,%d)>>>  GPU运行时间为：%fs\n",
                M, N, dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y, elapsedTime / 1000);

    // shared memory 版本
    printf("\n");
	printf("------------------------------------------------------------------------------------\n");
	printf("Computing matrix product using matrixMultiShared1 \n");
    
    elapsedTime = 0.0;
    cudaEventCreate(&gpustart);
    cudaEventCreate(&gpustop);
    cudaEventRecord(gpustart, 0);
    matrixMultiShared1<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
    // cudaDeviceSynchronize();
    cudaEventRecord(gpustop, 0);
    cudaEventSynchronize(gpustop);

    cudaEventElapsedTime(&elapsedTime, gpustart, gpustop);
    cudaEventDestroy(gpustart);
    cudaEventDestroy(gpustop);

    cudaMemcpy(deviceRef, d_C, Cxy * sizeof(float), cudaMemcpyDeviceToHost);
    checkResult(hostRef, deviceRef, Cxy);
    printf("Matrix_deviceRef: (%d×%d)  <<<(%d,%d),(%d,%d)>>>  GPU运行时间为：%fs\n",
		M, N, dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y, elapsedTime / 1000);


    printf("\n");
	printf("------------------------------------------------------------------------------------\n");
	printf("Computing matrix product using matrixMultiShared2 \n");
    
    elapsedTime = 0.0;
    cudaEventCreate(&gpustart);
    cudaEventCreate(&gpustop);
    cudaEventRecord(gpustart, 0);
    matrixMultiShared2<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
    // cudaDeviceSynchronize();
    cudaEventRecord(gpustop, 0);
    cudaEventSynchronize(gpustop);

    cudaEventElapsedTime(&elapsedTime, gpustart, gpustop);
    cudaEventDestroy(gpustart);
    cudaEventDestroy(gpustop);

    cudaMemcpy(deviceRef, d_C, Cxy * sizeof(float), cudaMemcpyDeviceToHost);
    checkResult(hostRef, deviceRef, Cxy);
    printf("Matrix_deviceRef: (%d×%d)  <<<(%d,%d),(%d,%d)>>>  GPU运行时间为：%fs\n",
		M, N, dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y, elapsedTime / 1000);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    free(h_A);
    free(h_B);
    free(hostRef);
    free(deviceRef);

    cudaDeviceReset();

}
