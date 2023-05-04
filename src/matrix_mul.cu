//计算两个维度的矩阵A和B的乘积C
//每个线程块负责计算一个正方形Csub的子矩阵
//在块内的每个线程计算Csub的一个元素
#include<cuda_runtime.h>
#include<stdio.h>
#include "freshman.h"
#define BLOCK_SIZE 16

__global__ void Muld(float* Ad, float* Bd, int hA, int wA, int wB, float* Cd);
//Host multiplication function
//Compute C = A * B 
//hA is the height of A
//wA is the width of A
//wB is the width of B  hB == wA  完成host内存和device内存的交互
void Mul(const float* A, const float* B, int hA, int wA, int wB, float* C) {
    int size;

    //Load A and B to the device
    float* Ad;
    size = hA*wA*sizeof(float);
    cudaMalloc((void**)&Ad, size);
    cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice);
    float* Bd;
    size = wA*wB*sizeof(float);
    cudaMalloc((void**)&Bd, size);
    cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice);
    
    //allocate C on the device
    float* Cd;
    size = hA*wB*sizeof(float);
    cudaMalloc((void**)&Cd, size);

    //compute the execution configuration
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(wB / dimBlock.x, hA / dimBlock.y);    //需要的线程数size和结果C的size相同

    //launch the device computation
    Muld<<<dimGrid, dimBlock>>>(Ad, Bd, hA, wA, wB, Cd);

    //read C from the device
    cudaMemcpy(C, Cd, size, cudaMemcpyDeviceToHost);

    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);

}

__global__ void Muld(float* Ad, float* Bd, int hA, int wA, int wB, float* Cd) {
    //block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    //thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    //index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    //index of the last sub-matrix of A processed by the block
    int aEnd = aBegin + wA - 1;

    //Step size used to iterate through the sub-matrices of A  ????
    int aStep = BLOCK_SIZE;

    //index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    //step size used to iterate through the sub-matrices of B   ????
    int bStep = BLOCK_SIZE * wB;

    //the element of the block sub-matrix that is computed by the thread
    float Csub = 0;

    //loop over all the sub-matrices of A and B required to compute the block sub-matrix
    for(int a = aBegin, b = bBegin; a < aEnd; a += aStep, b += bStep) {
        //shared memory for the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        printf("As: %p\n", As);

        //shared memory for the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        printf("Bs: %p\n", Bs);

        //load the matrices from global memory to shared memory
        //each thread loads one element of each matrix
        printf("a: %d, b: %d \n", a, b);
        As[ty][tx] = Ad[a + wA * ty + tx];
        Bs[ty][tx] = Bd[b + wB * ty + tx];

        //make sure the matrices are loaded
        __syncthreads();

        //multiply the matrices together
        //each thread computes one element of the block sub-matrix
        for(int k = 0; k<BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx]; 
        }
        
        //Synchronize to make sure that the preceding
        //computation is done before loading two new
        //sub-matrices of A and B in next iteration
        __syncthreads();

    }

    //write the block sub-matrix to global memory
    //each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    Cd[c + wB * ty + tx] = Csub;


}

int main(int argc, char* argv[]) {
    float* A_host, *B_host, *C_host;
    int hA = BLOCK_SIZE, wA = BLOCK_SIZE, wB = BLOCK_SIZE;
    if(argc == 4 ) {
        hA = atoi(argv[1]);
        wA = atoi(argv[2]);
        wB = atoi(argv[3]);
    }
    //const float* A, const float* B, int hA, int wA, int wB, float* C

    int size = hA * wA * sizeof(float);    
    A_host = (float*)malloc(size);
    printf("A_host size : %d\n", size);
    initialData(A_host, size/sizeof(float));

    size = wA * wB * sizeof(float);
    B_host = (float*)malloc(size);
    printf("B_host size : %d\n", size);
    initialData(B_host, size/sizeof(float));

    size = hA * wB * sizeof(float);
    C_host = (float*)malloc(size);
    printf("C_host size : %d\n", size);
    initialData(C_host, size/sizeof(float));

    Mul(A_host, B_host, hA, wA, wB, C_host);

    free(A_host);
    free(B_host);
    free(C_host);

    return 0;

}
