#include<cuda_runtime.h>

//matrix multiplication

//matrices are stored in row-major order
//M(row, col) = *(M.elements + row * M.width + col)
struct Matrix{
    int width;
    int height;
    float* elements;
};

//thread block size 
#define BLOCK_SIZE 16

//forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix A, const Matrix B, Matrix C);

void MatMul(const Matrix A, const Matrix B, Matrix C) {
    //load A and B to device memory
    Matrix d_A;
    d_A.width = A.width;
    d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
    
    Matrix d_B;
    d_B.width = B.width;
    d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

    //allocate C in device memory
    Matrix d_C;
    d_C.width = C.width;
    d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    //invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    //read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);

    //free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);

}

//Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
    //Each thread computes one element of C
    //by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for(int e = 0; e < A.width; e++) {
        Cvalue += A.elements[row * A.width + e] * B.elements[col + e * B.width];
    }
    C.elements[row * C.width + col] = Cvalue;
}

