#include<cuda_runtime.h>

//matrix multiplication

//matrices are stored in row-major order
//M(row, col) = *(M.elements + row * M.stride + col)  in this case stride==width
struct Matrix{
    int width;
    int height;
    int stride;
    float* elements;
};
//thread block size 
#define BLOCK_SIZE 16

//get a matrix element
__device__ float GetElement(const Matrix A, int row, int col) {
    return A.elements[row * A.stride + col];   //row * A.stride  stride == width
}

//set a matrix element
__device__ void SetElement(const Matrix A, int row, int col, float value) {
    A.elements[row * A.width + col] = value;
}

//get the BLOCK_SIZE * BLOCK_SIZE sub-matrix Asub of A that is 
//located col sub-matrices to the right and row sub-matrices down
//from the upper-left corner of A
//block row and col
__device__ Matrix GetSubMatrix(Matrix A, int row, int col) {
    Matrix Asub;
    Asub.width = BLOCK_SIZE;
    Asub.height = BLOCK_SIZE;
    Asub.stride = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                         + BLOCK_SIZE * col];
    return Asub;
}


//forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix A, const Matrix B, Matrix C);

void MatMul(const Matrix A, const Matrix B, Matrix C) {
    //load A and B to device memory
    Matrix d_A;
    d_A.width = d_A.stride = A.width;
    d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
    
    Matrix d_B;
    d_B.width = d_B.stride = B.width;
    d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

    //allocate C in device memory
    Matrix d_C;
    d_C.width = d_C.stride = C.width;
    d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    //invoke kernel  线程维度和目标矩阵的维度相同[A.height, B.width]
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
    //Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    //each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
    //Each thread computes one element of Csub
    //by accumulating results into Cvalue
    float Cvalue = 0;
    //Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;
    //loop over all the sub-matrices of A and B that are required to compute Csub
    //multiply each pair of sub-matrices together and accumulate the results
    for(int m = 0; m < (A.width/BLOCK_SIZE); m++) {
        //get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);
        //get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);
        //Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        //Load Asub and Bsub from device memory to shared memory
        //each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);
        //synchronize to make sure the sub-matrices are loaded before starting computation
        __syncthreads();
        //multiply Asub and Bsub together
        for(int e = 0; e < BLOCK_SIZE; ++e) {
            Cvalue += As[row][e] * Bs[e][col];
        }
        //synchronize to make sure that the preceding computation is done 
        //before loading two new sub-matrices of A and B in the next iteration
        __syncthreads();

    }
    //write to device memory
    //each thread writes one element
    SetElement(Csub, row, col, Cvalue);    

}

