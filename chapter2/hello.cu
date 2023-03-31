#include<stdio.h>
__global__ void hello() {
 printf("GPU: HELLO\n");
}

int main() {
    printf("CPU: HELLO\n"); 
    hello<<<1, 10>>>();
    cudaDeviceReset();
    return 0;
}
