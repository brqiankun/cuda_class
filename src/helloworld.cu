#include<stdio.h>
__global__ void hello_world(void) {   //__global__ 表示是可以在device上执行的核函数
    printf("GPU: Hello world!\n");
}

int main(int argc, char* argv[]) {
    printf("CPU: Hello world!\n");
    hello_world<<<1, 10>>>();   //<<<1, 10>>> 表示对设备进行设置的参数
    cudaDeviceReset();  //if no this line, it can not output hello world for gpu
    return 0;
}