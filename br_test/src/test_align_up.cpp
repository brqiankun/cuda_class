#include <stdio.h>

#define MEMORY_ALIGNMENT 4096
#define ALIGN_UP(x, size) (((size_t)x + (size - 1)) & (~(size - 1)))

int main() {
    int *a_UA = (int*)malloc(sizeof(int));
    int *a = (int*)ALIGN_UP(a_UA, MEMORY_ALIGNMENT);
}