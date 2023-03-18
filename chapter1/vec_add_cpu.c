#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>

#include "vec_add.h"


int main(int argc, char* argv[]) {
    int num_to_compute = atoi(argv[1]);
    printf("num to compute %d\n", num_to_compute);
    float *a_h, *b_h, *c_h;
    /***
     * memory alloc
    */
    int size = num_to_compute * sizeof(float);
    a_h = (float*)malloc(size);
    b_h = (float*)malloc(size);
    c_h = (float*)malloc(size);

    init_data(a_h, num_to_compute);
    init_data(b_h, num_to_compute);
    memset(c_h, 0, num_to_compute);
    struct timeval start, end;
    gettimeofday(&start, NULL);
    vec_add(a_h, b_h, c_h, num_to_compute);
    gettimeofday(&end, NULL);
    long timeuse_u = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    printf("func time elapse %f s\n", timeuse_u / 1000000.0);

}

void vec_add(const float* a_h, const float* b_h, float* c_h, const int num_to_compute) {
    for (int i = 0; i < num_to_compute; i++) {
        c_h[i] = a_h[i] + b_h[i];
    }
}

void init_data(float* p, int num) {
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < num; i++) {
        p[i] = (float)(rand()&0xffff) / 1000.0f;
    }
}