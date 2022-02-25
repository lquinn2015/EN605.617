#pragma strict

#include <stdio.h>
#include <assert.h>

typedef unsigned int uint32_t;
typedef unsigned char uint8_t;

cudaError_t checkCuda(cudaError_t result);

#define checkCudaKernel(x) do { \
        (x); \
        checkCuda( cudaPeekAtLastError() );\
    } while(0)


