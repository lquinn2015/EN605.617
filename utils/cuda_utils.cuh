#pragma strict

#include <stdio.h>
#include <assert.h>
#include "cufft.h"

typedef unsigned int uint32_t;
typedef unsigned char uint8_t;

#define checkCuda(ans) {checkCuda_z(ans, __FILE__, __LINE__);}
cudaError_t checkCuda_z(cudaError_t result, const char* file, int line);

#define checkCufft(ans) {checkCufft_z(ans, __FILE__, __LINE__);}
cufftResult_t checkCufft_z(cufftResult_t result, const char* file, int line);

#define checkCudaKernel(x) do { \
        (x); \
        checkCuda_z( cudaPeekAtLastError(), __FILE__,__LINE__);\
    } while(0)


