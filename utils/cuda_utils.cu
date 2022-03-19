#include "cuda_utils.cuh"
#include "cufft.h"


cudaError_t checkCuda_z(cudaError_t result, char const* file, int line)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n in %s:%d",
            cudaGetErrorString(result), file, line);
    assert(result == cudaSuccess);
  }
    #ifdef DEBUG  
  fprintf(stdout,"cuda result %d\n", result);
    #endif
  return result;
}

cufftResult_t checkCufft_z(cufftResult_t result, char const* file, int line)
{
  if (result != CUFFT_SUCCESS) {
    fprintf(stderr, "cufft Runtime Error: \n in %s:%d",
            file, line);
    assert(result == CUFFT_SUCCESS);
  }
    #ifdef DEBUG  
  fprintf(stdout,"cuFFT result %d\n", result);
    #endif
  return result;
}
