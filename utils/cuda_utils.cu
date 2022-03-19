#include "cuda_utils.cuh"
#include "cufft.h"



cudaError_t checkCuda_z(cudaError_t result, char* file, int line)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n in %s:%d",
            cudaGetErrorString(result), file, line);
    assert(result == cudaSuccess);
  }
  return result;
}


cufftResult_t checkCufft_z(cufftResult_t result, char* file, int line)
{
  if (result != CUFFT_SUCCESS) {
    fprintf(stderr, "cufft Runtime Error: \n in %s:%d",
            file, line);
    assert(result == cudaSuccess);
  }
  return result;
}
