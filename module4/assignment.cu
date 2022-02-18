//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>


#define checkCudaKernel(x) do { \
        (x); \
        checkCuda( cudaPeekAtLastError() ); \
    } while(0)

// Error checker
inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", 
            cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

// previous kernels

// kernel for intializing data quickly
__global__ void gen_data(int* a) {
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    a[tid] = tid;
}

// kernel for adding two numbers requires a thread for each unit of work
__global__ void gpu_add(int* a, int* b, int*c)
{
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    c[tid] = a[tid] + b[tid];
}

// kernel for subtracting two numbers requires a thread for each unit of work
__global__ void gpu_sub(int* a, int* b, int*c)
{
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    c[tid] = a[tid] - b[tid];
}

// kernel for multiplying two numbers requires a thread for each unit of work
__global__ void gpu_mult(int* a, int* b, int*c)
{
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    c[tid] = a[tid] * b[tid];
}

// kernel for moding two numbers requires a thread for each unit of work
__global__ void gpu_mod(int* a, int* b, int*c)
{
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    c[tid] = a[tid] % b[tid];
}




//////////////////////Caesar shift Section  start///////////////////////////////

__global__ void InitAlpha(int N, char* a) {
    
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    a[tid] = 'a' + (tid % 26);
}

// We do a CaesarShift Cipher but an alphine cipher would be way cooler
__global__ void CaesarShift(int N, char* str, int c){
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= N) return;
    char v = str[tid];
    v = (((v - 'a') + c ) % 26) + 'a';
    str[tid] = v;
}

// wrapper for printing results
// potential think about pulling the cuda memcpy into this for compactness
void print_result(int* arr, int N, char opt){

    int test_idx = rand() % N;
    printf("%c operation result c[%d]=%d\n", opt, test_idx, arr[test_idx]);
}


void RunGpuAdd(int N, int numBlocks, int blockSize, int* d_a, int* d_b,
               int* d_c, int* h_c){

    //printf("numBlocks: %d, blockSize %d, N: %d\n", numBlocks, blockSize, N);
    checkCudaKernel( (gpu_add<<<numBlocks, blockSize>>>(d_a, d_b, d_c)) );
    checkCuda( cudaMemcpy(h_c, d_c, N*sizeof(int), cudaMemcpyDeviceToHost) );
    print_result(h_c, N, '+');

}
void RunGpuSub(int N, int numBlocks, int blockSize, int* d_a, int* d_b, 
               int* d_c, int* h_c){
    checkCudaKernel( (gpu_sub<<<numBlocks, blockSize>>>(d_a, d_b, d_c)) );
    checkCuda( cudaMemcpy(h_c, d_c, N*sizeof(int), cudaMemcpyDeviceToHost) );
    print_result(h_c, N, '-');
}
void RunGpuMult(int N, int numBlocks, int blockSize, int* d_a, int* d_b, 
                int* d_c, int* h_c){
    checkCudaKernel( (gpu_mult<<<numBlocks, blockSize>>>(d_a, d_b, d_c)) );
    checkCuda( cudaMemcpy(h_c, d_c, N*sizeof(int), cudaMemcpyDeviceToHost) );
    print_result(h_c, N, '*');
}
void RunGpuMod(int N, int numBlocks, int blockSize, int* d_a, int* d_b, 
               int* d_c, int* h_c){
    checkCudaKernel( (gpu_mod<<<numBlocks, blockSize>>>(d_a, d_b, d_c)) );
    checkCuda( cudaMemcpy(h_c, d_c, N*sizeof(int), cudaMemcpyDeviceToHost) );
    print_result(h_c, N, '%');
}

void PrintCaesarStream(char* h_caesar, int testIdx){
    printf("Caesar result at idx %d is : ", testIdx);
    for(int i = 0; i < 26; i++){
        printf("%c", h_caesar[testIdx+i]);
    }
    printf("\n");
}

void PinnedMem(int N, int numBlocks, int blockSize, int shift, 
        cudaEvent_t* start, cudaEvent_t* stop ){
    
    // Setup a ton of Pinned memory
    int* d_a, *d_b, *d_c, *h_c;
    char* d_caesar, *h_caesar;

    h_c = (int*) malloc(N*sizeof(int));
    h_caesar = (char*) malloc(N*sizeof(char));

    checkCuda( cudaMallocHost(&d_caesar, N*sizeof(char)) );
    // Setup to exec the previous Kernels 
    checkCuda( cudaMallocHost(&d_a, N*sizeof(int)) );
    checkCuda( cudaMallocHost(&d_b, N*sizeof(int)) );    
    checkCuda( cudaMallocHost(&d_c, N*sizeof(int)) );

    printf("Allocation complete\n");
    checkCuda( cudaEventRecord(*start, 0) );

    // Execute Caesar Shifts first
    int testIdx = rand() % (N-26);

    checkCudaKernel( (InitAlpha<<<numBlocks, blockSize>>>(N, d_caesar)) ); 
    checkCudaKernel( (CaesarShift<<<numBlocks, blockSize>>>(N, d_caesar, shift)) );
    checkCuda( cudaMemcpy(h_caesar, d_caesar, N*sizeof(char), cudaMemcpyDeviceToHost) );
    printf("Caesar shifted %d \n", shift);
    PrintCaesarStream(h_caesar, testIdx);

    checkCudaKernel( (CaesarShift<<<numBlocks, blockSize>>>(N, d_caesar, 26-shift)) ); 
    checkCudaKernel( (cudaMemcpy(h_caesar, d_caesar, N*sizeof(char), cudaMemcpyDeviceToHost)) );
    printf("Caesar shifted back to original by %d \n", 26-shift);
    PrintCaesarStream(h_caesar, testIdx);

    checkCuda( cudaMemcpy(d_a, h_c, N*sizeof(int), cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_b, h_c, N*sizeof(int), cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_c, h_c, N*sizeof(int), cudaMemcpyHostToDevice) );

    // setup data for prveious kernels
    checkCudaKernel( (gen_data<<<numBlocks, blockSize>>>(d_a)) );
    checkCudaKernel( (gen_data<<<numBlocks, blockSize>>>(d_b)) );
    RunGpuAdd(N, numBlocks, blockSize, d_a, d_b, d_c, h_c);
    RunGpuSub(N, numBlocks, blockSize, d_a, d_b, d_c, h_c);
    RunGpuMult(N, numBlocks, blockSize, d_a, d_b, d_c, h_c);
    RunGpuMod(N, numBlocks, blockSize, d_a, d_b, d_c, h_c);
   
    checkCuda( cudaEventRecord(*stop, 0) );

    checkCuda( cudaFreeHost(d_caesar) );
    free(h_caesar);
    
    checkCuda( cudaFreeHost(d_a) );
    checkCuda( cudaFreeHost(d_b) );
    checkCuda( cudaFreeHost(d_c) );
    free(h_c);

}


// Setup pagged memory than run the Caesar shift function twice
void PaggedMem(int N, int numBlocks, int blockSize, int shift,
        cudaEvent_t* start, cudaEvent_t* stop ){

    // Setup a ton of pagged memory for use
    int* d_a, *d_b, *d_c, *h_c;
    char* d_caesar, *h_caesar;

    h_caesar = (char*) malloc(N*sizeof(char));
    checkCuda( cudaMalloc(&d_caesar, N*sizeof(char)) );
    
    // Setup to exec the previous Kernels 
    h_c = (int*) malloc(N*sizeof(int));
    checkCuda( cudaMalloc(&d_a, N*sizeof(int)) );
    checkCuda( cudaMalloc(&d_b, N*sizeof(int)) );    
    checkCuda( cudaMalloc(&d_c, N*sizeof(int)) );    

    printf("Allocation complete\n");
    checkCuda( cudaEventRecord(*start, 0) );

    // Execute Caesar Shifts first
    int testIdx = rand() % (N-26);

    checkCudaKernel( (InitAlpha<<<numBlocks, blockSize>>>(N, d_caesar)) ); 
    checkCudaKernel( (CaesarShift<<<numBlocks, blockSize>>>(N, d_caesar, shift)) );
    checkCuda( cudaMemcpy(h_caesar, d_caesar, N*sizeof(char), cudaMemcpyDeviceToHost) );
    printf("Caesar shifted %d\n", shift);
    PrintCaesarStream(h_caesar, testIdx);

    checkCudaKernel( (CaesarShift<<<numBlocks, blockSize>>>(N, d_caesar, 26-shift)) ); 
    checkCuda( cudaMemcpy(h_caesar, d_caesar, N*sizeof(char), cudaMemcpyDeviceToHost) );
    printf("Caesar shifted back to original with %d\n", 26-shift);
    PrintCaesarStream(h_caesar, testIdx);

    checkCuda( cudaMemcpy(d_a, h_c, N*sizeof(int), cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_b, h_c, N*sizeof(int), cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_c, h_c, N*sizeof(int), cudaMemcpyHostToDevice) );    

    // setup data for prveious kernels
    checkCudaKernel( (gen_data<<<numBlocks, blockSize>>>(d_a)) );
    checkCudaKernel( (gen_data<<<numBlocks, blockSize>>>(d_b)) );
    
    RunGpuAdd(N, numBlocks, blockSize, d_a, d_b, d_c, h_c);
    RunGpuSub(N, numBlocks, blockSize, d_a, d_b, d_c, h_c);
    RunGpuMult(N, numBlocks, blockSize, d_a, d_b, d_c, h_c);
    RunGpuMod(N, numBlocks, blockSize, d_a, d_b, d_c, h_c);
   
    checkCuda( cudaEventRecord(*stop, 0) );

    checkCuda( cudaFree(d_caesar) );
    free(h_caesar);     
    
    checkCuda( cudaFree(d_a) );
    checkCuda( cudaFree(d_b) );
    checkCuda( cudaFree(d_c) );
    free(h_c);
    
 
}



int main(int argc, char** argv)
{
	// read command line arguments
	srand(time(0));
	int N = 1 << 20;
	int blockSize = 256;
    int shift = 5;// for the caesar cipher
	
	if (argc >= 2) {
		N = atoi(argv[1]);
	}
    
    int totalThreads = N;

	if (argc >= 3) {
		blockSize = atoi(argv[2]);
        if(blockSize > N) {
            printf("You cant have more threads than tasks");
            return -1;
        }
	}
    if(argc >= 4) {
        shift = atoi(argv[3]);
    }

    int numBlocks = totalThreads / blockSize;


	// validate command line arguments
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;

		N = totalThreads;

		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}

    cudaDeviceProp prop; 
    cudaGetDeviceProperties(&prop, 0);
    printf("Cuda Device %s\n", prop.name);
    printf("Problem Size: %d, with grid of (%d,%d)\n", N, numBlocks, blockSize);

    float time;
    cudaEvent_t start, stop, pev1, pev2;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&pev1);
    cudaEventCreate(&pev2);

    // profile using Pinned Memory
    cudaEventRecord(start, 0);    
    PinnedMem(N, numBlocks, blockSize, shift, &pev1, &pev2);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Executing with Pinned memory took %f \n", time);
    cudaEventSynchronize(pev2);
    cudaEventElapsedTime(&time, pev1, pev2);
    printf("Pinned memory speed without allocs included %f \n", time);
   

    // profile using Page memory only
    cudaEventRecord(start, 0);    
    PaggedMem(N, numBlocks, blockSize, shift, &pev1, &pev2);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Executing with pagged memory took %f \n", time);
    cudaEventSynchronize(pev2);
    cudaEventElapsedTime(&time, pev1, pev2);
    printf("Pinned memory speed without allocs included %f \n", time);


}
