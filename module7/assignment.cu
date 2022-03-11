#include <stdio.h>
#include <stdlib.h>
#include "cuda_utils.cuh"



// basic math op kernels ensure 4|A| = |C| and offset <= 3|A|
__device__ void gpu_add(int* a, int* b, int *c, int offset){
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    c[offset + tid] = a[tid] + b[tid];
}
__device__ void gpu_sub(int* a, int* b, int *c, int offset){
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    c[offset + tid] = a[tid] - b[tid];
}
__device__ void gpu_mul(int* a, int* b, int *c, int offset){
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    c[offset + tid] = a[tid] * b[tid];
}
__device__ void gpu_xor(int* a, int* b, int *c, int offset){
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    c[offset + tid] = a[tid] ^ b[tid];
}
    
__global__ void mplex_kernel(int ksel, int *a, int *b, int *c, int offset){
    switch(ksel) {
        case 0: {
            gpu_add(a, b, c, offset);
            break;
        } case 1: {
            gpu_sub(a, b, c, offset);
            break;
        } case 2: {
            gpu_mul(a, b, c, offset);
            break;
        } case 3: {
            gpu_xor(a, b, c, offset);
            break;
        }
    }
}


void printResultsSync(int N, int* h_c, float t, int idx)
{
    printf("Sync kernels finished in %f ms", t);
    printf("A[%d] + B[%d] = %d \n", idx, idx, h_c[idx]);
    printf("A[%d] - B[%d] = %d \n", idx, idx, h_c[idx+N]);
    printf("A[%d] * B[%d] = %d \n", idx, idx, h_c[idx+N*2]);
    printf("A[%d] ^ B[%d] = %d \n", idx, idx, h_c[idx+N*3]);
}

void testSync(int N, int blockSize, int numBlocks, int testIdx,
        int *h_a, int *h_b,int *h_c,
        int *d_a, int *d_b, int *d_c)
{
    cudaEvent_t start, stop;
    checkCuda( cudaEventCreate(&start) );
    checkCuda( cudaEventCreate(&stop) );

    printf("SyncTest start\n");
    checkCuda( cudaEventRecord(start, 0) );
    // simulate new data coming in for parity
    for(int i = 0; i < 4; i++) 
    {
        printf("Memcpy input\n");
        checkCuda( cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice) );
        checkCuda( cudaMemcpy(d_b, h_b, N*sizeof(int), cudaMemcpyHostToDevice) );
        printf("Kernel %d exec\n", i); 
        checkCudaKernel( (mplex_kernel<<<N, blockSize>>>(i, d_a, d_b, d_c, N)) );
        printf("Memcpy result\n");
        checkCuda( cudaMemcpy(&h_c[N*i], &d_c[N*i], N*sizeof(int), cudaMemcpyDeviceToHost) );
    }

    printf("Kernels launched");
 
    float t;
    checkCuda( cudaEventRecord(stop, 0) );
    checkCuda( cudaEventElapsedTime(&t, start, stop));
    printResultsSync(N, h_c, t, testIdx);


}
    
void printResultsStream(int N, int* h_c, float t, int idx){
    
    printf("Stream kernels finished in %f ms", t);
    printf("A[%d] + B[%d] = %d \n", idx, idx, h_c[idx]);
    printf("A[%d] - B[%d] = %d \n", idx, idx, h_c[idx+N]);
    printf("A[%d] * B[%d] = %d \n", idx, idx, h_c[idx+N*2]);
    printf("A[%d] ^ B[%d] = %d \n", idx, idx, h_c[idx+N*3]);
}

void testStream(int N, int blockSize, int numBlocks, int testIdx,
        int *h_a, int *h_b, int *h_c,
        int *d_a, int *d_b, int *d_c)
{

    cudaEvent_t start, end;
    checkCuda( cudaEventCreate(&start) );
    checkCuda( cudaEventCreate(&end) );
    
    printf("Starting streaming approach \n");
    checkCuda( cudaEventRecord(start, 0));

    cudaStream_t streams[4]; // lets running everything in parallel
    for(int i = 0; i < 4; i++) {
        checkCuda( cudaStreamCreate(&streams[i]) );
    }

    for(int i = 0; i < 4; i++){
        checkCuda( cudaMemcpyAsync(d_a, h_a, sizeof(int) * N, cudaMemcpyHostToDevice, streams[i]) );
        checkCuda( cudaMemcpyAsync(d_b, h_b, sizeof(int) * N, cudaMemcpyHostToDevice, streams[i]) );
        checkCudaKernel( (mplex_kernel<<<N, blockSize, 0, streams[i]>>>(i, d_a, d_b, d_c, 0)) );
        checkCuda( cudaMemcpyAsync(&h_c[N*i], &d_c[N*i], sizeof(int)*N, cudaMemcpyHostToDevice, streams[i]) );
    }

    for(int i = 0; i < 4; i++) {
        checkCuda( cudaStreamSynchronize(streams[i]) ); // sync all threads
    }
    
    checkCuda( cudaEventRecord(end, 0) ); 
    float t;
    checkCuda( cudaEventElapsedTime(&t, start, end));
    printResultsStream(N, h_c, t, testIdx);


}


// print the specs of this machine and number of devices
void printDeviceSpecs(){
    
    cudaDeviceProp prop;
    checkCuda( cudaGetDeviceProperties(&prop, 0) );
    printf("Cuda Device %s\n", prop.name);

    int numDevices;
    checkCuda( cudaGetDeviceCount(&numDevices) );
    printf("you have %d devices\n", numDevices);
}

void allocateData(int N, int **h_a, int **h_b, int **h_c, int **d_a, int **d_b, int **d_c)
{
    checkCuda( cudaMallocHost((void **)h_a, sizeof(int) * N) );
    checkCuda( cudaMallocHost((void **)h_b, sizeof(int) * N) );
    checkCuda( cudaMallocHost((void **)h_c, sizeof(int) * 4 *N) );
    checkCuda( cudaMalloc((void**) d_a, sizeof(int) * 4 * N) ); // we need more space on recv
    checkCuda( cudaMalloc((void**) d_b, sizeof(int) * 4 * N) );
    checkCuda( cudaMalloc((void**) d_c, sizeof(int) * 4 * N) );
    
    // dummy data input
    for(int i = 0; i < N; i++){
        (*h_a)[i] = rand() %10;
        (*h_b)[i] = rand() %10;
    }
}

void freeData(int *h_a, int *h_b, int *h_c, int *d_a, int *d_b, int *d_c)
{
    checkCuda( cudaFreeHost(h_a) );
    checkCuda( cudaFreeHost(h_b) );
    checkCuda( cudaFreeHost(h_c) );
    checkCuda( cudaFree(d_a) );
    checkCuda( cudaFree(d_b) );
    checkCuda( cudaFree(d_c) );
}

int main(int argc, char** argv)
{
	// read command line arguments
    int N = (1 << 20);
	int blockSize = 256;
	
	if (argc >= 2) {
		N = atoi(argv[1]);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
	}
	int numBlocks = N/blockSize;

	// validate command line arguments
	if (N % blockSize != 0) {
		++numBlocks;
		N = numBlocks*blockSize;	
	}

    printDeviceSpecs(); 
    
    int *h_a, *h_b, *h_c;  // C is 4x len(a) 
    int *d_a, *d_b, *d_c;

    srand(time(NULL));
    int testIdx = rand() % N;
    printf("Allocating data\n");
    allocateData(N, &h_a, &h_b, &h_c, &d_a, &d_b, &d_c);
    printf("Allocating done running kernels\n");

    testSync(N, blockSize, numBlocks, testIdx, h_a, h_b, h_c, d_a, d_b, d_c);
    testStream(N, blockSize, numBlocks, testIdx, h_a, h_b, h_c, d_a, d_b, d_c); 

    printf("Free data\n");
    freeData(h_a, h_b, h_c, d_a, d_b, d_c);
    
}
