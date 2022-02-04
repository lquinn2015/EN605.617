//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>

__global__
void gpu_add(int* a, int* b, int*c)
{
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    printf("tid: val is %d\n", (a[tid]+b[tid]) );
    c[tid] = a[tid] + b[tid];
}

__global__
void gpu_sub(int* a, int* b, int*c)
{
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    c[tid] = a[tid] - b[tid];
}

__global__
void gpu_mult(int* a, int* b, int*c)
{
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    c[tid] = a[tid] * b[tid];
}

__global__
void gpu_mod(int* a, int* b, int*c)
{
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    c[tid] = a[tid] % b[tid];
}

__global__
void gen_data(int* a) {
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    a[tid] = tid;
}

void print_result(int* arr, int N, char opt){

    int test_idx = 5;// rand() % N;
    printf("%c operation result c[%d]=%d\n", opt, test_idx, arr[test_idx]);
}

int main(int argc, char** argv)
{
	// read command line arguments
	int totalThreads = 256;//(1 << 20);
	int blockSize = 256;
	
	if (argc >= 2) {
		totalThreads = atoi(argv[1]);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
	}

	int numBlocks = totalThreads/blockSize;
    int threadPerBlock = totalThreads / numBlocks;


	// validate command line arguments
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}

    int N = totalThreads; //= 1 << 20; // allocate 2^20 = 1024*1024 ~ 1 mil

    printf("numBlocks %d whith %d threads\n", numBlocks, threadPerBlock);

    int* c = (int*)malloc(N);
    int* dev_a, *dev_b, *dev_c;

    cudaMalloc((void**)&dev_a, sizeof(int)* N);
    cudaMalloc((void**)&dev_b, sizeof(int)* N);
    cudaMalloc((void**)&dev_c, sizeof(int)* N);

    cudaMemcpy(dev_a, c, N, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, c, N, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, N, cudaMemcpyHostToDevice);

    // generate inputs on GPU because they are going to be used there
    gen_data<<<numBlocks, threadPerBlock>>>(dev_a);
    gen_data<<<numBlocks, threadPerBlock>>>(dev_b);
   
    gpu_add<<<numBlocks, threadPerBlock>>>(dev_a,dev_b,dev_c);
    cudaMemcpy(c, dev_c, N*sizeof(long), cudaMemcpyDeviceToHost);
    print_result(c, N, '+'); 

    gpu_sub<<<numBlocks, threadPerBlock>>>(dev_a,dev_b,dev_c);
    cudaMemcpy(c, dev_c, N*sizeof(long), cudaMemcpyDeviceToHost);
    print_result(c, N, '-'); 

    gpu_mult<<<numBlocks, threadPerBlock>>>(dev_a,dev_b,dev_c);
    cudaMemcpy(c, dev_c, N*sizeof(long), cudaMemcpyDeviceToHost);
    print_result(c, N, '*'); 
    
    gpu_mod<<<numBlocks, threadPerBlock>>>(dev_a,dev_b,dev_c);
    cudaMemcpy(c, dev_c, N*sizeof(long), cudaMemcpyDeviceToHost);
    print_result(c, N, '%'); 
    
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    free(c);

}





