//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>

__global__
void gpu_add(int* a, int* b, int*c)
{
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
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

    int test_idx = rand() % N;
    printf("%c operation result c[%d]=%d", opt, test_idx, arr[test_idx]);
}

int main(int argc, char** argv)
{
	// read command line arguments
	int totalThreads = (1 << 20);
	int blockSize = 256;
	
	if (argc >= 2) {
		totalThreads = atoi(argv[1]);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
	}

	int numBlocks = totalThreads/blockSize;
    int threadPerBlock = totalThreads % blockSize;


	// validate command line arguments
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}

    int N = 1<<10; //= 1 << 20; // allocate 2^20 = 1024*1024 ~ 1 mil

    int* a;
    int* b;
    int* c;
    int* c_cpu = (int*) malloc(sizeof(int)*N);

    cudaMalloc((void**)&a, sizeof(int)* N);
    cudaMalloc((void**)&b, sizeof(int)* N);
    cudaMalloc((void**)&c, sizeof(int)* N);

    // generate inputs on GPU because they are going to be used there
    gen_data<<<numBlocks, threadPerBlock>>>(a);
    gen_data<<<numBlocks, threadPerBlock>>>(b);
   
    gpu_add<<<numBlocks, threadPerBlock>>>(a,b,c);
    cudaMemcpy(c_cpu, c, N*sizeof(long), cudaMemcpyDeviceToHost);
    print_result(c_cpu, N, '+'); 

    gpu_sub<<<numBlocks, threadPerBlock>>>(a,b,c);
    cudaMemcpy(c_cpu, c, N*sizeof(long), cudaMemcpyDeviceToHost);
    print_result(c_cpu, N, '-'); 

    gpu_mult<<<numBlocks, threadPerBlock>>>(a,b,c);
    cudaMemcpy(c_cpu, c, N*sizeof(long), cudaMemcpyDeviceToHost);
    print_result(c_cpu, N, '*'); 
    
    gpu_mod<<<numBlocks, threadPerBlock>>>(a,b,c);
    cudaMemcpy(c_cpu, c, N*sizeof(long), cudaMemcpyDeviceToHost);
    print_result(c_cpu, N, '%'); 
    
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    free(c_cpu);

}





