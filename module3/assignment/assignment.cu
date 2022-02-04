//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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
    printf("%c operation result c[%d]=%d\n", opt, test_idx, arr[test_idx]);
}

int main(int argc, char** argv)
{
    // read command line arguments

    int N = 1 << 20; // allocate 2^20 = 1024*1024 ~ 1 mil

    int totalThreads = (1 << 20);
    srand(time(0));
    int blockSize = 256;
    
    if (argc >= 2) {
        totalThreads = atoi(argv[1]);
    }
    if (argc >= 3) {
        blockSize = atoi(argv[2]);
    }
    
    if (argc >= 4) {
        N = atoi(argv[3]);
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

   
    printf("numBlocks %d whith %d threads\n", numBlocks, threadPerBlock);

    // allocate data
    int* c = (int*)malloc(N*sizeof(int));
    int* dev_a, *dev_b, *dev_c;
    cudaMalloc((void**)&dev_a, sizeof(int)* N);
    cudaMalloc((void**)&dev_b, sizeof(int)* N);
    cudaMalloc((void**)&dev_c, sizeof(int)* N);

    // init data
    //for(int i = 0; i < N; i++){
    //    c[i] = 0;                      
    //}
    
    cudaMemcpy(dev_a, c, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, c, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, N*sizeof(int), cudaMemcpyHostToDevice);

    // generate inputs on GPU because they are going to be used there
    gen_data<<<numBlocks, threadPerBlock>>>(dev_a);
    gen_data<<<numBlocks, threadPerBlock>>>(dev_b);
   
    gpu_add<<<numBlocks, threadPerBlock>>>(dev_a,dev_b,dev_c);
    cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);    
    print_result(c, N, '+'); 

    gpu_sub<<<numBlocks, threadPerBlock>>>(dev_a,dev_b,dev_c);
    cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);
    print_result(c, N, '-'); 

    gpu_mult<<<numBlocks, threadPerBlock>>>(dev_a,dev_b,dev_c);
    cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);
    print_result(c, N, '*'); 
    
    gpu_mod<<<numBlocks, threadPerBlock>>>(dev_a,dev_b,dev_c);
    cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);
    print_result(c, N, '%'); 
    
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    free(c);

}

