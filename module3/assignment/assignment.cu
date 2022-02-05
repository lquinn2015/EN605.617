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
    srand(time(0));
    int N = 1 << 20; // work 2 do

    if (argc >= 2){
        N = atoi(argv[1]);
    }
    int totalThreads = N;  // we want to have a thread for every unit of work
    int threadsPerBlock = 256; // how we divide work among blocks
    
    if (argc >= 3) {
        threadsPerBlock = atoi(argv[2]);
        if(threadsPerBlock > totalThreads) {
            printf("You cannot have more threads then tasks\n");
            return -1;
        }
    }
    
    int numBlocks = totalThreads / threadsPerBlock; // the number of blocks required

    // validate command line arguments
    if (totalThreads % threadsPerBlock != 0) {  // if this isn't zero we need more blocks
        ++numBlocks;
        totalThreads = numBlocks*threadsPerBlock;
    
        // we will over compute in order to skip bounds checking in the kernels
        N = totalThreads; 
    
        printf("Warning: Total thread count is not evenly divisible by the block size\n");
        printf("The total number of threads will be rounded up to %d\n", totalThreads);
    }

   
    printf("numBlocks %d whith %d threads used to solve for %d work units\n", numBlocks, threadsPerBlock, N);

    // allocate data
    int* c = (int*)malloc(N*sizeof(int));
    int* dev_a, *dev_b, *dev_c;
    cudaMalloc((void**)&dev_a, sizeof(int)* N);
    cudaMalloc((void**)&dev_b, sizeof(int)* N);
    cudaMalloc((void**)&dev_c, sizeof(int)* N);

    cudaMemcpy(dev_a, c, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, c, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, N*sizeof(int), cudaMemcpyHostToDevice);

    // generate inputs on GPU because they are going to be used there
    gen_data<<<numBlocks, threadsPerBlock>>>(dev_a);
    gen_data<<<numBlocks, threadsPerBlock>>>(dev_b);
   
    gpu_add<<<numBlocks, threadsPerBlock>>>(dev_a,dev_b,dev_c);
    cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);    
    print_result(c, N, '+'); 

    gpu_sub<<<numBlocks, threadsPerBlock>>>(dev_a,dev_b,dev_c);
    cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);
    print_result(c, N, '-'); 

    gpu_mult<<<numBlocks, threadsPerBlock>>>(dev_a,dev_b,dev_c);
    cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);
    print_result(c, N, '*'); 
    
    gpu_mod<<<numBlocks, threadsPerBlock>>>(dev_a,dev_b,dev_c);
    cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);
    print_result(c, N, '%'); 
    
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    free(c);

}

