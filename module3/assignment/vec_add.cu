
#include <stdio.h>
#include <stdlib.h>

__global__ void vec_add(int* a, int* b, int* c){

    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    c[tid] = a[tid] + b[tid];
    printf("Tid %d, c[tid]=%d\n", tid, c[tid]);
}

int main(){

    int N = 16; 
    int div = 4;
    int numBlocks = N / div;
    int tPB = N/ numBlocks;
    printf("%d Blocks with %d threads each\n", numBlocks, tPB);

    int* a = (int*)malloc(sizeof(int)*N);
    int * dev_a, *dev_b, *dev_c;
    cudaMalloc((void**)&dev_a, sizeof(int)*N);
    cudaMalloc((void**)&dev_b, sizeof(int)*N);
    cudaMalloc((void**)&dev_c, sizeof(int)*N);

    for(int i = 0; i < N; i++)
        a[i] = i*i;
    
    cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, a, N*sizeof(int), cudaMemcpyHostToDevice);
    vec_add<<<numBlocks, tPB>>>(dev_a, dev_b, dev_c);
    
    cudaMemcpy(a, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_c);
    for(int i = 0; i< N; i++){
        printf("%d\n", a[i]);
    } 
}



