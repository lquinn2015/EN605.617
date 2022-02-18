#include <stdio.h>
#include <assert.h>


#define NUM_ROUNDS 30
#define NUM_OF_OPTS 4
#define NUM_OF_MODES 5
char gMode2Str[NUM_OF_MODES][30] = {"global", "shmem", "constant", "literal", "Constant and shmem"};
char gOpt2Char[NUM_OF_OPTS] = {'+', '-', '*', '%'};

typedef unsigned int uint32_t;


__constant__ static const uint32_t const_M1 = 0xF0F0F0F0;
__constant__ static const uint32_t const_M2 = 0x0FF00FF0;
__constant__ static const uint32_t const_M3 = 0xFF00FF00;
__constant__ static const uint32_t const_M4 = 0x0000FFFF;
__constant__ static const uint32_t const_M5 = 0x80000001;


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


// general structure all device functions were formally kernels
// they have a global c var which always writes back to global mem
// however they have a sid value which represents a shared mem id 
// if they have a shared mem its transparent if not tid == sid
// and no change happens

// As a note sid -> something could be shared mem if you want
//      and as such can right or left operands 
//      but one uses a tid in the right hand they will crash


__device__ void gpu_scramble_literal(const int tid, const int sid, int *a, int *b, int *c){


    uint32_t v = 0;
    for(int i = 0; i < NUM_ROUNDS; i++) {
        v ^= ((a[sid] ^ (b[sid]*3)) & 0xF0F0F0F0) >> 3;
        v ^= ((a[sid] ^ (b[sid]*3)) & 0xF0F0F0F0) << 5;
        v ^= ((a[sid] ^ (b[sid]*3)) & 0xF0F0F0F0) >> 3;
        v ^= ((a[sid] ^ (b[sid]*3)) & 0xF0F0F0F0) << 5;
        v ^= 0x80000001;
    }
    c[tid] = v;
}

__device__ void gpu_scramble_const(const int tid, const int sid,  int *a, int *b, int *c){
    
    uint32_t v = 0;
    for(int i = 0; i < NUM_ROUNDS; i++) {
        v ^= ((a[sid] ^ (b[sid]*3)) & const_M1) >> 3;
        v ^= ((a[sid] ^ (b[sid]*3)) & const_M2) << 5;
        v ^= ((a[sid] ^ (b[sid]*3)) & const_M3) >> 3;
        v ^= ((a[sid] ^ (b[sid]*3)) & const_M4) << 5;
        v ^= const_M5;
    }
    c[tid] = v;
}


__device__ void gpu_add(const int tid, const int sid, int* a, int* b, int*c)
{
    c[tid] = a[sid] + b[sid];
}

__device__ void gpu_sub(const int tid, const int sid, int* a, int* b, int*c)
{
    c[tid] = a[sid] - b[sid];
}

__device__ void gpu_mult(const int tid, const int sid, int* a, int* b, int*c)
{
    c[tid] = a[sid] * b[sid];
}

__device__ void gpu_mod(const int tid, const int sid, int* a, int* b, int*c)
{
    c[tid] = a[sid] % b[sid];
}

__global__ void MultiMemKern(int mode, int *d_a, int *d_b, int *d_k1, int *d_k2, int *d_k3, int *d_k4){

    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    extern __shared__ int shmem[];

    uint32_t sid = tid % blockDim.x;
    
    switch(mode) {

        case 0: {// global
            d_a[tid] = tid;
            d_b[tid] = tid;
            gpu_add (tid, tid, d_a, d_b, d_k1);
            gpu_sub (tid, tid, d_a, d_b, d_k2);
            gpu_mult(tid, tid, d_a, d_b, d_k3);
            gpu_mod (tid, tid, d_a, d_b, d_k4);
            break;
        }case 1: {// shared
            shmem[sid] = tid;
            gpu_add (tid, sid, shmem, shmem, d_k1);
            gpu_sub (tid, sid, shmem, shmem, d_k2);
            gpu_mult(tid, sid, shmem, shmem, d_k3);
            gpu_mod (tid, sid, shmem, shmem, d_k4);
            break;
        } case 2: {// constant
            d_a[tid] = tid;
            d_b[tid] = tid;
            gpu_scramble_literal(tid, tid, d_a, d_b, d_k1);
            break;
        } case 3: {// literals
            d_a[tid] = tid;
            d_b[tid] = tid;
            gpu_scramble_const(tid, tid, d_a, d_b, d_k1);
            break;
        } case 4: { // constants and shmem
            shmem[tid] = tid;
            gpu_scramble_const(tid, sid, shmem, shmem, d_k1);
            break;
        }
    }

}

void print_result(char opt, int *d_ptr, int* h_ptr, int N){

    cudaMemcpy(h_ptr, d_ptr, N*sizeof(int), cudaMemcpyDeviceToHost);

    int test_idx = rand() % N;
    printf("%c operation result c[%d]=%d\n", opt, test_idx, h_ptr[test_idx]);
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
		
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", N);
	}
    
    int shmem_size = blockSize*sizeof(int);
    if (shmem_size > 48 * (2<<10) ) {
        printf("Canceling run block size to big for memory requirements\n");
        return -1;
    }
    


    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Cuda Device %s\n", prop.name);
    printf("Problem Size: %d, with grid of (%d,%d)\n", N, numBlocks, blockSize);

    cudaEvent_t start,stop;
    checkCuda( cudaEventCreate(&start) );
    checkCuda( cudaEventCreate(&stop) );


    int *h_c = (int*)  malloc( N * sizeof(int));
    int *d_a, *d_b, *d_k1, *d_k2, *d_k3, *d_k4;

    // indulge my memory hunger
    checkCuda( cudaMalloc(&d_a, N*sizeof(int)) ); 
    checkCuda( cudaMalloc(&d_b, N*sizeof(int)) ); 
    checkCuda( cudaMalloc(&d_k1, N*sizeof(int)) ); 
    checkCuda( cudaMalloc(&d_k2, N*sizeof(int)) ); 
    checkCuda( cudaMalloc(&d_k3, N*sizeof(int)) ); 
    checkCuda( cudaMalloc(&d_k4, N*sizeof(int)) ); 
    
    int* h_dptr[4] = {d_k1, d_k2, d_k3, d_k4};

    float delta;
    
    for(int mode = 0; mode<NUM_OF_MODES; mode++) {

        checkCuda( cudaEventRecord(start, 0) );
        printf("Launching kernel with %s mode\n", gMode2Str[mode] );
        checkCudaKernel(( 
            MultiMemKern<<<numBlocks, blockSize, shmem_size>>>(mode, d_a, d_b, 
                                                               d_k1, d_k2, d_k3, d_k4) 
        ));
        checkCuda( cudaEventRecord(stop, 0) );
        
        checkCuda( cudaEventSynchronize(stop) );
        checkCuda( cudaEventElapsedTime(&delta, start, stop) );
        printf("The %s mem mode took %f to calculate the following\n", gMode2Str[mode], delta);

        if(mode < 2){
            for(int opt = 0; opt < NUM_OF_OPTS; opt++) {
                print_result(gOpt2Char[opt], h_dptr[opt], h_c, N);
            }
        } else {
            print_result('s', d_k1, h_c, N);
        }
    
    }


    free(h_c);
    checkCuda( cudaFree(&d_a ) );
    checkCuda( cudaFree(&d_b ) );
    checkCuda( cudaFree(&d_k1) );
    checkCuda( cudaFree(&d_k2) );
    checkCuda( cudaFree(&d_k3) );
    checkCuda( cudaFree(&d_k4) );



}
