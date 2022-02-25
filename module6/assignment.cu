#include <stdio.h>
#include "cuda_utils.cuh"

__constant__ uint8_t c_ZERO = 0;
__constant__ uint8_t c_ONE = 1;
__constant__ uint8_t c_EIGHT = 8;
__constant__ uint8_t c_FFMASK = 0xff;
__constant__ uint8_t c_HIGHMASK = 0x80;
__constant__ uint8_t c_POLY_1B = 0x1b; // irreducible poly in GF(2^8)

__constant__ uint8_t c_MSB  = 0x18;
__constant__ uint8_t c_MLSB = 0x10;
__constant__ uint8_t c_LMSB = 0x08;

#define K2TEST 3
char KMODE[K2TEST][30] = {"Reg_test", "AntiReg_test", "Fmul_test"};

// FIPS 197 fmul in 2^8 mod poly(1b)
__device__ void fmul_reg(uint8_t* a, uint8_t* b, uint32_t* c){

    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint8_t p_i = a[tid];
    uint8_t rb = b[tid];
   
    uint8_t i = c_ZERO;
    uint8_t p_it = p_i;
    uint8_t ret = c_ZERO;
    while(i < c_EIGHT){
        // mult
        if( c_ONE & rb == c_ONE) ret ^= p_i;
        rb = rb >> c_ONE;
        // xtime p_i 
        p_it = p_i << c_ONE;
        if(p_i & c_HIGHMASK == c_ONE) p_it ^= c_POLY_1B;
        p_i = p_it & c_FFMASK;
        i++;
    }
    c[tid] = ret;
}

__global__ void initData(uint8_t* a, uint8_t *b){
    
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    a[tid] = (uint8_t)(tid & c_FFMASK);
    b[tid] = (uint8_t)(tid & c_FFMASK);
}
// use registers to speed up this calc i/e copy gmem to reg than reg to gmem
__device__ void regTest(uint8_t* a, uint8_t *b, uint32_t *c)
{
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    uint8_t ra = (uint8_t) a[tid];
    uint8_t rb = (uint8_t) b[tid];
    uint32_t plus = (uint32_t) (ra + rb) << c_MSB;
    uint32_t sub  = (uint32_t) (ra - rb) << c_MLSB;
    uint32_t mult = (uint32_t) (ra * rb) << c_LMSB;
    uint32_t mod  = (uint32_t) (ra % rb);
    uint32_t res = plus || sub || mult || mod; 
    c[tid] = res;
}

// use as few registers as possible global data only
__device__ void antiRegTest(uint8_t* a, uint8_t *b, uint32_t *c)

{
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    c[tid] = 
       (uint32_t) ((a[tid] + b[tid]) << c_MSB  ||
                  (a[tid] - b[tid]) << c_MLSB ||
                  (a[tid] * b[tid]) << c_LMSB ||
                  (a[tid] % b[tid]));
}

__global__ void MultKernel(uint8_t *a, uint8_t *b, uint32_t *c, int mode){

    switch(mode){
        case 0:
            regTest(a,b,c);
            break;
        case 1:
            antiRegTest(a,b,c);
            break;
        case 2:
            fmul_reg(a,b,c);
            break;
    }

}
        

void print_result(int mode, uint32_t* h_d, int N, float prememcpy, float postmemcpy){

    int tid = rand() % N;
    printf("Timing of  %s prememcpy: %f     postmemcpy: %f \n", KMODE[mode], 
            prememcpy, postmemcpy);
    if(mode == 0 || mode == 1) {
        uint8_t add = ((h_d[tid] >> 0x18) & 0xff);
        uint8_t sub = ((h_d[tid] >> 0x10) & 0xff);
        uint8_t mul = ((h_d[tid] >> 0x08) & 0xff);
        uint8_t mod = ((h_d[tid] >> 0x00) & 0xff);
        printf("Tid = %d \n", tid);
        printf("    %d + %d = %d \n",tid, tid, add);
        printf("    %d + %d = %d \n",tid, tid, sub);
        printf("    %d + %d = %d \n",tid, tid, mul);
        printf("    %d + %d = %d \n",tid, tid, mod);
    }
    
}

int main(int argc, char** argv)
{
	// read command line arguments
    srand(time(NULL));
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

    cudaDeviceProp prop;
    checkCuda( cudaGetDeviceProperties(&prop, 0) );
    printf("Cuda Device %s\n", prop.name);
    printf("Problem Size: %d with grid of ( %d , %d )\n", N, numBlocks, blockSize);

    cudaEvent_t s1,s2,s3;
    checkCuda( cudaEventCreate(&s1) );
    checkCuda( cudaEventCreate(&s2) );
    checkCuda( cudaEventCreate(&s3) );

    uint32_t *h_c = (uint32_t*) malloc(N * sizeof(uint32_t));
    uint8_t *d_a, *d_b;
    uint32_t *d_c;

    checkCuda( cudaMalloc(&d_a, N*sizeof(uint8_t)) );
    checkCuda( cudaMalloc(&d_b, N*sizeof(uint8_t)) );
    checkCuda( cudaMalloc(&d_c, N*sizeof(uint32_t)) );

    checkCudaKernel( (initData<<<numBlocks, blockSize>>>(d_a, d_b)) );

    printf("Data setup startin runs\n");

    float d1,d2; 
    for(int i = 0; i < K2TEST; i++) {

        printf("Starting test %s\n", KMODE[i]);
        checkCuda( cudaEventRecord(s1, 0) );    
        printf("Cuda Event start");
        checkCudaKernel( 
            (MultKernel<<<numBlocks, blockSize>>>(d_a, d_b, d_c, i))
        );
        checkCuda( cudaEventRecord(s2, 0) );
        checkCuda( cudaMemcpy(h_c, d_c, N*sizeof(uint32_t), cudaMemcpyDeviceToHost) );    
        checkCuda( cudaEventRecord(s3, 0) );

        checkCuda( cudaEventSynchronize(s2) );
        checkCuda( cudaEventSynchronize(s3) );
        checkCuda( cudaEventElapsedTime(&d1, s1, s2) ); 
        checkCuda( cudaEventElapsedTime(&d2, s2, s3) ); 

        print_result(i, h_c, N, d1, d2);
        printf("%s now finished \n", KMODE[i]);
    }
 
    free(h_c); 
    checkCuda( cudaFree(d_a) );
    checkCuda( cudaFree(d_b) );
    checkCuda( cudaFree(d_c) );
    checkCuda( cudaEventDestroy(s1) );
    checkCuda( cudaEventDestroy(s2) );
    checkCuda( cudaEventDestroy(s3) );
 
    
}
