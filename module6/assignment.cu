#include <stdio.h>
#include <stdlib.h>
#include "cuda_utils.cuh"

__constant__ uint32_t c_ZERO = 0;
__constant__ uint32_t c_ONE = 1;
__constant__ uint32_t c_EIGHT = 8;
__constant__ uint32_t c_FFMASK = 0xff;
__constant__ uint32_t c_HIGHMASK = 0x80;
__constant__ uint32_t c_POLY_1B = 0x1b; // irreducible poly in GF(2^8)

__constant__ uint32_t c_MSB  = 0x18;
__constant__ uint32_t c_MLSB = 0x10;
__constant__ uint32_t c_LMSB = 0x08;

#define K2TEST 4
char KMODE[K2TEST][30] = {"Reg_test", "AntiReg_test", "Fmul_test", "FmulAntiReg_test"};

// FIPS 197 fmul in 2^8 mod poly(1b)
__device__ void fmul_gmem(uint32_t* a, uint32_t* b, uint32_t* c){

    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    c[tid] = 0;
    for(int i = 0; i < 8; i++){
        if( (b[tid] & 1) == 1) {
            c[tid] ^= b[tid];
        }
        
        if((a[tid] & 0x80) != 0){
            a[tid] = ((a[tid] << 1) ^ 0x1b) & 0xff;
        } else {
            a[tid] = (a[tid] << 1) & 0xff;
        }
        b[tid] = b[tid] >> 1;
        
    } 
}

// FIPS 197 fmul in 2^8 mod poly(1b)
__device__ void fmul_reg(uint32_t* d_a, uint32_t* d_b, uint32_t* c){

    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint32_t a = d_a[tid];
    uint32_t b  = d_b[tid];
   
    uint32_t p_i = a;
    uint32_t p_it = p_i;
    uint32_t ret = c_ZERO;
    for(int i = 0; i < 8; i++)
    {
        if((b & c_ONE) == c_ONE){
            ret = ret ^ p_i;
        }
        // p_i = xtime(p_i);
            p_it = p_i << c_ONE;
            if((p_i & c_HIGHMASK) != c_ZERO){
               p_it = p_it^c_POLY_1B;
            }
            p_i = p_it & c_FFMASK;
     
        b = b >> c_ONE;
    }
    c[tid] = ret;
}

__global__ void initData(uint32_t* a, uint32_t *b){
    
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    a[tid] = (tid & c_FFMASK);
    b[tid] = (tid & c_FFMASK);
}
// use registers to speed up this calc i/e copy gmem to reg than reg to gmem
__device__ void regTest(uint32_t* a, uint32_t *b, uint32_t *c)
{
    const int tid = ((blockIdx.x * blockDim.x) + threadIdx.x);

    uint32_t ra = a[tid];
    uint32_t rb = b[tid];
    uint32_t add = ((uint32_t) (ra + rb));
    uint32_t sub = ((uint32_t) (ra - rb));
    uint32_t mul = ((uint32_t) (ra * rb));
    uint32_t mod =  (uint32_t) (ra % rb);
    c[tid*4+0] = add;
    c[tid*4+1] = sub;
    c[tid*4+2] = mul;
    c[tid*4+3] = mod;
}

// use as few registers as possible global data only
__device__ void antiRegTest(uint32_t* a, uint32_t *b, uint32_t *c)

{
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    c[tid*4+0] = a[tid] + b[tid];
    c[tid*4+1] = a[tid] - b[tid];
    c[tid*4+2] = a[tid] * b[tid];
    c[tid*4+3] = a[tid] % b[tid];

}

__global__ void MultKernel(uint32_t *a, uint32_t *b, uint32_t *c, int mode){

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
        case 3:
            fmul_gmem(a,b,c);
            break;
    }

}
        

void print_result(int mode, uint32_t* h_d, int N, float prememcpy, float postmemcpy){

    int tid = rand() % N;
    uint32_t ltid = tid %256;
    printf("Timing of  %s prememcpy: %f     postmemcpy: %f \n", KMODE[mode], 
            prememcpy, postmemcpy);
    if(mode == 0 || mode == 1) {
        tid *= 4; // get to c space since cspace 4x the size of tid space
        uint32_t add = h_d[tid+0];
        uint32_t sub = h_d[tid+1];
        uint32_t mul = h_d[tid+2];
        uint32_t mod = h_d[tid+3];
        printf("Tid = %d mod 256 = %d \n", tid, ltid);
        printf("    %d + %d = %d \n",ltid, ltid, add);
        printf("    %d - %d = %d \n",ltid, ltid, sub);
        printf("    %d * %d = %d \n",ltid, ltid, mul);
        printf("    %d m %d = %d \n",ltid, ltid, mod);
    } else {
        uint32_t res = h_d[tid];
        printf("Tid = %d mod 256 = %d \n", tid, ltid);
        printf("    FMUL(%02x, %02x) = %02x\n", ltid, ltid, res);
        
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

    uint32_t *h_c = (uint32_t*) malloc(4*N * sizeof(uint32_t));
    uint32_t *d_a, *d_b, *d_c;

    checkCuda( cudaMalloc(&d_a, N*sizeof(uint32_t)) );
    checkCuda( cudaMalloc(&d_b, N*sizeof(uint32_t)) );
    checkCuda( cudaMalloc(&d_c, 4*N*sizeof(uint32_t)) );

    checkCudaKernel( (initData<<<numBlocks, blockSize>>>(d_a, d_b)) );

    printf("Data setup startin runs\n");

    float d1,d2; 
    for(int i = 0; i < K2TEST; i++) {

        printf("Starting test %s\n", KMODE[i]);
        checkCuda( cudaEventRecord(s1, 0) );    
        checkCudaKernel( 
            (MultKernel<<<numBlocks, blockSize>>>(d_a, d_b, d_c, i))
        );
        checkCuda( cudaEventRecord(s2, 0) );
        checkCuda( cudaMemcpy(h_c, d_c, 4*N*sizeof(uint32_t), cudaMemcpyDeviceToHost) );    
        checkCuda( cudaEventRecord(s3, 0) );

        checkCuda( cudaEventSynchronize(s2) );
        checkCuda( cudaEventSynchronize(s3) );
        checkCuda( cudaEventElapsedTime(&d1, s1, s2) ); 
        checkCuda( cudaEventElapsedTime(&d2, s1, s3) ); 

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
