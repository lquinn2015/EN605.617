#include "assignment.cuh" // important globals are defined here read it

__global__ void testKern(){
    printf("hello from test kern\n");
}

__global__ void findMaxMag(int n, cuFloatComplex *arr,  float *db)
{
    //assert(c_FIND_MAX_CACHESIZE >= blockDim.x);
    __shared__ float cache[c_FIND_MAX_CACHESIZE];

    unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx == 0) printf("Hello\n"); 
    unsigned stride = gridDim.x * blockDim.x;
    unsigned offset = 0;
    if(threadIdx.x == 0) {
        printf("My stride is %d and n is %d\n", stride, n);
    }
    
    float *max = &db[n]; // db has a max at n
    int* mutex = (int*) &db[n+1]; // and lock at 0;
    
    float tmp = -1.0;
    while(idx + offset < n){
        tmp = fmaxf(tmp, cuCabsf(arr[idx+offset]));
        offset += stride;
    }
    cache[threadIdx.x] = tmp;
    __syncthreads();

    //reduce in the block
    unsigned int i = blockDim.x/2; 
    while(i != 0){
        if(threadIdx.x < i){
            cache[threadIdx.x] = fmaxf(cache[threadIdx.x], cache[threadIdx.x+i]);
        }
        __syncthreads();
        i /= 2;
    }
    // reduce among all blocks
    if(threadIdx.x == 0){
        while(atomicCAS(mutex, 0, 1) != 0); // lock
        *max =fmaxf(*max, cache[0]);
        atomicExch(mutex, 0); // unlock
    }
}

__global__ void fft2amp(int n, cuFloatComplex *fft, float *db)
{
    float dbMax = db[n];
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned stride = gridDim.x * blockDim.x;
    if(threadIdx.x == 0) {
        printf("My stride is %d\n", stride);
    }
    int idx = tid;
    while( idx < n){
        db[idx] = c_dBAdjustment * log10(cuCabsf(fft[idx])/dbMax) ;
        idx += stride; 
    }
}

cuFloatComplex* readData(int *n, double *&idata, double *&qdata)
{
    // IQ data from FMcapture1.dat
    FILE* f = fopen("FMcapture1.dat", "r");
    fseek(f, 0, SEEK_END);
    int samples = ftell(f) / 2; // IQ samples are 8 bit unsigned  values
    rewind(f);
    unsigned char* data = (unsigned char*) malloc(2*samples * sizeof(char));
    idata  = (double*) malloc(samples * sizeof(double));
    qdata  = (double*) malloc(samples * sizeof(double));
    fread(data, 1, samples*2, f);
    cuFloatComplex *z = (cuFloatComplex *) malloc(sizeof(cuFloatComplex) * samples);
    
    for(int i = 0; i < samples; i++){
        z[i] = make_cuFloatComplex( (float)data[2*i] - 127.0, (float)data[2*i+1] -127.0 );
        idata[i] = data[2*i] - 127.0;
        qdata[i] = data[2*i+1] - 127.0;
    }
    free(data); 
    fclose(f);
    *n = samples;
    return z;
}

void plot_xy_data(double* x, double *y, int n)
{
    fprintf(gnuplot, "set term wxt %d size 500,500\n", cplot++ );
    fprintf(gnuplot, "plot '-' \n");

    for(int i = 0; i < n; i++){
        fprintf(gnuplot,"%lf, %lf\n", x[i], y[i]);
    }
    fprintf(gnuplot, "e\n");
}

void plotfft(float f_c, float f_s, int n, float* db){

    float Fc_Mhz = f_c / 1e6; // div by 10^6 to shift to mhz units
    float Fs_Mhz = f_s / 1e6;

    float lowF = Fc_Mhz - Fs_Mhz/2; 
    float highF = Fc_Mhz + Fs_Mhz/2;

    fprintf(gnuplot, "set xtics ('%.1f' 1, '%.1f' %d, '%.1f' %d)\n", lowF, Fc_Mhz, n/2, highF, n-1);
    fprintf(gnuplot, "plot '-' smooth frequency with linespoints lt -1 notitle\n");
    for(int i = 0; i < n; i++){
        fprintf(gnuplot,"%d  %f\n", i, db[i]);
    }
    fprintf(gnuplot, "e\n");
    fflush(gnuplot);

}


    
void create_fft(cuFloatComplex *z, int n, int offset, cudaStream_t s,
    float f_c, // freqency center 
    float f_s  // sample rate 
){
    
    printf("Starting FFT\n");
    cufftComplex *d_sig, *d_fft;
    float * d_db; 
    
    // setup data
    checkCuda( cudaMalloc((void**)&d_sig, sizeof(cufftComplex) * n) );
    checkCuda( cudaMalloc((void**)&d_fft, sizeof(cufftComplex) * n) );
    checkCuda( cudaMalloc((void**)&d_db, sizeof(float) * n + 2) ); // lock and max space
    checkCuda( cudaMemsetAsync(d_db, 0, sizeof(float) * n +2, s) );
    checkCuda( cudaMemcpyAsync(d_sig, &z[offset], n*sizeof(cufftComplex), cudaMemcpyHostToDevice, s) );

    // setup FFT
    printf("Running FFT \n");
    cufftHandle plan;
    checkCufft( cufftPlan1d(&plan, n, CUFFT_C2C, 1) ); // issuing 1 FFT of the size sample
    checkCufft( cufftSetStream(plan, s) );
    checkCufft( cufftExecC2C(plan, d_sig, d_fft, CUFFT_FORWARD) ); // execute the plan
    checkCuda( cudaStreamSynchronize(s) );
    checkCuda( cudaDeviceSynchronize() ); // this is required?

    testKern<<<1,1,0,s>>>();
    checkCuda( cudaStreamSynchronize(s) );

    // we have a FFT we need to normalize the db data so it makes sense
    checkCudaKernel( (findMaxMag<<<2,1024, 0, s>>>(n, d_fft, d_db)) );
    checkCudaKernel( (fft2amp<<<1, 1024, 0, s>>>(n, d_fft, d_db)) );
    float * db = (float*) malloc(n*sizeof(float) + 2); 

    // db is display as  0,1,2..Fs/2 -Fs/2 ... -3 -2. -1 reorder it 
    checkCuda( cudaMemcpyAsync(db, &d_db[n/2], n/2*sizeof(float),cudaMemcpyDeviceToHost,s) );
    checkCuda( cudaMemcpyAsync(&db[n/2], d_db, n/2*sizeof(float),cudaMemcpyDeviceToHost,s) );
    checkCuda( cudaStreamSynchronize(s) );

    // plot and release results
    printf("plotting fft\n");
    plotfft(f_c,f_s, n, db);

    printf("Free data\n");
    checkCufft( cufftDestroy(plan) );
    checkCuda( cudaFree(d_sig) );
    checkCuda( cudaFree(d_fft) );
    free(db);
}

//////////////// cuRand Section ////////////////////////////////////////////////


__global__ void kern_gen_noise(cuFloatComplex* z, int n, int seed)
{
    const unsigned long tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    curandState_t state;
    curand_init(seed, 0, tid*2, &state);
    
    unsigned int idx = tid; 
    unsigned int stride = gridDim.x*blockDim.x; // #blocks * blockSize
    while(idx < n) {
        unsigned char i = curand(&state) % 127;
        unsigned char q = curand(&state) % 127;
        z[tid] = make_cuFloatComplex(i,q);
        idx += stride;
    }
}

cuFloatComplex* genNoise(cudaStream_t s, int n)
{
    cuFloatComplex *z, *d_z;
    z = (cuFloatComplex *) malloc(sizeof(cuFloatComplex)*n); 
    checkCuda( cudaMalloc((void**)&d_z, sizeof(cuFloatComplex)*n) );
    
    int blocks = n / 1024 + 1;  // n < 1024 than blocks  = 0 so add    
 
    checkCudaKernel( (kern_gen_noise<<<blocks, 1024, 0, s>>>(d_z, n, (int)time(NULL))) );
    checkCuda( cudaMemcpyAsync(z, d_z, n*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost,s) );
    checkCuda( cudaStreamSynchronize(s) );

    checkCuda( cudaFree(d_z) );
    return z;
}

//////////////// cuRand Section end ////////////////////////////////////////////

int main(int argc, char** argv)
{
    int n;
    double *idata, *qdata;
    cuFloatComplex *z = readData(&n, idata, qdata); // we have n complex numbers now
    free(idata); free(qdata); // unused

    #ifdef DPLOT
    gnuplot = popen("gnuplot -persistent", "w");
    #else
    gnuplot = fopen("gplot", "w"); // with live ploting off write theplots to a file
    #endif

    cudaStream_t s;
    checkCuda( cudaStreamCreate(&s) );
    // FFT from actual data
    printf("Calculating fft of normal IQ dat\n");
    create_fft(z, 5000, 0, s, 100.122e6, 2.5e6);
    free(z);

    // FFT from random noise
    printf("Gen noise\n");
    cuFloatComplex *noise = genNoise(s, 5000);
    printf("Calculating fft of noise IQ dat\n");
    for(int i = 5000-10; i <5000; i++){
        printf("z = %f i*%f\n",  cuCrealf(noise[i]), cuCimagf(noise[i]));
    }
    create_fft(noise, 5000, 0, s, 100.122e6, 2.5e6); 
    
    free(noise);
    checkCuda( cudaStreamDestroy(s) );
}
