#include "assignment.cuh" // important globals are defined here read it


__global__ void phaseShift(int n, cuFloatComplex *S, float shiftF)
{
    
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;
    unsigned int idx = tid;

    cuFloatComplex shiftVec = make_cuFloatComplex(cosf(shiftF), sinf(shiftF));


    while(idx < n){

        S[idx] = cuCmulf(shiftVec, S[idx]);

        idx += stride;
    }

}

// IT MUST BE TRUE THAT S has BLACKMAN_LPF_200KHz_len samples before it. This can be filled
// with zero or really anything it doesn't matter because of long runs those values will be
// filled with valid samples 
__global__ void blackmanFIR_200KHz( int n, cuFloatComplex *S,
                                           cuFloatComplex *R)
{
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int idx = tid;

    while(idx < n)
    {
        float I = 0;
        float Q = 0; 
        for(int k = 0; k < c_BLACKMAN_LPF_200KHz_len; k++)
        {
            cuFloatComplex F = S[idx-k];
            I += c_BLACKMAN_LPF_200KHz[k] * cuCrealf(F);
            Q += c_BLACKMAN_LPF_200KHz[k] * cuCimagf(F);
        }

        R[idx] = make_cuFloatComplex(I, Q);
        idx += stride; 
    }
}

__global__ void findMaxMag(int n, cuFloatComplex *arr,  float *db)
{
    //assert(c_FIND_MAX_CACHESIZE >= blockDim.x);
    __shared__ float cache[c_FIND_MAX_CACHESIZE];

    unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned stride = gridDim.x * blockDim.x;
    unsigned offset = 0;
    
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

void plotfft(float f_c, float f_s, int n, float* db, const char* title){

    float Fc_Mhz = f_c / 1e6; // div by 10^6 to shift to mhz units
    float Fs_Mhz = f_s / 1e6;

    float lowF = Fc_Mhz - Fs_Mhz/2; 
    float highF = Fc_Mhz + Fs_Mhz/2;
    
    fprintf(gnuplot, "set term wxt %d size 500,500\n", cplot++);
    fprintf(gnuplot, "set ylabel 'loss dB'; set xlabel 'freq Mhz'; set xtics ('%.1f' 1, '%.1f' %d, '%.1f' %d)\n", lowF, Fc_Mhz, n/2, highF, n-1);
    fprintf(gnuplot, "plot '-' smooth frequency with linespoints lt -1 title '%s' \n", title);
    for(int i = 0; i < n; i++){
        fprintf(gnuplot,"%d  %f\n", i, db[i]);
    }
    fprintf(gnuplot, "e\n");
    fflush(gnuplot);

}


    
void create_fft(cuFloatComplex *z, int n, int offset, cudaStream_t s,
    float f_c, // freqency center 
    float f_s,  // sample rate 
    const char* title
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
    checkCufft( cufftDestroy(plan) ); // brick the plan after being sued

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
    plotfft(f_c,f_s, n, db, title);

    printf("Free data\n");
    checkCuda( cudaFree(d_sig) );
    checkCuda( cudaFree(d_fft) );
    checkCuda( cudaFree(d_db)  );
    free(db);
}


int main(int argc, char** argv)
{
    int n;
    double *idata, *qdata;
    cuFloatComplex *z = readData(&n, idata, qdata); // we have n complex numbers now
    free(idata); free(qdata); // unused

    for(int i = n-5; i < n; i++){
        printf("z[%d] = %f + i*%f \n", i, cuCrealf(z[i]), cuCimagf(z[i]));
    }

    #ifdef DPLOT
    gnuplot = popen("gnuplot -persistent", "w");
    #else
    gnuplot = fopen("gplot", "w"); // with live ploting off write theplots to a file
    #endif

    cudaStream_t s;
    checkCuda( cudaStreamCreate(&s) );
    
    // FFT raw data
    printf("Calculating fft of normal IQ dat\n");
    //create_fft(z, 5000, 0, s, 100.122e6, 2.5e6, "FM FFT");

    // allocate some data with filter primer space
    cuFloatComplex *d_preFilter, *d_z, *d_r;
    checkCuda( cudaMalloc((void**)&d_preFilter, sizeof(cuFloatComplex) * (n+ BLACKMAN_LPF_200KHz_len)));
    checkCuda( cudaMalloc((void**)&d_r, sizeof(cuFloatComplex)*n) );
    checkCuda( cudaMemsetAsync(d_preFilter, 0, sizeof(float) * BLACKMAN_LPF_200KHz_len, s) );
    d_z = d_preFilter+BLACKMAN_LPF_200KHz_len; // samples start after filter primers vars
    checkCuda( cudaMemcpyAsync(d_z, &z[0], n*sizeof(cuFloatComplex), cudaMemcpyHostToDevice,s) );
    
    // phase shift the data
    checkCudaKernel( (phaseShift<<<8,1024,0, s>>>(n, d_z, -0.0712)) );
    // FFT from actual data
    printf("Calculating fft of shifted IQ dat\n");
    create_fft(z, 5000, 0, s, 100.122e6, 2.5e6, "FM FFT Shift");
    for(int i = n-5; i < n; i++){
        printf("z[%d] = %f + i*%f \n", i, cuCrealf(z[i]), cuCimagf(z[i]));
    }
    
    checkCuda( cudaStreamSynchronize(s) );

    checkCudaKernel( (blackmanFIR_200KHz<<<8,1024, 0, s>>>(n, d_z, d_r)) );
    checkCuda( cudaMemcpyAsync(&z[0], d_r, n*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost,s) );
   
    checkCuda( cudaStreamSynchronize(s) );
    checkCuda( cudaFree(d_r) );
    checkCuda( cudaFree(d_preFilter) );
 
    for(int i = n-5; i < n; i++){
        printf("z[%d] = %f + i*%f \n", i, cuCrealf(z[i]), cuCimagf(z[i]));
    }
    

    // FFT from actual data
    printf("Calculating fft of shifted filtered IQ dat\n");
    create_fft(z, 5000, 0, s, 100.122e6, 2.5e6, "FM FFT Shift & filter");
    free(z);

    
    checkCuda( cudaStreamDestroy(s) );
}
