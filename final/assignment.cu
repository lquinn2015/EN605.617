#include "assignment.cuh" // important globals are defined here read it


__global__ void freqShift(int n, cuFloatComplex *S, 
     float shiftF, float intialPhase, float sampleF)
{
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;
    unsigned int idx = tid;

    float dt = shiftF / sampleF;

    while(idx < n)
    {

        float f_t = dt * (float)idx;
        cuFloatComplex shiftVec = make_cuFloatComplex(cospif(2*f_t), sinpif(2*f_t)); 
        S[idx] = cuCmulf(shiftVec, S[idx]);

        idx += stride;
    }

}

__global__ void pdsC2R(int n, cuFloatComplex *sig, float *r)
{
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;
    unsigned int idx = tid;

    while(idx < n) {
        
        if(idx+2 > n) break;
        cuFloatComplex k = sig[idx];
        cuFloatComplex k1 = sig[idx+1];
        cuFloatComplex k2 = sig[idx+2];

        float p =  (cuCrealf(k1) * ( cuCimagf(k) - cuCimagf(k2)))
                 - (cuCimagf(k1) * ( cuCrealf(k) - cuCrealf(k2)));
        //        ----------------------------------------------
        p =  p * (1 / (cuCrealf(k) * cuCrealf(k) + cuCimagf(k) * cuCimagf(k)));
        
        r[idx] = p;
        idx += stride;
    }
 
}
__global__ void decimateC2C(int n, int dec, cuFloatComplex *S, 
        cuFloatComplex *R)
{

    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int idx = tid;
    while(idx*dec < n) 
    {
        R[idx] = S[idx*dec];
        idx += stride;
    } 
}

__global__ void decimateR2R(int n, int dec, float *S, float *R)
{
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int idx = tid;
    while(idx *dec < n) 
    {
        
        R[idx] = S[idx*dec];
        idx += stride;
    } 
}

__global__ void scaleVec(int n, float *s, float *normal)
{
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int idx = tid;

    float norm = *normal;

    while(idx < n)
    {
        s[idx] = 10000 * s[idx] / norm;
        idx += stride;
    }
}

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
            if(idx-k < 0 ) continue; // ease of impl ignore lower samples
            cuFloatComplex F = S[idx-k];
            I += c_BLACKMAN_LPF_200KHz[k] * cuCrealf(F);
            Q += c_BLACKMAN_LPF_200KHz[k] * cuCimagf(F);
        }

        R[idx] = make_cuFloatComplex(I, Q);
        idx += stride; 
    }
}

__global__ void findMaxR2RMag(int n, float *arr,  float *db){
    
    __shared__ float cache[c_FIND_MAX_CACHESIZE];
    unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned stride = gridDim.x * blockDim.x;
    unsigned offset = 0;
    
    float *max = &db[n]; // db has a max at n
    int* mutex = (int*) &db[n+1]; // and lock at 0;
    
    float tmp = -1.0;
    while(idx + offset < n){
        tmp = fmaxf(tmp, abs(arr[idx+offset]));
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


__global__ void findMaxC2RMag(int n, cuFloatComplex *arr,  float *db)
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

void plot_wave(float* wave, int n)
{
    fprintf(gnuplot, "set term wxt %d size 500,500\n", cplot++ );
    fprintf(gnuplot, "plot '-' \n");
    
    for(int i = 0; i < n; i++){
        fprintf(gnuplot,"%d, %d\n", i, (int16_t)wave[i]);
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

void printSampleC(cudaStream_t s, cuFloatComplex* d_c, int n){

    cuFloatComplex *c = (cuFloatComplex *) malloc(sizeof(cuFloatComplex)* n);
    checkCuda( cudaMemcpyAsync(c, d_c, sizeof(cuFloatComplex)*n, cudaMemcpyDeviceToHost, s) );
    checkCuda( cudaStreamSynchronize(s) );
    for(int i = 0; i < n; i++){
        printf("z[%d] = %f  +  i*%f\n", i, cuCrealf(c[i]), cuCimagf(c[i]));
    }
    free(c);
}
void printSampleR(cudaStream_t s, float* d_c, int n){

    float *c = (float *) malloc(sizeof(float)* n);
    checkCuda( cudaMemcpyAsync(c, d_c, sizeof(float)*n, cudaMemcpyDeviceToHost, s) );
    checkCuda( cudaStreamSynchronize(s) );
    for(int i = 0; i < n; i++){
        printf("(d)Theta[%d] = %f \n", i, c[i]);
    }
    free(c);
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
    checkCudaKernel( (findMaxC2RMag<<<2,1024, 0, s>>>(n, d_fft, d_db)) );
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


float* fm_demod(cuFloatComplex *signal, int *n_out, float freq_drift, float freq_sr) 
{
    // setup
    int n = *n_out;
    printf("FM demodulating %d at %f sample rate\n", n, freq_sr);

    cudaStream_t s;
    checkCuda( cudaStreamCreate(&s) );
    cuFloatComplex *d_ca, *d_cb;
    float *d_ra, *d_rb;
    checkCuda( cudaMalloc((void**)&d_ca, sizeof(cuFloatComplex)*n) );
    checkCuda( cudaMalloc((void**)&d_cb, sizeof(cuFloatComplex)*n) );
    checkCuda( cudaMalloc((void**)&d_ra, sizeof(float)*n) );
    checkCuda( cudaMalloc((void**)&d_rb, sizeof(float)*n) );
    checkCuda( cudaMemcpyAsync(d_ca, &signal[0], n*sizeof(cuFloatComplex), cudaMemcpyHostToDevice,s) );
   
    
    printf("Shifting signal to baseband\n");
    // exec
    // center by removing drift
    checkCudaKernel( (freqShift<<<8,1024,0, s>>>(n, d_ca, freq_drift, 0, freq_sr)) );

    printf("Filtering at baseband 200KHz\n");
    // filter out noise
    checkCudaKernel( (blackmanFIR_200KHz<<<8,1024,0, s>>>(n, d_ca, d_cb)) );

    printf("Decimating signal\n");
    // Decimate to bandwidth = 200Khz
    int dec_rate = int(freq_sr / 2e5);
    float freq_sr_d1 = freq_sr / dec_rate;
    checkCudaKernel( (decimateC2C<<<8, 1024, 0, s>>>(n, dec_rate, d_cb, d_ca)));
    int n_d1 = n / dec_rate; // trunction keeps us in band

    printf("Signal decimated %d -> %d at rate of %d\n", n, n_d1, dec_rate);
    printf("Polar discrimnate\n");
    
    // potential plot a constellation for debug
    // polar discriminate to demoulate the signal this is a C2R operation
    checkCudaKernel( (pdsC2R<<<8, 1024, 0, s>>>(n_d1, d_ca, d_ra)) );

    printf("Convert to audio sampler rate\n");
    
    // skiping de-emphasis fitler and just decimate to audio
    dec_rate = int(freq_sr_d1/ 44100.0); //audio samples will be at ~44.1Khz
    //float freq_sr_d2 = freq_sr_d1 / dec_rate;
    checkCudaKernel( (decimateR2R<<<8, 1024, 0, s>>>(n, dec_rate, d_ra, d_rb)) );
    int n_d2 = n_d1 / dec_rate; // stay in band
    
    printf("Decimated %d -> %d at a rate of %d\n", n_d1, n_d2, dec_rate);

    // scale volume
    printf("Finding max Mag on %d samples\n", n_d2);
   
    checkCuda( cudaMemsetAsync(d_ra, 0, (n_d2+2)*sizeof(float), s) ); 
    checkCudaKernel( (findMaxR2RMag<<<8,1024, 0, s>>>(n_d2, d_rb, d_ra)) ); 
    printf("Scaling vector\n");
    //checkCudaKernel( (scaleVec<<<8, 1024, 0, s>>>(n_d2, d_rb, &d_ra[n_d2])) );
    
    printf("Copying data back to sig_out\n");
    *n_out = n_d2; // log the final samples count
    float *sig_out = (float*) malloc(n_d2 * sizeof(float));
    checkCuda( cudaMemcpyAsync(sig_out, d_rb, n_d2*sizeof(float), cudaMemcpyDeviceToHost,s) );
    printf("Cleanup\n");

    // release resources
    checkCuda( cudaStreamSynchronize(s) );
    checkCuda( cudaStreamDestroy(s) );
    checkCuda( cudaFree(d_ca) );
    checkCuda( cudaFree(d_cb) );
    checkCuda( cudaFree(d_ra) );
    checkCuda( cudaFree(d_rb) );

    return sig_out;
}

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

    // Sample Rate ends up being 44Khz by convention
    printf("Running fm_demod on signal\n");
    float *audio = fm_demod(z, &n, 0.178e6, 2.5e6); 

    FILE* ad = fopen("audio.out", "w+");
    printf("Printing audio samples 2 a file\n");
    for(int i = 0; i<n; i++){
        int16_t sample = (int16_t) audio[i];
        fwrite( &sample, sizeof(sample), 1, ad);
    }
    fclose(ad);

    plot_wave(audio, n);


    free(audio);   
}
