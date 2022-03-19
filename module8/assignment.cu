#include <stdio.h>
#include <stdlib.h>
#include "cuComplex.h"
#include "cuda_utils.cuh"
#include <cufft.h>

// globals
static FILE* gnuplot;

__global__ void fft2amp(int n, cuFloatComplex *fft, float *db){

    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int idx = tid;
    while( idx < n){
        db[idx] = cuCabsf(fft[idx]);
        idx += tid;
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
static int cplot = 0;
void plot_xy_data(double* x, double *y, int n)
{
    fprintf(gnuplot, "set term wxt %d size 500,500\n", cplot++ );
    fprintf(gnuplot, "plot '-' \n");

    for(int i = 0; i < n; i++){
        fprintf(gnuplot,"%lf, %lf\n", x[i], y[i]);
    }
    fprintf(gnuplot, "e\n");
}

    
// user job to insure that z[offset+n] does not overboubd 
void create_fft(cuFloatComplex *z, int n, int offset, cudaStream_t s){
    
    cufftComplex *d_sig, *d_fft;
    float * d_db; 
    checkCuda( cudaMalloc((void**)&d_sig, sizeof(cufftComplex) * n) );
    checkCuda( cudaMalloc((void**)&d_fft, sizeof(cufftComplex) * n) );
    checkCuda( cudaMalloc((void**)&d_db, sizeof(float) * n) );

    checkCuda( cudaMemcpyAsync(d_sig, &z[offset], n*sizeof(cufftComplex), cudaMemcpyHostToDevice, s) );
    
    cufftHandle plan;
    checkCufft( cufftPlan1d(&plan, n, CUFFT_C2C, 1) ); // issuing 1 FFT of the size sample
    checkCufft( cufftSetStream(plan, s) );
    checkCufft( cufftExecC2C(plan, d_sig, d_fft, CUFFT_FORWARD) ); // execute the plan

    // we have a FFT we need to extract and plot the amplitude of it now
    checkCudaKernel( (fft2amp<<<1, 1024, 0, s>>>(n, d_fft, d_db)) );
    float * db = (float*) malloc(n*sizeof(float)); 
    checkCuda( cudaMemcpyAsync(db, d_db, n*sizeof(float), cudaMemcpyDeviceToHost, s) );
    

    checkCuda( cudaStreamSynchronize(s) );

    fprintf(gnuplot, "plot '-' smooth frequency with linespoints lt -1 notitle");
    for(int i = 0; i < n; i++){
        fprintf(gnuplot,"%d  %f\n", i, db[i]);
    }
    fprintf(gnuplot, "e\n");
    
    checkCufft( cufftDestroy(plan) );
    checkCuda( cudaFree(d_sig) );
    checkCuda( cudaFree(d_fft) );
    free(db);

}


int main(int argc, char** argv)
{
    int n;
    double *idata, *qdata;
    cuFloatComplex *z = readData(&n, idata, qdata); // we have n complex numbers now

    gnuplot = popen("gnuplot -persistent", "w");
    
       

    

}
