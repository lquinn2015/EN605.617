#pragma once

#include <stdio.h>
#include <stdlib.h>
#include "cuComplex.h"
#include "cuda_utils.cuh"
#include <time.h>

// cuFFT
#include <cufft.h>

// cuRAND
#include <curand.h>
#include <curand_kernel.h>

// our handle to gnuplot this can be a live handle if 
// DPLOT flag exists, i.e `make DPLOT=1` will plot the data OTG
static FILE* gnuplot;

#ifdef __FIND_MAX_CACHESIZE__
__constant__ const int c_FIND_MAX_CACHESIZE = __FIND_MAX_CACHE_SIZE__;
#else
__constant__ const int c_FIND_MAX_CACHESIZE = 1024;
#endif

__constant__ float c_BLACKMAN_LPF_200KHz[] = {
    0.000000000000000000, -0.000011466433343440, -0.000048159680438716, -0.000070951498745996,
    0.000000000000000000, 0.000226394896924650, 0.000570593070542985, 0.000843882357908127,
    0.000742644459879189, -0.000000000000000001, -0.001387330856962112, -0.002974017320060804,
    -0.003876072294410999, -0.003078546062261788, 0.000000000000000002, 0.004911281747189231,
    0.009897689392343489, 0.012239432700612913, 0.009302643950572093, -0.000000000000000005,
    -0.013947257358995015, -0.027649479941983943, -0.034045985830080311, -0.026173578588735643,
    0.000000000000000007, 0.043288592274982135, 0.096612134128292462, 0.148460098539443669,
    0.186178664190551207, 0.199977588313552862, 0.186178664190551235, 0.148460098539443669,
    0.096612134128292462, 0.043288592274982128, 0.000000000000000007, -0.026173578588735653,
    -0.034045985830080325, -0.027649479941983936, -0.013947257358995019, -0.000000000000000005,
    0.009302643950572094, 0.012239432700612920, 0.009897689392343489, 0.004911281747189235,
    0.000000000000000002, -0.003078546062261790, -0.003876072294411004, -0.002974017320060808,
    -0.001387330856962112, -0.000000000000000001, 0.000742644459879189, 0.000843882357908127,
    0.000570593070542984, 0.000226394896924650, 0.000000000000000000, -0.000070951498745996,
    -0.000048159680438716, -0.000011466433343440, 0.000000000000000000
}; 

#define BLACKMAN_LPF_200KHz_len (59)
__constant__ int c_BLACKMAN_LPF_200KHz_len = BLACKMAN_LPF_200KHz_len;

/*
    This function uses my blackman coefficents above and applies a FIR filter tuned for
        200Khz. We will filter the I and Q components with a linear Blackman FIR filter
        seperately. 
   
    The filter will require some time to settle but we do our best
*/
__global__ void blackmanFIR_200KHz(int n, cuFloatComplex *S, cuFloatComplex *R); 


/*
    To do a frequency shift in the frequency domain you need to multiply by a complex 
        signal in the time domain. I.e
    
       w(t) * Z(t) --> w(F) +  Z(F)
        w(t) = cos(2pi * Freq(t)) + sin(2pi * freq(t)) = e^(2pi i F_offset / Fs)
        

*/
__global__ void freqShift(int n, cuFloatComplex *S, float shiftF, float intialPhase, float sampleF);


/*
    Decimates a signal by dec_rate i.e take only every dec_rate sample
        C2C is complex to complex
        C2R is complex to real currently unimpl
        R2R is real to real
*/
__global__ void decimateC2C(int n, int dec_rate, cuFloatComplex *S, cuFloatComplex *R);
__global__ void decimateR2R(int n, int dec_rate, float *S, float *R);

/*
    a default PDS requires me to write a filter after it instead following
    https://www.embedded.com/dsp-tricks-frequency-demodulation-algorithms/#:~:text=An%20often%20used%20technique%20for,in%20Figure%2013%E2%80%9360%20below%20.

    We can skip the arctan by actually computing the derivate i.e

    theta(t) = atan( Q(t)/I(t))  let  r(t) = Q(t) / I(t) 
    
    d/dt theta(z) =  1 / ( 1 + r(t)^2) * r'(t)
            =  1/(1+Q^2/I^2) * IQ'-QI' / I^2    mul by (I^2/I^2)
            =  (IQ' - QI')  /  I^2 + Q^2
        
        dTheta(t)/dt =   (Q(k)-Q(k-2))I(k-1)      *     1/(I^2 + Q^2)
                        -(I(k)-I(k-2))Q(k-1)  
        
*/
__global__ void pdsC2R(int n, cuFloatComplex *S, float *R);

__global__ void scaleVec(int n, float* vec, float *normal);


/*
    This function finds the Max Magnitutde in the arr and inserts it in db[n]
        
    Constaints:
        db must be n+2 in size as d[n] = max  d[n+1] = arrya lock
        Will assert if __FIND_MAX_CACHESIZE__ < block size

*/
__global__ void findMaxC2RMag(int n, cuFloatComplex *arr, float *db);
__global__ void findMaxR2RMag(int n, float *arr, float *db);


__constant__ float c_dBAdjustment = 20.0;
/*
    This function takes a max from db[n] and generates a normalized db mapping
*/
__global__ void fft2amp(int n, cuFloatComplex *fft, float *db);


/*
    Reads in data from FMcapture1.dat it inserts values into a returned cuFloatComplex
        and idata and qdata

    Must free idata, qdata, and returned cuFloatComplex
*/
cuFloatComplex* readData(int *n, double *&idata, double *&qdata);


/*
    Will plot the given x,y data of size n in a new terminal
*/
static int cplot = 0;
void plot_xy_data(double* x, double *y, int n);

/*
    This creates a normalized FFT using cuda cores and outputs the n points to gnuplot
        It does the final file I/O on host side which will halt progress
    
    improvements this would be faster if knew the size of the FFT and used
        persistent memory in cuda
*/
void create_fft(cuFloatComplex *z, int n, int offset, cudaStream_t s,
    float f_c, // freqency center
    float f_s  // sample rate
);

cuFloatComplex* genNoise(cudaStream_t s, int n);



