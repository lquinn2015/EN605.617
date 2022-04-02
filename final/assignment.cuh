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

/*
    This function finds the Max Magnitutde in the arr and inserts it in db[n]
        
    Constaints:
        db must be n+2 in size as d[n] = max  d[n+1] = arrya lock
        Will assert if __FIND_MAX_CACHESIZE__ < block size

*/
__global__ void findMaxMag(int n, cuFloatComplex *arr, float *db);


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
