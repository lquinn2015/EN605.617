# Overview 

The goal of this project is to decode radio signalling data from a RTL-SDR. This is a brand
of Softward defined radio that is really popular. The device is capabable of doing the 
Radio front end signal capture and it is going to be our job to ingest this data and attempt
to generate a FM audio output. 

# Understand the input

We have a interesting data source from FMCapture1.dat. This file contains IQ sample data
from a RTL-SDR. IQ data which was sampled with the follow parameters

Freq center = 100.112 Mhz  = F_c
Freq sample rate = 2.5 Mhz = F_r
Number of samples = 25Million = n     the sample is 10seconds long

Thus we have 25M IQ samples which each IQ sample is 2 uint8_t they need to be converted
to signed data since the really represent the complex value of the signal at a given time
i.e 

z = I + iQ = a cos(theta) + i b sin(theta) = re^(i theta)

    Here theta is the sample's frequency at the time that number was sampled

# Theory to decoding

I'll now outline the plane to decode and the motivation for the underlying cuda kernels used

## Sample data:  X_1 = RTL_sample(f_c, f_r, n) 

This is the data the RTL-SDR will generate when we tune it. Lets give it length n 

## Mix to Baseband: X_2 = X_1 * e^(i f_d)   

I know aprori there is a FM signal at 100.3, 101.1,... MHz. In order to get the signal out
We need to center samples to f_t or our target frequency. This means we need to do a phase
shift or multiple our complex numbers by the given downshift amount. Because we are sampling
at 2.5 MHz our delta shift freq should be in MHz. 
 Thus the following parameters make sense

f_t = 100.3 ->  f_d = f_t - f_c  =  100.3 - 100.122 =  0.178 MHz 
f_t =  99.5 ->  f_d = f_t - f_c  =   99.5 - 100.122 = -0.622 Mhz
f_t = 101.1 ->  f_d = f_t - f_c  =  101.1 - 100.122 =  0.978 Mhz

These are the strongest signals in our data. This is clear if you look at a spectrogram of 
the data. 

## Low Pass Filter:  X_3 = conv(X_2, H)

This is the first math heavy step. We want to smooth the data an remove high frequency
artifacts and other stations from our output. To do this we need to run a FIR filter on our
data. Or convolute our data with a blackman window. This is equal to computing the following

X_3[n] = SUM(k=0, n=|H|, H[k] * X[n-k])

This is is a discrete sum and the H matrix is a series of coefficients for our filter. They 
are independent of the target freqency so we can treat them as constants no matter the station
we want to tune to. This is because we are always acting on the Baseband freqency we generated
with the last step. I used a reference to generate those constants. 

## Downsample  X_4 = X_3[0:n:D]

We have a lot of data because we sampled at 2.5 Mhz FM bandwidth is only around 200 Khz. Thus
at this point we can reduce the data we have by downsampling without any lost. We simply splice
the array of data and reduce it to be smaller. Notable the following are chaning f_r is now 
lower for all future calculation

D = Decimation rate = round_down(f_s /f_fmbw) =  round_down(2.5MHz/ 200 KHz) = 12
X_4 = X_3[0:n:D]

Thus we end up with 12x less data whose sample rate is at 200 KHz

## Demodulate:  X_5 = Freq_Disc(X_4)

In order to demodulate the signal we are going to use a frequency discriminator The equation
looks as follows

X_5[n] =  ComplexConjugate(X_4[n+1]) * X_4[n] 


## De-Emphasize 

This might be skipable to some extent but according to wikipedia Noise will dispropotionally
effect higher frequenies more. So FM transmitters boost there higher frequencies in order to
counter this. Thus we can boost quality by using a preemphaize filter. This is a stretch goal
for me though because its not a big boost in quality but requires figuring out how to implement
this type of filter which is different from the filter we made before. 


## Decimate  to audio   X_6 = X_5[0:n:D]

We are trying to only get the mono data which is only at 44-48 Khz, so downsample again to
get to the correct bandwidth

f_r'' = f_r' / D = f_r / 12 / D ->  D = 4.75 ... 5 

## Play X_6

X_6 is mono data we just need to play it now.   





# References 
[1] http://www.aaronscher.com/wireless_com_SDR/RTL_SDR_AM_spectrum_demod.html
    This is the source of our sample data mainly because he did this in MatLab
    but also he has pictures so it is the control to our experiment. He used a lot of
    Predefines so his work has many limitations and it runs in MatLab!

[2] https://witestlab.poly.edu/blog/capture-and-decode-fm-radio/
    This lady describes how to do this in the python world but also has helped form a 
    great block diagram for the things that need to be down in a vector like notation
    which was critical to generating the pipeline for data. In this project

[3] https://tomroelandts.com/articles/how-to-create-a-simple-low-pass-filter

    This man showed me how to generate a Blackman windows coefficents I used his tool
    to generate filters used in the code

[4] https://sestevenson.wordpress.com/implementation-of-fir-filtering-in-c-part-1/

    This man helped me understand the equations needed to generate a Filter and did a 
    basic c implementation which was important because the math involved is really difficult
    I am not an EE person but I am a Math person. If someone can tell me the general idea
    I can probably implement it. 
