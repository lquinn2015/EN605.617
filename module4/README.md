### Results

Rather than a graph this time I wanted to highlight results from the profiler
and by timing both tests using runtime enviroments event system. The data is
from running all previous tests along with running the caesar cipher and the
outputs of the data was verified manually. The commited results and prov_results
are where these numbers come from but a make command will overwrite them

# Data

Pinned Mem
ExecTime:                           21.22 ms
ExecTime without allocs and frees:  11.83 ms
cudaHostMalloc: 1.707ms * 4 =       6.828 ms
cudaHostFree:   589us * 4   =       2.356 ms       

PageMem
ExecTime:                           11.15 ms
ExecTime without allocs and frees:  9.043 ms
cudMalloc: 132.98us * 4 =           .5319 ms
cudaFree:  383.33us * 4 =           1.533 ms       

#Understand Data
The first thing to take away from the data is that there was a slowdown, 
and a signifcant one at that by just using pinned data however the majority
of this slowdown comes from having to allocate host pages. This is because
cuda has to coordinate with the host operating system and allocate continuous
physical memory to use as pinned memory i.e this requires evicting many pages
however when setup this is going to be a 1 time cost. This is why pinned memory
can be better. However even when subtracting out the malloc and free costs we 
are still seeing a delta of around 2.8ms. 

Well based on results from the last assignemnt we can see by digging in deeper
to the kernels seem to be taking 7 times longer in general. This is signifcant but
explainable because the results from the pinned memory are already shared to the cpu
this takes time and thus this explains the time difference. This also highlights
a mistake I am making and will keep because its instructive. In a sense I am making
my program pay the memcpy from DtoH twice for my pinned application. Now this wouldn't
make it fast but it highlights that the speed up is there but requires me to do more
transfers than I am doing. And I don't think its by much. Around 2^24 transfers of data
would likely be faster given that I only allocate 2^20 bytes for Pinned memory. This may
or may not be doable but provides complex way to gainning performance. 

### Stretch problem
There are a lot of issues with the code but highlights are below
1. The code never reads data from the input files so it won't be encrypting anything
2. While the code leaves comments for freeing it never actually frees some things. 
    Infact it frees things that are not allocated which will likely cause issues
3. I would also like to see how the array size in bytes was calculated because I think
    that an error could happen if the key file and text file are not the same size.
4. Expanding on 3 if the size of the two arrays are not the same and something stops
    early you might be leaving part of you input not encrypted which would be really bad
    if this application was used in security. 
5. Another issue i see is that the get_time function doesn't exists. The author should
    consider using cudaEventRecord(cudaEvent_t, 0); api function instead of making one.
6. Finally no error detection is don't the cudaAPI can fail to do things and there is
    no code to detect that or bugs to instrument the code to prevent leaky errors from 
    causing critical failures. 
