
# Overview 

I did a pretty quick modification rather choosing to focus on my project this week. I think 
openCL lends it solve to issue kernels after doing the rather complex setup. So I pulled
out all of kernel specifics and pushed them into a Helper funciton. That function just
runs our one kernel however it can do both the original and our custom manhattan distance
convolution filter. This is possible by passing the alloc size and array members as void
pointers. 

The results were instresting clearly openCL is doing a lot of caching because on the second
run of my code the execution time dropped by over 50%. In fact the first run was such an
outlier that my average was greater than every run execept the first. It would be interesting
to know if you could preflight your kernel and cache it with a small sample set to improve
performance. 


# timing results
run[0] took 725 ms                                                     
run[1] took 338 ms                                                     
run[2] took 327 ms                                                     
run[3] took 323 ms                                                     
run[4] took 323 ms                                                     
run[5] took 326 ms                                                     
run[6] took 322 ms                                                     
run[7] took 328 ms                                                     
run[8] took 324 ms                                                     
run[9] took 323 ms                                                     
Total Exec for 10 runs was 3659 ms, with an average of 365 ms 
