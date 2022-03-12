#!/usr/bin/awk -f

BEGIN {
    curr=0;
}

/problem of size / {
    ++curr
    wsize[curr]=$5
    blksize[curr]=$8
}

/Sync kernels / {
    sync[curr]=$5
}

/Stream kernels / {
    stream[curr]=$5
}

END {

    {print "workSize, blocksize, reg_test, AntiReg, Fmul, AntiRegFmul"}
    for(i = 1; i < curr+1; i++){
        {print wsize[i] "," blksize[i] "," sync[i] "," stream[i]}
    }

}

