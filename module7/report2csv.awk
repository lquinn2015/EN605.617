#!/usr/bin/awk -f

BEGIN {
    curr=0;
}

/problem of size / {
    ++curr
    wsize[curr]=$4
    blksize[curr]=$7
}

/Sync kernels / {
    sync[curr]=$4
}

/Stream kernels / {
    stream[curr]=$4
}

END {

    {print "workSize, blocksize, reg_test, AntiReg, Fmul, AntiRegFmul"}
    for(i = 1; i < curr+1; i++){
        {print wsize[i] "," blksize[i] "," sync[i] "," stream[i]}
    }

}

