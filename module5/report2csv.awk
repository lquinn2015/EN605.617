#!/usr/bin/awk -f

BEGIN {
    curr=0;
}

/Problem Size:/ {
    ++curr
    wsize[curr]=$10
    blksize[curr]=$3
}

/ shmem mem mode/ {
    shmem[curr]=$2
    shmemt[curr]=$6
}
/global mem mode/ {
    gmem[curr]=$2
    gmemt[curr]=$6
}
/literal/ {
    lmem[curr]=$2
    lmemt[curr]=$6
}
/constant/ {
    cmem[curr]=$2
    cmemt[curr]=$6
}
/Constant_/ {
    csmem[curr]=$2
    csmemt[curr]=$6
}


END {

    {print "workSize, blocksize, gmem, shmem, lmem, cmem, con_shmem"}
    for(i = 1; i < curr+1; i++){
        
        {print wsize[i] "," blksize[i] "," gmemt[i] "," shmemt[i] "," lmemt[i] "," cmemt[i] "," csmemt[i] }
    }

}

