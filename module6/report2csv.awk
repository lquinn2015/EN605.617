#!/usr/bin/awk -f

BEGIN {
    curr=0;
}

/Problem Size:/ {
    ++curr
    wsize[curr]=$3
    blksize[curr]=$10
}

/ Reg_test/ {
    rt_t1[curr]=$4
    rt_t2[curr]=$6
}
/ AntiReg_test/ {
    art_t1[curr]=$4
    art_t2[curr]=$6
}
/ Fmul_test/ {
    ft_t1[curr]=$4
    ft_t2[curr]=$6
}

/ FmulAntiReg_test/ {
    aft_t1[curr]=$4
    aft_t2[curr]=$6
}



END {

    {print "workSize, blocksize, reg_test, AntiReg, Fmul, AntiRegFmul"}
    for(i = 1; i < curr+1; i++){
        {print wsize[i] "," blksize[i] "," rt_t1[i] "," art_t1[i] "," ft_t1[i] "," aft_t1[i]}
    }

}

