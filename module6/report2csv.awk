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
    rt_t1[curr]=$5
    rt_t2[curr]=$7
}
/ AntiReg_test/ {
    art_t1[curr]=$5
    art_t2[curr]=$7
}
/ Fmul_test/ {
    ft_t1[curr]=$5
    ft_t2[curr]=$7
}

/ FmulAntiReg_test/ {
    aft_t1[curr]=$5
    aft_t2[curr]=$7
}



END {

    {print "workSize, blocksize, reg_test, AntiReg, Fmul, AntiRegFmul"}
    for(i = 1; i < curr+1; i++){
        {print wsize[i] "," blksize[i] "," rt_t1[i] "," art_t1[i] "," ft_t1[i] "," aft_t1[i]}
    }

}

