#!/usr/bin/awk -f

BEGIN {
    curr=0;
}

/command: \.\/prog1 / {
    ++curr
    p1[curr]=$9
    p2[curr]=$10
}

/CUDA memcpy HtoD/ {
    h2d[curr]=$2
}

/CUDA memcpy DtoH/ {
    d2h[curr]=$2
}

/gpu_add/ {
    addt[curr]=$2
}

/gpu_mod/ {
    mod[curr]=$2
}

/gpu_mult/ {
    mult[curr]=$2

}

/gpu_sub/ {
    subt[curr]=$2
}

/gpu_mixed/ {
    cond[curr]=$2
}

END {

    {print "Program, workSize, blocksize, gpu_add, gpu_sub, gpu_mult, gpu_mod, gpu_cond, gpu_memcpy(d2h), gpu_memcpy(h2d)  "}
    for(i = 1; i < curr+1; i++){
        {print "prog1," p1[i] "," p2[i] "," addt[i] "," subt[i] "," mult[i] "," mod[i] "," cond[i] "," d2h[i] "," h2d[i]  }
    }

}

