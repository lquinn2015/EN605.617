#!/usr/bin/awk -f

BEGIN {
    curr=0;
}

/Problem/{
    ++curr;
    wsize[curr]=$3
}
/seconds/{
    ctime[curr]=$3
}

END{
    {print "worksize, time"}
    for(i=1; i<curr+1; i++){
        {print wsize[i] "," ctime[i] }
    }
}
