#!/usr/bin/awk -f

BEGIN {
    curr=0
}

/thrust\.exe/ {
    ++curr
    wsize[curr]=sub("/-p/","",$2)
}

/basic test/ {
    basic[curr]=$5
}
/fast test/ {
    fast[curr]=$5
}

END {
    {print "workSize, slowTime, fastTime"}
    for(i = 1; i < curr+1; i++){
        {print wsize[i] "," basic[i] "," fast[i]}
    }
}
