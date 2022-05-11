
## Overview

a.out --problem_size=XXXXX --eq=5 --et=XX

We are going to launch int(eq) kernels of size int(problem_size) 
    switch(et){

    case 0:
        no dependencies

    case 1: 
        A->B->C->D

    case 2:  
        0->2->6
        1->3->5
    }

