
## Overview

To test the problem space i focus on events that forum a Directed Acylic Graph. With
the additional restriction that kernels only depend on at most 1 event for easy of 
development in C. In order to get output out I use CL callbacks attached to events. 

I support 3 modes 
    mode 0:
        no dependencies
    mode 1: 
        A->B->C->D ... 
    mode 2:  
        0->2->6 ...
        1->3->5 ...

I also have robust inputs that support the number of kernels to fire

--help output
Usage: async.exe [OPTION...]

  -c, --callback=callback    Mode for call backs
  -n, --size=size            Problem size
  -o, --order=order          Order of operations
  -q, --queued=queued        Number of events/kernels to queue up
  -?, --help                 Give this help list
      --usage                Give a short usage message

Mandatory or optional arguments to long options are also mandatory or optional
for any corresponding short options.

