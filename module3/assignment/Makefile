

PROG1 = prog1.cu


all: $(PROG1)
	nvcc $(PROG1) -o prog1
	echo "Test 1 fulfills reqs 1,2,3,4 and 7"
	nvprof ./prog1 1048576 32 2>&1 > results1
	nvprof ./prog1 1048576 64 2>&1 >> results1
	nvprof ./prog1 1048576 128 2>&1 >> results1
	nvprof ./prog1 1048576 192 2>&1 >> results1
	nvprof ./prog1 1048576 256 2>&1 >> results1
	nvprof ./prog1 1048576 512 2>&1 >> results1
	nvprof ./prog1 1048576 1024 2>&1 >> results
	echo "Test two for req 6,8 i.e vary threads"
	nvprof ./prog1 4096 256 2>&1 > results2
	nvprof ./prog1 8192 64 2>&1 >> results2


	
