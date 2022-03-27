
UTIL_SOURCE = ../utils/cuda_utils.cu 
INCLUDES = -I../utils/ -lcuda -lcudart -lcufft 

TSRC = thrust.cu
GSRC = 

r1 = thrust
a1 = thrust.exe
r2 = nvgraph
a2 = nvgraph.exe


all: $(r1) $(r2) 
	echo "$(a1) and $(a2) tests have run"

$(r1): $(UTIL_SOURCE) $(TSRC)
	nvcc $(INCLUDES) $(UTIL_SOURCE) $(TSRC) -ccbin g++ -o $(a1)
	./$(a1) -p1024    > thrust.out
	./$(a1) -p2048	 >> thrust.out
	./$(a1) -p16384  >> thrust.out
	./$(a1) -p65536  >> thrust.out
	./$(a1) -p131072 >> thrust.out
	./$(a1) -p262144 >> thrust.out
	./$(a1) -p524288 >> thrust.out
	t2csv.awk thrust.out > thrust.csv
	

$(r2): $(UTIL_SOURCE) $(GSRC)
	echo "Skip for now"

clean: 
	rm $(a1) $(a2)

