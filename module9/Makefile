
UTIL_SOURCE = ../utils/cuda_utils.cu 
INCLUDES = -I../utils/ -lcuda -lcudart -lcufft 

TSRC = thrust.cu
GSRC = 

a1 = thrust.exe
a2 = nvgraph.exe


all: $(a1) $(a2)
	echo "$(a1) and $(a2) tests have run"

$(a1): $(UTIL_SOURCE) $(TSRC)
	nvcc $(INCLUDES) $(UTIL_SOURCE) $(TSRC) -ccbin g++ -o $(a1)
	./$(a1) -p1024    > thrust.out
	./$(a1) -p2048	 >> thrust.out
	./$(a1) -p16384  >> thrust.out
	./$(a1) -p65536  >> thrust.out
	./$(a1) -p131072 >> thrust.out
	./$(a1) -p262144 >> thrust.out
	./$(a1) -p524288 >> thrust.out
	t2csv.awk thrust.out > thrust.csv
	

$(a2): $(UTIL_SOURCE) $(GSRC)
	echo "Skip for now"

clean: 
	rm $(a1) $(a2)

