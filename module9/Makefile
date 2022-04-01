
UTIL_SOURCE = ../utils/cuda_utils.cu 
INCLUDES = -I../utils/ -I../common/inc/ -lcuda -lcudart -lcufft -lnvgraph
NPP_INCLUDES = -L./common/FreeImage/lib/ -L./common/FreeImage/lib/linux -L./common/FreeImage/lib/linux/x86_64 -lnppi_static -lnppc_static -lculibos -lfreeimage -I./common/UtilNPP -I./common/FreeImage/include/


TSRC = thrust.cu
GSRC = graph.cu
NSRC = npp.cu

r1 = thrust
a1 = thrust.exe
r2 = nvgraph
a2 = nvgraph.exe
r3 = npp
a3 = npp.exe

all: $(r3) $(r2) $(r1) 
	echo "$(a1) and $(a2) and $(a3) tests have run"

$(r3): $(UTIL_SOURC) $(NSRC)
	nvcc $(INCLUDES) $(NPP_INCLUDES) $(UTIL_SOURCE) $(NSRC) -ccbin g++ -o $(a3)
	./$(a3) > npp.out

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
	nvcc $(INCLUDES) $(UTIL_SOURCE) $(GSRC) -ccbin g++ -o $(a2)
	./$(a2) > graph.out

clean: 
	rm $(a1) $(a2) $(a3)

