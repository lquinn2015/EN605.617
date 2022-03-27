
UTIL_SOURCE = ../utils/cuda_utils.cu 
INCLUDES = -I../utils/ -lcuda -lcudart -lcufft

TSRC = thrust.cu

all: $(UTIL_SOURCE) $(TSRC)
	nvcc $(INCLUDES) $(UTIL_SOURCE) $(TSRC) -ccbin g++ -o thrust_assignment.exe
