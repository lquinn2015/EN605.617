all: assignment.cu
	nvcc assignment.cu -o assignment.exe
	nvprof --log-file prov_results ./assignment.exe > results
	
