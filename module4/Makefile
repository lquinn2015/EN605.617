all: assignment.cu
	nvcc assignment.cu -o assignment.exe
	nvprof --log_file prov_results ./assignment.exe > results
	
