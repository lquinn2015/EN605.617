all: assignment.cl assignment.cpp
	nvcc assignment.cpp -lOpenCL -o assignment.exe
	./assignment.exe 1024
	./assignment.exe 2048
	./assignment.exe 4096
	./assignment.exe 8128
	./assignment.exe 16000
	./assignment.exe 32000
	./assignment.exe 64000
	./assignment.exe 128000
	./assignment.exe 256000

