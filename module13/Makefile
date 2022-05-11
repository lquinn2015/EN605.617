all: async.cpp
	nvcc async.cpp -lOpenCL -o async.exe
	./async.exe 
	./async.exe  -q 5 
	./async.exe  -q 5 -o 1
	./async.exe  -q 5 -o 2
	./async.exe  -q 6 -o 2

