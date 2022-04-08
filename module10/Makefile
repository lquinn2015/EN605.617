all: assignment.cl assignment.cpp
	nvcc assignment.cpp -lOpenCL -o assignment.exe
	./assignment.exe 1024	>  data
	./assignment.exe 2048	>> data
	./assignment.exe 4096	>> data
	./assignment.exe 8128	>> data
	./assignment.exe 16000	>> data
	./assignment.exe 32000	>> data
	./assignment.exe 64000	>> data
	./assignment.exe 128000	>> data
	./assignment.exe 256000	>> data
	./report2csv.awk data > data.csv

