all: assignment.cu
	nvcc assignment.cu -o assignment.exe
	./assignment.exe 1048576 32   > results 
	./assignment.exe 1048576 64   >> results 
	./assignment.exe 1048576 128  >> results 
	./assignment.exe 1048576 192  >> results 
	./assignment.exe 1048576 256  >> results 
	./assignment.exe 1048576 512  >> results 
	./assignment.exe 1048576 1024 >> results 
	./report2csv.awk > report.csv
