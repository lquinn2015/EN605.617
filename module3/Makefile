

PROG1 = prog1.cu


all: $(PROG1)
	nvcc $(PROG1) -o assignment.exe
	echo "Execute program with varying arguments"
	nvprof --log-file nvprof_run_1 ./assignment.exe 1048576 32 2>&1 > results 
	nvprof --log-file nvprof_run_2 ./assignment.exe 1048576 64 2>&1 >> results
	nvprof --log-file nvprof_run_3 ./assignment.exe 1048576 128 2>&1 >> results
	nvprof --log-file nvprof_run_4 ./assignment.exe 1048576 192 2>&1 >> results
	nvprof --log-file nvprof_run_5 ./assignment.exe 1048576 256 2>&1 >> results
	nvprof --log-file nvprof_run_6 ./assignment.exe 1048576 512 2>&1 >> results
	nvprof --log-file nvprof_run_7 ./assignment.exe 1048576 1024 2>&1 >> results
	nvprof --log-file nvprof_run_8 ./assignment.exe 4096 256 2>&1 >> results
	nvprof --log-file nvprof_run_9 ./assignment.exe 8192 64 2>&1 >> results
	cat nvprof_run_* > report
	sed 's/[nmu]s//g' report > report_1
	./report2csv.awk report_1 > report.csv	
	rm nvprof_run_*	
	rm report_1
	
