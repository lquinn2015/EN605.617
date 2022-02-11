

PROG1 = prog1.cu


all: $(PROG1)
	nvcc $(PROG1) -o prog1
	echo "Execute program with varying arguments"
	nvprof --log-file nvprof_run_1 ./prog1 1048576 32 2>&1 > results 
	nvprof --log-file nvprof_run_2 ./prog1 1048576 64 2>&1 >> results
	nvprof --log-file nvprof_run_3 ./prog1 1048576 128 2>&1 >> results
	nvprof --log-file nvprof_run_4 ./prog1 1048576 192 2>&1 >> results
	nvprof --log-file nvprof_run_5 ./prog1 1048576 256 2>&1 >> results
	nvprof --log-file nvprof_run_6 ./prog1 1048576 512 2>&1 >> results
	nvprof --log-file nvprof_run_7 ./prog1 1048576 1024 2>&1 >> results
	nvprof --log-file nvprof_run_8 ./prog1 4096 256 2>&1 >> results
	nvprof --log-file nvprof_run_9 ./prog1 8192 64 2>&1 >> results
	cat nvprof_run_* > report
	sed 's/[nmu]s//g' report > report_1
	./report2csv.awk report_1 > report.csv	
	rm nvprof_run_*	
	rm report_1
	
