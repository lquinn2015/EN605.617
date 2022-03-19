
SOURCE = ../utils/cuda_utils.cu assignment.cu 
INCLUDES = -I../utils/ -lcuda -lcudart -lcufft

STRETCH = module7_stretch_problem.cu

all: $(SOURCE)
	nvcc $(INCLUDES) $(SOURCE) -DD$(PLOT) -D$(DEBUG) -o assignment.exe 
	./assignment.exe
