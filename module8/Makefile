
SOURCE = ../utils/cuda_utils.cu assignment.cu 
INCLUDES = -I ../utils/ -lcufft -lcudart 

STRETCH = module7_stretch_problem.cu

all: $(SOURCE)
	nvcc $(INCLUDES) $(SOURCE) -o assignment.exe 
