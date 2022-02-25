
SOURCE = ../utils/cuda_utils.cu assignment.cu 
INCLUDES = -I ../utils/

all: $(SOURCE)
	nvcc $(INCLUDES) $(SOURCE) -o assignment.exe
	./assignment.exe
