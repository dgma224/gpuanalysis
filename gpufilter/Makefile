CUDA_PATH ?= "/usr/local/cuda-9.0"

HOST_COMPILER ?= g++ -std=c++11
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

INCLUDES := -I/home/david/NVIDIA_CUDA-9.0_Samples/common/inc
LIBRARIES :=

SM ?= 50
GENCODE_FLAGS += -gencode arch=compute_$(SM),code=compute_$(SM)

LIBRARIES += -lcufft

#Building Code Here
all: build
build: gpuanalyzer

gpuanalyzer.o:gpuanalyzer.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(GENCODE_FLAGS) -o $@ -c $<

gpuanalyzer: gpuanalyzer.o
	$(EXEC) $(NVCC) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

clean: 
	rm -f gpuanalyzer gpuanalyzer.o
