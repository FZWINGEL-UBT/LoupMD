NVCC ?= /usr/local/cuda/bin/nvcc
CUDA_ARCH ?= sm_120
NVCCFLAGS = -O3 -std=c++17 -arch=$(CUDA_ARCH) -Iinclude -Isrc -diag-suppress=1046 -Xcompiler -fPIC
LDFLAGS = -lcublas -lcublasLt -lcudart -ldl

SRCS = src/bundle.cu src/model_loader.cu src/generic.cu src/polar.cu src/neighbors.cu src/interaction0.cu src/integrator.cu src/sgemm.cu src/engine.cu
OBJS = $(SRCS:.cu=.o)

all: libloupmd.so

%.o: %.cu include/loupmd.h src/model.hpp src/kernels_common.cuh src/generated_kernels.cuh src/generated_kernels_signature.h src/bundle.hpp src/generic.hpp src/polar.hpp src/generated_generic.cuh
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

libloupmd.so: $(OBJS)
	$(NVCC) -shared -arch=$(CUDA_ARCH) $(OBJS) $(LDFLAGS) -o $@

clean:
	rm -f $(OBJS) libloupmd.so
