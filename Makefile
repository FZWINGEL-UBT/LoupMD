NVCC ?= /usr/local/cuda/bin/nvcc
CUDA_ARCH ?= sm_120
# One object directory per target architecture. Objects of another -arch link and load without complaint, and every
# kernel launch then fails with an invalid device function, which once looked like a physics bug.
OBJDIR = build/$(CUDA_ARCH)
NVCCFLAGS = -O3 -std=c++17 -arch=$(CUDA_ARCH) -Iinclude -Isrc -diag-suppress=1046 -Xcompiler -fPIC
LDFLAGS = -lcublas -lcublasLt -lcudart -ldl

SRCS = src/bundle.cu src/model_loader.cu src/generic.cu src/polar.cu src/neighbors.cu src/interaction0.cu src/integrator.cu src/sgemm.cu src/engine.cu
OBJS = $(patsubst src/%.cu,$(OBJDIR)/%.o,$(SRCS))
HDRS = include/loupmd.h src/model.hpp src/kernels_common.cuh src/generated_kernels.cuh src/generated_kernels_signature.h src/bundle.hpp src/generic.hpp src/polar.hpp src/generated_generic.cuh

.PHONY: all objects clean

# `make` builds for CUDA_ARCH and installs the library at ./libloupmd.so, where python/loupmd.py looks for it.
# `make CUDA_ARCH=sm_89 objects` only compiles, for linking on the machine that has that GPU.
all: $(OBJDIR)/libloupmd.so
	cp $(OBJDIR)/libloupmd.so libloupmd.so

objects: $(OBJS)

$(OBJDIR)/%.o: src/%.cu $(HDRS) | $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJDIR):
	mkdir -p $@

$(OBJDIR)/libloupmd.so: $(OBJS)
	$(NVCC) -shared -arch=$(CUDA_ARCH) $(OBJS) $(LDFLAGS) -o $@

clean:
	rm -rf build libloupmd.so
