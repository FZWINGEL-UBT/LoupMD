NVCC ?= /usr/local/cuda/bin/nvcc
CUDA_ARCH ?= sm_120
# One object directory per target architecture. Objects of another -arch link and load without complaint, and every
# kernel launch then fails with an invalid device function, which once looked like a physics bug.
OBJDIR = build/$(CUDA_ARCH)
NVCCFLAGS = -O3 -std=c++17 -arch=$(CUDA_ARCH) -Iinclude -Isrc -diag-suppress=1046 -Xcompiler -fPIC -MMD -MP
LDFLAGS = -lcublas -lcublasLt -lcudart -ldl

SRCS = src/bundle.cu src/model_loader.cu src/generic.cu src/generated_generic.cu src/polar.cu src/neighbors.cu src/interaction0.cu src/integrator.cu src/sgemm.cu src/engine.cu src/jit.cu src/kernel_ir.cu
OBJS = $(patsubst src/%.cu,$(OBJDIR)/%.o,$(SRCS))
DEPS = $(OBJS:.o=.d)

.PHONY: all objects clean

# `make` builds for CUDA_ARCH and installs the library at ./libloupmd.so, where python/loupmd.py looks for it.
# `make CUDA_ARCH=sm_89 objects` only compiles, for linking on the machine that has that GPU.
all: $(OBJDIR)/libloupmd.so
	cp $(OBJDIR)/libloupmd.so libloupmd.so

objects: $(OBJS)

$(OBJDIR)/%.o: src/%.cu | $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -MF $(@:.o=.d) -c $< -o $@
$(OBJDIR):
	mkdir -p $@

# Kernel headers the library also compiles at run time (src/jit.cu, JIT_DESIGN.md): each one embedded as a raw string.
JIT_HDRS = src/kernels_common.cuh src/generated_sh16.cuh src/layer0_kernels.cuh
JIT_INCS = $(patsubst src/%,$(OBJDIR)/jit_src/%.inc,$(JIT_HDRS))
$(OBJDIR)/jit_src/%.inc: src/% | $(OBJDIR)
	mkdir -p $(dir $@)
	{ printf 'R"LOUPJIT('; cat $<; printf ')LOUPJIT"\n'; } > $@
$(OBJDIR)/jit.o: $(JIT_INCS)
$(OBJDIR)/jit.o: NVCCFLAGS += -I$(OBJDIR)

$(OBJDIR)/libloupmd.so: $(OBJS)
	$(NVCC) -shared -arch=$(CUDA_ARCH) $(OBJS) $(LDFLAGS) -o $@

clean:
	rm -rf build libloupmd.so

-include $(DEPS)
