NVCC ?= /usr/local/cuda/bin/nvcc
CUDA_ARCH ?= sm_120
# One object directory per target architecture. Objects of another -arch link and load without complaint, and every
# kernel launch then fails with an invalid device function, which once looked like a physics bug.
OBJDIR = build/$(CUDA_ARCH)
NVCCFLAGS = -O3 -std=c++17 -arch=$(CUDA_ARCH) -Iinclude -Isrc -diag-suppress=1046 -Xcompiler -fPIC -MMD -MP
LDFLAGS = -lcublas -lcublasLt -lcudart -ldl

SRCS = src/bundle.cu src/model_loader.cu src/generic.cu src/generated_generic.cu src/polar.cu src/neighbors.cu src/interaction0.cu src/integrator.cu src/minimize.cu src/cell.cu src/thermo.cu src/sgemm.cu src/tile_plan.cu src/engine.cu \
       src/dispersion/d3_pair.cu
OBJS = $(patsubst src/%.cu,$(OBJDIR)/%.o,$(SRCS))
DEPS = $(OBJS:.o=.d)

.PHONY: all objects clean

# `make` builds for CUDA_ARCH and installs the library at ./libloupmd.so, where the loupmd package looks for it.
# `make CUDA_ARCH=sm_89 objects` only compiles, for linking on the machine that has that GPU.
all: $(OBJDIR)/libloupmd.so
	cp $(OBJDIR)/libloupmd.so libloupmd.so

objects: $(OBJS)

$(OBJDIR)/%.o: src/%.cu | $(OBJDIR)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -MF $(@:.o=.d) -c $< -o $@
$(OBJDIR):
	mkdir -p $@

$(OBJDIR)/libloupmd.so: $(OBJS)
	$(NVCC) -shared -arch=$(CUDA_ARCH) $(OBJS) $(LDFLAGS) -o $@

# The library of the wheel (PLAN.md S2-T02): every kernel for each GPU generation the release supports, in one
# file; a GPU loads the code of its own architecture. `make wheel-lib` writes build/wheel/libloupmd.so and leaves
# ./libloupmd.so alone. WHEEL_ARCHS narrows the list (for example WHEEL_ARCHS="89 120" for a quick build).
WHEEL_ARCHS ?= 80 86 89 90 100 120
WHEEL_GENCODE = $(foreach a,$(WHEEL_ARCHS),-gencode arch=compute_$(a),code=sm_$(a))
WHEELDIR = build/wheel
WHEEL_OBJS = $(patsubst src/%.cu,$(WHEELDIR)/%.o,$(SRCS))

.PHONY: wheel-lib
wheel-lib: $(WHEELDIR)/libloupmd.so

$(WHEELDIR)/%.o: src/%.cu | $(WHEELDIR)
	@mkdir -p $(dir $@)
	$(NVCC) -O3 -std=c++17 $(WHEEL_GENCODE) -Iinclude -Isrc -diag-suppress=1046 -Xcompiler -fPIC -MMD -MP -MF $(@:.o=.d) -c $< -o $@
$(WHEELDIR):
	mkdir -p $@

$(WHEELDIR)/libloupmd.so: $(WHEEL_OBJS)
	$(NVCC) -shared $(WHEEL_OBJS) $(LDFLAGS) -o $@

clean:
	rm -rf build libloupmd.so

-include $(DEPS) $(WHEEL_OBJS:.o=.d)
