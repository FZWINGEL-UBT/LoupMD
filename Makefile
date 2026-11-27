NVCC ?= /usr/local/cuda/bin/nvcc
CUDA_ARCH ?= sm_120
# One object directory per target architecture. Objects of another -arch link and load without complaint, and every
# kernel launch then fails with an invalid device function, which once looked like a physics bug.
OBJDIR = build/$(CUDA_ARCH)
NVCCFLAGS = -O3 -std=c++17 -arch=$(CUDA_ARCH) -Iinclude -Isrc -diag-suppress=1046 -Xcompiler -fPIC -MMD -MP $(CPPFLAGS)
LDFLAGS = -lcublas -lcublasLt -lcudart -ldl

SRCS = src/bundle.cu src/generic.cu src/ilmlip.cu src/ilmlip_lrsk_fast.cu src/generated_generic.cu src/polar.cu src/neighbors.cu src/integrator.cu src/minimize.cu src/cell.cu src/thermo.cu src/sgemm.cu src/hgemm2.cu src/tile_plan.cu src/engine.cu \
       src/jit.cu src/kernel_ir.cu src/dispersion/d3_pair.cu
OBJS = $(patsubst src/%.cu,$(OBJDIR)/%.o,$(SRCS))
DEPS = $(OBJS:.o=.d)

.PHONY: all objects clean

# `make` builds for CUDA_ARCH and installs the library at ./libloupmd.so, where the loupmd package looks for it.
# `make CUDA_ARCH=sm_89 objects` only compiles, for linking on the machine that has that GPU.
all: $(OBJDIR)/libloupmd.so
	cp $(OBJDIR)/libloupmd.so libloupmd.so

objects: $(OBJS)

# Strict teacher library: independent objects and artifact; never installs over the MD library.
.PHONY: label-lib
label-lib:
	$(MAKE) OBJDIR=build/$(CUDA_ARCH)-label CPPFLAGS="$(CPPFLAGS) -DLOUP_CONV_TABLE_FP32=1" build/$(CUDA_ARCH)-label/libloupmd.so

$(OBJDIR)/%.o: src/%.cu | $(OBJDIR)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -MF $(@:.o=.d) -c $< -o $@
$(OBJDIR):
	mkdir -p $@

# Kernel headers the library also compiles at run time (src/jit.cu, JIT_DESIGN.md): each one embedded as a raw string.
JIT_HDRS = src/kernels_common.cuh src/generated_sh16.cuh src/layer0_kernels.cuh src/generated_conv.cuh
JIT_INCS = $(patsubst src/%,$(OBJDIR)/jit_src/%.inc,$(JIT_HDRS))
$(OBJDIR)/jit_src/%.inc: src/% | $(OBJDIR)
	mkdir -p $(dir $@)
	{ printf 'R"LOUPJIT('; cat $<; printf ')LOUPJIT"\n'; } > $@
$(OBJDIR)/jit.o: $(JIT_INCS)
$(OBJDIR)/jit.o: NVCCFLAGS += -I$(OBJDIR)

$(OBJDIR)/libloupmd.so: $(OBJS)
	$(NVCC) -shared -arch=$(CUDA_ARCH) $(OBJS) $(LDFLAGS) -o $@

# The library of the wheel (PLAN.md S2-T02): every kernel for each GPU generation the release supports, in one
# file; a GPU loads the code of its own architecture. `make wheel-lib` writes build/wheel/libloupmd.so and leaves
# ./libloupmd.so alone. WHEEL_ARCHS narrows the list (for example WHEEL_ARCHS="89 120" for a quick build).
WHEEL_ARCHS ?= 80 86 89 90 100 120
ifeq ($(strip $(WHEEL_ARCHS)),)
# An empty WHEEL_ARCHS (from the environment, say) is the default list, not a build without -gencode.
override WHEEL_ARCHS = 80 86 89 90 100 120
endif
WHEEL_GENCODE = $(foreach a,$(WHEEL_ARCHS),-gencode arch=compute_$(a),code=sm_$(a))
WHEELDIR = build/wheel
WHEEL_OBJS = $(patsubst src/%.cu,$(WHEELDIR)/%.o,$(SRCS))

.PHONY: wheel-lib
wheel-lib: $(WHEELDIR)/libloupmd.so

$(WHEELDIR)/%.o: src/%.cu | $(WHEELDIR)
	@mkdir -p $(dir $@)
	$(NVCC) -O3 -std=c++17 $(WHEEL_GENCODE) -Iinclude -Isrc -I$(WHEELDIR) -diag-suppress=1046 -Xcompiler -fPIC -MMD -MP -MF $(@:.o=.d) -c $< -o $@
$(WHEELDIR):
	mkdir -p $@
# src/jit.cu embeds the kernel headers here as above (JIT_HDRS)
WHEEL_JIT_INCS = $(patsubst src/%,$(WHEELDIR)/jit_src/%.inc,$(JIT_HDRS))
$(WHEELDIR)/jit_src/%.inc: src/% | $(WHEELDIR)
	mkdir -p $(dir $@)
	{ printf 'R"LOUPJIT('; cat $<; printf ')LOUPJIT"\n'; } > $@
$(WHEELDIR)/jit.o: $(WHEEL_JIT_INCS)

$(WHEELDIR)/libloupmd.so: $(WHEEL_OBJS)
	$(NVCC) -shared $(WHEEL_OBJS) $(LDFLAGS) -o $@

# Loup-S training library (loupgym/fasttrain/cuda.py): its own objects and artifact, never part of libloupmd.so.
# src/train/strain/generated_train.cuh comes from tools/dev/gen_strain_kernels.py.
TRAIN_SRCS = src/train/strain/strain.cu
TRAIN_OBJS = $(patsubst src/%.cu,$(OBJDIR)/%.o,$(TRAIN_SRCS))
TRAIN_CC = $(patsubst sm_%,%,$(CUDA_ARCH))
TRAIN_GENCODE ?= -gencode arch=compute_$(TRAIN_CC),code=$(CUDA_ARCH)
TRAIN_LEAN ?= 0
TRAIN_DEFS = $(if $(filter 1,$(TRAIN_LEAN)),-DLST_LEAN=1,)
.PHONY: train-lib train-cli lrsk-train-lib
$(OBJDIR)/train/strain/strain.o: src/train/strain/strain.cu | $(OBJDIR)
	@mkdir -p $(dir $@)
	$(NVCC) -O3 -std=c++17 $(TRAIN_GENCODE) $(TRAIN_DEFS) -Iinclude -Isrc -Isrc/train/strain -diag-suppress=1046 -Xcompiler -fPIC -MMD -MP $(CPPFLAGS) -MF $(@:.o=.d) -c $< -o $@
train-lib: $(OBJDIR)/libloupstrain.so
$(OBJDIR)/libloupstrain.so: $(TRAIN_OBJS)
	$(NVCC) -shared $(TRAIN_GENCODE) $(TRAIN_OBJS) -lcublas -lcudart -o $@

LRSK_TRAIN_SRCS = src/train/lrsk/lrsk_train.cu src/train/lrsk/lrsk_full_train.cu
LRSK_TRAIN_OBJS = $(patsubst src/train/lrsk/%.cu,$(OBJDIR)/train/lrsk/%.o,$(LRSK_TRAIN_SRCS))
$(OBJDIR)/train/lrsk/lrsk_train.o: src/train/lrsk/lrsk_train.cu | $(OBJDIR)
	@mkdir -p $(dir $@)
	$(NVCC) -O3 -std=c++17 $(TRAIN_GENCODE) -Iinclude -Isrc -Isrc/train/lrsk -diag-suppress=1046 -Xcompiler -fPIC -MMD -MP $(CPPFLAGS) -MF $(@:.o=.d) -c $< -o $@
$(OBJDIR)/train/lrsk/lrsk_full_train.o: src/train/lrsk/lrsk_full_train.cu | $(OBJDIR)
	@mkdir -p $(dir $@)
	$(NVCC) -O3 -std=c++17 $(TRAIN_GENCODE) -Iinclude -Isrc -Isrc/train/lrsk -diag-suppress=1046 -Xcompiler -fPIC -MMD -MP $(CPPFLAGS) -MF $(@:.o=.d) -c $< -o $@
lrsk-train-lib: build/libloup_lrsk_full_train_v3.so
build/libloup_lrsk_full_train_v3.so: $(LRSK_TRAIN_OBJS)
	@mkdir -p build
	$(NVCC) -shared $(TRAIN_GENCODE) $(LRSK_TRAIN_OBJS) -lcublas -lcudart -o $@

# Full pure-CUDA foundation MACE fine-tuning library (libloupmd_ft.so): zero PyTorch!
MACE_FT_OBJS = $(OBJDIR)/train/mace_ft/mace_ft_train.o $(OBJDIR)/bundle.o $(OBJDIR)/jit.o $(OBJDIR)/polar.o $(OBJDIR)/hgemm2.o
$(OBJDIR)/train/mace_ft/mace_ft_train.o: src/train/mace_ft/mace_ft_train.cu | $(OBJDIR)
	@mkdir -p $(dir $@)
	$(NVCC) -O3 -std=c++17 $(TRAIN_GENCODE) -Iinclude -Isrc -Isrc/train/mace_ft -diag-suppress=1046 -Xcompiler -fPIC -MMD -MP $(CPPFLAGS) -MF $(@:.o=.d) -c $< -o $@
.PHONY: mace-ft-lib
mace-ft-lib: $(OBJDIR)/libloupmd_ft.so
	cp $(OBJDIR)/libloupmd_ft.so libloupmd_ft.so
$(OBJDIR)/libloupmd_ft.so: $(MACE_FT_OBJS)
	$(NVCC) -shared $(TRAIN_GENCODE) $(MACE_FT_OBJS) -lcublas -lcudart -ldl -o $@

train-cli: $(OBJDIR)/loupstrain_cli
$(OBJDIR)/loupstrain_cli: src/train/strain/loupstrain_cli.cu $(OBJDIR)/libloupstrain.so | $(OBJDIR)
	$(NVCC) -O3 -std=c++17 $(TRAIN_GENCODE) -Iinclude -Isrc -Isrc/train/strain $< -Xcompiler -pthread -Xlinker -rpath -Xlinker '$$ORIGIN' -Xlinker -L$(OBJDIR) -Xlinker -lloupstrain -lcudart -o $@

clean:
	rm -rf build libloupmd.so

-include $(DEPS) $(WHEEL_OBJS:.o=.d) $(TRAIN_OBJS:.o=.d)
