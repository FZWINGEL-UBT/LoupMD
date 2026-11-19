# One kernel IR and a run-time compiler (NVRTC)

Status: stage 1 is built and measured on `develop` (S143, 19 November 2026). It is a tie, so `LOUP_JIT` stays off by
default. This note is the plan. The ledger (`gen-jit-layer0`) and `STATUS_AND_PROFILE.md` hold the measurements.

## Stage 1 result (5060 Ti)

- **Speed.** Compiling layer 0's constants in shrinks the static SASS by 5-37% at the same registers but buys no
  time: `first_bwd_edge` +0.2% / +1.8% / -2.4% and `first_fwd` -0.5% / -1.9% / 0% on OMAT-0 generic, OFF23-S and
  MH-1. The edge backward is bound by its load/store pipe, and S142's `J_FIX = 65` kernel had already folded the one
  constant that mattered.
- **The grid.** Compiling the spline grid in as literals is slower (+2.5%). Seven bands become select chains, where
  the argument costs one indexed constant load per field. So the IR does not bake the grid.
- **Correctness.** Forces agree with the built-in kernels to within the order of the floating-point atomics. The 3x
  gate passes with `LOUP_JIT=1` on five families.

What follows for stages 2 and 3:

- A constant is worth compiling in only where it removes work on the critical path, not just instructions.
- The conv kernels' `C = 0` templates (widths without an AOT instance) are the first stage-2 candidate. The first
  step there is to measure, on one model, what the `C = 128` instance gains over the `C = 0` template.
- Stage 3 (products without `nvcc`) is about coverage, not speed.

## Why

Today a kernel's shape is fixed when the library is built:

- **Generic path.** `python/gen_generic.py` emits template kernels for every model in `/home/fzwin/models`
  (products by signature, conv kernels by `(lx, C)`). `generic.cu` then picks one of a few instances per layer
  from tables written by hand.
- **Specialised path.** `gen_kernels.py` emits OMAT-0 medium's kernels for one shape only.
- **Release.** A model outside the build set needs `loupmd build-kernels`, which runs `nvcc`.

Everything that is known only at load (the table width `J`, species count `S`, interval count, spline grid,
pair symmetry, `sh_perm`, which layers are pruned) reaches the kernels as run-time arguments. S142's `J_FIX = 65`
measured what baking one of them in is worth: 1,984 → 1,568 static instructions and 424 → 375 µs per force call
for `first_bwd_edge` (OMAT-0 generic SrTiO3, 5060 Ti). Each such constant has so far needed its own template row
and dispatch case.

## Shape of the design

1. **IR.** At load, `GenericModel` lowers each layer to a typed record holding every constant a kernel could bake
   in (`src/kernel_ir.hpp`):
   - widths and degrees;
   - the spline grid;
   - pair symmetry and the species count;
   - ZBL and stress flags;
   - pruning;
   - later, the factored product program.

   One record type serves every path. A kernel family reads only its own fields.
2. **Kernel sources are compiled into the library as text.** Each kernel lives in one header that is used twice:
   - `nvcc` compiles it ahead of time as today (the AOT path, kept as the fallback);
   - the library also embeds it as a string (Makefile rule `jit_src_*.inc`).

   A constant is baked in through `LOUP_BAKE_*` macros. With the macro undefined, the value is the kernel argument,
   as now. With it defined (by the JIT prelude), the value is a `constexpr`. There is one source of truth and no
   second copy to keep in step.
3. **JIT runtime (`src/jit.cu`).**
   - Loading:
     - NVRTC is loaded with `dlopen` (`libnvrtc.so.13`, `.12`, or `LOUP_NVRTC`), so the library has no new link
       dependency.
     - The cubin is loaded with `cudaLibraryLoadData`.
     - Kernels are fetched with `cudaLibraryGetKernel` and launched with `cudaLaunchKernel`. That works inside graph
       capture, and the kernel keeps the AOT kernel's parameter list.
   - Disk cache: `<LOUP_CACHE_DIR or ~/.cache/loupmd>/jit/<key>.cubin` plus the lowered names. The key hashes the
     source, headers, options, NVRTC version and target architecture.
   - Failure handling: if NVRTC is missing or a compile fails, the library warns once and keeps the AOT kernel.
4. **Switch.** `LOUP_JIT=1` turns the JIT kernels on and `LOUP_JIT=0` forces AOT. During stage 1 the default is
   off. A family becomes default-on only after the 3x gate and an `a b b a` MD A/B on the same library and plans
   (`LOUP_JIT=0` vs `1`).

## Stages

| stage | scope | done when |
|---|---|---|
| 1 (built, S143) | JIT runtime and disk cache; IR for layer 0; `first_bwd_edge` and `first_fwd` with `S`, `I`, `J`, `Jt`, `K0`, `sh_perm` and `pair_sym` baked in (the grid: measured, rejected) | forces at the cross-process level, gates, nsys A/B and tests: done; a tie, so off by default |
| 2 | conv forward and backward kernels from the embedded generator output, with `C`, `S`, `I`, the grid and pruning baked in; the hand-written dispatch tables (`LOUP_LAUNCH_*`) replaced by one IR → kernel lookup | gate on every generic family, A/B |
| 3 | products: the bundle carries the factored program (`export_generic.py`), and a C++ emitter or the embedded Python emitter's output builds the kernel at load. A model outside the build set then runs without `nvcc` (replaces `build-kernels`' `nvcc` step) | a model absent from `/home/fzwin/models` passes the gate |
| 4 | epilogue fusion chosen from the IR (gate, readout, density into the GEMM epilogues), which today are fixed per kind | A/B per family |
| 5 | the specialised OMAT-0 path as one lowering of the same IR | see below |

## What this does not change

- Pointers stay run-time arguments. Baked pointers freed no registers (idea 29's census), and they would force
  recompiles on every reallocation and graph re-record.
- Accuracy: the JIT kernel runs the same arithmetic in the same order. Bitwise checks are not possible across
  processes, because the floating-point atomics differ from run to run. Stage 1 therefore compared against two
  built-in runs, with conv tuning off: built-in against JIT has to stay within built-in against built-in.

## Open point for the owner

The owner's earlier decision keeps the specialised OMAT-0 kernels built and gated. Stage 5 would retire them in
favour of the IR, so it needs the owner's decision first. Stages 1 to 4 keep both paths.
