# Install

LoupMD is a Python package with a compiled C++/CUDA library. It runs on Linux x86_64 with an NVIDIA GPU, and in
WSL2 on Windows. It needs Python 3.10 or newer. The runtime needs no PyTorch and no CUDA compiler.

## What you need

- **Linux x86_64** with glibc 2.28 or newer: the wheel is tagged `manylinux_2_28_x86_64`. On Windows, use WSL2 with
  the NVIDIA driver installed on Windows. The package does not run on native Windows or macOS.
- **An NVIDIA GPU of compute capability 8.0 or newer** and a driver that runs CUDA 13. The wheel's library holds
  code for compute capability 8.0, 8.6, 8.9, 9.0, 10.0 and 12.0, for example the A100, the RTX 30 and 40 series,
  the H100, the B200 and the RTX 50 series.
- **Python 3.10 or newer.**

The tensor-core GEMM kernels ship for compute capability 8.x, 9.x and 12.x. On a 10.x GPU the engine runs on its
other GEMMs, somewhat slower, and `loupmd doctor` prints a warning.

The 0.1.0a1 wheel has been run on one GPU only, an RTX 5060 Ti (compute capability 12.0):
[the release verification](validation/release-0.1.0.md) has the results. Its code for compute capability 8.0, 8.6,
8.9, 9.0 and 10.0 is compiled from the same sources and has not been run from the wheel. On any GPU,
`loupmd validate` compares a model with MACE float64 on that machine.

## Install the wheel

The release wheel is built from a commit of this repository with `tools/release/build_wheel.sh` (it needs docker
and git, no GPU). The wheel lands in `dist/`:

```bash
tools/release/build_wheel.sh            # dist/loupmd-0.1.0a1-py3-none-manylinux_2_28_x86_64.whl
pip install dist/loupmd-0.1.0a1-py3-none-manylinux_2_28_x86_64.whl
```

A PyPI release is not published yet. The package name is `loupmd` and the command it installs is `loupmd`.

The install pulls in numpy (1.26 or newer), ASE (3.23 or newer), `tomli` on Python 3.10, and on Linux the NVIDIA
CUDA libraries.

### CUDA run-time libraries

The engine loads the CUDA 13 runtime, cuBLAS and cuBLASLt. They come from NVIDIA's pip wheels through one
dependency:

```text
cuda-toolkit[cublas,cudart,cufft]==13.0.3
```

The version is the one torch 2.14.1+cu130 pins, so torch (in the `export` extra, below) and the engine load the
same libraries. You do not need a CUDA toolkit on the system. If the libraries are not on the loader path, LoupMD
loads them from the NVIDIA wheels in the same Python environment.

cuFFT is used only by the split sum of the D3 dispersion term (`long_range="split"`, and the split candidates of
`long_range="auto"`). It is opened the first time a mesh is made. Without it the split sum is refused with the line
to install it, and `long_range="auto"` leaves the split candidates out. See [dispersion](guide/dispersion.md).

### The export extra

A model runs on the engine as a bundle (a `.loup` file), exported once from the MACE checkpoint for the elements of
your structure and kept in the cache. The export needs PyTorch and MACE:

```bash
pip install "dist/loupmd-0.1.0a1-py3-none-manylinux_2_28_x86_64.whl[export]"
```

| Extra | Installs | Needed for |
|---|---|---|
| `export` | torch 2.14.1, mace-torch 0.3.16, cuequivariance 0.12.0, cuequivariance-torch 0.12.0 | exporting a checkpoint to a bundle, `loupmd validate`, `loupmd build-kernels` |
| `dev` | pytest, build | running the tests |
| `docs` | mkdocs-material | building this site |

The export runs in a separate process with the same Python interpreter, so install the extra into the environment
that runs LoupMD. Bundles already in the cache, or exported elsewhere, run without it.

## Models

LoupMD runs the released MACE foundation models by label, for example `omat0-medium`, `mpa0-medium`, `mh1-omat`,
`off23-medium` or `polar1-s`. `loupmd models` lists every label with its file, head, training functional and
licence. [Models](guide/models.md) describes the families, and [the label table](reference/models.md) lists them.

A label's checkpoint file is looked for in this order:

1. the path given with `--checkpoint` (Python: `checkpoint=`);
2. the folder named by `LOUPMD_MODELS`;
3. `<cache root>/models/`, where downloads go.

Nothing is downloaded unless you ask. Fetch a file once with `loupmd models --download LABEL`, or pass `--download`
(Python: `download=True`) to a run:

```bash
loupmd models --download omat0-medium
```

A download is checked against the SHA-256 in the label table and is not kept if it differs. A MACE checkpoint is a
pickle and runs code when it is loaded, so a file outside the table needs `--sha256 HEX` or `--trusted`.

Many models are under the Academic Software License (non-commercial use only). The run report and `loupmd models`
name the licence of each.

The cache root is `$LOUPMD_CACHE`, else `~/.cache/loupmd`. It holds `models/`, `bundles/`, `plans/`, `gate_refs/` and
`lib/`, and must lie outside the Python environment. `loupmd cache list`, `size` and `prune` manage it. See
[environment variables](reference/environment.md).

## Check the setup

`loupmd doctor` checks the installation and changes nothing:

```bash
loupmd doctor
```

It prints one line per check, `[ ok ]`, `[warn]` or `[FAIL]`, with a fix for anything that is not ok:

- the NVIDIA GPU and driver (through `nvidia-smi`), and that each GPU has compute capability 8.0 or newer;
- the CUDA runtime libraries `libcudart.so.13`, `libcublas.so.13` and `libcublasLt.so.13`;
- `libloupmd` and its C interface version;
- the tensor-core GEMM kernels for each GPU;
- that the cache root can be written;
- the model files found, and that their SHA-256 matches the label table (`--no-hashes` skips the hashing);
- whether the export extra (torch, mace, e3nn) is importable.

The exit status is 1 if a check failed. In Python, `print(lm.runtime_info())` shows the package and library
versions, the CUDA runtime that was loaded and the GPUs.

## Build from source

The library builds with `make` in Linux or WSL2. It needs `nvcc` and cuBLAS from a CUDA toolkit; the Makefile
looks for `nvcc` at `/usr/local/cuda/bin/nvcc` (set `NVCC=` otherwise). Build for your GPU's architecture:

```bash
make -j4 CUDA_ARCH=sm_120         # RTX 50 series (the default)
make -j4 CUDA_ARCH=sm_89          # RTX 40 series
make -j4 CUDA_ARCH=sm_80          # A100
```

This writes `build/<arch>/libloupmd.so` and copies it to `./libloupmd.so` at the root of the checkout. Then install
the package from the checkout:

```bash
pip install -e ".[export]"
```

The package uses, in this order: the library named by `LOUP_LIB`, the one inside the installed package
(`loupmd/_lib/`, filled only by the wheel build), and `libloupmd.so` at the root of the checkout. The tensor-core
kernels are read from `cubins/` beside the library.

Objects of different architectures are kept apart in `build/<arch>/`: objects of another architecture link without
complaint and then fail at every kernel launch. `make clean` removes `build/` and `./libloupmd.so`.

`make wheel-lib` builds the library of the wheel, with code for all six architectures, into
`build/wheel/libloupmd.so`. `WHEEL_ARCHS="89 120"` narrows it for a quick build. `tools/release/build_wheel.sh` runs
this inside a CUDA 13.3 image on Rocky Linux 8 and packs the wheel.

A model whose symmetric contraction the library was not built with needs its own kernels. `loupmd build-kernels
MODEL` compiles them with `nvcc` into a plugin in the cache, which the engine then finds by itself. It needs a CUDA
toolkit and the export extra.

Next: [the quickstart](quickstart.md).
