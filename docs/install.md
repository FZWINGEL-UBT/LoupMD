# Install

LoupMD is a Python package with a compiled C++/CUDA library. It runs on Linux x86_64 with an NVIDIA GPU, and in
WSL2 on Windows. It needs Python 3.10 or newer. The runtime needs no PyTorch and no CUDA compiler.

```bash
pip install "loupmd[export]"
loupmd doctor
```

The first line installs the engine together with PyTorch and MACE, which export a model the first time it runs.
The second checks the GPU, the driver and the libraries. The rest of this page explains each part.

## What you need

- **Linux x86_64** with glibc 2.28 or newer: the wheel is tagged `manylinux_2_28_x86_64`. On Windows, use WSL2 with
  the NVIDIA driver installed on Windows. The package does not run on native Windows or macOS.
- **An NVIDIA GPU of compute capability 8.0 or newer** and a driver that runs CUDA 13. The wheel's library holds
  code for compute capability 8.0, 8.6, 8.9, 9.0, 10.0 and 12.0, for example the A100, the RTX 30 and 40 series,
  the H100, the B200 and the RTX 50 series.
- **Python 3.10 or newer.**

The tensor-core GEMM kernels ship for compute capability 8.x, 9.x and 12.x. On a 10.x GPU the engine runs on its
other GEMMs, somewhat slower, and `loupmd doctor` prints a warning.

The 0.0.2 wheel has been run on one GPU only, an RTX 5060 Ti (compute capability 12.0); the
[release record](https://github.com/FZWINGEL-UBT/LoupMD/blob/release/v1/docs/validation/release-0.0.2.md) in the
repository has the results. Its code for compute capability 8.0, 8.6,
8.9, 9.0 and 10.0 is compiled from the same sources and has not been run from the wheel. On any GPU,
`loupmd validate` compares a model with MACE float64 on that machine.

## Install with pip

The package is on PyPI under the name `loupmd`, and the command it installs is `loupmd`. Install it into a virtual
environment:

```bash
python3 -m venv ~/loupmd-env
source ~/loupmd-env/bin/activate
pip install "loupmd[export]"
```

`pip install loupmd` without the extra installs the engine alone. That is enough for a machine that only runs
models whose bundles are already in the cache ([the export extra](#the-export-extra) below).

The install pulls in numpy (1.26 or newer), ASE (3.23 or newer), `tomli` on Python 3.10, and on Linux the NVIDIA
CUDA libraries. An environment with the export extra takes about 6 GB, most of it PyTorch and its CUDA libraries;
the engine alone takes about 1.6 GB.

pip installs a wheel only: there is no source distribution, since the library is compiled with the CUDA compiler.
On a platform the wheel does not cover, pip answers that it finds no matching distribution.

Each version is also a [release on GitHub](https://github.com/FZWINGEL-UBT/LoupMD/releases) with the same wheel,
an archive of the sources and a `SHA256SUMS` file. To install from a downloaded file, check it and name it to pip;
the dependencies still come from a package index:

```bash
sha256sum -c --ignore-missing SHA256SUMS
pip install "./loupmd-0.0.2-py3-none-manylinux_2_28_x86_64.whl[export]"
```

### CUDA run-time libraries

The engine loads the CUDA 13 runtime, cuBLAS and cuBLASLt. They come from NVIDIA's pip wheels through one
dependency:

```text
cuda-toolkit[cublas,cudart,cufft]==13.2.2
```

The version is the one torch 2.15.0 pins, so torch (in the `export` extra, below) and the engine load the same
libraries. On Python 3.10 it is 13.0.3, the version of torch 2.14.1. You do not need a CUDA toolkit on the system.
If the libraries are not on the loader path, LoupMD loads them from the NVIDIA wheels in the same Python
environment.

cuFFT is used only by the split sum of the D3 dispersion term (`long_range="split"`, and the split candidates of
`long_range="auto"`). It is opened the first time a mesh is made. Without it the split sum is refused with the line
to install it, and `long_range="auto"` leaves the split candidates out. See [dispersion](guide/dispersion.md).

### The export extra

A model runs on the engine as a bundle (a `.loup` file), exported once from the MACE checkpoint for the elements of
your structure and kept in the cache. The export needs PyTorch and MACE:

```bash
pip install "loupmd[export]"
```

| Extra | Installs | Needed for |
|---|---|---|
| `export` | torch 2.15.0, mace-torch 0.3.17, graph_longrange 0.4.4, cuequivariance 0.12.0, cuequivariance-torch 0.12.0 | exporting a checkpoint to a bundle, the accuracy gate (`check="gate"`, `loupmd validate`), `loupmd build-kernels` |

torch 2.15 and mace-torch 0.3.17 need Python 3.11 or newer. On Python 3.10 the extra installs torch 2.14.1 and
mace-torch 0.3.16, the last versions that support it. `graph_longrange` is the package MACE's POLAR-1 models are
built on; the `polar1-*` labels cannot be exported without it.

The export runs in a separate process with the same Python interpreter, so install the extra into the environment
that runs LoupMD. Bundles already in the cache, or exported elsewhere, run without it. The extras `dev` and `docs`
are for work on LoupMD itself ([development](development.md)).

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

The cache root is `$LOUPMD_CACHE`, else `~/.cache/loupmd`. It holds `models/`, `bundles/`, `gate_refs/` and `lib/`
and the engine's GEMM plan files (`plans_*.bin`) in the root itself, and must lie outside the Python environment.
`loupmd cache list`, `size` and `prune` manage it. See [environment variables](reference/environment.md).

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

## Building it yourself

The wheel from PyPI is all a run needs. [Development](development.md) describes how to build the library from
source for one GPU, run the tests, and build the wheel and the release files.

One case needs a compiler on a user's machine: a model whose symmetric contraction the wheel's library was not
built with needs its own kernels. `loupmd build-kernels MODEL` compiles them with `nvcc` into a plugin in the
cache, which the engine then finds by itself. It needs a CUDA toolkit and the export extra; see
[kernel plugins](guide/models.md#kernel-plugins).

Next: [the quickstart](quickstart.md).
