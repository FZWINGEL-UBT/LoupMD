# Development

This page is for work on LoupMD itself: building the library from source, running the tests, and building the
wheel and the release files. To use LoupMD, `pip install "loupmd[export]"` is all you need ([install](install.md)).

## Get the source

```bash
git clone https://github.com/FZWINGEL-UBT/LoupMD.git
cd LoupMD
```

The checkout holds the engine (`src/`, `include/`), the Python package (`loupmd/`), the kernel generators
(`codegen/`), the tensor-core GEMM kernels (`cubins/`), the tests (`tests/`), this site (`docs/`) and the build
and release scripts (`tools/`). The generated kernel sources (`src/generated_*`) are tracked, so a build does not
run the generators.

## Build the library

The library builds with `make` in Linux or WSL2. It needs `nvcc` and cuBLAS from a CUDA 13 toolkit; the Makefile
looks for `nvcc` at `/usr/local/cuda/bin/nvcc` (set `NVCC=` otherwise). Build for your GPU's architecture:

```bash
make -j4 CUDA_ARCH=sm_120         # RTX 50 series (the default)
make -j4 CUDA_ARCH=sm_89          # RTX 40 series
make -j4 CUDA_ARCH=sm_80          # A100
```

This writes `build/<arch>/libloupmd.so` and copies it to `./libloupmd.so` at the root of the checkout. Then install
the package from the checkout, with the extras for exporting models and for the tests:

```bash
pip install -e ".[export,dev]"
loupmd doctor
```

The package uses, in this order: the library named by `LOUP_LIB`, the one inside the installed package
(`loupmd/_lib/`, filled only by the wheel build), and `libloupmd.so` at the root of the checkout. The tensor-core
kernels are read from `cubins/` beside the library. `LOUP_LIB=/path/to/libloupmd.so` runs any command on another
build, for example to compare two.

Objects of different architectures are kept apart in `build/<arch>/`: objects of another architecture link without
complaint and then fail at every kernel launch. `make clean` removes `build/` and `./libloupmd.so`. Do not rebuild
`./libloupmd.so` while a process has it loaded.

## Run the tests

```bash
pytest -q tests/cpu                       # no GPU, no library, no PyTorch: a minute or two
pytest -q tests/gpu -m fast               # the fast GPU tier, under ten minutes
pytest -q tests/gpu -m "not acceptance"   # the fast and the full tier
pytest -q tests/gpu -m acceptance         # the accuracy gate per model family, and the examples
```

`tests/cpu` runs the package on a stand-in engine. `tests/gpu` needs a GPU, `./libloupmd.so` (or `LOUP_LIB`), the
export extra and the model files in the folder that `LOUPMD_MODELS` names; a test whose model file is not there is
skipped.

`loupmd validate --model LABEL` runs the accuracy gate for one model outside the test suite
([models](guide/models.md#the-accuracy-gate)).

## Build the wheel

The wheel holds one library with code for compute capability 8.0, 8.6, 8.9, 9.0, 10.0 and 12.0. It is built in a
container, so the host needs docker and git, and neither a GPU nor a CUDA toolkit:

```bash
tools/release/build_wheel.sh            # dist/loupmd-<version>-py3-none-manylinux_2_28_x86_64.whl
pip install "dist/loupmd-<version>-py3-none-manylinux_2_28_x86_64.whl[export]"
```

The script builds `git archive` of a commit, so nothing uncommitted is in the wheel. The build image (NVIDIA's
CUDA 13.3 development image on Rocky Linux 8, about 12 GB) is made on first use from
`tools/release/Dockerfile.wheel`. Rocky Linux 8 has glibc 2.28, which is what makes the library a
`manylinux_2_28` one.

| Variable | Meaning | Default |
|---|---|---|
| `LOUPMD_WHEEL_REF` | the commit to build | `HEAD` |
| `LOUPMD_WHEEL_ARCHS` | the compute capabilities, for a quick build (for example `"89 120"`) | all six |
| `LOUPMD_WHEEL_IMAGE` | the name of the build image | `loupmd-wheel-build:cuda13.3-rocky8` |
| `LOUPMD_WHEEL_JOBS` | the number of `make` jobs in the container | 4 |
| `DOCKER` | the docker command | `docker` |

Inside the container the build checks its own result: that the library holds code for every target, that it asks
for no newer glibc or libstdc++ symbols than the tag allows, and that the wheel holds every file of the package
and nothing else. The CUDA libraries are not in the wheel. They come from NVIDIA's pip wheels, which the package
pins ([install](install.md#cuda-run-time-libraries)).

`make wheel-lib` builds only the wheel's library, into `build/wheel/libloupmd.so`, on a machine with a CUDA
toolkit. `WHEEL_ARCHS="89 120"` narrows it.

## Make a release

The version is written in two places that must agree: `__version__` in `loupmd/__init__.py` and `LOUP_VERSION` in
`include/loupmd.h`. `CHANGELOG.md` has a section per version.

```bash
tools/release/make_release.sh            # the release files of HEAD
tools/release/make_release.sh v0.0.1     # of a tag or a commit
```

This writes `dist/release-<version>-<commit>/` with the wheel, a source tarball (`git archive` of the commit) and
`SHA256SUMS`, and runs `twine check --strict` on the wheel. It never uploads. There is no source distribution: an
sdist could not build the CUDA library on a user's machine, so the wheel is the package.

Two builds of one commit give libraries of different bytes; the build is not reproducible bit for bit. Test the
wheel you upload: install it into a new environment and run `loupmd doctor`, `loupmd bench --quick` and
`loupmd validate`.

## Build this site

```bash
pip install -r tools/docs/requirements.txt
mkdocs build --strict        # or: mkdocs serve
```

`docs/reference/cli.md` is generated from the command-line parser. After a change to a command or a flag, write it
again; the CPU tests fail when the page is out of date:

```bash
python tools/dev/gen_cli_reference.py
```

The Python reference is read from the docstrings of `loupmd/`. A push to the release branch that touches `docs/`,
`mkdocs.yml` or `loupmd/` builds the site and publishes it (`.github/workflows/docs.yml`).
