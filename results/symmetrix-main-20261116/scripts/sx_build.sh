#!/bin/bash
# Build symmetrix main (b907f04, v0.3.0 + build fixes): only the _mace_foundation Python module, with the flags of
# symmetrix/source/cpp/mace_foundation/CMakeLists.txt in a Release build (no Kokkos; the foundation module needs none).
set -euo pipefail
REV=b907f04
SRC=$HOME/src/symmetrix-$REV
OUT=$HOME/src/symmetrix-eval/$REV
P=/home/fzwin/environments/mace_cueq_latest/bin/python
NVCC=/usr/local/cuda/bin/nvcc
if [ ! -d "$SRC" ]; then
  git clone -q https://github.com/wcwitt/symmetrix "$SRC"
fi
cd "$SRC"
git checkout -q "$REV"
git submodule update --init -q libsymmetrix/external/json symmetrix/external/pybind11
git log --oneline -1
PYINC=$($P -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
EXT=$($P -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
rm -rf "$OUT"
mkdir -p "$OUT/obj" "$OUT/symmetrix/cli"
F=symmetrix/source/cpp/mace_foundation
INC="-I$SRC/libsymmetrix/source -I$SRC/libsymmetrix/external/json/single_include -I$SRC/symmetrix/external/pybind11/include -I$PYINC"
CXXF="-O3 -DNDEBUG -std=c++20 -fPIC -fvisibility=hidden -march=native -fopenmp -DMACE_FOUNDATION_CUDA"
time g++ $CXXF $INC -c "$SRC/$F/bindings.cpp" -o "$OUT/obj/bindings.o"
time g++ $CXXF -ffast-math -fno-finite-math-only $INC -c "$SRC/$F/cpu.cpp" -o "$OUT/obj/cpu.o"
time $NVCC -O3 -DNDEBUG -std=c++20 --expt-relaxed-constexpr --use_fast_math -arch=sm_120 \
  -Xcompiler=-fPIC,-fvisibility=hidden -DMACE_FOUNDATION_CUDA $INC -c "$SRC/$F/cuda.cu" -o "$OUT/obj/cuda.o"
g++ -shared -fopenmp "$OUT/obj/bindings.o" "$OUT/obj/cpu.o" "$OUT/obj/cuda.o" \
  -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 -lcudart -lcublas -o "$OUT/symmetrix/_mace_foundation$EXT"
S=$SRC/symmetrix/source/symmetrix
cp "$S/experimental.py" "$S/extract_mace_data.py" "$S/extract_mace_nonlinear.py" "$S/extract_mace_polar.py" "$OUT/symmetrix/"
cp "$S/cli/extract_mace.py" "$OUT/symmetrix/cli/"
# The package __init__ imports the Kokkos module (.symmetrix), which this build leaves out.
: > "$OUT/symmetrix/__init__.py"
ls -la "$OUT/symmetrix"
sha256sum "$OUT/symmetrix/_mace_foundation$EXT"
