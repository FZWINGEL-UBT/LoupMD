"""``pyproject.toml`` and ``loupmd.runtime_info()`` (plan task S2-T01). No GPU needed."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def project():
    try:
        import tomllib as toml
    except ImportError:
        toml = pytest.importorskip("tomli")
    return toml.loads((ROOT / "pyproject.toml").read_text())


def test_name_version_and_entry_point(project):
    meta = project["project"]
    assert meta["name"] == "loupmd" and meta["dynamic"] == ["version"]
    assert project["tool"]["setuptools"]["dynamic"]["version"] == {"attr": "loupmd.__version__"}
    assert meta["scripts"] == {"loupmd": "loupmd.cli:main"}
    assert meta["requires-python"] == ">=3.10"
    assert project["build-system"]["build-backend"] == "setuptools.build_meta"


def test_the_runtime_needs_neither_torch_nor_a_compiler(project):
    base = project["project"]["dependencies"]
    names = {d.split(";")[0].split("[")[0].split("=")[0].split(">")[0].split("<")[0].strip().lower() for d in base}
    assert names == {"numpy", "ase", "tomli", "cuda-toolkit"}      # tomli: run files on Python 3.10 only
    # The CUDA 13 runtime and cuBLAS exactly as torch 2.14.1+cu130 pins them, so both resolve to the same wheels.
    assert "cuda-toolkit[cublas,cudart]==13.0.3; platform_system == 'Linux'" in base


def test_the_extras(project):
    extras = project["project"]["optional-dependencies"]
    assert set(extras) == {"export", "dev", "docs"}
    export = {d.split("=")[0].strip() for d in extras["export"]}
    assert {"torch", "mace-torch"} <= export
    assert all("==" in d for d in extras["export"]), "the export extra pins the versions it was tested with"
    assert any(d.startswith("pytest") for d in extras["dev"])


def test_licence_and_the_files_that_ship(project):
    meta = project["project"]
    assert meta["license"] == "MIT AND BSD-3-Clause"
    for name in meta["license-files"]:
        assert (ROOT / name).is_file(), name
    assert {"LICENSE", "NOTICE.md", "cubins/LICENSE.cutlass"} <= set(meta["license-files"])
    data = project["tool"]["setuptools"]["package-data"]["loupmd"]
    assert {"_lib/*.so", "_lib/cubins/*", "bench/reference.json", "bench/structures/*"} <= set(data)
    assert (ROOT / "loupmd" / "bench" / "reference.json").is_file()
    assert project["tool"]["setuptools"]["packages"]["find"]["include"] == ["loupmd*"]


def test_the_version_is_a_plain_assignment_setuptools_can_read_without_importing():
    import re

    import loupmd

    text = (ROOT / "loupmd" / "__init__.py").read_text()
    found = re.findall(r'^__version__ = "([^"]+)"', text, flags=re.M)
    assert found == [loupmd.__version__]


# ------------------------------------------------------------------------------------------- runtime_info

def test_runtime_info_names_the_gpus_and_the_library(monkeypatch):
    import loupmd
    from loupmd import _core, _gpu, runtime

    monkeypatch.setattr(_gpu, "query_gpus", lambda: [("NVIDIA GeForce RTX 5060 Ti", "621.14", 120)])
    monkeypatch.setattr(_core, "library_path", lambda: Path("/opt/x/libloupmd.so"))
    monkeypatch.setattr(_core, "library_version", lambda: "1.0.0a1+0123456789abcdef")
    monkeypatch.setattr(runtime, "_loaded", lambda part: "/usr/local/cuda/lib64/libcudart.so.13")
    info = loupmd.runtime_info()
    assert info.version == loupmd.__version__ and info.abi_version == _core.ABI_VERSION
    assert info.library == "/opt/x/libloupmd.so" and info.library_version == "1.0.0a1+0123456789abcdef"
    assert info.library_error is None and info.cuda_runtime == "/usr/local/cuda/lib64/libcudart.so.13"
    gpu, = info.gpus
    assert (gpu.index, gpu.name, gpu.driver, gpu.compute_capability) == (0, "NVIDIA GeForce RTX 5060 Ti", "621.14", "12.0")
    text = str(info)
    assert f"loupmd {loupmd.__version__}" in text and "library 1.0.0a1+0123456789abcdef (/opt/x/libloupmd.so)" in text
    assert "GPU 0: NVIDIA GeForce RTX 5060 Ti, compute capability 12.0, driver 621.14" in text
    with pytest.raises(Exception):
        info.version = "2"                                            # a frozen record


def test_runtime_info_reports_a_library_that_does_not_load_and_a_machine_without_a_gpu(monkeypatch):
    import loupmd
    from loupmd import _core, _gpu
    from loupmd.errors import BackendError

    def no_gpu():
        raise OSError("nvidia-smi was not found on the PATH")

    def no_library():
        raise BackendError("cannot load /opt/x/libloupmd.so", why="libcudart.so.13: cannot open shared object file")

    monkeypatch.setattr(_gpu, "query_gpus", no_gpu)
    monkeypatch.setattr(_core, "library_path", lambda: Path("/opt/x/libloupmd.so"))
    monkeypatch.setattr(_core, "library_version", no_library)
    info = loupmd.runtime_info()
    assert info.gpus == () and info.gpu_error == "nvidia-smi was not found on the PATH"
    assert info.library_version is None and "cannot load /opt/x/libloupmd.so" in info.library_error
    text = str(info)
    assert "no GPU found: nvidia-smi was not found" in text and "library not loaded: cannot load" in text


def test_importing_loupmd_and_asking_for_the_runtime_imports_no_torch():
    code = ("import sys, loupmd; info = loupmd.runtime_info(); print(info); "
            "assert 'torch' not in sys.modules and 'mace' not in sys.modules, sorted(sys.modules)")
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]
    assert "loupmd " in done.stdout


def test_the_cuda_runtime_of_the_nvidia_wheels_is_preloaded_when_the_loader_does_not_find_it(tmp_path, monkeypatch):
    """The layout of nvidia-cuda-runtime 13.0.96 and nvidia-cublas 13.1.1.3: ``site-packages/nvidia/cu13/lib``."""
    import ctypes
    import importlib.util
    import types

    from loupmd import _core

    lib = tmp_path / "nvidia" / "cu13" / "lib"
    lib.mkdir(parents=True)
    for name in ("libcudart.so.13", "libcublasLt.so.13", "libcublas.so.13", "libnvrtc.so.13"):
        (lib / name).write_bytes(b"")
    loaded = []
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: types.SimpleNamespace(
        submodule_search_locations=[str(tmp_path / "nvidia")]) if name == "nvidia" else None)
    monkeypatch.setattr(ctypes, "CDLL", lambda path, mode=0: loaded.append((Path(path).name, mode)))
    _core._preload_nvidia_libs()
    assert loaded == [("libcudart.so.13", ctypes.RTLD_GLOBAL), ("libcublasLt.so.13", ctypes.RTLD_GLOBAL),
                      ("libcublas.so.13", ctypes.RTLD_GLOBAL)]
