"""Interface version, library version and loup_get_info (plan task S1-T02).

The first four tests load at most the library file and need no GPU; the rest create an engine.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

import pytest
from conftest import ROOT, find_model

import loupmd
from loupmd import LoupEngine, ensure_loup_bundle
from loupmd import _core

HEADER = (ROOT / "include" / "loupmd.h").read_text()


def test_header_and_package_agree_on_both_versions():
    assert re.search(r'#define LOUP_VERSION "([^"]+)"', HEADER).group(1) == loupmd.__version__
    assert int(re.search(r"#define LOUP_ABI_VERSION (\d+)", HEADER).group(1)) == _core.ABI_VERSION


def test_library_reports_its_interface_and_build():
    lib = _core._load_lib()
    assert lib.loup_abi_version() == _core.ABI_VERSION == 1
    version = loupmd.library_version()
    release, build = version.split("+")
    assert release == loupmd.__version__
    assert re.fullmatch(r"[0-9a-f]{16}", build), version
    assert loupmd.library_version() == version


STUB = {
    "another_version": "int loup_abi_version(void) { return 2; }\n",
    "no_version_call": "int loup_create(void) { return 0; }\n",
}
LOAD = (
    "import sys; sys.path.insert(0, sys.argv[1])\n"
    "from loupmd import BackendError, _core\n"
    "try:\n"
    "    _core._load_lib()\n"
    "except BackendError as err:\n"
    "    print(str(err))\n"
    "else:\n"
    "    print('LOADED')\n"
)


@pytest.mark.parametrize("kind,found", [("another_version", 2), ("no_version_call", 0)])
def test_a_library_of_another_interface_version_is_refused(tmp_path, kind, found):
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        pytest.skip("no C compiler for the stub library")
    src = tmp_path / "stub.c"
    src.write_text(STUB[kind])
    stub = tmp_path / "libloupmd_stub.so"
    subprocess.run([cc, "-shared", "-fPIC", str(src), "-o", str(stub)], check=True)
    out = subprocess.run([sys.executable, "-c", LOAD, str(ROOT)], env=dict(os.environ, LOUP_LIB=str(stub)),
                         capture_output=True, text=True, check=True).stdout
    assert f"{stub} has C interface version {found}, and this loupmd package needs version 1" in out, out
    assert "Next: Rebuild the library" in out


def _gpu_name():
    return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", "0"],
                          capture_output=True, text=True, check=True).stdout.strip()


def test_info_of_a_model_on_the_specialised_kernels(spec_bundle):
    info = LoupEngine(spec_bundle).info()
    assert info.abi == 1
    assert info.specialised and not info.has_long_range
    assert (info.n_layers, info.n_species, info.r_max) == (2, 3, 6.0)
    assert info.kernel_signature.startswith("omat0m:C128:")
    assert info.gpu_name == _gpu_name()
    major, minor = divmod(info.compute_capability, 10)
    assert major >= 8 and 0 <= minor <= 9


def test_info_of_a_model_on_the_generic_path():
    eng = LoupEngine(ensure_loup_bundle(find_model("mace-omat-0-small.model"), [8, 22, 38]))
    info = eng.info()
    assert not info.specialised and not info.has_long_range
    assert (info.n_layers, info.n_species) == (2, 3)
    signatures = info.kernel_signature.split(";")
    assert len(signatures) == 2 and all(signatures) and "?" not in info.kernel_signature


def test_info_marks_a_long_range_model():
    eng = LoupEngine(ensure_loup_bundle(find_model("MACE-POLAR-1-S.model"), [11, 17]))
    info = eng.info()
    assert info.has_long_range and not info.specialised and info.n_species == 2


def test_get_info_without_an_output_struct_is_refused(sto_engine):
    assert sto_engine._lib.loup_get_info(sto_engine._h, None) == -1
    assert sto_engine.last_error() == "loup_get_info: out is NULL"
