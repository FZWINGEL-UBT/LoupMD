"""Python ctypes wrapper for libloupmd.so."""
from __future__ import annotations

import ctypes
import os
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np


class LoupMDStats(ctypes.Structure):
    _fields_ = [
        ("step_ms", ctypes.c_double),
        ("rebuilds", ctypes.c_int),
        ("final_potential_energy_ev", ctypes.c_double),
        ("final_kinetic_energy_ev", ctypes.c_double),
        ("final_temperature_k", ctypes.c_double),
        ("tune_ms", ctypes.c_double),
    ]


_LIB = None


def _preload_nvidia_libs():
    import importlib.util

    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        return
    for base in spec.submodule_search_locations:
        for sub in ("cu13/lib", "cublas/lib", "cuda_runtime/lib"):
            d = Path(base) / sub
            for name in ("libcudart.so.13", "libcublasLt.so.13", "libcublas.so.13",
                         "libcudart.so.12", "libcublasLt.so.12", "libcublas.so.12"):
                f = d / name
                if f.exists():
                    try:
                        ctypes.CDLL(str(f), mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        pass


def _load_lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    # LOUP_LIB selects another build (keep it beside libloupmd.so: cubins are found relative to the library).
    so_path = Path(os.environ.get("LOUP_LIB") or Path(__file__).resolve().parent.parent / "libloupmd.so")
    try:
        lib = ctypes.CDLL(str(so_path))
    except OSError:
        # cuBLAS / cudart from the nvidia pip wheels (as PyTorch ships them) when they are not on the loader path
        _preload_nvidia_libs()
        lib = ctypes.CDLL(str(so_path))
    lib.loup_create.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.loup_create.restype = ctypes.c_void_p
    lib.loup_destroy.argtypes = [ctypes.c_void_p]
    lib.loup_destroy.restype = None
    lib.loup_set_system.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_double,
    ]
    lib.loup_set_system.restype = ctypes.c_int
    lib.loup_evaluate.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.loup_evaluate.restype = ctypes.c_int
    lib.loup_set_state.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.loup_set_state.restype = ctypes.c_int
    lib.loup_run_nve.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(LoupMDStats),
    ]
    lib.loup_run_nve.restype = ctypes.c_int
    lib.loup_run_langevin.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_ulonglong,
        ctypes.POINTER(LoupMDStats),
    ]
    lib.loup_run_langevin.restype = ctypes.c_int
    lib.loup_get_state.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.loup_get_state.restype = ctypes.c_int
    lib.loup_profile_force.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.loup_profile_force.restype = ctypes.c_int
    _LIB = lib
    return lib


def _model_digest(model_path: str) -> str:
    """sha256 of the model file, memoised per (path, size, mtime) so a large model is hashed once."""
    st = os.stat(model_path)
    memo_key = (str(Path(model_path).resolve()), st.st_size, st.st_mtime_ns)
    if memo_key not in _DIGESTS:
        h = hashlib.sha256()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _DIGESTS[memo_key] = h.hexdigest()
    return _DIGESTS[memo_key]


_DIGESTS: dict = {}
_PY = Path(__file__).resolve().parent
# Everything on LoupMD's side that decides a bundle's contents: the exporters, the container writer and the compiled
# kernels' signature. Their hash is part of the cache key, so a changed exporter or regenerated kernels never find
# an old bundle, with no version number to remember. (A FastMACE change to the product basis is caught separately:
# the loader compares the product signatures a bundle carries with the compiled kernels'.)
EXPORTER_SOURCES = [_PY / name for name in ("export.py", "export_omat0.py", "export_generic.py", "export_polar.py",
                                            "bundle.py")] + [_PY.parent / "src" / "generated_kernels_signature.h"]


def exporter_fingerprint(paths=None) -> str:
    """Hash of the files in ``paths`` (default EXPORTER_SOURCES), by name and content; a missing file counts as empty."""
    h = hashlib.sha256()
    for p in EXPORTER_SOURCES if paths is None else paths:
        p = Path(p)
        h.update(p.name.encode() + b"\0" + (p.read_bytes() if p.is_file() else b"") + b"\0")
    return h.hexdigest()[:16]


def ensure_loup_bundle(model_path: str, atomic_numbers: list[int], cache_dir: str | None = None,
                       head: str | None = None, generic: bool | None = None, charge: int | None = None,
                       spin_multiplicity: int | None = None) -> Path:
    """The cached bundle for (model contents, species, head, path, charge / spin, exporter sources), exporting it on
    a miss.

    The specialised OMAT-0-medium kernels are used when the model has that architecture; ``generic=True`` (or the
    environment variable LOUP_GENERIC=1) forces the generic path. ``head`` selects a multi-head model's head
    (default: omat_pbe if present, else the last head). ``charge`` and ``spin_multiplicity`` (total charge in e and
    2S + 1) apply to models with a total charge / spin embedding (MACE-OMOL-0; default 0 and 1); the export folds
    them into the bundle, so each setting is its own bundle. Other models refuse them."""
    def integral(name, v):  # categorical model inputs: 0.5 must not round to 0, 1.0 must share 1's bundle
        if v is None:
            return None
        if isinstance(v, bool) or int(v) != v:
            raise ValueError(f"{name} must be an integer, got {v!r}")
        return int(v)

    charge = integral("charge", charge)
    spin_multiplicity = integral("spin_multiplicity", spin_multiplicity)
    z_sorted = sorted({int(z) for z in atomic_numbers})
    if generic is None:
        generic = os.environ.get("LOUP_GENERIC", "0") == "1"
    cdir = Path(cache_dir or Path.home() / ".cache" / "loupmd")
    cdir.mkdir(parents=True, exist_ok=True)
    tag = (f"{_model_digest(model_path)}|{z_sorted}|{head}|{int(generic)}|{charge}|{spin_multiplicity}|"
           f"{exporter_fingerprint()}")
    key = hashlib.sha256(tag.encode()).hexdigest()[:16]
    out = cdir / f"{Path(model_path).stem}_{'_'.join(map(str, z_sorted))}_{key}.loup"
    if not out.is_file():
        exporter = Path(__file__).resolve().parent / "export.py"
        cmd = [sys.executable, str(exporter), "--model", model_path,
               "--elements", ",".join(map(str, z_sorted)), "--output", str(out)]
        if head is not None:
            cmd += ["--head", head]
        if generic:
            cmd.append("--generic")
        if charge is not None:
            cmd += ["--charge", str(charge)]
        if spin_multiplicity is not None:
            cmd += ["--spin-multiplicity", str(spin_multiplicity)]
        subprocess.run(cmd, check=True)
    return out


def _array(x, shape: tuple, name: str) -> np.ndarray:
    """``x`` as a C-contiguous float64 array of exactly ``shape``. The library reads and writes by the atom count
    it holds, so a wrong shape here would be an out-of-bounds access there: refuse it instead."""
    a = np.ascontiguousarray(x, dtype=np.float64)
    if a.shape != shape:
        raise ValueError(f"{name} has shape {a.shape}, expected {shape}")
    return a


class LoupEngine:
    def __init__(self, loup_path: str | Path, device_id: int = 0):
        self._lib = _load_lib()
        self._h = self._lib.loup_create(str(loup_path).encode(), int(device_id))
        if not self._h:
            raise RuntimeError(f"Failed to create LoupEngine from {loup_path}")
        self._n = 0

    def __del__(self):
        if getattr(self, "_h", None):
            self._lib.loup_destroy(self._h)
            self._h = None

    def set_system(self, atomic_numbers, masses_amu=None, skin: float = 0.5):
        z = np.ascontiguousarray(atomic_numbers, dtype=np.int32)
        if z.ndim != 1:
            raise ValueError(f"atomic_numbers has shape {z.shape}, expected (n_atoms,)")
        n = int(z.shape[0])
        m = _array(masses_amu, (n,), "masses_amu") if masses_amu is not None else np.ones(n, dtype=np.float64)
        rc = self._lib.loup_set_system(
            self._h, n,
            z.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            m.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            float(skin),
        )
        if rc != 0:
            raise RuntimeError("loup_set_system failed")
        self._n = n  # only now: a rejected call keeps the previous system, and later calls size their arrays by it

    def evaluate(self, positions, cell, compute_stress: bool = True):
        pos = _array(positions, (self._n, 3), "positions")
        cel = _array(cell, (3, 3), "cell")
        energy = ctypes.c_double()
        forces = np.empty((self._n, 3), dtype=np.float64)
        stress = np.empty((3, 3), dtype=np.float64) if compute_stress else None
        rc = self._lib.loup_evaluate(
            self._h,
            pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            cel.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            1 if compute_stress else 0,
            ctypes.byref(energy),
            forces.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            stress.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if compute_stress else None,
        )
        if rc != 0:
            raise RuntimeError("loup_evaluate failed")
        return float(energy.value), forces, stress

    def set_state(self, positions, velocities, cell):
        pos = _array(positions, (self._n, 3), "positions")
        vel = _array(velocities, (self._n, 3), "velocities") if velocities is not None else None
        cel = _array(cell, (3, 3), "cell")
        rc = self._lib.loup_set_state(
            self._h,
            pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            vel.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if vel is not None else None,
            cel.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if rc != 0:
            raise RuntimeError("loup_set_state failed")

    def run_nve(self, n_steps: int, dt_fs: float = 1.0) -> LoupMDStats:
        st = LoupMDStats()
        rc = self._lib.loup_run_nve(self._h, int(n_steps), float(dt_fs), ctypes.byref(st))
        if rc != 0:
            raise RuntimeError("loup_run_nve failed")
        return st

    def run_langevin(self, n_steps: int, dt_fs: float = 1.0, temp_k: float = 300.0,
                     friction_per_fs: float = 0.01, seed: int = 1) -> LoupMDStats:
        st = LoupMDStats()
        rc = self._lib.loup_run_langevin(
            self._h, int(n_steps), float(dt_fs), float(temp_k), float(friction_per_fs), int(seed), ctypes.byref(st)
        )
        if rc != 0:
            raise RuntimeError("loup_run_langevin failed")
        return st

    def get_state(self):
        pos = np.empty((self._n, 3), dtype=np.float64)
        vel = np.empty((self._n, 3), dtype=np.float64)
        frc = np.empty((self._n, 3), dtype=np.float64)
        en = ctypes.c_double()
        rc = self._lib.loup_get_state(
            self._h,
            pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            vel.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            frc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(en),
        )
        if rc != 0:
            raise RuntimeError("loup_get_state failed")
        return pos, vel, frc, float(en.value)

    def profile_force(self, warmup: int = 50, n_calls: int = 200) -> float:
        us = ctypes.c_double()
        rc = self._lib.loup_profile_force(self._h, int(warmup), int(n_calls), ctypes.byref(us))
        if rc != 0:
            raise RuntimeError("loup_profile_force failed")
        return float(us.value)
