"""ctypes binding of libloupmd's C interface (include/loupmd.h).

Nothing here imports torch. A failed library call raises ``BackendError`` with the library's own message
(``loup_last_error``); an argument the binding refuses before the library reads it raises ``InputError``.
"""
from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import BackendError, InputError

_c_double_p = ctypes.POINTER(ctypes.c_double)
_c_int_p = ctypes.POINTER(ctypes.c_int)

ABI_VERSION = 1  # LOUP_ABI_VERSION of include/loupmd.h this binding is written against


class LoupMDStats(ctypes.Structure):
    _fields_ = [
        ("step_ms", ctypes.c_double),
        ("rebuilds", ctypes.c_int),
        ("final_potential_energy_ev", ctypes.c_double),
        ("final_kinetic_energy_ev", ctypes.c_double),
        ("final_temperature_k", ctypes.c_double),
        ("dangerous_builds", ctypes.c_int),
        ("tune_ms", ctypes.c_double),
        ("pipelined_steps", ctypes.c_int),
    ]


class LoupInfo(ctypes.Structure):
    _fields_ = [
        ("abi", ctypes.c_int),
        ("compute_capability", ctypes.c_int),
        ("n_layers", ctypes.c_int),
        ("has_long_range", ctypes.c_int),
        ("specialised", ctypes.c_int),
        ("n_species", ctypes.c_int),
        ("r_max", ctypes.c_double),
        ("gpu_name", ctypes.c_char * 64),
        ("kernel_signature", ctypes.c_char * 160),
    ]


@dataclass(frozen=True)
class EngineInfo:
    """What an engine handle runs (``loup_get_info``)."""

    abi: int
    compute_capability: int   # of the handle's GPU, major * 10 + minor
    n_layers: int
    has_long_range: bool
    specialised: bool         # the specialised kernels; False on the generic path
    n_species: int
    r_max: float              # A
    gpu_name: str
    kernel_signature: str


_LIB = None
_PACKAGE = Path(__file__).resolve().parent


def library_path() -> Path:
    """The library this process uses: ``LOUP_LIB`` if set, else the one shipped inside the package, else the one
    ``make`` installs at the root of a source checkout."""
    override = os.environ.get("LOUP_LIB")
    if override:
        return Path(override)
    shipped = _PACKAGE / "_lib" / "libloupmd.so"
    return shipped if shipped.is_file() else _PACKAGE.parent / "libloupmd.so"


def _preload_nvidia_libs():
    """cuBLAS and the CUDA runtime from the nvidia pip wheels, for when they are not on the loader path."""
    import importlib.util

    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        return
    for base in spec.submodule_search_locations:
        for sub in ("cu13/lib", "cublas/lib", "cuda_runtime/lib"):
            for name in ("libcudart.so.13", "libcublasLt.so.13", "libcublas.so.13",
                         "libcudart.so.12", "libcublasLt.so.12", "libcublas.so.12"):
                f = Path(base) / sub / name
                if f.exists():
                    try:
                        ctypes.CDLL(str(f), mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        pass


def _load_lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    so_path = library_path()
    try:
        lib = ctypes.CDLL(str(so_path))
    except OSError:
        _preload_nvidia_libs()
        try:
            lib = ctypes.CDLL(str(so_path))
        except OSError as exc:
            raise BackendError(f"cannot load {so_path}", why=str(exc),
                               next_step="Build the library with `make`, or point LOUP_LIB at a build.") from None
    # The interface version first: nothing else of a library of another version is touched.
    abi = 0  # a build older than the version call
    if hasattr(lib, "loup_abi_version"):
        lib.loup_abi_version.argtypes = []
        lib.loup_abi_version.restype = ctypes.c_int
        abi = int(lib.loup_abi_version())
    if abi != ABI_VERSION:
        raise BackendError(
            f"{so_path} has C interface version {abi}, and this loupmd package needs version {ABI_VERSION}",
            why="the library and the Python package come from different builds",
            next_step="Rebuild the library from this source tree (`make`), or unset LOUP_LIB.")
    lib.loup_version.argtypes = []
    lib.loup_version.restype = ctypes.c_char_p
    lib.loup_last_error.argtypes = [ctypes.c_void_p]
    lib.loup_last_error.restype = ctypes.c_char_p
    lib.loup_get_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(LoupInfo)]
    lib.loup_create.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.loup_create.restype = ctypes.c_void_p
    lib.loup_destroy.argtypes = [ctypes.c_void_p]
    lib.loup_destroy.restype = None
    lib.loup_set_system.argtypes = [ctypes.c_void_p, ctypes.c_int, _c_int_p, _c_double_p, ctypes.c_double]
    lib.loup_evaluate.argtypes = [ctypes.c_void_p, _c_double_p, _c_double_p, ctypes.c_int, _c_double_p, _c_double_p,
                                  _c_double_p]
    lib.loup_set_state.argtypes = [ctypes.c_void_p, _c_double_p, _c_double_p, _c_double_p]
    lib.loup_run_nve.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.POINTER(LoupMDStats)]
    lib.loup_run_langevin.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_double,
                                      ctypes.c_double, ctypes.c_ulonglong, ctypes.POINTER(LoupMDStats)]
    lib.loup_get_state.argtypes = [ctypes.c_void_p, _c_double_p, _c_double_p, _c_double_p, _c_double_p]
    lib.loup_profile_force.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, _c_double_p]
    for name in ("loup_set_system", "loup_evaluate", "loup_set_state", "loup_run_nve", "loup_run_langevin",
                 "loup_get_state", "loup_profile_force", "loup_get_info"):
        getattr(lib, name).restype = ctypes.c_int
    _LIB = lib
    return lib


def library_version() -> str:
    """``loup_version()`` of the loaded library: the release, ``+``, and 16 hex digits identifying the build."""
    return _load_lib().loup_version().decode("ascii")


def last_error(handle=None) -> str:
    """The library's message for the last failed call on ``handle``; without a handle, this thread's last failed
    create. Empty if that call succeeded."""
    text = _load_lib().loup_last_error(handle)
    return text.decode("utf-8", errors="replace") if text else ""


def _fail(call: str, handle=None):
    raise BackendError(last_error(handle) or f"{call}: failed without a message", call=call)


def _array(x, shape: tuple, name: str) -> np.ndarray:
    """``x`` as a C-contiguous float64 array of exactly ``shape``. The library reads and writes by the atom count
    it holds, so a wrong shape here would be an out-of-bounds access there: refuse it instead."""
    a = np.ascontiguousarray(x, dtype=np.float64)
    if a.shape != shape:
        raise InputError(f"{name} has shape {a.shape}, expected {shape}")
    return a


def _ptr(a):
    return a.ctypes.data_as(_c_double_p) if a is not None else None


class LoupEngine:
    """One engine handle: a bundle on one GPU, then a system, then single points or MD."""

    def __init__(self, loup_path: str | Path, device_id: int = 0):
        self._lib = _load_lib()
        self._n = 0
        self._h = self._lib.loup_create(os.fsencode(str(loup_path)), int(device_id))
        if not self._h:
            _fail("loup_create")

    def __del__(self):
        if getattr(self, "_h", None):
            self._lib.loup_destroy(self._h)
            self._h = None

    def last_error(self) -> str:
        return last_error(self._h)

    def info(self) -> EngineInfo:
        raw = LoupInfo()
        self._check(self._lib.loup_get_info(self._h, ctypes.byref(raw)), "loup_get_info")
        return EngineInfo(
            abi=raw.abi, compute_capability=raw.compute_capability, n_layers=raw.n_layers,
            has_long_range=bool(raw.has_long_range), specialised=bool(raw.specialised), n_species=raw.n_species,
            r_max=raw.r_max, gpu_name=raw.gpu_name.decode("utf-8", errors="replace"),
            kernel_signature=raw.kernel_signature.decode("ascii", errors="replace"))

    def _check(self, rc: int, call: str):
        if rc != 0:
            _fail(call, self._h)

    def set_system(self, atomic_numbers, masses_amu=None, skin: float = 0.5):
        z = np.ascontiguousarray(atomic_numbers, dtype=np.int32)
        if z.ndim != 1:
            raise InputError(f"atomic_numbers has shape {z.shape}, expected (n_atoms,)")
        n = int(z.shape[0])
        m = _array(masses_amu, (n,), "masses_amu") if masses_amu is not None else np.ones(n, dtype=np.float64)
        self._check(self._lib.loup_set_system(self._h, n, z.ctypes.data_as(_c_int_p), _ptr(m), float(skin)),
                    "loup_set_system")
        self._n = n  # only now: a rejected call keeps the previous system, and later calls size their arrays by it

    def evaluate(self, positions, cell, compute_stress: bool = True):
        pos = _array(positions, (self._n, 3), "positions")
        cel = _array(cell, (3, 3), "cell")
        energy = ctypes.c_double()
        forces = np.empty((self._n, 3), dtype=np.float64)
        stress = np.empty((3, 3), dtype=np.float64) if compute_stress else None
        self._check(self._lib.loup_evaluate(self._h, _ptr(pos), _ptr(cel), 1 if compute_stress else 0,
                                            ctypes.byref(energy), _ptr(forces), _ptr(stress)), "loup_evaluate")
        return float(energy.value), forces, stress

    def set_state(self, positions, velocities, cell):
        pos = _array(positions, (self._n, 3), "positions")
        vel = _array(velocities, (self._n, 3), "velocities") if velocities is not None else None
        cel = _array(cell, (3, 3), "cell")
        self._check(self._lib.loup_set_state(self._h, _ptr(pos), _ptr(vel), _ptr(cel)), "loup_set_state")

    def run_nve(self, n_steps: int, dt_fs: float = 1.0) -> LoupMDStats:
        st = LoupMDStats()
        self._check(self._lib.loup_run_nve(self._h, int(n_steps), float(dt_fs), ctypes.byref(st)), "loup_run_nve")
        return st

    def run_langevin(self, n_steps: int, dt_fs: float = 1.0, temp_k: float = 300.0,
                     friction_per_fs: float = 0.01, seed: int = 1) -> LoupMDStats:
        st = LoupMDStats()
        self._check(self._lib.loup_run_langevin(self._h, int(n_steps), float(dt_fs), float(temp_k),
                                                float(friction_per_fs), int(seed), ctypes.byref(st)),
                    "loup_run_langevin")
        return st

    def get_state(self):
        pos = np.empty((self._n, 3), dtype=np.float64)
        vel = np.empty((self._n, 3), dtype=np.float64)
        frc = np.empty((self._n, 3), dtype=np.float64)
        en = ctypes.c_double()
        self._check(self._lib.loup_get_state(self._h, _ptr(pos), _ptr(vel), _ptr(frc), ctypes.byref(en)),
                    "loup_get_state")
        return pos, vel, frc, float(en.value)

    def profile_force(self, warmup: int = 50, n_calls: int = 200) -> float:
        us = ctypes.c_double()
        self._check(self._lib.loup_profile_force(self._h, int(warmup), int(n_calls), ctypes.byref(us)),
                    "loup_profile_force")
        return float(us.value)
