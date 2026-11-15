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


# Integrator kinds and cell modes of include/loupmd.h.
LOUP_NVE, LOUP_LANGEVIN, LOUP_CSVR, LOUP_NHC = 0, 1, 2, 3
LOUP_BERENDSEN_NVT, LOUP_BERENDSEN_NPT, LOUP_BAOAB_NPT, LOUP_MTK_NPT = 4, 5, 6, 7
LOUP_CELL_HYDROSTATIC, LOUP_CELL_FULL = 0, 1


class LoupIntegrator(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint),
        ("kind", ctypes.c_int),
        ("dt_fs", ctypes.c_double),
        ("temp_k", ctypes.c_double),
        ("temp_rate_k_per_step", ctypes.c_double),
        ("friction_per_fs", ctypes.c_double),
        ("tau_t_fs", ctypes.c_double),
        ("mass_temp_k", ctypes.c_double),
        ("pressure_gpa", ctypes.c_double),
        ("tau_p_fs", ctypes.c_double),
        ("compressibility_per_gpa", ctypes.c_double),
        ("chain_length", ctypes.c_int),
        ("chain_loops", ctypes.c_int),
        ("p_chain_length", ctypes.c_int),
        ("p_chain_loops", ctypes.c_int),
        ("cell_mode", ctypes.c_int),
        ("cell_mask", ctypes.c_int * 3),
        ("fix_com", ctypes.c_int),
        ("thermo_every", ctypes.c_int),
        ("seed", ctypes.c_ulonglong),
    ]


LOUP_EXEC_AUTO, LOUP_EXEC_WHOLE, LOUP_EXEC_TILED, LOUP_EXEC_WAVEFRONT = 0, 1, 2, 3
LOUP_EVAL_STRESS, LOUP_EVAL_ENERGY_ONLY, LOUP_EVAL_PER_ATOM = 1, 2, 4


LOUP_MIN_LBFGS = 0


class LoupMinimizer(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint),
        ("kind", ctypes.c_int),
        ("fmax_ev_a", ctypes.c_double),
        ("maxstep_a", ctypes.c_double),
        ("memory", ctypes.c_int),
        ("reserved", ctypes.c_int),
        ("damping", ctypes.c_double),
        ("alpha", ctypes.c_double),
    ]


class LoupMinStats(ctypes.Structure):
    _fields_ = [
        ("steps", ctypes.c_int),
        ("converged", ctypes.c_int),
        ("total_steps", ctypes.c_int),
        ("rebuilds", ctypes.c_int),
        ("force_passes", ctypes.c_int),
        ("repeated_passes", ctypes.c_int),
        ("max_force_ev_a", ctypes.c_double),
        ("potential_energy_ev", ctypes.c_double),
        ("force_ms", ctypes.c_double),
        ("optimizer_ms", ctypes.c_double),
    ]


class LoupThermoRow(ctypes.Structure):
    _fields_ = [
        ("step", ctypes.c_longlong),
        ("e_pot", ctypes.c_double),
        ("e_kin", ctypes.c_double),
        ("temp_k", ctypes.c_double),
        ("pressure_gpa", ctypes.c_double),
        ("volume_a3", ctypes.c_double),
        ("conserved", ctypes.c_double),
    ]


THERMO_DTYPE = np.dtype([("step", "<i8"), ("e_pot", "<f8"), ("e_kin", "<f8"), ("temp_k", "<f8"),
                         ("pressure_gpa", "<f8"), ("volume_a3", "<f8"), ("conserved", "<f8")])


class LoupMemoryPlan(ctypes.Structure):
    _fields_ = [
        ("whole_bytes", ctypes.c_size_t),
        ("tiled_global_bytes", ctypes.c_size_t),
        ("tile_workspace_bytes", ctypes.c_size_t),
        ("free_bytes", ctypes.c_size_t),
        ("receivers", ctypes.c_int),
        ("mode", ctypes.c_int),
    ]


@dataclass(frozen=True)
class MemoryPlan:
    """Device memory a system would take (``loup_plan_memory``), in bytes."""

    whole_bytes: int
    tiled_global_bytes: int
    tile_workspace_bytes: int
    free_bytes: int
    receivers: int
    mode: int               # LOUP_EXEC_WHOLE: fits whole; LOUP_EXEC_TILED: needs the capacity mode


def integrator(kind: int, dt_fs: float = 1.0, **fields) -> LoupIntegrator:
    """A ``LoupIntegrator`` with ``size`` set and every field not named left at zero."""
    it = LoupIntegrator()
    it.size = ctypes.sizeof(LoupIntegrator)
    it.kind = int(kind)
    it.dt_fs = float(dt_fs)
    for name, value in fields.items():
        if name == "cell_mask":
            it.cell_mask = (ctypes.c_int * 3)(*[int(v) for v in value])
        elif name in ("size", "kind", "dt_fs") or name not in {f[0] for f in LoupIntegrator._fields_}:
            raise InputError(f"LoupIntegrator has no settable field {name!r}")
        else:
            setattr(it, name, value)
    return it


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
    try:
        _bind(lib)
    except AttributeError as exc:  # a symbol of this binding is missing: an older build of the same version
        raise BackendError(
            f"{so_path} lacks a call this loupmd package uses ({exc})",
            why="the library was built from an older source tree than the Python package",
            next_step="Rebuild the library from this source tree (`make`), or unset LOUP_LIB.") from None
    _LIB = lib
    return lib


def _bind(lib):
    """Argument and result types of every call the package uses."""
    lib.loup_version.argtypes = []
    lib.loup_version.restype = ctypes.c_char_p
    lib.loup_last_error.argtypes = [ctypes.c_void_p]
    lib.loup_last_error.restype = ctypes.c_char_p
    lib.loup_get_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(LoupInfo)]
    lib.loup_run.argtypes = [ctypes.c_void_p, ctypes.POINTER(LoupIntegrator), ctypes.c_int,
                             ctypes.POINTER(LoupMDStats)]
    lib.loup_export_state.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                      ctypes.POINTER(ctypes.c_size_t)]
    lib.loup_import_state.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
    lib.loup_plan_memory.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.POINTER(LoupMemoryPlan)]
    lib.loup_read_thermo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.loup_create.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.loup_create.restype = ctypes.c_void_p
    lib.loup_destroy.argtypes = [ctypes.c_void_p]
    lib.loup_destroy.restype = None
    lib.loup_set_system.argtypes = [ctypes.c_void_p, ctypes.c_int, _c_int_p, _c_double_p, ctypes.c_double]
    lib.loup_evaluate.argtypes = [ctypes.c_void_p, _c_double_p, _c_double_p, ctypes.c_int, _c_double_p, _c_double_p,
                                  _c_double_p]
    lib.loup_set_state.argtypes = [ctypes.c_void_p, _c_double_p, _c_double_p, _c_double_p]
    lib.loup_set_fixed_atoms.argtypes = [ctypes.c_void_p, _c_int_p]
    lib.loup_set_fixed_atoms.restype = ctypes.c_int
    lib.loup_evaluate2.argtypes = [ctypes.c_void_p, _c_double_p, _c_double_p, ctypes.c_int, _c_double_p, _c_double_p,
                                   _c_double_p, _c_double_p]
    lib.loup_run_nve.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.POINTER(LoupMDStats)]
    lib.loup_run_langevin.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_double,
                                      ctypes.c_double, ctypes.c_ulonglong, ctypes.POINTER(LoupMDStats)]
    lib.loup_get_state.argtypes = [ctypes.c_void_p, _c_double_p, _c_double_p, _c_double_p, _c_double_p]
    lib.loup_profile_force.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, _c_double_p]
    lib.loup_minimize.argtypes = [ctypes.c_void_p, ctypes.POINTER(LoupMinimizer), ctypes.c_int,
                                  ctypes.POINTER(LoupMinStats)]
    lib.loup_minimize.restype = ctypes.c_int
    lib.loup_get_step_displacement.argtypes = [ctypes.c_void_p, _c_double_p]
    lib.loup_get_step_displacement.restype = ctypes.c_int
    lib.loup_set_cell.argtypes = [ctypes.c_void_p, _c_double_p, ctypes.c_int]
    lib.loup_set_cell.restype = ctypes.c_int
    lib.loup_get_cell.argtypes = [ctypes.c_void_p, _c_double_p]
    lib.loup_get_cell.restype = ctypes.c_int
    lib.loup_set_reference_volume.argtypes = [ctypes.c_void_p, ctypes.c_double]
    lib.loup_set_reference_volume.restype = ctypes.c_int
    lib.loup_debug_csvr_noise.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_ulonglong, ctypes.c_int,
                                          ctypes.c_int, _c_double_p, _c_double_p]
    lib.loup_debug_csvr_noise.restype = ctypes.c_int
    for name in ("loup_set_system", "loup_evaluate", "loup_set_state", "loup_run_nve", "loup_run_langevin",
                 "loup_get_state", "loup_profile_force", "loup_get_info", "loup_run", "loup_export_state",
                 "loup_import_state", "loup_plan_memory", "loup_read_thermo", "loup_evaluate2"):
        getattr(lib, name).restype = ctypes.c_int


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
        from . import kernels

        self._lib = _load_lib()
        self._n = 0
        os.environ[kernels.PLUGIN_PATH_VARIABLE] = kernels.plugin_search_path()   # where the engine finds plugins
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

    def evaluate_energy(self, positions, cell) -> float:
        """The energy alone, from a pass without gradients (``LOUP_EVAL_ENERGY_ONLY``). The forces on the device
        are not those of these positions afterwards."""
        pos = _array(positions, (self._n, 3), "positions")
        cel = _array(cell, (3, 3), "cell")
        energy = ctypes.c_double()
        self._check(self._lib.loup_evaluate2(self._h, _ptr(pos), _ptr(cel), LOUP_EVAL_ENERGY_ONLY,
                                             ctypes.byref(energy), None, None, None), "loup_evaluate2")
        return float(energy.value)

    def set_state(self, positions, velocities, cell):
        pos = _array(positions, (self._n, 3), "positions")
        vel = _array(velocities, (self._n, 3), "velocities") if velocities is not None else None
        cel = _array(cell, (3, 3), "cell")
        self._check(self._lib.loup_set_state(self._h, _ptr(pos), _ptr(vel), _ptr(cel)), "loup_set_state")

    def set_fixed_atoms(self, fixed=None):
        """Hold atoms in place (ASE's ``FixAtoms``): ``fixed`` is a mask of length ``n_atoms`` or ``None`` to free
        all. Fixed atoms keep their positions, have zero velocity and get no thermostat noise; the temperatures the
        engine reports count ``3 (N - n_fixed)`` degrees of freedom."""
        if fixed is None:
            self._check(self._lib.loup_set_fixed_atoms(self._h, None), "loup_set_fixed_atoms")
            return
        mask = np.ascontiguousarray(np.asarray(fixed, dtype=bool), dtype=np.int32)
        if mask.shape != (self._n,):
            raise InputError(f"fixed has shape {mask.shape}, expected ({self._n},)")
        self._check(self._lib.loup_set_fixed_atoms(self._h, mask.ctypes.data_as(_c_int_p)), "loup_set_fixed_atoms")

    def run(self, n_steps: int, it: LoupIntegrator) -> LoupMDStats:
        """``n_steps`` steps of the integrator ``it`` (see ``integrator``)."""
        st = LoupMDStats()
        self._check(self._lib.loup_run(self._h, ctypes.byref(it), int(n_steps), ctypes.byref(st)), "loup_run")
        return st

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

    def debug_csvr_noise(self, seed: int, first_step: int, n_steps: int, ndof: int):
        """Testing only: the standard normal and the chi-square variate (``ndof - 1`` degrees of freedom) CSVR
        draws in each of ``n_steps`` steps from ``first_step`` on, as two arrays."""
        normal = np.empty(int(n_steps), dtype=np.float64)
        chi2 = np.empty(int(n_steps), dtype=np.float64)
        self._check(self._lib.loup_debug_csvr_noise(self._h, int(seed), int(first_step), int(n_steps), int(ndof),
                                                    _ptr(normal), _ptr(chi2)), "loup_debug_csvr_noise")
        return normal, chi2

    def minimize(self, max_steps: int, fmax: float = 0.05, maxstep_A: float = 0.2, memory: int = 100,
                 damping: float = 1.0, alpha: float = 70.0) -> LoupMinStats:
        """Up to ``max_steps`` LBFGS steps on the device from the positions on the handle (ASE's LBFGS without
        line search), stopping when the largest force on a free atom is below ``fmax`` (eV/A). A call with the
        same settings continues the relaxation of the one before; ``max_steps=0`` only evaluates. ``get_state``
        reads the positions, forces and energy."""
        settings = LoupMinimizer(size=ctypes.sizeof(LoupMinimizer), kind=LOUP_MIN_LBFGS, fmax_ev_a=float(fmax),
                                 maxstep_a=float(maxstep_A), memory=int(memory), damping=float(damping),
                                 alpha=float(alpha))
        stats = LoupMinStats()
        self._check(self._lib.loup_minimize(self._h, ctypes.byref(settings), int(max_steps), ctypes.byref(stats)),
                    "loup_minimize")
        return stats

    def set_cell(self, cell, scale_atoms: bool = True):
        """A new cell (3 x 3, rows are the cell vectors, in A) for the positions on the handle; with
        ``scale_atoms`` the atoms keep their fractional coordinates, as ASE's ``set_cell(scale_atoms=True)``.
        Velocities stay; forces are those of the old cell until the next evaluation or run."""
        cel = _array(cell, (3, 3), "cell")
        self._check(self._lib.loup_set_cell(self._h, _ptr(cel), 1 if scale_atoms else 0), "loup_set_cell")

    def get_cell(self) -> np.ndarray:
        """The cell of the positions on the handle: the one last given, or where a variable-cell run took it."""
        out = np.empty((3, 3), dtype=np.float64)
        self._check(self._lib.loup_get_cell(self._h, _ptr(out)), "loup_get_cell")
        return out

    def set_reference_volume(self, volume_A3: float):
        """The volume the cell limits of a variable-cell run refer to (``set_state`` sets it to the volume it is
        given; a resumed run passes the first run's)."""
        self._check(self._lib.loup_set_reference_volume(self._h, float(volume_A3)), "loup_set_reference_volume")

    def step_displacement(self) -> float:
        """The largest distance in A an atom moved in one MD step lately (the displacement tracker's two windows):
        what the neighbour skin has to cover for the pipelined loop, which runs while four times this is below
        half the skin."""
        out = ctypes.c_double()
        self._check(self._lib.loup_get_step_displacement(self._h, ctypes.byref(out)), "loup_get_step_displacement")
        return float(out.value)

    def thermo_pending(self) -> int:
        """Unread rows in the device's thermo buffer."""
        n = ctypes.c_int()
        self._check(self._lib.loup_read_thermo(self._h, None, 0, ctypes.byref(n)), "loup_read_thermo")
        return n.value

    def read_thermo(self, max_rows: int | None = None) -> np.ndarray:
        """The oldest unread thermo rows (all of them by default) as a structured array (``THERMO_DTYPE``); they
        leave the device's buffer."""
        want = self.thermo_pending() if max_rows is None else int(max_rows)
        out = np.zeros(max(want, 0), dtype=THERMO_DTYPE)
        n = ctypes.c_int()
        self._check(self._lib.loup_read_thermo(self._h, out.ctypes.data if want > 0 else None, want, ctypes.byref(n)),
                    "loup_read_thermo")
        return out[: n.value]

    def plan_memory(self, n_atoms: int, edges_per_atom: float = 0.0) -> MemoryPlan:
        """Device memory for ``n_atoms`` of the current material, scaled from the current system's buffers.
        ``edges_per_atom`` defaults to the current system's."""
        raw = LoupMemoryPlan()
        self._check(self._lib.loup_plan_memory(self._h, int(n_atoms), float(edges_per_atom), ctypes.byref(raw)),
                    "loup_plan_memory")
        return MemoryPlan(raw.whole_bytes, raw.tiled_global_bytes, raw.tile_workspace_bytes, raw.free_bytes,
                          raw.receivers, raw.mode)

    def export_state(self) -> bytes:
        """Positions, velocities, cell, step counter and displacement tracker as one versioned block of bytes."""
        size = ctypes.c_size_t()
        self._check(self._lib.loup_export_state(self._h, None, 0, ctypes.byref(size)), "loup_export_state")
        buf = ctypes.create_string_buffer(size.value)
        self._check(self._lib.loup_export_state(self._h, buf, size.value, ctypes.byref(size)), "loup_export_state")
        return buf.raw[: size.value]

    def import_state(self, blob: bytes):
        """Restore a block from ``export_state`` on a handle that holds the same system."""
        blob = bytes(blob)
        self._check(self._lib.loup_import_state(self._h, blob, len(blob)), "loup_import_state")

    def profile_force(self, warmup: int = 50, n_calls: int = 200) -> float:
        us = ctypes.c_double()
        self._check(self._lib.loup_profile_force(self._h, int(warmup), int(n_calls), ctypes.byref(us)),
                    "loup_profile_force")
        return float(us.value)
