"""The engine of a ``Simulation`` whose system does not fit the GPU whole (plan task S5-T12, PLAN.md D65).

:class:`TiledEngine` has the interface ``Simulation`` uses of ``LoupEngine`` and gets its forces from the tiled call,
which keeps its tile lists across steps until an atom has moved skin/2 (S5-T06). The integrator runs on the host:
velocity Verlet (NVE) and ASE's Langevin step, with fixed atoms and the centre of mass held at rest as the device
loop does. Thermostats other than Langevin and every barostat are refused in tiled mode until the device integrators
run on the tiled state (D65). The Langevin noise follows ASE's formulas with its own random stream (seeded from the
run's seed and the step), so a tiled run does not repeat a whole-system run's trajectory; a restart continues the
stream.

A dispersion term (``set_dispersion``, PLAN.md S6-D09, D76) is computed for the whole system by the library's D3
term kept across calls (``loupmd._d3.Term``): it needs no neighbour list and holds about 200 bytes per atom, so it
fits where the model has to be tiled. Its energy, forces and stress are added to the tiled call's.
"""
from __future__ import annotations

import struct
import time

import numpy as np

from . import _core
from .errors import BackendError, PlanRefused

_MAGIC = b"LOUPTILE"
_VERSION = 1
_SUPPORTED = {_core.LOUP_NVE: "NVE", _core.LOUP_LANGEVIN: "Langevin"}


class TiledEngine:
    """Engine interface over ``engine.evaluate_tiled``: ``engine`` is a ``LoupEngine`` with no system set."""

    def __init__(self, engine, receivers: int = 0, memory_fraction: float = 0.9, wavefront: bool = False):
        self._eng = engine
        self.receivers = int(receivers)          # 0 until the first call takes the plan's cap
        self.memory_fraction = float(memory_fraction)
        self.mode = _core.LOUP_EXEC_WAVEFRONT if wavefront else _core.LOUP_EXEC_TILED
        self.wavefront = False                   # whether the last call ran the wavefront schedule (S5-T08)
        self.tiles = 0
        self._n = 0
        if hasattr(engine, "info"):            # what callers ask with hasattr (the record's kernel path)
            self.info = engine.info
        self.dispersion = None
        self._d3 = None

    # ------------------------------------------------------------------------------------------------- system
    def set_system(self, numbers, masses, skin=0.5):
        self._numbers = np.ascontiguousarray(numbers, dtype=np.int32)
        self._masses = np.asarray(masses, dtype=np.float64)
        self._n = len(self._numbers)
        self.skin = float(skin)
        self._fixed = None
        self._pos = self._vel = self._cell = None
        self._forces = self._energy = None
        self._md_steps = 0
        self._noise_step = 0                     # the Langevin stream's position: kept by restarts, as the device's
        self._rows = []
        self._step_disp = 0.0
        self._close_d3()
        self._eng.set_execution(self.mode, self.receivers, self.memory_fraction)

    def set_dispersion(self, record=None):
        """The D3 term of a resolved record added to every force call (``None``: removed)."""
        self._close_d3()
        self.dispersion = record
        self._forces = self._energy = None

    def _close_d3(self):
        if self._d3 is not None:
            self._d3.close()
            self._d3 = None

    def set_fixed_atoms(self, fixed=None):
        self._fixed = None if fixed is None else np.asarray(fixed, dtype=bool).copy()
        if self._fixed is not None and self._vel is not None:
            self._vel[self._fixed] = 0.0

    def set_reference_volume(self, volume):
        raise PlanRefused("a barostat in tiled mode", why="the tiled mode has no variable cell yet (PLAN.md D65)",
                          next_step="Run NVT or NVE in tiled mode, or a cell that fits the GPU whole.")

    def set_state(self, positions, velocities, cell):
        self._pos = np.array(positions, dtype=np.float64)
        self._vel = np.array(velocities, dtype=np.float64)
        self._cell = np.array(cell, dtype=np.float64)
        if self._fixed is not None:
            self._vel[self._fixed] = 0.0
        self._forces = self._energy = None
        self._md_steps = 0

    def get_cell(self):
        return self._cell.copy()

    # ------------------------------------------------------------------------------------------------- forces
    def _force_call(self):
        e, f, s, tiles = self._eng.evaluate_tiled(self._numbers, self._pos, self._cell, self.receivers,
                                                  skin=self.skin)
        done = self._eng.last_execution()
        if not self.receivers:
            self.receivers = int(done.receivers)
        self.wavefront = done.mode == _core.LOUP_EXEC_WAVEFRONT
        self.tiles = int(tiles)
        if not np.isfinite(e):
            raise BackendError(f"loup_run: non-finite energy ({e}) in the tiled call", call="loup_run")
        if self.dispersion is not None:
            from . import _d3

            if self._d3 is None:
                self._d3 = _d3.Term(self.dispersion, self._numbers, self._cell)
            de, df, ds = self._d3.compute(self._pos, self._cell, stress=s is not None)
            e, f = e + de, f + df
            if s is not None:
                s = s + ds
        self._energy, self._forces, self._stress = float(e), f, s
        return self._energy, f, s

    def evaluate(self, positions, cell, compute_stress=True):
        self._pos = np.array(positions, dtype=np.float64)
        self._cell = np.array(cell, dtype=np.float64)
        e, f, s = self._force_call()
        return e, f.copy(), (s.copy() if compute_stress else None)

    def last_execution(self):
        return self._eng.last_execution()

    # ------------------------------------------------------------------------------------------------- dynamics
    def _dof(self, fix_com):
        free = self._n - (int(self._fixed.sum()) if self._fixed is not None else 0)
        return 3 * free - (3 if fix_com else 0)

    def run(self, n_steps, it):
        from ase import units

        if n_steps <= 0:
            raise BackendError(f"loup_run: n_steps must be positive, got {n_steps}", call="loup_run")
        if it.kind not in _SUPPORTED:
            raise PlanRefused(f"integrator kind {it.kind} in tiled mode",
                              why="the tiled mode integrates NVE and Langevin on the host; the other thermostats "
                                  "and the barostats wait for the device integrators on the tiled state (PLAN.md "
                                  "D65)",
                              next_step="Use NVE or Langevin, or a cell that fits the GPU whole.")
        if self._forces is None:
            self._force_call()
        dt = it.dt_fs * units.fs
        m = self._masses[:, None]
        inv_m = 1.0 / m
        if self._fixed is not None:
            inv_m = np.where(self._fixed[:, None], 0.0, inv_m)
        fix_com = bool(it.fix_com)
        total_mass = float(self._masses.sum())
        dof = self._dof(fix_com)
        v, x = self._vel, self._pos
        if fix_com and it.kind == _core.LOUP_NVE:
            v -= (self._masses[:, None] * v).sum(axis=0) / total_mass
        langevin = it.kind == _core.LOUP_LANGEVIN
        if langevin:
            fr = it.friction_per_fs / units.fs
            c1 = dt / 2.0 - dt * dt * fr / 8.0
            c2 = dt * fr / 2.0 - dt * dt * fr * fr / 8.0
        largest = 0.0
        t0 = time.perf_counter()
        for i in range(1, n_steps + 1):
            f = self._forces
            if langevin:
                target = max(it.temp_k + i * it.temp_rate_k_per_step, 0.0)
                sigma = np.sqrt(2.0 * units.kB * target * fr / m)
                c3 = np.sqrt(dt) * sigma / 2.0 - dt ** 1.5 * fr * sigma / 8.0
                c5 = dt ** 1.5 * sigma / (2.0 * np.sqrt(3.0))
                c4 = fr / 2.0 * c5
                rng = np.random.default_rng([int(it.seed), self._noise_step])
                xi, eta = rng.standard_normal((self._n, 3)), rng.standard_normal((self._n, 3))
                rnd_pos = c5 * eta
                rnd_vel = c3 * xi - c4 * eta
                if self._fixed is not None:
                    rnd_pos[self._fixed] = 0.0
                    rnd_vel[self._fixed] = 0.0
                if fix_com:
                    rnd_pos -= (m * rnd_pos).sum(axis=0) / total_mass
                    rnd_vel -= (m * rnd_vel).sum(axis=0) / total_mass
                v += c1 * f * inv_m - c2 * v + rnd_vel
                step = dt * v + rnd_pos
                if self._fixed is not None:
                    step[self._fixed] = 0.0
                    v[self._fixed] = 0.0
                x += step
                e, f, _ = self._force_call()
                v += c1 * f * inv_m - c2 * v + rnd_vel
                if fix_com:
                    v -= (m * v).sum(axis=0) / total_mass
                self._noise_step += 1
            else:
                v += 0.5 * dt * f * inv_m
                step = dt * v
                x += step
                e, f, _ = self._force_call()
                v += 0.5 * dt * f * inv_m
            largest = max(largest, float(np.sqrt((step * step).sum(axis=1).max())))
            self._md_steps += 1
            if it.thermo_every and self._md_steps % it.thermo_every == 0:
                e_kin = 0.5 * float((m * v * v).sum())
                self._rows.append((self._md_steps, e, e_kin, 2.0 * e_kin / (dof * units.kB) if dof > 0 else 0.0,
                                   0.0, abs(float(np.linalg.det(self._cell))), e + e_kin))
        wall = (time.perf_counter() - t0) * 1e3 / n_steps
        self._step_disp = largest
        e_kin = 0.5 * float((m * v * v).sum())
        stats = _core.LoupMDStats()
        stats.step_ms = wall
        stats.rebuilds = 0
        stats.final_potential_energy_ev = self._energy
        stats.final_kinetic_energy_ev = e_kin
        stats.final_temperature_k = 2.0 * e_kin / (dof * units.kB) if dof > 0 else 0.0
        stats.pipelined_steps = 0
        stats.dangerous_builds = 0
        return stats

    def step_displacement(self):
        return self._step_disp

    def thermo_pending(self):
        return len(self._rows)

    def read_thermo(self, max_rows=None):
        rows, self._rows = self._rows, []
        return np.array(rows, dtype=_core.THERMO_DTYPE)

    def get_state(self):
        if self._forces is None:
            self._force_call()
        return self._pos.copy(), self._vel.copy(), self._forces.copy(), self._energy

    # ------------------------------------------------------------------------------------------------- restarts
    def export_state(self) -> bytes:
        head = struct.pack("<8sIqq", _MAGIC, _VERSION, self._n, self._noise_step)
        return head + self._pos.tobytes() + self._vel.tobytes() + self._cell.tobytes()

    def import_state(self, blob):
        blob = bytes(blob)
        size = struct.calcsize("<8sIqq")
        if len(blob) < size:
            raise BackendError("loup_import_state: the state block is too short for the tiled mode",
                               call="loup_import_state")
        magic, version, n, noise = struct.unpack("<8sIqq", blob[:size])
        if magic != _MAGIC or version != _VERSION:
            raise BackendError("loup_import_state: not a tiled-mode state block (written by the whole-system "
                               "engine, or another version)", call="loup_import_state")
        if n != self._n or len(blob) != size + 8 * (6 * n + 9):
            raise BackendError(f"loup_import_state: the state block holds {n} atoms, the system {self._n}",
                               call="loup_import_state")
        a = np.frombuffer(blob[size:], dtype=np.float64)
        pos, cell = a[:3 * n].reshape(n, 3), a[6 * n:].reshape(3, 3)
        # The same positions and cell (a Simulation restarts the step count this way on a thermo step): the forces
        # stay, so no tiled call is repeated.
        same = (self._forces is not None and self._pos is not None and np.array_equal(pos, self._pos)
                and np.array_equal(cell, self._cell))
        self._pos, self._cell = pos.copy(), cell.copy()
        self._vel = a[3 * n:6 * n].reshape(n, 3).copy()
        self._noise_step = int(noise)
        if not same:
            self._forces = self._energy = None
        self._md_steps = 0
        self._rows = []
