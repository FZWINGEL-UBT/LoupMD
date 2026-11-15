"""``Simulation`` (plan tasks S3-T04 and S3-T05) on a fake engine: springs integrated with velocity Verlet in
numpy, with the engine's thermo buffer, state block and step counters. Deterministic, so runs can be compared
exactly. Needs neither a GPU nor the library; ``test_simulation_gpu.py`` has the real runs."""
from __future__ import annotations

import json
import pickle

import numpy as np
import pytest
from ase import units
from ase.build import bulk

import loupmd
from loupmd import _core, integrators, output, simulation
from loupmd.config import IntegratorConfig, Output
from loupmd.errors import BackendError, InputError, NonFiniteError, PlanRefused, RestartMismatch

K = 2.0          # eV/A^2: every atom on a spring to its lattice site


class SpringEngine:
    """NVE (and "Langevin" without noise or friction) on springs. Mirrors what Simulation uses of LoupEngine."""

    made = []
    poison_at = None      # an md step whose energy is reported as NaN

    def __init__(self, bundle, device_id=0):
        self.bundle, self.device_id = bundle, device_id
        self.log, self.rows, self.md_steps, self.forces_current = [], [], 0, False
        self.total_steps = 0                       # survives set_state and import_state: for the tests only
        SpringEngine.made.append(self)

    # --- system and state
    def set_system(self, numbers, masses, skin=0.5):
        self.numbers, self.masses, self.n = np.array(numbers), np.array(masses, dtype=float), len(numbers)

    def set_state(self, positions, velocities, cell):
        self.sites = getattr(self, "sites", None)
        self.pos, self.vel, self.cell = np.array(positions, dtype=float), np.array(velocities, dtype=float), np.array(cell)
        if self.sites is None:
            self.sites = self.pos.copy()
        self.md_steps, self.rows, self.forces_current = 0, [], False
        self.log.append("set_state")

    def _forces(self):
        return -K * (self.pos - self.sites)

    def _energy(self):
        return 0.5 * K * float(((self.pos - self.sites) ** 2).sum()) - 3.0 * self.n

    def evaluate(self, positions, cell, compute_stress=True):
        assert np.array_equal(positions, self.pos)
        self.forces_current = True
        self.log.append("evaluate")
        return self._energy(), self._forces(), None

    # --- dynamics
    def run(self, n_steps, it):
        assert n_steps > 0
        if not self.forces_current:
            self.log.append("force pass")
        dt = it.dt_fs * units.fs
        inverse_mass = 1.0 / self.masses[:, None]
        forces = self._forces()
        for _ in range(n_steps):
            self.vel += 0.5 * dt * forces * inverse_mass
            self.pos += dt * self.vel
            forces = self._forces()
            self.vel += 0.5 * dt * forces * inverse_mass
            self.md_steps += 1
            self.total_steps += 1
            if it.thermo_every and self.md_steps % it.thermo_every == 0:
                e_kin = 0.5 * float((self.masses[:, None] * self.vel ** 2).sum())
                e_pot = float("nan") if self.md_steps == SpringEngine.poison_at else self._energy()
                self.rows.append((self.md_steps, e_pot, e_kin, 2 * e_kin / (3 * self.n * units.kB), 0.0, 0.0,
                                  e_pot + e_kin))
        self.forces_current = True
        self.log.append(("run", n_steps, it.kind, it.thermo_every))
        return _core.LoupMDStats(step_ms=0.5, rebuilds=1, final_temperature_k=300.0, pipelined_steps=n_steps - 1)

    def thermo_pending(self):
        return len(self.rows)

    def read_thermo(self, max_rows=None):
        out = np.array(self.rows, dtype=_core.THERMO_DTYPE)
        self.rows = []
        return out

    def get_state(self):
        return self.pos.copy(), self.vel.copy(), self._forces(), self._energy()

    def export_state(self):
        return pickle.dumps((self.n, self.pos, self.vel, self.cell, self.sites))

    def import_state(self, blob):
        n, pos, vel, cell, sites = pickle.loads(blob)
        if n != self.n:
            raise BackendError(f"loup_import_state: the state blob holds {n} atoms, the system {self.n}",
                               call="loup_import_state")
        self.pos, self.vel, self.cell, self.sites = pos.copy(), vel.copy(), cell.copy(), sites.copy()
        self.md_steps, self.rows, self.forces_current = 0, [], False
        self.log.append("import_state")


@pytest.fixture
def setup(tmp_path):
    checkpoint = tmp_path / "some.model"
    checkpoint.write_bytes(b"never unpickled here")
    SpringEngine.made, SpringEngine.poison_at = [], None

    def ensure_bundle(model_path, numbers, head=None, charge=None, spin_multiplicity=None, environ=None):
        return model_path.with_name("bundle.loup")

    def force_field(path=checkpoint):
        return loupmd.load(str(path), trusted=True, engine_factory=SpringEngine, ensure_bundle=ensure_bundle,
                           probe=lambda p: None)

    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((2, 1, 1))          # 16 atoms
    atoms.set_velocities(np.random.default_rng(3).normal(0.0, 0.02, atoms.positions.shape))

    def make(directory=None, integrator=None, **more):
        out = Output(str(tmp_path / directory), **more) if directory else None
        return loupmd.Simulation(atoms, force_field(), integrator=integrator or integrators.NVE(timestep_fs=1.0),
                                 output=out, seed=7)

    make.atoms, make.force_field, make.tmp_path, make.checkpoint = atoms, force_field, tmp_path, checkpoint
    return make


def _steps(directory):
    return [int(row[0]) for row in output.read_thermo(directory)[1]]


# ------------------------------------------------------------------------------------------------ the run

def test_a_run_writes_fastmace_s_rows_and_frames(setup):
    sim = setup("run")
    summary = sim.run(steps=100)
    directory = setup.tmp_path / "run"
    assert sim.step == 100 and summary.steps == 100 and summary.pipelined_steps == 95 and summary.rebuilds == 5
    assert _steps(directory) == [0, 20, 40, 60, 80, 100]                     # step 0, then every 20
    frames = output.read_frames(directory)
    assert len(frames) == 6
    engine, = SpringEngine.made
    position, velocity, forces, energy = engine.get_state()
    assert np.array_equal(frames[-1].positions, position) and frames[-1].get_potential_energy() == energy
    assert np.allclose(frames[-1].get_momenta(), setup.atoms.get_masses()[:, None] * velocity)
    assert np.array_equal(frames[0].positions, setup.atoms.positions)
    header, rows = output.read_thermo(directory)
    assert header == list(output.THERMO_HEADER)
    last = rows[-1]
    assert float(last[1]) == pytest.approx(0.1) and last[2] == "constant" and last[3] == "nan"   # NVE: no target
    assert float(last[5]) == pytest.approx(energy / 16)
    assert float(last[7]) == pytest.approx(float(last[5]) + float(last[6]))
    # The first row is made on the host from the state at step 0; energy is conserved to the integrator's order.
    assert abs(float(rows[0][7]) - float(last[7])) < 1e-5
    # One force evaluation before the first step, none per call after it; chunks of 20 steps with thermo on.
    assert engine.log[:2] == ["set_state", "evaluate"] and "force pass" not in engine.log
    assert [entry for entry in engine.log if isinstance(entry, tuple)] == [("run", 20, _core.LOUP_NVE, 20)] * 5
    # No restart point was due (every 10,000 steps) and none was asked for: one chunk, no restart file.
    assert sorted(p.name for p in directory.iterdir()) == ["simulation.json", "thermo.0.csv", "trajectory.0.traj"]


def test_rows_between_frames_come_from_the_device_buffer(setup):
    sim = setup("run", thermo_every=5, trajectory_every=20, restart_every=None)
    sim.run(steps=40)
    directory = setup.tmp_path / "run"
    assert _steps(directory) == list(range(0, 41, 5)) and len(output.read_frames(directory)) == 3
    assert [e for e in SpringEngine.made[0].log if isinstance(e, tuple)] == [("run", 20, _core.LOUP_NVE, 5)] * 2
    assert not (directory / "restart.npz").exists()


def test_without_a_trajectory_the_chunks_are_as_long_as_the_buffer_allows(setup, monkeypatch):
    monkeypatch.setattr(simulation, "THERMO_ROWS_PER_CALL", 10)
    sim = setup("run", thermo_every=2, trajectory_every=None, restart_every=None)
    sim.run(steps=50)
    assert _steps(setup.tmp_path / "run") == list(range(0, 51, 2)) and output.read_frames(setup.tmp_path / "run") == []
    assert [e[1] for e in SpringEngine.made[0].log if isinstance(e, tuple)] == [20, 20, 10]


def test_two_calls_equal_one(setup):
    one = setup("one")
    one.run(steps=100)
    two = setup("two")
    two.run(steps=60)
    two.run(steps=40)
    assert output.read_thermo(setup.tmp_path / "one") == output.read_thermo(setup.tmp_path / "two")
    for a, b in zip(output.read_frames(setup.tmp_path / "one"), output.read_frames(setup.tmp_path / "two")):
        assert np.array_equal(a.positions, b.positions) and np.array_equal(a.get_momenta(), b.get_momenta())


def test_a_run_without_output_writes_nothing_and_gives_the_atoms(setup):
    sim = setup()
    sim.run(steps=30)
    engine, = SpringEngine.made
    assert [e for e in engine.log if isinstance(e, tuple)] == [("run", 30, _core.LOUP_NVE, 0)]
    current = sim.atoms
    assert np.array_equal(current.positions, engine.pos) and current.get_potential_energy() == engine._energy()
    assert np.allclose(current.get_velocities(), engine.vel)
    assert not np.array_equal(current.positions, setup.atoms.positions)       # the caller's atoms are untouched
    assert list(setup.tmp_path.iterdir()) == [setup.checkpoint]


def test_langevin_reaches_the_engine_with_its_settings(setup):
    sim = setup("run", integrator=integrators.Langevin(timestep_fs=2.0, temperature_K=350.0, friction_per_fs=0.02))
    sim.run(steps=20)
    engine, = SpringEngine.made
    assert [e for e in engine.log if isinstance(e, tuple)] == [("run", 20, _core.LOUP_LANGEVIN, 20)]
    rows = output.read_thermo(setup.tmp_path / "run")[1]
    assert rows[-1][3] == "350.0" and float(rows[-1][1]) == pytest.approx(0.04)
    record = json.loads((setup.tmp_path / "run" / "simulation.json").read_text())
    assert record["integrator"] == {"ensemble": "nvt", "scheme": "langevin", "timestep_fs": 2.0,
                                    "temperature_K": 350.0, "friction_per_fs": 0.02, "seed": 7}
    assert record["seed"] == 7 and record["model"]["sha256"] == sim.force_field.sha256


# ----------------------------------------------------------------------------------------------- refusals

def test_argument_errors_come_before_anything_is_exported(setup):
    ff = setup.force_field()
    with pytest.raises(InputError, match="integrator"):
        loupmd.Simulation(setup.atoms, ff, integrator="langevin")
    with pytest.raises(InputError, match="force_field"):
        loupmd.Simulation(setup.atoms, "omat0-medium", integrator=integrators.NVE(timestep_fs=1.0))
    with pytest.raises(InputError, match="output"):
        loupmd.Simulation(setup.atoms, ff, integrator=integrators.NVE(timestep_fs=1.0), output="run/")
    with pytest.raises(InputError, match="seed"):
        loupmd.Simulation(setup.atoms, ff, integrator=integrators.NVE(timestep_fs=1.0), seed=-1)
    sim = setup()
    for steps in (0, -5, 2.5):
        with pytest.raises(InputError, match="steps must be a positive whole number"):
            sim.run(steps=steps)
    with pytest.raises(PlanRefused, match="fixcm"):
        integrators.Langevin(temperature_K=300, friction_per_fs=0.01, fixcm=True).to_config(seed=1)
    assert SpringEngine.made == []


def test_a_directory_that_holds_a_run_is_refused(setup):
    setup("run").run(steps=20)
    with pytest.raises(InputError, match="already holds output chunks"):
        setup("run").run(steps=20)


def test_a_non_finite_energy_stops_the_run(setup):
    SpringEngine.poison_at = 40
    sim = setup("run")
    with pytest.raises(NonFiniteError, match="step 40"):
        sim.run(steps=100)
    assert sim.status == "non-finite" and _steps(setup.tmp_path / "run") == [0, 20]
    with pytest.raises(NonFiniteError):
        sim.run(steps=20)                                           # it stays stopped


# ------------------------------------------------------------------------------------------------ restart

def test_restart_points_end_chunks_and_resume_continues_the_run(setup):
    whole = setup("whole", restart_every=40)
    whole.run(steps=100)
    part = setup("part", restart_every=40)
    part.run(steps=50)                                              # a restart point at 40
    part.save_restart()                                             # and one asked for at 50
    saved = SpringEngine.made[-1].export_state()
    part.close()
    directory = setup.tmp_path / "part"
    assert output.chunk_indices(directory) == [0, 1, 2]
    resumed = loupmd.Simulation.resume(directory, force_field=setup.force_field())
    assert resumed.step == 50 and SpringEngine.made[-1].export_state() == saved      # the state, bitwise
    resumed.run(steps=50)
    assert output.read_thermo(directory) == output.read_thermo(setup.tmp_path / "whole")
    for a, b in zip(output.read_frames(directory), output.read_frames(setup.tmp_path / "whole")):
        assert np.array_equal(a.positions, b.positions) and np.array_equal(a.get_momenta(), b.get_momenta())
    assert resumed.atoms.positions.tolist() == whole.atoms.positions.tolist()


def test_a_restart_between_thermo_steps_keeps_the_rows_on_multiples_of_the_cadence(setup):
    whole = setup("whole", restart_every=None)
    whole.run(steps=100)
    part = setup("part", restart_every=None)
    part.run(steps=50)
    part.save_restart()
    part.close()
    resumed = loupmd.Simulation.resume(setup.tmp_path / "part", force_field=setup.force_field())
    resumed.run(steps=50)
    assert _steps(setup.tmp_path / "part") == [0, 20, 40, 60, 80, 100]
    engine = SpringEngine.made[-1]
    # 10 steps without the buffer to step 60, the state moved onto a fresh step count there, then 20 + 20.
    assert [e for e in engine.log if e != "set_state"] == ["import_state", "force pass", ("run", 10, 0, 0),
                                                           "import_state", "force pass", ("run", 20, 0, 20),
                                                           ("run", 20, 0, 20)]
    assert np.array_equal(resumed.atoms.positions, whole.atoms.positions)
    assert output.read_thermo(setup.tmp_path / "part") == output.read_thermo(setup.tmp_path / "whole")


def test_resume_rebuilds_the_run_from_its_directory(setup, monkeypatch):
    sim = setup("run", integrator=integrators.Langevin(temperature_K=300.0, friction_per_fs=0.01), thermo_every=10)
    sim.run(steps=40)
    sim.save_restart()
    sim.close()
    made = {}

    def fake_load(model, head=None, **options):
        made.update(model=model, head=head, options=options)
        return setup.force_field()

    monkeypatch.setattr(simulation, "load", fake_load)
    resumed = loupmd.Simulation.resume(setup.tmp_path / "run")
    assert made["model"] == str(setup.checkpoint.resolve()) and made["options"]["sha256"] == sim.force_field.sha256
    assert resumed.step == 40 and resumed.seed == 7 and resumed.output.thermo_every == 10
    assert resumed.config == IntegratorConfig(temperature_K=300.0, friction_per_fs=0.01, seed=7)
    assert np.array_equal(resumed.atoms.numbers, setup.atoms.numbers) and all(resumed.atoms.pbc)


def test_a_restart_of_another_model_or_atom_count_is_refused(setup, tmp_path):
    sim = setup("run")
    sim.run(steps=20)
    sim.save_restart()
    sim.close()
    directory = setup.tmp_path / "run"
    other = tmp_path / "other.model"
    other.write_bytes(b"another file")
    with pytest.raises(RestartMismatch) as caught:
        loupmd.Simulation.resume(directory, force_field=setup.force_field(other))
    assert "was written with the model of SHA-256" in str(caught.value) and "other.model" in str(caught.value)
    # A restart whose state block holds another number of atoms than its own record.
    data = dict(np.load(directory / "restart.npz", allow_pickle=False))
    data["blob"] = np.frombuffer(pickle.dumps((15,) + pickle.loads(data["blob"].tobytes())[1:]), dtype=np.uint8)
    np.savez(directory / "restart.npz", **data)
    with pytest.raises(RestartMismatch, match="does not fit"):
        loupmd.Simulation.resume(directory, force_field=setup.force_field())
    with pytest.raises(RestartMismatch, match="no restart"):
        loupmd.Simulation.resume(tmp_path / "nothing-here", force_field=setup.force_field())
    assert _steps(directory) == [0, 20]                             # a refused resume deletes nothing


def test_save_restart_without_output_needs_a_path(setup, tmp_path):
    sim = setup()
    sim.run(steps=10)
    with pytest.raises(InputError, match="path"):
        sim.save_restart()
    target = sim.save_restart(tmp_path / "state" / "restart.npz")
    data = np.load(target, allow_pickle=False)
    assert int(data["step"]) == 10 and str(data["format"]) == "loupmd.restart/1"
    assert str(data["model_sha256"]) == sim.force_field.sha256 and int(data["output_chunk"]) == -1
