"""``Simulation`` with a barostat (plan task S4-T07) on the fake engine of tests/test_simulation.py, taught a
made-up equation of state: the cell is the engine's, rows carry volume and pressure, frames and the restart carry
the cell, and the engine's stop at the cell limits becomes ``CellDivergedError``. No GPU needed."""
from __future__ import annotations

import numpy as np
import pytest
from ase import units
from ase.build import bulk
from ase.constraints import FixAtoms
from test_simulation import SpringEngine

import loupmd
from loupmd import BackendError, CellDivergedError, InputError, Output, PlanRefused, _core, integrators, output, report
from loupmd.config import IntegratorConfig

BULK_GPA = 100.0            # the fake's bulk modulus


class CellEngine(SpringEngine):
    """The spring engine with a cell that answers pressure: P = B (V_eq / V - 1) plus the ideal-gas part, with
    V_eq 5% above the starting volume. A Berendsen NPT call scales cell, positions and spring sites as ASE's
    ``NPTBerendsen`` does."""

    diverge_at = None       # an md step at which the run stops as the engine does at its cell limits

    def set_state(self, positions, velocities, cell):
        super().set_state(positions, velocities, cell)
        self.cell = np.array(cell, dtype=float)
        self.v_eq = 1.05 * abs(np.linalg.det(self.cell))
        self.reference_volume = abs(np.linalg.det(self.cell))

    def get_cell(self):
        return self.cell.copy()

    def set_reference_volume(self, volume):
        self.reference_volume = float(volume)
        self.log.append(("reference volume", float(volume)))

    def _volume(self):
        return abs(float(np.linalg.det(self.cell)))

    def _virial_pressure(self):
        return BULK_GPA * (self.v_eq / self._volume() - 1.0)

    def _pressure(self):
        e_kin = 0.5 * float((self.masses[:, None] * self.vel ** 2).sum())
        return self._virial_pressure() + 2.0 * e_kin / (3.0 * self._volume()) / units.GPa

    def evaluate(self, positions, cell, compute_stress=True):
        energy, forces, _ = super().evaluate(positions, cell, compute_stress)
        stress = -self._virial_pressure() * units.GPa * np.eye(3) if compute_stress else None
        return energy, forces, stress

    def export_state(self):
        import pickle

        return pickle.dumps((self.n, self.pos, self.vel, self.cell, self.sites, self.v_eq))

    def import_state(self, blob):
        import pickle

        state = pickle.loads(blob)
        super().import_state(pickle.dumps(state[:5]))
        self.v_eq = state[5]
        if not hasattr(self, "reference_volume"):
            self.reference_volume = self._volume()

    def run(self, n_steps, it):
        if it.kind != _core.LOUP_BERENDSEN_NPT:
            return super().run(n_steps, it)
        dt = it.dt_fs * units.fs
        inverse_mass = self._inverse_mass()
        dof = 3 * self.n - (3 if it.fix_com else 0)
        forces = self._forces()
        for _ in range(n_steps):
            if CellEngine.diverge_at is not None and self.md_steps + 1 == CellEngine.diverge_at:
                raise BackendError(f"loup_run: the cell diverged: its volume is {21 * self.reference_volume:g} A^3, "
                                   f"outside [V0 / 20, 20 V0] with V0 = {self.reference_volume:g} A^3",
                                   call="loup_run")
            scale = 1.0 - it.dt_fs / it.tau_p_fs * it.compressibility_per_gpa / 3.0 * (it.pressure_gpa - self._pressure())
            factors = np.full(3, scale)
            if it.cell_mode == _core.LOUP_CELL_FULL:
                factors = np.where(np.array(list(it.cell_mask)) == 1, scale, 1.0)
            self.cell = factors[:, None] * self.cell                   # cubic cell: each vector by its factor
            self.pos, self.sites = self.pos * factors, self.sites * factors
            self.vel += 0.5 * dt * forces * inverse_mass
            self.pos += dt * self.vel
            forces = self._forces()
            self.vel += 0.5 * dt * forces * inverse_mass
            self.md_steps += 1
            self.total_steps += 1
            if it.thermo_every and self.md_steps % it.thermo_every == 0:
                e_kin = 0.5 * float((self.masses[:, None] * self.vel ** 2).sum())
                self.rows.append((self.md_steps, self._energy(), e_kin, 2 * e_kin / (dof * units.kB),
                                  self._pressure(), self._volume(), self._energy() + e_kin))
        self.forces_current = True
        self.log.append(("run", n_steps, it.kind, it.thermo_every))
        self.targets.append((n_steps, it.temp_k, it.temp_rate_k_per_step))
        return _core.LoupMDStats(step_ms=0.5, rebuilds=1, final_temperature_k=300.0, pipelined_steps=0)


NPT = dict(timestep_fs=1.0, temperature_K=300.0, taut_fs=50.0, pressure_GPa=0.0, taup_fs=20.0,
           compressibility_per_GPa=1.0 / BULK_GPA)


@pytest.fixture
def setup(tmp_path):
    checkpoint = tmp_path / "some.model"
    checkpoint.write_bytes(b"never unpickled here")
    SpringEngine.made, SpringEngine.poison_at, CellEngine.diverge_at = [], None, None

    def force_field():
        return loupmd.load(str(checkpoint), trusted=True, engine_factory=CellEngine,
                           ensure_bundle=lambda model_path, numbers, **more: model_path.with_name("bundle.loup"),
                           probe=lambda p: None)

    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((2, 2, 2))            # 64 atoms, cubic
    atoms.set_velocities(np.random.default_rng(3).normal(0.0, 0.02, atoms.positions.shape))

    def make(directory=None, integrator=None, structure=None, **more):
        out = Output(str(tmp_path / directory), **more) if directory else None
        return loupmd.Simulation(structure or atoms, force_field(),
                                 integrator=integrator or integrators.Berendsen(**NPT), output=out, seed=7)

    make.atoms, make.tmp_path, make.force_field = atoms, tmp_path, force_field
    return make


def test_the_record_and_its_struct():
    record = integrators.Berendsen(**NPT)
    config = record.to_config(seed=1)
    assert (config.ensemble, config.scheme, config.pressure_GPa, config.taup_fs, config.mask) == \
        ("npt", "berendsen", 0.0, 20.0, None)
    it = config.to_struct(thermo_every=5)
    assert (it.kind, it.tau_t_fs, it.tau_p_fs, it.pressure_gpa, it.cell_mode, it.thermo_every) == \
        (_core.LOUP_BERENDSEN_NPT, 50.0, 20.0, 0.0, _core.LOUP_CELL_HYDROSTATIC, 5)
    assert it.compressibility_per_gpa == pytest.approx(0.01)
    masked = integrators.Berendsen(**NPT, mask=[1, 1, 0]).to_config(seed=1)       # a run file's list
    it = masked.to_struct()
    assert masked.mask == (1, 1, 0) and it.cell_mode == _core.LOUP_CELL_FULL and list(it.cell_mask) == [1, 1, 0]
    assert integrators.from_data(integrators.to_data(record)) == record
    assert not integrators.Berendsen(timestep_fs=1.0, temperature_K=300.0, taut_fs=50.0).barostat

    for fields, reason in ((dict(pressure_GPa=None), "the Berendsen barostat needs pressure_GPa"),
                           (dict(taup_fs=None, compressibility_per_GPa=None),
                            "the Berendsen barostat needs taup_fs, compressibility_per_GPa"),
                           (dict(taup_fs=0.0), "taup_fs must be positive and finite, got 0.0"),
                           (dict(pressure_GPa=float("nan")), "pressure_GPa must be finite"),
                           (dict(mask=(0, 0, 0)), "mask frees no axis"),
                           (dict(mask=(1, 2, 0)), "mask must be three entries of 0 or 1")):
        with pytest.raises(InputError, match=reason):
            integrators.Berendsen(**{**NPT, **fields})
    with pytest.raises(InputError, match="pressure_GPa belongs to a barostat, and the ensemble is 'nvt'"):
        IntegratorConfig(ensemble="nvt", scheme="langevin", temperature_K=300.0, friction_per_fs=0.01,
                         pressure_GPa=1.0)
    with pytest.raises(InputError, match="npt with scheme 'baoab' is not in this version"):
        IntegratorConfig(ensemble="npt", scheme="baoab", temperature_K=300.0)


def test_an_npt_run_writes_cell_volume_and_pressure(setup):
    sim = setup("run", thermo_every=10, trajectory_every=50, restart_every=100)
    sim.run(steps=200)
    directory = setup.tmp_path / "run"
    engine, = SpringEngine.made
    header, rows = output.read_thermo(directory)
    assert header == list(output.THERMO_HEADER + output.NPT_COLUMNS)
    assert [int(row[0]) for row in rows] == list(range(0, 201, 10))
    volumes = np.array([float(row[8]) for row in rows]) * 64
    pressures = np.array([float(row[9]) for row in rows])
    start = setup.atoms.get_volume()
    assert volumes[0] == pytest.approx(start, rel=1e-12) and volumes[-1] == pytest.approx(engine._volume(), rel=1e-12)
    assert volumes[1] > volumes[0] and volumes[-1] > 1.03 * start               # towards V_eq = 1.05 V
    assert pressures[0] == pytest.approx(engine_start_pressure(setup.atoms), rel=1e-9)   # the host's row at step 0
    assert pressures[-1] == pytest.approx(engine._pressure(), rel=1e-12) and abs(pressures[-1]) < abs(pressures[0])

    frames = output.read_frames(directory)
    assert len(frames) == 5
    assert np.allclose(frames[0].cell.array, setup.atoms.cell.array)
    assert np.allclose(frames[-1].cell.array, engine.get_cell(), rtol=1e-12)
    assert frames[2].get_volume() > frames[1].get_volume() > frames[0].get_volume()
    current = sim.atoms
    assert np.array_equal(current.cell.array, engine.get_cell()) and np.array_equal(current.positions, engine.pos)

    with np.load(directory / "restart.npz") as data:
        assert np.array_equal(data["cell"], engine.get_cell())
        assert float(data["reference_volume_A3"]) == pytest.approx(start, rel=1e-15)
    sim.close()

    again = loupmd.Simulation.resume(directory, force_field=setup.force_field())
    resumed = SpringEngine.made[-1]
    assert ("reference volume", pytest.approx(start)) in [entry for entry in resumed.log if isinstance(entry, tuple)]
    assert np.array_equal(resumed.get_cell(), engine.get_cell())
    again.run(steps=50)
    header, rows = output.read_thermo(directory)
    assert header == list(output.THERMO_HEADER + output.NPT_COLUMNS) and int(rows[-1][0]) == 250
    assert float(rows[-1][8]) * 64 == pytest.approx(resumed._volume(), rel=1e-12)
    again.close()


def engine_start_pressure(atoms):
    """What the host row of step 0 holds: the fake's virial pressure at V = V_eq / 1.05 plus the ideal-gas part."""
    e_kin = float(atoms.get_kinetic_energy())
    return BULK_GPA * 0.05 + 2.0 * e_kin / (3.0 * atoms.get_volume()) / units.GPa


def test_rows_between_the_engine_s_own_cadence_carry_pressure(setup):
    """A restart cadence that is no multiple of the thermo cadence makes the simulation write rows from states it
    reads back; a barostat's rows then take their stress from one more evaluation."""
    sim = setup("odd", thermo_every=10, trajectory_every=None, restart_every=25)
    sim.run(steps=60)
    header, rows = output.read_thermo(setup.tmp_path / "odd")
    engine, = SpringEngine.made
    assert [int(row[0]) for row in rows] == list(range(0, 61, 10))
    assert all(np.isfinite(float(row[9])) and float(row[8]) > 0 for row in rows)
    assert float(rows[-1][9]) == pytest.approx(engine._pressure(), rel=1e-9)
    sim.close()


def test_the_cell_limits_and_other_refusals(setup):
    CellEngine.diverge_at = 30
    sim = setup("diverged", thermo_every=10)
    with pytest.raises(CellDivergedError, match="the cell left its limits between steps") as caught:
        sim.run(steps=100)
    assert "outside [V0 / 20, 20 V0]" in str(caught.value.why) and 30 <= caught.value.step <= 40
    CellEngine.diverge_at = None

    fixed = setup.atoms.copy()
    fixed.set_constraint(FixAtoms(indices=[0, 1]))
    with pytest.raises(PlanRefused, match="Berendsen's barostat with fixed atoms"):
        setup(structure=fixed)
    with pytest.raises(PlanRefused, match="a barostat with fixed atoms"):
        loupmd.Simulation(fixed, setup.force_field(),
                          integrator=integrators.Berendsen(**NPT, fixcm=False).to_config(seed=1))


def test_the_report_names_the_barostat():
    config = integrators.Berendsen(**NPT, mask=(1, 1, 0)).to_config(seed=1)
    import dataclasses

    line = report._physics_lines({"integrator": dataclasses.asdict(config), "schedule": None, "step": 10,
                                  "seed": 1, "counters": {"steps": 10}})[0]
    assert line.startswith("NPT, Berendsen thermostat and barostat (experimental")
    assert "pressure 0 GPa" in line and "mask (1, 1, 0)" in line and "time constants 50 fs and 20 fs" in line


def test_the_ase_drop_ins_take_ase_s_arguments_and_move_the_cell(setup):
    from loupmd.ase import md as ase_md

    atoms = setup.atoms.copy()
    start = atoms.get_volume()
    dyn = ase_md.NPTBerendsen(atoms, 1.0 * units.fs, temperature_K=300, pressure_au=1.0 * units.GPa,
                              taut=50 * units.fs, taup=20 * units.fs, compressibility_au=0.01 / units.GPa,
                              fixcm=False, model=setup.force_field())
    config = dyn.simulation.config
    assert (config.ensemble, config.scheme, config.mask, config.fixcm) == ("npt", "berendsen", None, False)
    assert (config.pressure_GPa, config.taup_fs, config.compressibility_per_GPa, config.tau_fs) == \
        (pytest.approx(1.0), pytest.approx(20.0), pytest.approx(0.01), pytest.approx(50.0))
    assert dyn.run(100) is True
    engine = SpringEngine.made[-1]
    assert np.array_equal(atoms.cell.array, engine.get_cell()) and atoms.get_volume() > 1.02 * start
    assert np.array_equal(atoms.positions, engine.pos)

    masked = ase_md.Inhomogeneous_NPTBerendsen(setup.atoms.copy(), 1.0 * units.fs, temperature_K=300,
                                               pressure_au=0.0, compressibility_au=0.01 / units.GPa,
                                               mask=(1, 0, 1), model=setup.force_field())
    config = masked.simulation.config
    assert config.mask == (1, 0, 1) and config.fixcm is True
    assert (config.tau_fs, config.taup_fs) == (pytest.approx(500.0), pytest.approx(1000.0))      # ASE's defaults
    masked.run(40)
    cell = SpringEngine.made[-1].get_cell()
    assert cell[1, 1] == setup.atoms.cell.array[1, 1] and cell[0, 0] != setup.atoms.cell.array[0, 0]

    with pytest.raises(InputError, match="NPTBerendsen needs pressure_au, compressibility_au"):
        ase_md.NPTBerendsen(atoms, units.fs, temperature_K=300, model=setup.force_field())
    with pytest.raises(InputError, match="pressure= .bar. and compressibility= .1/bar. are ASE's deprecated units"):
        ase_md.NPTBerendsen(atoms, units.fs, temperature_K=300, pressure=1.0, model=setup.force_field())


def test_lm_md_takes_a_barostat_s_record(setup, tmp_path):
    record = integrators.Berendsen(**NPT)
    result = loupmd.md(setup.atoms, model=setup.force_field(), steps=40, integrator=record,
                       output=str(tmp_path / "quick"), seed=3)
    assert result.simulation.config.ensemble == "npt" and result.metadata["arguments"]["ensemble"] == "NPT"
    assert result.metadata["arguments"]["integrator"]["pressure_GPa"] == 0.0
    assert "NPT, Berendsen thermostat and barostat (experimental" in result.summary
    header, rows = output.read_thermo(tmp_path / "quick")
    assert header == list(output.THERMO_HEADER + output.NPT_COLUMNS) and int(rows[-1][0]) == 40
    with pytest.raises(PlanRefused, match="constant-pressure MD is not in this version"):
        loupmd.md(setup.atoms, model=setup.force_field(), steps=40, temperature_K=300, ensemble="NPT")
