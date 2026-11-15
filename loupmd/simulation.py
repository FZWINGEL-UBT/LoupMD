"""``lm.Simulation``: an MD run with output, restart and resume.

The engine owns the loop. ``run`` hands it the steps in chunks that end where the host has something to do: a
trajectory frame, a restart point, the end of the call. Thermo rows are written by the device inside the captured
step (``LoupIntegrator.thermo_every``) and read once per chunk; rows and frames go to disk on a background thread.

Steps are counted from the start of the run (``Simulation.step``). The engine counts from its last ``set_state`` or
``import_state`` and writes a thermo row where its own count is a multiple of the cadence; a run resumed between
two thermo steps therefore first runs to the next one without the buffer and moves the state onto a fresh count
there (export and import of the state block, which changes nothing else), so rows stay on multiples of the cadence.
"""
from __future__ import annotations

import dataclasses
import json
import math
import secrets
from dataclasses import dataclass
from pathlib import Path

from . import output as _output
from .config import IntegratorConfig, Output
from .errors import BackendError, InputError, LoupError, NonFiniteError, RestartMismatch
from .forcefield import ForceField, load

RESTART_NAME = "restart.npz"
SIMULATION_NAME = "simulation.json"
RESTART_FORMAT = "loupmd.restart/1"
SIMULATION_FORMAT = "loupmd.simulation/1"
# Thermo rows one engine call may write: the device buffer holds 4,096 unread rows (src/thermo.hpp).
THERMO_ROWS_PER_CALL = 4000


@dataclass(frozen=True)
class RunSummary:
    """What one ``run`` call did."""

    steps: int
    ms_per_step: float            # the engine's step time, averaged over the call's chunks
    pipelined_steps: int
    rebuilds: int
    dangerous_builds: int
    final_temperature_K: float


class Simulation:
    """``lm.Simulation(atoms, force_field, integrator=lm.integrators.Langevin(...), output=lm.Output("run/"),
    seed=1)``.

    - ``atoms``: the starting structure with its momenta (a structure at rest starts at rest); it is not changed;
    - ``force_field``: what ``lm.load`` returned;
    - ``integrator``: an ``lm.integrators`` record;
    - ``output``: an ``lm.Output`` (directory and cadences), or ``None`` for no files;
    - ``seed``: the thermostat's noise seed; ``None`` draws one (``Simulation.seed``, written to
      ``simulation.json``);
    - ``skin``: the neighbour-list skin in A.

    Nothing is exported and no engine is made before the first ``run``; argument errors come before that.
    """

    def __init__(self, atoms, force_field, *, integrator, output=None, seed=None, skin=0.5):
        from ase import Atoms

        if not isinstance(atoms, Atoms):
            raise InputError(f"atoms must be an ase.Atoms, got {type(atoms).__name__}")
        if not isinstance(force_field, ForceField):
            raise InputError(f"force_field must be what lm.load returns, got {type(force_field).__name__}",
                             next_step="ff = lm.load(model); lm.Simulation(atoms, ff, ...)")
        if output is not None and not isinstance(output, Output):
            raise InputError(f"output must be an lm.Output or None, got {type(output).__name__}",
                             next_step='output=lm.Output("run/")')
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 63):
            raise InputError(f"seed must be an int from 0 to 2**63 - 1 or None, got {seed!r}")
        if isinstance(integrator, IntegratorConfig):
            self.seed = integrator.seed if seed is None else seed
            self.config = dataclasses.replace(integrator, seed=self.seed)
        elif hasattr(integrator, "to_config"):
            self.seed = secrets.randbits(62) if seed is None else seed
            self.config = integrator.to_config(seed=self.seed)
        else:
            raise InputError(f"integrator must be an lm.integrators record, got {type(integrator).__name__}",
                             next_step="integrator=lm.integrators.Langevin(temperature_K=300, friction_per_fs=0.01)")
        self.force_field, self.output, self.skin = force_field, output, float(skin)
        self.step = 0
        self.status = "new"                # "new", "running", "idle", "non-finite", "closed"
        self._atoms = atoms.copy()
        self._engine = None
        self._chunks = self._writer = None
        self._base = 0                     # the step at which the engine's own count was zero
        self._resume = None                # (state block, step, output indices) of a resumed run
        self._stopped = None               # the error that ended the run

    # ---------------------------------------------------------------------------------------------- start

    def _start(self):
        import numpy as np

        atoms = self._atoms
        if self.output is not None:        # first: a directory that holds a run is refused before any export
            self._chunks = _output.ChunkedOutput(self.output.directory, atoms,
                                                 resume_from=self._resume[2] if self._resume else None)
            self._writer = _output.AsyncWriter()
        engine = self.force_field.new_engine(atoms, self.skin)
        self._masses = np.asarray(atoms.get_masses(), dtype=float)
        if self._resume is not None:
            blob, self.step, _ = self._resume
            try:
                engine.import_state(blob)
            except BackendError as error:
                raise RestartMismatch("the restart's state does not fit its own record",
                                      why=str(error).splitlines()[0],
                                      next_step="Use the restart file as it was written, or start a new run.") from None
            self._base = self.step
            self._engine = engine
        else:
            positions, velocities = atoms.positions, atoms.get_velocities()
            engine.set_state(positions, velocities, atoms.cell.array)
            energy, forces, _ = engine.evaluate(positions, atoms.cell.array, compute_stress=False)
            self._engine = engine
            if self.output is not None:
                self._write_record()
                self._submit_row(self._host_row(0, energy, velocities))
                if self.output.trajectory_every is not None:
                    self._submit_frame(positions.copy(), velocities, forces, energy)
        self.status = "idle"

    def _write_record(self):
        from . import __version__

        ff = self.force_field
        record = {
            "format": SIMULATION_FORMAT, "loupmd": __version__,
            "model": {"label": ff.label, "path": str(ff.path), "head": ff.head, "sha256": ff.sha256,
                      "charge": ff.charge, "spin_multiplicity": ff.spin_multiplicity},
            "integrator": dataclasses.asdict(self.config),
            "output": dataclasses.asdict(self.output),
            "seed": self.seed, "skin": self.skin, "atoms": len(self._atoms),
        }
        text = json.dumps(record, indent=2) + "\n"
        _output.write_atomically(Path(self.output.directory) / SIMULATION_NAME,
                                 lambda handle: handle.write(text.encode("utf-8")))

    # ------------------------------------------------------------------------------------------- output

    def _target(self):
        return float("nan") if self.config.temperature_K is None else float(self.config.temperature_K)

    def _row(self, step, e_pot, e_kin, temperature):
        n = len(self._atoms)
        return (int(step), step * self.config.timestep_fs / 1000.0, _output.CONSTANT_PHASE, self._target(),
                float(temperature), e_pot / n, e_kin / n, (e_pot + e_kin) / n)

    def _host_row(self, step, e_pot, velocities):
        """A row made from the state on the host, as the device makes its rows: used at step 0 and where the
        engine's count starts anew."""
        from ase import units

        e_kin = 0.5 * float((self._masses[:, None] * velocities ** 2).sum())
        return self._row(step, e_pot, e_kin, 2 * e_kin / (3 * len(self._atoms) * units.kB))

    def _check_finite(self, row):
        for name, value in zip(_output.THERMO_HEADER, row):
            if isinstance(value, float) and not math.isfinite(value) and name != "target_K":
                raise NonFiniteError(f"non-finite {name} ({value}) at step {row[0]}",
                                     why="the run stops at a non-finite value; it is never patched over")

    def _submit_row(self, row):
        self._check_finite(row)
        self._writer.submit(self._chunks.write_row, row)

    def _submit_frame(self, positions, velocities, forces, energy):
        self._writer.submit(self._chunks.write_frame, positions, velocities, forces, energy)

    # ---------------------------------------------------------------------------------------------- run

    def run(self, steps) -> RunSummary:
        """``steps`` more MD steps. Returns what the call did; ``Simulation.step`` counts all steps of the run."""
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise InputError(f"steps must be a positive whole number, got {steps!r}")
        if self._stopped is not None:
            raise self._stopped
        if self.status == "closed":
            raise InputError("this simulation is closed", next_step="Resume it: lm.Simulation.resume(directory).")
        if self._engine is None:
            self._start()
        try:
            return self._advance(steps)
        except NonFiniteError as error:
            self.status, self._stopped = "non-finite", error
            self._settle()
            raise

    def _settle(self):
        """After an error: what was written before it stays; nothing is written after it."""
        if self._writer is not None:
            try:
                self._writer.drain()
            except LoupError:
                pass
            self._chunks.flush()

    def _advance(self, steps) -> RunSummary:
        engine, out = self._engine, self.output
        thermo = out.thermo_every if out is not None else None
        frames = out.trajectory_every if out is not None else None
        restarts = out.restart_every if out is not None else None
        end = self.step + steps
        wall_ms, pipelined, rebuilds, dangerous, temperature = 0.0, 0, 0, 0, float("nan")
        self.status = "running"
        while self.step < end:
            n = end - self.step
            for every in (frames, restarts):
                if every is not None:
                    n = min(n, every - self.step % every)
            buffered = thermo is not None and self._base % thermo == 0
            if buffered:
                n = min(n, thermo * THERMO_ROWS_PER_CALL)
            elif thermo is not None:
                n = min(n, thermo - self.step % thermo)          # to the next thermo step, without the buffer
            try:
                stats = engine.run(n, self.config.to_struct(thermo_every=thermo if buffered else 0))
            except BackendError as error:
                if "non-finite" in str(error):
                    raise NonFiniteError(f"non-finite energy between steps {self.step} and {self.step + n}",
                                         why=str(error).splitlines()[0],
                                         next_step=NonFiniteError.default_next_step) from None
                raise
            self.step += n
            wall_ms += stats.step_ms * n
            pipelined += stats.pipelined_steps
            rebuilds += stats.rebuilds
            dangerous += stats.dangerous_builds
            temperature = stats.final_temperature_k
            if buffered:
                for row in engine.read_thermo():
                    self._submit_row(self._row(self._base + int(row["step"]), float(row["e_pot"]),
                                               float(row["e_kin"]), float(row["temp_k"])))
            state = None
            if thermo is not None and not buffered and self.step % thermo == 0:
                state = engine.get_state()
                self._submit_row(self._host_row(self.step, state[3], state[1]))
                engine.import_state(engine.export_state())      # the engine's count starts anew on a thermo step
                self._base = self.step
            if frames is not None and self.step % frames == 0:
                positions, velocities, forces, energy = state or engine.get_state()
                self._submit_frame(positions, velocities, forces, energy)
            if restarts is not None and self.step % restarts == 0:
                self.save_restart()
        if self._writer is not None:
            self._writer.drain()
        self.status = "idle"
        return RunSummary(steps=steps, ms_per_step=wall_ms / steps, pipelined_steps=pipelined, rebuilds=rebuilds,
                          dangerous_builds=dangerous, final_temperature_K=temperature)

    # ------------------------------------------------------------------------------------------ the state

    @property
    def atoms(self):
        """The structure as it is now: a copy with positions, momenta and a calculator holding the potential
        energy and the forces. Before the first ``run`` it is the starting structure."""
        atoms = self._atoms.copy()
        if self._engine is None:
            return atoms
        from ase.calculators.singlepoint import SinglePointCalculator

        positions, velocities, forces, energy = self._engine.get_state()
        atoms.set_positions(positions)
        atoms.set_velocities(velocities)
        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
        return atoms

    def save_restart(self, path=None) -> Path:
        """Write the restart file (``restart.npz`` in the output directory unless ``path`` is given). With output,
        this is a chunk boundary: rows and frames up to this step are closed and fsynced first."""
        import numpy as np

        if self._engine is None:
            raise InputError("nothing has run yet, so there is no state to save")
        if path is None:
            if self.output is None:
                raise InputError("save_restart needs a path when the simulation has no output directory",
                                 next_step='sim.save_restart("restart.npz")')
            path = Path(self.output.directory) / RESTART_NAME
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        indices = dict(output_chunk=-1, output_rows=0, output_frames=0)
        if self._chunks is not None:
            self._writer.drain()
            indices = self._chunks.restart_point()
        positions, velocities, _, energy = self._engine.get_state()
        atoms = self._atoms
        arrays = dict(format=np.array(RESTART_FORMAT), step=np.array(self.step, dtype=np.int64),
                      blob=np.frombuffer(self._engine.export_state(), dtype=np.uint8),
                      model_sha256=np.array(self.force_field.sha256), seed=np.array(self.seed, dtype=np.int64),
                      numbers=np.asarray(atoms.numbers, dtype=np.int64), masses=self._masses,
                      pbc=np.asarray(atoms.pbc, dtype=bool), cell=np.asarray(atoms.cell.array, dtype=float),
                      positions=positions, velocities=velocities, potential_energy=np.array(energy),
                      **{key: np.array(value, dtype=np.int64) for key, value in indices.items()})
        return _output.write_atomically(path, lambda handle: np.savez(handle, **arrays))

    def close(self):
        """Finish writing and close the output files. The run can be resumed from its last restart."""
        if self._writer is not None:
            self._writer.close(raise_errors=self._stopped is None)
            self._chunks.close()
            self._writer = self._chunks = None
        self._engine = None
        self.status = "closed"

    # --------------------------------------------------------------------------------------------- resume

    @classmethod
    def resume(cls, directory, *, force_field=None, restart=None):
        """Continue the run of ``directory`` from its restart file: ``sim = lm.Simulation.resume("run/")``.

        The model is loaded again as ``simulation.json`` names it (a label through the label table; a file by its
        path with the recorded SHA-256), or pass ``force_field``. A restart of another model, a state that does
        not fit, or output chunks that do not fit the restart raise ``RestartMismatch`` before anything is
        changed.
        """
        import numpy as np
        from ase import Atoms

        directory = Path(directory)
        restart = Path(restart) if restart is not None else directory / RESTART_NAME
        record_file = directory / SIMULATION_NAME
        if not restart.is_file() or not record_file.is_file():
            missing = restart if not restart.is_file() else record_file
            raise RestartMismatch(f"no restart to resume in {directory}: {missing.name} is missing",
                                  next_step="A run writes it every Output.restart_every steps and when "
                                            "save_restart() is called.")
        record = json.loads(record_file.read_text())
        with np.load(restart, allow_pickle=False) as data:
            data = {key: data[key] for key in data.files}
        if str(data["format"]) != RESTART_FORMAT or record.get("format") != SIMULATION_FORMAT:
            raise RestartMismatch(f"{restart} has format {str(data['format'])!r} and {record_file.name} "
                                  f"{record.get('format')!r}; this version reads {RESTART_FORMAT!r} and "
                                  f"{SIMULATION_FORMAT!r}")
        saved_sha = str(data["model_sha256"])
        model = record["model"]
        if force_field is None:
            try:
                if model["label"] is not None:
                    force_field = load(model["label"], model["head"], charge=model["charge"],
                                       spin_multiplicity=model["spin_multiplicity"])
                else:
                    force_field = load(model["path"], model["head"], sha256=saved_sha, charge=model["charge"],
                                       spin_multiplicity=model["spin_multiplicity"])
            except LoupError as error:
                raise RestartMismatch(f"the model of the restart in {directory} cannot be loaded again",
                                      why=str(error).splitlines()[0],
                                      next_step="Pass force_field= with the model the run was written with.") from None
        if force_field.sha256 != saved_sha:
            raise RestartMismatch(
                f"the restart in {directory} was written with the model of SHA-256 {saved_sha}; "
                f"{force_field.path.name} has {force_field.sha256}",
                next_step=RestartMismatch.default_next_step)
        atoms = Atoms(numbers=data["numbers"], positions=data["positions"], cell=data["cell"], pbc=data["pbc"])
        atoms.set_masses(data["masses"])
        atoms.set_velocities(data["velocities"])
        out = Output(**record["output"]) if record.get("output") else None
        if out is not None and Path(out.directory).resolve() != directory.resolve():
            out = dataclasses.replace(out, directory=str(directory))     # the directory was moved or is relative
        sim = cls(atoms, force_field, integrator=IntegratorConfig(**record["integrator"]), output=out,
                  seed=int(data["seed"]), skin=float(record["skin"]))
        indices = {key: int(data[key]) for key in _output.OUTPUT_KEYS}
        sim._resume = (data["blob"].tobytes(), int(data["step"]), indices if out is not None else None)
        sim._start()
        return sim
