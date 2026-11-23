"""Each native engine reservation advances one run. Reports name immutable endpoint records."""
from __future__ import annotations

from contextlib import closing, contextmanager
import csv
import os
from pathlib import Path

import numpy as np
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import Stationary

from loupmd.output import chunk_paths
from loupmd.quick import draw_initial_velocities

from .observable_io import digest, file_hash, geometry, load_model, model_args, publish, record


def add_commands(sub):
    p = sub.add_parser('md', help='start one model/seed/ensemble in a NEW directory (GPU reservation)')
    model_args(p)
    p.add_argument('--input', type=Path, required=True, help='ASE structure; frame 0. Use equilibrated structures for NVE.')
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--ensemble', choices=['nvt', 'npt', 'nve'], required=True)
    p.add_argument('--temperature-K', type=float, required=True, help='also sets Maxwell-Boltzmann initial velocities for NVE')
    p.add_argument('--timestep-fs', type=float, required=True)
    p.add_argument('--stride', type=int, default=1, help='uniform sampling interval in MD steps, including step zero')
    p.add_argument('--friction-per-fs', type=float, help='Langevin NVT only (default .01/fs)')
    p.add_argument('--pressure-GPa', type=float, help='target pressure, required for isotropic MTK NPT')
    p.add_argument('--tdamp-fs', type=float, help='NPT thermostat damping time (default 100 fs)')
    p.add_argument('--pdamp-fs', type=float, help='NPT barostat damping time (default 1000 fs)')
    p.add_argument('--steps', type=int, required=True, help='chunk length; start small and size to the reservation')
    p = sub.add_parser('extend', help='resume EXACT thermostat/barostat/engine state at a named endpoint')
    p.add_argument('--previous', type=Path, required=True, help='latest segment-STEP.json (cannot fork or replay)')
    p.add_argument('--steps', type=int, required=True, help='additional steps, not a target step')


@contextmanager
def exclusive_writer(directory):
    # Native jobs run under Linux. The OS releases the lock even when a reservation kills the writer.
    import fcntl
    with (directory / 'writer.lock').open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def run(opts):
    import loupmd as lm

    if opts.steps < 1:
        raise ValueError('--steps must be positive')
    prev = None
    if opts.command == 'md':
        directory = opts.directory
        at = read(opts.input, index=0)
        system = digest(geometry(at))
        if (len(at) < 2 or not 0 <= opts.seed < 2**63 or not np.isfinite(opts.temperature_K)
                or opts.temperature_K <= 0 or opts.stride <= 0):
            raise ValueError('need at least two atoms, positive temperature/cadence and a nonnegative 63-bit seed')
        if opts.ensemble != 'npt' and any(x is not None for x in [opts.pressure_GPa, opts.tdamp_fs, opts.pdamp_fs]):
            raise ValueError('barostat options require NPT')
        if opts.ensemble != 'nvt' and opts.friction_per_fs is not None:
            raise ValueError('Langevin friction requires NVT')
        at.info.clear()
        # Do not let velocities from an input trajectory collapse the two independent seeds.
        at.set_momenta(np.zeros((len(at), 3)))
        draw_initial_velocities(at, opts.temperature_K, opts.seed)
        Stationary(at, preserve_temperature=False)
        params = {'timestep_fs': opts.timestep_fs, 'fixcm': True}
        if opts.ensemble == 'npt':
            if opts.pressure_GPa is None:
                raise ValueError('NPT needs an explicit target pressure')
            params.update(temperature_K=opts.temperature_K, pressure_GPa=opts.pressure_GPa,
                          tdamp_fs=100. if opts.tdamp_fs is None else opts.tdamp_fs,
                          pdamp_fs=1000. if opts.pdamp_fs is None else opts.pdamp_fs, cell_mode='hydrostatic')
            integrator = lm.integrators.MTK(**params)
        elif opts.ensemble == 'nvt':
            params.update(temperature_K=opts.temperature_K, friction_per_fs=.01 if opts.friction_per_fs is None else opts.friction_per_fs)
            integrator = lm.integrators.Langevin(**params)
        else:
            integrator = lm.integrators.NVE(**params)
        ff, model, build = load_model(opts, at)
        protocol = dict(system=system, ensemble=opts.ensemble, initial_K=opts.temperature_K, engine_build=build,
                        timestep_fs=opts.timestep_fs, stride=opts.stride, integrator=type(integrator).__name__,
                        parameters=params)
        origin = dict(kind='observable-md-v1', role=opts.role, model=model, protocol=protocol, seed=opts.seed,
                      initial_momenta_sha256=digest(at.get_momenta().tolist()))
        directory.mkdir(exist_ok=False, parents=True)
        publish(directory/'origin.json', origin)
        out = lm.Output(directory=directory, thermo_every=opts.stride, restart_every=None,
                        trajectory_every=opts.stride if opts.ensemble != 'nve' else None)
    else:
        directory = opts.previous.parent
        prev = record(opts.previous, 'segment')
        origin = record(directory/'origin.json', 'observable-md-v1')
        if prev['origin'] != digest(origin) or origin['protocol']['engine_build'] != lm.library_version():
            raise ValueError('wrong origin or engine build for continuation')
    with exclusive_writer(directory):
        if prev is None:
            sim = lm.Simulation(at, ff, integrator=integrator, seed=opts.seed, output=out)
            start, index = 0, 0
        else:
            latest = max(int(p.stem.split('-')[1]) for p in directory.glob('segment-*.json'))
            if latest != prev['end_step']:
                raise ValueError('previous endpoint is not the latest sealed segment; refusing to rewind')
            saved = directory / f"state-{prev['restart_sha256']}.npz"
            if file_hash(saved) != prev['restart_sha256']:
                raise ValueError('native state does not match the sealed segment')
            if file_hash(directory/'simulation.json') != prev['simulation_sha256']:
                raise ValueError('native integration settings or model record changed since the previous segment')
            sim = lm.Simulation.resume(directory, restart=saved)
            start, index = prev['end_step'], prev['chunk'] + 1
        with closing(sim):
            if sim.step != start:
                raise ValueError('restart step mismatch')
            sim.run(steps=opts.steps)
            restart = sim.save_restart()
            end = sim.step
        thermo, traj = chunk_paths(directory, index)
        state_id = file_hash(restart)
        # The native writer atomically replaces restart.npz. Keep the inode of
        # each committed state so a killed next job can never destroy it.
        state_path = directory / f'state-{state_id}.npz'
        try:
            os.link(restart, state_path)
        except FileExistsError:
            if file_hash(state_path) != state_id:
                raise ValueError('sealed native state was modified')
        endpoint = dict(kind='segment', origin=digest(origin), previous=digest(prev) if prev else None,
                        start_step=start, end_step=end, chunk=index, restart_sha256=state_id,
                        simulation_sha256=file_hash(directory/'simulation.json'),
                        thermo_sha256=file_hash(thermo), trajectory_sha256=file_hash(traj))
        path = directory / f'segment-{end}.json'
        publish(path, endpoint)
    print(path)


class Snapshot:
    """Read a sealed prefix, independent of later extensions of the run."""
    def __init__(self, path):
        self.path = Path(path)
        self.directory = self.path.parent
        self.origin = record(self.directory/'origin.json', 'observable-md-v1')
        self.protocol = self.origin['protocol']
        key = digest(self.origin)
        s = record(self.path, 'segment')
        self.end = s['end_step']
        self.id = digest(s)
        self.chunks = []
        while True:
            if s['origin'] != key or s['start_step'] >= s['end_step'] or s['chunk'] < 0:
                raise ValueError('invalid segment chain')
            self.chunks.append(s)
            if s['chunk'] == 0:
                if s['start_step'] or s['previous'] is not None:
                    raise ValueError('missing initial segment')
                break
            parent = record(self.directory/f"segment-{s['start_step']}.json", 'segment')
            if (digest(parent), parent['end_step'], parent['chunk'] + 1) != (s['previous'], s['start_step'], s['chunk']):
                raise ValueError('broken segment chain')
            s = parent
        self.chunks.reverse()

    def data(self, *, thermo):
        count = 0
        stride = self.protocol['stride']
        for s in self.chunks:
            csvfile, traj = chunk_paths(self.directory, s['chunk'])
            path, key = (csvfile, 'thermo_sha256') if thermo else (traj, 'trajectory_sha256')
            if file_hash(path) != s[key]:
                raise ValueError(f'chunk content changed: {path}')
            if thermo:
                with path.open(newline='') as f:
                    for row in csv.DictReader(f):
                        if int(row['step']) != count * stride:
                            raise ValueError('gap or duplicate thermo step')
                        count += 1
                        yield row
            elif path.stat().st_size:
                with Trajectory(str(path)) as trajectory:
                    for at in trajectory:
                        count += 1
                        yield at
        if count != 1 + self.end // stride:
            raise ValueError('missing or duplicated samples')
