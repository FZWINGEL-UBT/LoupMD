"""Stage 0 instruments: analytical trajectories and ASE forces, no engine or torch."""
import numpy as np
import pytest
from ase import Atoms, units
from ase.calculators.emt import EMT
from ase.build import bulk

from loupgym.distill import observables as obs
from loupgym.distill import observable_static as static


def frame(x=0.0, side=10.):
    at = Atoms('HHe', scaled_positions=[[x % 1, 0, 0], [(x + .1) % 1, 0, 0]],
               cell=[side] * 3, pbc=True)
    return at


def test_rdf_unlike_and_like_unique_pairs_and_half_cell():
    at = Atoms('H2He', positions=[[0, 0, 0], [1, 0, 0], [0, 2, 0]], cell=[10] * 3, pbc=True)
    edges = np.array([0., 1.5, 2.5])
    shells = 4 * np.pi / 3 * np.diff(edges**3)
    rdf = obs.partial_rdf(at, edges)
    np.testing.assert_allclose(rdf['H-H'] * shells / 1000, [1, 0])
    np.testing.assert_allclose(rdf['H-He'] * shells / 1000, [0, 1])
    assert 'He-He' not in rdf
    at.set_cell([[10, 0, 0], [9, 1, 0], [0, 0, 10]])
    with pytest.raises(ValueError, match='height'):
        obs.partial_rdf(at, edges)


def test_wrapping_and_barostat_affine_change_are_not_diffusion():
    disp = obs.unwrapped_path([frame(.95), frame(.05), frame(.15)])
    np.testing.assert_allclose(disp[:, 0, 0], [0, 1, 2])
    np.testing.assert_allclose(obs.unwrapped_path([frame(.8, 10), frame(.8, 11)]), 0, atol=1e-14)


def test_equal_blocks_lag_diffusion_and_npt_units():
    # Ballistic test path: MSD(t)=t² at 1 Å/fs; the requested 1..2 fs fit has slope 3 Å²/fs.
    frames = [frame(.9 + i * .1) for i in range(8)]
    out = obs.statistics(iter(frames), sample_fs=1, block_frames=4, lags=[1, 2],
                         edges=[0., 2.], diffuse=['He'], npt=True)
    assert out['blocks'] == 2
    for key, expected in [('rms_displacement_A:He', [1, 2]), ('diffusion_cm2_s:He', 0.05),
                          ('cell_lengths_A', [10, 10, 10]),
                          ('density_g_cm3', frames[0].get_masses().sum() * units._amu * 1e27 / 1000)]:
        np.testing.assert_allclose(out['metrics'][key]['mean'], expected)
        np.testing.assert_allclose(out['metrics'][key]['block_sem'], 0, atol=1e-14)
    with pytest.raises(ValueError, match='blocks'):
        obs.statistics(frames[:4], 1, 4, [1, 2], [0, 2], [], True)


def test_energy_centering_time_and_atom_units():
    # Total energy per atom; 1 meV/atom per ps despite a large arbitrary offset.
    r = obs.energy_metrics(np.arange(4) * 250., -10 + np.arange(4) * .00025)
    assert r['drift_meV_atom_ps'] == pytest.approx(1)
    assert r['rms_meV_atom'] == pytest.approx(.25 * np.std(np.arange(4)))
    with pytest.raises(ValueError):
        obs.energy_metrics([0, 0], [1, 2])


def test_empirical_spread_is_not_block_error_or_fixed_tolerance():
    def seed(v): return {'metrics': {'rdf:H-He': {'mean': v, 'block_sem': [5, 5]}}}
    r = obs.spread([seed([0, 1]), seed([2, 1])], [seed([2, 1]), seed([3, 1.1])])['rdf:H-He']
    np.testing.assert_allclose(r['teacher_seed_spread'], [2, 0])
    np.testing.assert_allclose(r['student_minus_teacher'], [1.5, .05])
    assert r['within_teacher_spread'] == [True, False]


def test_relax_and_cartesian_force_column_shards():
    at = bulk('Al', 'fcc', a=4.2, cubic=True)
    at.calc = EMT()
    result = static.relax(at, pressure_GPa=0, fmax=.01, steps=100)
    assert result['converged'] and at.cell.lengths()[0] < 4.2
    shards = [static.columns(at, delta=.01, start=i, stop=i+3) for i in range(0, len(at)*3, 3)]
    modes = static.frequencies(at, shards)
    assert len(modes) == 12 and max(modes) > 1  # THz, not eV
    with pytest.raises(ValueError, match='missing'):
        static.frequencies(at, shards[1:])
    with pytest.raises(ValueError, match='overlap'):
        static.frequencies(at, shards + [shards[0]])


def test_signed_frequencies_and_fixed_design_tolerance():
    r = static.compare_values([4, 4, 4], [4.004, 4, 4], [-1., .01, 10], [-.85, .09, 10.29])
    assert r['lattice_pass'] == [True] * 3
    assert r['phonon_pass'] == [False, True, True]
    assert r['tolerance_source'].startswith('design')


def test_json_artifacts_never_overwrite(tmp_path):
    path = tmp_path / 'result.json'
    obs.publish(path, {'sample': 1})
    with pytest.raises(FileExistsError):
        obs.publish(path, {'sample': 2})
    assert obs.record(path)['sample'] == 1


@pytest.fixture
def cli_md(tmp_path, monkeypatch):
    # Native Simulation, restart, trajectory and thermo writers, using the same
    # harmonic CPU engine as the checkout's simulation tests.
    from fake_core import SpringEngine
    from ase.io import write
    import loupmd as lm
    from loupmd import simulation

    model = tmp_path / 'springs.model'
    model.write_bytes(b'CPU test engine, never exported')
    load = lm.load

    def spring_load(*args, **kw):
        return load(str(model), trusted=True, probe=lambda p: None, engine_factory=SpringEngine,
                    ensure_bundle=lambda path, numbers, **kwargs: path.with_name('bundle.loup'))

    monkeypatch.setattr(lm, 'load', spring_load)
    monkeypatch.setattr(simulation, 'load', spring_load)
    monkeypatch.setattr(lm, 'library_version', lambda: 'CPU spring test')
    SpringEngine.made, SpringEngine.poison_at = [], None
    structure = tmp_path / 'periodic.xyz'
    write(structure, frame(), format='extxyz')

    def run(role, seed, ensemble='nvt', dt=1, stride=1, steps=7):
        directory = tmp_path / f'{role}-{seed}-{ensemble}-{dt}'
        obs.main(['md', '--model', str(model), '--input', str(structure), '--directory', str(directory),
                  '--role', role, '--seed', str(seed), '--ensemble', ensemble, '--temperature-K', '300',
                  '--timestep-fs', str(dt), '--stride', str(stride), '--steps', str(steps)])
        return str(directory / f'segment-{steps}.json')
    return run


def test_paired_statistics_cli_and_immutable_prefix(cli_md, tmp_path):
    teacher = [cli_md('teacher', i) for i in (7, 8)]
    student = [cli_md('student', i) for i in (8, 7)]
    obs.main(['extend', '--previous', teacher[0], '--steps', '3'])
    with pytest.raises(ValueError, match='rewind'):
        obs.main(['extend', '--previous', teacher[0], '--steps', '3'])
    dest = tmp_path / 'spread.json'
    args = ['statistics', '--teacher', *teacher, '--student', *student, '--output', str(dest),
            '--discard-frames', '0', '--block-frames', '4', '--lags', '1', '2', '--rmax-A', '2', '--bins', '3']
    obs.main(args)
    out = obs.record(dest)
    assert [r['run']['seed'] for r in out['student']] == [7, 8]
    assert out['teacher'][0]['run']['end_step'] == 7
    assert all(out['comparisons']['rms_displacement_A:H']['within_teacher_spread'])
    args[2] = teacher[1]
    args[args.index(str(dest))] = str(tmp_path/'bad.json')
    with pytest.raises(ValueError, match='different seeds'):
        obs.main(args)


def test_nve_two_step_cli_checks_physical_duration(cli_md, tmp_path):
    production = [cli_md(role, 10, 'nve', 1, 1, 7) for role in ('teacher', 'student')]
    small = [cli_md(role, 10, 'nve', .5, 2, 14) for role in ('teacher', 'student')]
    dest = tmp_path/'energies.json'
    obs.main(['nve', '--production', *production, '--conserving', *small, '--output', str(dest)])
    data = obs.record(dest)
    for mode in ('production', 'candidate_conserving'):
        assert data[mode]['teacher']['time_fs'] == [0, 7]
        assert all(data[mode]['no_worse_than_teacher'].values())
    obs.main(['extend', '--previous', small[0], '--steps', '2'])
    wrong = tmp_path/'teacher-10-nve-0.5'/'segment-16.json'
    with pytest.raises(ValueError, match='physical duration'):
        obs.main(['nve', '--production', production[0], '--conserving', str(wrong),
                  '--output', str(tmp_path/'wrong.json')])
