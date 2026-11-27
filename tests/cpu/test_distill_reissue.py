"""Publish only gate-qualified teacher labels, on exactly the original frozen geometries."""
import hashlib
import json

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

from loupgym.distill import reissue


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    base = tmp_path / 'original'
    reports = tmp_path / 'gate'
    base.mkdir()
    reports.mkdir()
    library = tmp_path / 'corrected.so'
    library.write_bytes(b'corrected')
    splits = {}
    for i, split in enumerate(('train', 'valid', 'teacher-test')):
        path = base / f'{split}.xyz'
        trajectory = f'{split}-trajectory'
        rows = []
        for j in range(2):
            a = Atoms('SrTiO3', positions=np.arange(15).reshape(5, 3) / 11 + i + j / 10,
                      cell=np.diag([8., 9.5, 10.]) + i / 4, pbc=True)
            a.set_momenta(a.positions / 5)
            a.new_array('REF_forces', a.positions.copy())
            a.info.update(split=split, sampler='teacher', integrator='langevin', md_seed=i,
                          trajectory_id=trajectory, frame_id=f'{trajectory}:{j}', step=j, engine_build='former',
                          teacher_head='omat_pbe', teacher_sha256=reissue.TEACHER_SHA256,
                          sampler_head='omat_pbe', sampler_sha256=reissue.TEACHER_SHA256,
                          REF_energy=-1., REF_stress=np.eye(3))
            write(path, a, format='extxyz', append=j != 0)
            labels = {'REF_forces': (a.positions * np.pi).tolist(), 'REF_energy': -.01234567891234,
                      'REF_stress': (np.eye(3) / 7).tolist()}
            passed = {'passed': True, 'failed': [], 'atoms': len(a)}
            rows.append(dict(frame_id=a.info['frame_id'], labels=labels, sequential=passed, fresh=passed,
                             sequential_vs_fresh=dict(energy_abs_eV_atom=1e-9, force_max_abs_eV_A=1e-7,
                                                      stress_max_abs_eV_A3=1e-9)))
        splits[split] = (2, digest(path))
        report = dict(schema='loupmd.frozen-cell-gate.v1', passed=True, frames=2, source_sha256=digest(path),
                      teacher_head='omat_pbe', teacher_sha256=reissue.TEACHER_SHA256, library_version='corrected',
                      library_sha256=digest(library), bundle_sha256='2' * 64, rows=rows)
        (reports / f'{split}.json').write_text(json.dumps(report))
    monkeypatch.setattr(reissue, 'FROZEN', splits)
    return base, reports, library, tmp_path / 'base-labels-v2'


def issue(inputs):
    base, reports, lib, out = inputs
    reissue.main(['--base', str(base), '--reports', str(reports), '--library', str(lib), '--output', str(out)])


def test_release_preserves_sampling_and_exact_qualified_values(inputs):
    base, reports, _, out = inputs
    issue(inputs)
    manifest = json.loads((out / 'manifest.json').read_text())
    for split, (count, source_sha) in reissue.FROZEN.items():
        old, new, gate = base / f'{split}.xyz', out / f'{split}.xyz', reports / f'{split}.json'
        assert digest(old) == source_sha == manifest['splits'][split]['source_sha256']
        assert digest(new) == manifest['splits'][split]['output_sha256']
        assert digest(gate) == manifest['splits'][split]['gate_sha256']
        assert len(read(new, ':')) == count
        for a, b, row in zip(read(old, ':'), read(new, ':'), json.loads(gate.read_text())['rows'], strict=True):
            for key in ('frame_id', 'trajectory_id', 'md_seed', 'engine_build', 'teacher_sha256', 'split'):
                assert a.info[key] == b.info[key]
            for key in ('numbers', 'positions', 'momenta'):
                np.testing.assert_array_equal(a.arrays[key], b.arrays[key])
            np.testing.assert_array_equal(a.cell, b.cell)
            assert b.info['label_release'] == 'base-labels-v2'
            np.testing.assert_array_equal(b.arrays['REF_forces'], row['labels']['REF_forces'])
            np.testing.assert_array_equal(b.info['REF_stress'], row['labels']['REF_stress'])
            assert b.info['REF_energy'] == row['labels']['REF_energy']
    with pytest.raises(FileExistsError):
        issue(inputs)


@pytest.mark.parametrize('fault', ['source', 'source-receipt', 'library', 'head', 'bundle', 'overall', 'frame',
                                   'count', 'fresh', 'sequential', 'nonfinite', 'force-shape', 'stress'])
def test_bad_final_split_does_not_release_partial_data(inputs, fault):
    base, reports, _, out = inputs
    path = reports / 'teacher-test.json'
    report = json.loads(path.read_text())
    row = report['rows'][-1]
    if fault == 'source':
        with (base / 'teacher-test.xyz').open('ab') as stream:
            stream.write(b'\n')
    elif fault == 'source-receipt':
        report['source_sha256'] = '3' * 64
    elif fault == 'library':
        report['library_sha256'] = '3' * 64
    elif fault == 'head':
        report['teacher_head'] = 'matpes_r2scan'
    elif fault == 'bundle':
        report['bundle_sha256'] = '3' * 64
    elif fault == 'overall':
        report['passed'] = False
    elif fault == 'frame':
        report['rows'].reverse()
    elif fault == 'count':
        report['rows'].pop()
    elif fault in ('fresh', 'sequential'):
        row[fault]['passed'] = False
    elif fault == 'nonfinite':
        row['labels']['REF_energy'] = float('inf')
    elif fault == 'force-shape':
        row['labels']['REF_forces'].pop()
    else:
        row['labels']['REF_stress'][0][1] = 1
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        issue(inputs)
    assert not out.exists()
    assert not list(out.parent.glob('.base-labels-v2.*'))


@pytest.mark.parametrize('quantity,error', [('energy_abs_eV_atom', 2e-5),
                                           ('force_max_abs_eV_A', 2e-5),
                                           ('stress_max_abs_eV_A3', 2e-7)])
def test_individual_stock_passes_do_not_excuse_history_dependence(inputs, quantity, error):
    _, reports, _, out = inputs
    p = reports / 'valid.json'
    report = json.loads(p.read_text())
    report['rows'][1]['sequential_vs_fresh'][quantity] = error
    p.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='sequential vs fresh'):
        issue(inputs)
    assert not out.exists()


def test_changed_sampling_cell_requires_investigation_not_new_labels(inputs, monkeypatch):
    base, reports, _, out = inputs
    path = base / 'valid.xyz'
    atoms = read(path, ':')
    atoms[1].set_cell(atoms[1].cell.array * 1.01, scale_atoms=False)
    write(path, atoms, format='extxyz')
    pin = digest(path)
    monkeypatch.setitem(reissue.FROZEN, 'valid', (2, pin))
    gate = reports / 'valid.json'
    record = json.loads(gate.read_text())
    record['source_sha256'] = pin
    gate.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='investigate sampling'):
        issue(inputs)
    assert not out.exists()
