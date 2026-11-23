"""The labelled 320-atom curve geometry whose stress failed only after radial-table packing."""
import os
from pathlib import Path

import pytest
from ase.io import read
from gpu_support import find_model

from loupmd import load
from loupmd.validate import gate, references


@pytest.mark.skipif(os.environ.get('LOUP_TEST_FP32_TABLE') != '1', reason='requires make label-lib, not the packed MD library')
def test_unpacked_teacher_agrees_with_stock_on_low_noise_stress():
    path = find_model('mace-mh-1.model')
    atoms = read(Path(__file__).parent / 'data/sto320-curve-replica365.xyz')
    name = 'sto320:target:replica365'
    ref64, ref32, stress = references.compute(path, [(name, atoms)], 'omat_pbe', 'cuda', None, None)
    teacher = load('mh1', checkpoint=path, head='omat_pbe')
    result = gate.compare(name, {}, len(atoms), teacher.evaluate(atoms, stress=True), ref64[0], ref32[0], stress[0])
    assert result.passed, result
