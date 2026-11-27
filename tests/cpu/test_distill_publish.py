import json

import pytest

from loupgym.distill.publish import finish
from loupmd.errors import GateFailed


def test_failed_physics_gate_preserves_evidence_without_a_bundle(tmp_path):
    source = tmp_path / 'cache.loup'
    source.write_bytes(b'candidate')
    out = tmp_path / 'release'
    out.mkdir()
    with pytest.raises(GateFailed):
        finish(out, source, {'passed': False, 'rows': [{'passed': False, 'failed': ['stress']}]})
    assert json.loads((out/'gate.json').read_text())['rows'][0]['failed'] == ['stress']
    assert not (out/'student.loup').exists()
