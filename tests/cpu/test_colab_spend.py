import json

import pytest

from tools.dev.colab_spend import tally


def receipt(root, name, *, session, gpu='L4', **fields):
    path = root / name
    path.mkdir(parents=True)
    (path / 'allocation.json').write_text(json.dumps(dict(account='1', session=session, gpu=gpu, **fields)))
    return path


def test_charge_only_our_allocated_time_and_honor_manual_stop(tmp_path):
    first = receipt(tmp_path, 'old', session='one', allocated=100, stopped=8300)
    (first / 'recovery').mkdir()
    (first / 'recovery/stopped-unix').write_text('3700')  # controller returned much later
    receipt(tmp_path, 'active', session='two', gpu='T4', allocated=3700)
    receipt(tmp_path, 'denied', session='three', start=33, stopped=34)
    report = tally(tmp_path, now=5500)
    assert report['accounts']['1']['estimated_cu'] == pytest.approx(1.94 + 1.15 / 2)
    assert report['accounts']['1']['active_sessions'] == ['two']
    assert report['accounts']['1']['no_allocation_receipt'] == ['three']
    assert report['accounts']['2']['estimated_cu'] == 0


def test_copied_receipt_cannot_double_charge_campaign(tmp_path):
    for run in ['original', 'copy']:
        receipt(tmp_path, run, session='same', allocated=1, stopped=10)
    with pytest.raises(ValueError, match='duplicate'):
        tally(tmp_path, now=20)


def test_negative_allocation_interval_is_not_a_credit(tmp_path):
    receipt(tmp_path, 'bad', session='early', allocated=100, stopped=20)
    with pytest.raises(ValueError, match='interval'):
        tally(tmp_path, now=600)
