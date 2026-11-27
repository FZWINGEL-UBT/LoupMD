"""Stage 0 measurements: synthetic time lines only, never initialise CUDA/NVML."""
import hashlib
import json
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from tools.dev import student_census as census


def _database(path):
    with sqlite3.connect(path) as db:
        db.executescript('''
            CREATE TABLE StringIds(id INTEGER, value TEXT);
            INSERT INTO StringIds VALUES(1, 'generic_gemm<float>');
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start, end, demangledName, gridX, gridY, gridZ,
                blockX, blockY, blockZ, staticSharedMemory, dynamicSharedMemory, deviceId, globalPid);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
                (100, 120, 1, 5,1,1, 32,2,1, 0,128, 0,17),
                (105, 110, 1, 5,1,1, 32,2,1, 0,128, 0,17),
                (115, 130, 1, 5,1,1, 32,1,2, 0,128, 0,17),
                (140, 145, 1, 5,1,1, 32,2,1, 0,256, 0,17);
        ''')


def test_node_sums_do_not_double_count_overlap_or_assign_gemms(tmp_path):
    path = tmp_path / 'trace.sqlite'
    _database(path)
    result = census.parse_trace(path, 2)
    assert result['kernel_sum_ns'] == 45
    assert result['kernel_union_ns'] == 35
    assert result['kernel_span_ns'] == 45
    assert result['inter_kernel_gap_ns'] == 10
    assert result['overlap_ns'] == 10
    assert result['launches'] == 4 and result['launches_per_step'] == 2
    assert 'ms_per_step' not in result  # traced span is not the sustained step
    assert [(r['launches'], r['sum_ns']) for r in result['kernels']] == [(2, 25), (1, 15), (1, 5)]
    assert {r['name'] for r in result['kernels']} == {'generic_gemm<float>'}
    assert result['kernels'][0]['sum_ms_per_step'] == 25 / 2e6


@pytest.mark.parametrize('mutation', ["UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET deviceId=2 WHERE start=105",
    "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET globalPid=2 WHERE start=105", 'DELETE FROM StringIds',
    'DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL', 'UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET end=start-1'])
def test_invalid_trace_not_partial_result(tmp_path, mutation):
    path = tmp_path / 'trace.sqlite'
    _database(path)
    with sqlite3.connect(path) as db:
        db.execute(mutation)
    with pytest.raises(ValueError):
        census.parse_trace(path, 8)


def test_missing_sqlite_never_created(tmp_path):
    path = tmp_path / 'missing.sqlite'
    with pytest.raises((FileNotFoundError, sqlite3.OperationalError)):
        census.parse_trace(path, 8)
    assert not path.exists()


@pytest.mark.parametrize('name, cap', [('NVIDIA L4', 89), ('NVIDIA RTX 5060 Ti', 120),
                                       ('NVIDIA RTX 4000 Ada Generation', 86)])
def test_only_benchmark_card_is_allowed(name, cap):
    with pytest.raises(ValueError, match='RTX 4000 Ada'):
        census.require_ada(name, cap)
    census.require_ada('NVIDIA RTX 4000 Ada Generation', 89)


@pytest.mark.parametrize('atoms', [320, 5000, 40000])
def test_physical_state(atoms):
    at = census.hot_cell(atoms)
    assert len(at) == atoms and all(at.pbc)
    assert np.bincount(at.numbers)[[38, 22, 8]].tolist() == [atoms // 5, atoms // 5, 3 * atoms // 5]
    assert 240 < at.get_temperature() < 360
    assert np.array_equal(at.get_velocities(), census.hot_cell(atoms).get_velocities())


def test_latter_half_is_by_elapsed_time_not_block_count():
    # The 3rd block straddles the wall-time midpoint. Only two complete blocks count.
    blocks = [{'end_ns': end, 'steps': n} for end, n in [(250_000_000, 30), (500_000_000, 30),
              (2_000_000_000, 80), (3_000_000_000, 80), (3_250_000_000, 20)]]
    r = census.timing_summary(0, blocks)
    assert r['read_start_ns'] == 2_000_000_000
    assert r['read_steps'] == 100 and r['ms_per_step'] == 12.5
    with pytest.raises(ValueError, match='3 s'):
        census.timing_summary(0, blocks[:2])


def test_large_cell_burn_in_cannot_mask_a_cold_timing_window():
    blocks = [{'end_ns': i, 'final_temperature_K': t}
              for i, t in enumerate((300., 300., 301., 283., 285., 287.))]
    with pytest.raises(ValueError, match='temperature'):
        census._measurement_temperature(blocks, 2)
    # A single cold final endpoint is not the temperature of an entire window.
    blocks[-3]['final_temperature_K'] = 299.
    blocks[-2]['final_temperature_K'] = 301.
    assert census._measurement_temperature(blocks, 2) == 299.


@pytest.mark.parametrize('temperature', [290., 310., 315.])
def test_census_uses_the_operational_band_not_an_iid_temperature_interval(temperature):
    blocks = [{'end_ns': i, 'final_temperature_K': temperature} for i in range(10)]
    assert census._measurement_temperature(blocks, 4) == temperature


def test_process_memory_is_pid_specific_not_board_memory():
    rows = [SimpleNamespace(pid=7, usedGpuMemory=1234), SimpleNamespace(pid=8, usedGpuMemory=2**28)]
    assert census.process_bytes(rows, 7) == 1234
    assert census.process_bytes(rows, 99) == 0  # no context yet
    rows[0].usedGpuMemory = 2**64 - 1
    with pytest.raises(ValueError, match='memory'):
        census.process_bytes(rows, 7)


def test_sweep_separates_wall_and_node_processes(tmp_path, monkeypatch):
    commands = []
    bundle = tmp_path / 's64.loup'
    out = tmp_path / 'records'

    def record(command, check):
        assert check
        cmd = list(map(str, command))
        commands.append(cmd)
        if 'time' in cmd or 'trace' in cmd:
            child = cmd[cmd.index('trace' if 'trace' in cmd else 'time'):]
            n = int(child[child.index('--atoms') + 1])
            p = child[child.index('--output') + 1]
            with open(p, 'w') as f:
                json.dump({'atoms': n, 'bundle_sha256': 'a' * 64, 'gpu_uuid': 'test', 'build': 'test',
                           'profiled_steps': 16, 'memory': {'baseline_bytes': 200,
                           'sampled_peak_bytes': 200 + 1000 * n}}, f)
        elif cmd[:2] == ['nsys', 'export']:
            _database(cmd[cmd.index('--output') + 1])

    monkeypatch.setattr(census.subprocess, 'run', record)
    census.main(['sweep', str(bundle), '--sha256', 'a' * 64, '--output', str(out), '--trace-steps', '16'])
    for n in (320, 5000, 40000):
        direct, node = [cmd for cmd in commands if '--atoms' in cmd and str(n) in cmd]
        assert direct[0] != 'nsys' and 'time' in direct
        assert node[:2] == ['nsys', 'profile'] and 'trace' in node
        assert {'--cuda-graph-trace=node', '--capture-range=cudaProfilerApi'} <= set(node)
    assert json.loads((out / 'summary.json').read_text())['memory_slope_bytes_per_atom_5000_to_40000'] == 1000


def test_only_exported_bytes_matching_digest_are_accepted(tmp_path):
    path = tmp_path / 'student.loup'
    path.write_bytes(b'checkpoint')
    with pytest.raises(ValueError, match='exported'):
        census._verified_bundle(path, hashlib.sha256(path.read_bytes()).hexdigest())
    path.write_bytes(b'LOUPC001fixture')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='SHA256'):
        census._verified_bundle(path, '0' * 64)
    assert census._verified_bundle(path, digest) == digest


@pytest.mark.parametrize('fail', [False, True])
def test_only_requested_steps_inside_profiler_range(monkeypatch, fail):
    calls = []

    def api(name):
        def call():
            calls.append(name)
            return 0
        return call

    lib = SimpleNamespace(cudaProfilerStart=api('start'), cudaProfilerStop=api('stop'),
                          cudaDeviceSynchronize=api('sync'))
    monkeypatch.setattr(census.ctypes, 'CDLL', lambda _: lib)

    def run(n, seed):
        calls.append((n, seed))
        if fail:
            raise RuntimeError('trajectory failed')
        return 'stats'

    if fail:
        with pytest.raises(RuntimeError, match='trajectory failed'):
            census._capture(run, 16, 7)
        assert calls == ['sync', 'start', (16, 7), 'stop']
    else:
        assert census._capture(run, 16, 7) == 'stats'
        assert calls == ['sync', 'start', (16, 7), 'sync', 'stop']
