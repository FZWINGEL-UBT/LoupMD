"""The run-time compiled layer-0 kernels (LOUP_JIT=1, JIT_DESIGN.md stage 1) against the built-in ones.

LOUP_JIT is read once per process, so each side runs in a subprocess of its own. Conv tuning is off (LOUP_CONV_TUNE=0)
so both sides run the same conv variant; what is left between two processes is the order of floating-point atomics
(MH-1 NaCl216, measured 19 November 2026: two built-in runs 2.3e-7 eV/A apart, built-in against JIT 1.6e-7).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from conftest import ROOT, find_model

_CHILD = r"""
import json, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from ase.build import bulk
from loupmd import LoupEngine, ensure_loup_bundle
at = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((3, 3, 3))
at.positions += np.random.default_rng(5).normal(0.0, 0.05, at.positions.shape)
eng = LoupEngine(ensure_loup_bundle(sys.argv[2], [11, 17]))
eng.set_system(at.get_atomic_numbers(), at.get_masses())
e, f, s = eng.evaluate(at.positions, at.cell.array)
print("RESULT " + json.dumps({"e": e, "f": np.asarray(f).tolist(), "s": np.asarray(s).tolist(), "n": len(at)}))
"""


def _run(model: str, cache: Path, **env) -> tuple[dict, str]:
    e = dict(os.environ, LOUP_CONV_TUNE="0", LOUP_CACHE_DIR=str(cache), LOUP_DEBUG_JIT="1", **env)
    p = subprocess.run([sys.executable, "-c", _CHILD, str(ROOT / "python"), model], env=e, capture_output=True,
                       text=True, timeout=600)
    assert p.returncode == 0, p.stderr[-2000:]
    line = next(x for x in p.stdout.splitlines() if x.startswith("RESULT "))
    return json.loads(line[7:]), p.stderr


def _close(a: dict, b: dict):
    n = a["n"]
    assert abs(a["e"] - b["e"]) / n < 1e-6
    assert np.abs(np.array(a["f"]) - np.array(b["f"])).max() < 2e-6
    assert np.abs(np.array(a["s"]) - np.array(b["s"])).max() < 1e-8


def test_jit_layer0_matches_built_in_kernels_and_is_cached(tmp_path):
    model = find_model("mace-mh-1.model")  # ZBL, 65 radial columns: the three-column edge backward
    # the built-in side shares the cache folder (its GEMM plans), so both sides run the same plans
    ref, _ = _run(model, tmp_path, LOUP_JIT="0")
    jit, err = _run(model, tmp_path, LOUP_JIT="1")
    assert "kernel(s) compiled" in err, err[-2000:]
    _close(ref, jit)
    files = list((tmp_path / "jit").glob("layer0_*.bin"))
    assert len(files) == 1, files
    again, err2 = _run(model, tmp_path, LOUP_JIT="1")
    assert "from the disk cache" in err2, err2[-2000:]
    _close(ref, again)


def test_jit_without_nvrtc_keeps_built_in_kernels(tmp_path):
    model = find_model("mace-mh-1.model")
    ref, _ = _run(model, tmp_path, LOUP_JIT="0")
    out, err = _run(model, tmp_path, LOUP_JIT="1", LOUP_NVRTC=str(tmp_path / "no_such_libnvrtc.so"))
    assert "using the built-in kernels" in err, err[-2000:]
    assert not (tmp_path / "jit").exists()
    _close(ref, out)
