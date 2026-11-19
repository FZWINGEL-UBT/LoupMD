"""One force call of MODEL on SYSTEM inside cudaProfilerStart/Stop, after 20 warm-up calls (ncu driver, run with
--profile-from-start off): python prof_eval.py MODEL SYSTEM [--generic] [--stress]"""
import ctypes
import sys

import numpy as np

sys.path.insert(0, "/mnt/d/Margraf/LoupMD/python")
import bench_family as bf  # noqa: E402
from loupmd import LoupEngine, ensure_loup_bundle  # noqa: E402

model, name = sys.argv[1], sys.argv[2]
generic = "--generic" in sys.argv[3:]
stress = "--stress" in sys.argv[3:]
at = bf.system(name)
at.positions += np.random.default_rng(7).normal(0.0, 0.05, at.positions.shape)
eng = LoupEngine(ensure_loup_bundle(model, at.get_atomic_numbers().tolist(), generic=generic or None))
eng.set_system(at.get_atomic_numbers(), at.get_masses(), skin=0.5)
for _ in range(20):
    eng.evaluate(at.positions, at.cell.array, compute_stress=stress)
cudart = ctypes.CDLL("libcudart.so.13")
cudart.cudaProfilerStart()
e, f, s = eng.evaluate(at.positions, at.cell.array, compute_stress=stress)
cudart.cudaProfilerStop()
print(f"PROF-EVAL {model.split('/')[-1]} {name} {len(at)} atoms: E {e:.6f}")
