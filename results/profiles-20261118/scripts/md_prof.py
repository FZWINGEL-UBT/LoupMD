"""One MD segment of MODEL on SYSTEM inside cudaProfilerStart/Stop (nsys capture): python md_prof.py MODEL SYSTEM STEPS"""
import ctypes
import sys

import numpy as np

sys.path.insert(0, "/mnt/d/Margraf/LoupMD/python")
import bench_family as bf  # noqa: E402
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution  # noqa: E402
from loupmd import LoupEngine, ensure_loup_bundle  # noqa: E402

model, name, steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
at = bf.system(name)
MaxwellBoltzmannDistribution(at, temperature_K=300.0, rng=np.random.default_rng(42))
eng = LoupEngine(ensure_loup_bundle(model, at.get_atomic_numbers().tolist()))
eng.set_system(at.get_atomic_numbers(), at.get_masses(), skin=0.5)
eng.set_state(at.positions, at.get_velocities(), at.cell.array)
eng.run_langevin(100, dt_fs=1.0, temp_k=300.0, friction_per_fs=0.01, seed=1)
cudart = ctypes.CDLL("libcudart.so.13")
cudart.cudaProfilerStart()
r = eng.run_langevin(steps, dt_fs=1.0, temp_k=300.0, friction_per_fs=0.01, seed=2)
cudart.cudaProfilerStop()
print(f"PROF {model.split('/')[-1]} {name} {len(at)} atoms: {steps} steps, {r.rebuilds} rebuilds, {r.step_ms:.4f} ms/step")
