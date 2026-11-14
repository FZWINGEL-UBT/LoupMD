"""Shared fixtures: model path, bundles, standard structures.

LOUP_TEST_MODEL selects the model (default: the OMAT-0 medium file on the local workstation or on leeloo).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))  # the loupmd package, ahead of python/loupmd.py (see loupmd/_legacy.py)
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from loupmd import LoupEngine, ensure_loup_bundle  # noqa: E402

MODEL_CANDIDATES = [
    "/home/fzwin/models/mace-omat-0-medium.model",
    "/mnt/homes/zwingel/ResearchModule/Models/mace-omat-0-medium.model",
]


MODELS_DIRS = [os.environ.get("LOUP_MODELS_DIR", ""), "/home/fzwin/models", "/mnt/homes/zwingel/ResearchModule/Models"]


def find_model(name: str) -> str:
    """Path of a model file in LOUP_MODELS_DIR or a workstation's model directory; skips the test if it is absent."""
    for d in MODELS_DIRS:
        if d and os.path.exists(os.path.join(d, name)):
            return os.path.join(d, name)
    pytest.skip(f"{name} not found (set LOUP_MODELS_DIR)")


@pytest.fixture(scope="session")
def model_path() -> str:
    path = os.environ.get("LOUP_TEST_MODEL") or next((p for p in MODEL_CANDIDATES if os.path.exists(p)), None)
    if path is None:
        pytest.skip("no model file (set LOUP_TEST_MODEL)")
    return path


def srtio3(rep: int):
    from ase import Atoms

    a = 3.905
    return Atoms(
        symbols=["Sr", "Ti", "O", "O", "O"],
        scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],
        cell=[a, a, a],
        pbc=True,
    ).repeat((rep, rep, rep))


@pytest.fixture(scope="session")
def sto_bundle(model_path):
    return ensure_loup_bundle(model_path, [8, 22, 38])


@pytest.fixture
def sto_engine(sto_bundle):
    return LoupEngine(sto_bundle, device_id=0)


def maxwell_velocities(atoms, temp_k: float, seed: int) -> np.ndarray:
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

    at = atoms.copy()
    MaxwellBoltzmannDistribution(at, temperature_K=temp_k, rng=np.random.default_rng(seed))
    Stationary(at)
    return at.get_velocities()
