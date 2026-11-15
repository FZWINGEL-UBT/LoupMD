"""What the GPU tests share: the model files, the standard structures, velocities.

Model files are found in one place only: the folder ``LOUPMD_MODELS`` names. Structures come from the package
(``loupmd.bench.suites``).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from loupmd.bench import suites

ROOT = Path(__file__).resolve().parents[2]
MODELS_VARIABLE = "LOUPMD_MODELS"
TEST_MODEL_VARIABLE = "LOUPMD_TEST_MODEL"
DEFAULT_MODEL = "mace-omat-0-medium.model"


def find_model(name: str) -> str:
    """The path of a model file in the folder ``LOUPMD_MODELS`` names; skips the test when it is not there."""
    folder = os.environ.get(MODELS_VARIABLE)
    if not folder:
        pytest.skip(f"{MODELS_VARIABLE} names no folder with the model files ({name})")
    path = Path(os.path.expanduser(folder)) / name
    if not path.is_file():
        pytest.skip(f"{name} is not in {MODELS_VARIABLE} ({folder})")
    return str(path)


def srtio3(rep: int):
    """Cubic SrTiO3, ``rep`` cells along each axis (5 atoms per cell), as the benchmark's."""
    return suites.srtio3(rep)


def water_box():
    """The 648-atom water box shipped with the package."""
    from ase.io import read

    return read(suites.structure_file("water"))


def maxwell_velocities(atoms, temp_k: float, seed: int) -> np.ndarray:
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

    at = atoms.copy()
    MaxwellBoltzmannDistribution(at, temperature_K=temp_k, rng=np.random.default_rng(seed))
    Stationary(at)
    return at.get_velocities()
