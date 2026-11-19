"""Runs python/test_gate.py (3x float32 rule) per supported model family; slow (exports and float64 references).

The default model (LOUP_TEST_MODEL or MACE-OMAT-0 medium) runs the oxide set with each forced conv forward variant.
The other families run on their FastMACE gate sets when their model file exists in LOUP_MODELS_DIR (default: the
workstation's or leeloo's model directory).
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest
from conftest import ROOT

MODELS_DIRS = [os.environ.get("LOUP_MODELS_DIR", ""), "/home/fzwin/models",
               "/mnt/homes/zwingel/ResearchModule/Models"]

FAMILIES = [
    ("mace-omat-0-small.model", ["--set", "oxide"]),
    ("MACE-OFF23_small.model", ["--set", "organic"]),
    ("MACE-OFF23_medium.model", ["--set", "organic"]),
    ("mace-mh-1.model", ["--set", "crystals"]),
    ("mace-mh-1.model", ["--set", "organic", "--head", "omol"]),
    ("MACE-POLAR-1-S.model", ["--set", "crystals"]),
    ("MACE-POLAR-1-M.model", ["--set", "crystals"]),
    # 1024 hidden channels and the charge / spin embedding folded into the bundle (neutral singlet and a cation)
    ("MACE-omol-0-extra-large-1024.model", ["--set", "organic"]),
    ("MACE-omol-0-extra-large-1024.model", ["--set", "organic", "--charge", "1", "--spin-multiplicity", "2"]),
]


def _run(args):
    proc = subprocess.run([sys.executable, str(ROOT / "python" / "test_gate.py"), *args], capture_output=True,
                          text=True)
    print(proc.stdout[-6000:])
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]


def test_accuracy_gate(model_path):
    _run([model_path])


@pytest.mark.parametrize("name,args", FAMILIES, ids=[f"{n}{''.join(a)}" for n, a in FAMILIES])
def test_family_gate(name, args):
    path = next((os.path.join(d, name) for d in MODELS_DIRS if d and os.path.exists(os.path.join(d, name))), None)
    if path is None:
        pytest.skip(f"{name} not found")
    _run([path, *args])
