"""The bundle cache: the ``.loup`` file for a model, an element set, a head and a charge / spin setting.

A bundle is exported once and reused. Its file name carries a key over everything that decides its contents (the
model file's SHA-256, the elements, the head, the path, charge and spin, and the exporter's own sources), so a
changed exporter never finds an old bundle. Two processes that ask for the same bundle export it once: the export
runs under a file lock and the file appears under its name only when it is complete.

Until the layout move (PLAN.md S0-T03) the exporter is ``python/export.py`` of the source checkout and the key is
computed by ``python/loupmd.py``; bundles that file wrote into the cache root are found here as well.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

from . import _legacy, cache
from .errors import InputError, LoupError


def _integral(name, value):
    """Charge and spin multiplicity are categorical model inputs: 0.5 must not become 0, and 1.0 is 1."""
    if value is None:
        return None
    if isinstance(value, bool) or int(value) != value:
        raise InputError(f"{name} must be an integer, got {value!r}")
    return int(value)


def bundle_name(model_path, atomic_numbers, head=None, generic=None, charge=None, spin_multiplicity=None) -> str:
    """The file name of the bundle: model stem, elements and the 16-digit key."""
    charge = _integral("charge", charge)
    spin_multiplicity = _integral("spin_multiplicity", spin_multiplicity)
    z_sorted = sorted({int(z) for z in atomic_numbers})
    if generic is None:
        generic = os.environ.get("LOUP_GENERIC", "0") == "1"
    tag = (f"{_legacy._model_digest(str(model_path))}|{z_sorted}|{head}|{int(bool(generic))}|{charge}|"
           f"{spin_multiplicity}|{_legacy.exporter_fingerprint()}")
    key = hashlib.sha256(tag.encode()).hexdigest()[:16]
    return f"{Path(model_path).stem}_{'_'.join(map(str, z_sorted))}_{key}.loup"


def _export_command(model_path, z_sorted, output, head, generic, charge, spin_multiplicity, exporter):
    exporter = exporter or [sys.executable, str(_legacy._SOURCE.parent / "export.py")]
    cmd = [*exporter, "--model", str(model_path), "--elements", ",".join(map(str, z_sorted)), "--output", str(output)]
    if head is not None:
        cmd += ["--head", head]
    if generic:
        cmd.append("--generic")
    if charge is not None:
        cmd += ["--charge", str(charge)]
    if spin_multiplicity is not None:
        cmd += ["--spin-multiplicity", str(spin_multiplicity)]
    return cmd


def ensure_bundle(model_path, atomic_numbers, *, head=None, generic=None, charge=None, spin_multiplicity=None,
                  environ=None, exporter=None) -> Path:
    """The cached bundle, exporting it on a miss.

    ``exporter`` replaces the export command's first words (``[python, export.py]``); the tests use it.
    """
    model_path = Path(model_path)
    if not model_path.is_file():
        raise InputError(f"model file {model_path} not found")
    charge = _integral("charge", charge)
    spin_multiplicity = _integral("spin_multiplicity", spin_multiplicity)
    z_sorted = sorted({int(z) for z in atomic_numbers})
    if generic is None:
        generic = os.environ.get("LOUP_GENERIC", "0") == "1"
    name = bundle_name(model_path, z_sorted, head, generic, charge, spin_multiplicity)
    out = cache.cache_dir("bundles", environ=environ) / name
    old_layout = cache.cache_root(environ) / name       # where python/loupmd.py keeps its bundles
    for existing in (out, old_layout):
        if existing.is_file():
            cache.mark_in_use(existing)
            return existing
    with cache.locked(out):
        if not out.is_file():                           # another process may have exported it while we waited
            part = out.with_name(out.name + f".part.{os.getpid()}")
            cmd = _export_command(model_path, z_sorted, part, head, generic, charge, spin_multiplicity, exporter)
            try:
                done = subprocess.run(cmd, capture_output=True, text=True)
                if done.returncode != 0 or not part.is_file():
                    tail = (done.stderr or done.stdout or "").strip().splitlines()[-6:]
                    raise LoupError(f"the export of {model_path.name} for elements {z_sorted} failed",
                                    why="; ".join(tail) or f"the exporter ended with status {done.returncode}",
                                    next_step="Run the command by hand to see its full output: " + " ".join(cmd))
                os.replace(part, out)
            finally:
                part.unlink(missing_ok=True)
    cache.mark_in_use(out)
    return out
