"""Exporter and bundle-cache checks (CPU only, except the fixture bundle).

The exporters set CUDA_VISIBLE_DEVICES for themselves, so they are driven in a subprocess, as ensure_loup_bundle does.
"""
from __future__ import annotations

import re
import subprocess
import sys

import pytest
from conftest import ROOT

import loupmd

DRIVER = """
import sys
sys.path.insert(0, {python_dir!r})
import export_omat0 as ex
{patch}
ex.export_model({model!r}, [8, 22, 38], ex.Path({out!r}))
"""


def _export_with(patch: str, model_path: str, tmp_path):
    src = DRIVER.format(python_dir=str(ROOT / "python"), patch=patch, model=model_path, out=str(tmp_path / "x.loup"))
    proc = subprocess.run([sys.executable, "-W", "ignore", "-c", src], capture_output=True, text=True)
    return proc, (tmp_path / "x.loup").exists()


def test_specialised_export_refuses_an_uncertified_spline(model_path, tmp_path):
    """The interaction-1 spline must be at least as accurate as the float32 network (certificate <= 1)."""
    proc, written = _export_with(
        "ex.rspline.fit = lambda *a, **k: (None, {'worst': 1.5})", model_path, tmp_path)
    assert proc.returncode != 0 and not written
    assert "interaction 1 radial spline: certificate 1.500" in proc.stderr, proc.stderr[-600:]


def test_specialised_export_refuses_an_uncertified_first_table(model_path, tmp_path):
    """The interaction-0 table must be at least as accurate as the float32 network (certificate <= 1)."""
    proc, written = _export_with(
        "ex.rad.certify = lambda *a, **k: {'worst': 1.5}", model_path, tmp_path)
    assert proc.returncode != 0 and not written
    assert "interaction 0 radial table: certificate 1.500" in proc.stderr, proc.stderr[-600:]


def test_specialised_bundle_signature_names_the_product_basis(sto_bundle):
    """The signature ties a bundle to the compiled kernels' factored product programs, not only to the shapes."""
    from bundle import read_bundle

    sig = read_bundle(sto_bundle)[0]["kernel_signature"]
    assert re.search(r":p0=[0-9a-f]{16}:p1=[0-9a-f]{16}$", sig), sig
    header = (ROOT / "src" / "generated_kernels_signature.h").read_text()
    assert f'"{sig}"' in header


def test_bundle_cache_key_follows_the_exporter_sources(tmp_path):
    """A changed exporter must not find the bundles of the old one (no version number to remember to bump)."""
    a, b = tmp_path / "a.py", tmp_path / "b.py"
    a.write_text("x = 1\n")
    b.write_text("y = 2\n")
    first = loupmd.exporter_fingerprint([a, b])
    assert first == loupmd.exporter_fingerprint([a, b])
    b.write_text("y = 3\n")
    assert first != loupmd.exporter_fingerprint([a, b])
    # the real key covers every exporter, the container writer and the compiled kernels' signature
    names = {p.name for p in loupmd.EXPORTER_SOURCES}
    assert {"export.py", "export_omat0.py", "export_generic.py", "export_polar.py", "bundle.py",
            "generated_kernels_signature.h"} <= names
