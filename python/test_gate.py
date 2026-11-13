#!/usr/bin/env python3
"""Accuracy gate for LoupMD against MACE float64 and float32 references (3x float32 rule).

  python test_gate.py [MODEL] [--head H] [--generic] [--set oxide|organic|crystals]

Structure sets: "oxide" (NaCl, SrTiO3, NbCl5; the default), "organic" (urea, naphthalene, paracetamol, the
648-atom water box; MACE-OFF's set) and "crystals" (NaCl, NaNbO3, NbCl5 R2SCAN cells; FastMACE's MH-1 / POLAR set).
For the specialised OMAT-0-medium bundle every structure is evaluated with the default kernel choices; the NbCl5 2x2x2 cell (576 atoms, above the 512-atom
switch to the receiver-owned w' kernel) is also evaluated with every forced spline variant, both w' block sizes and the
Morton atom reorder on and off, and the small cells with the receiver-owned w' forced, so every kernel path the
engine can pick is gated (the per-system tuning otherwise only picks one of them per run).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.io import read as ase_read
from mace.calculators import MACECalculator

from loupmd import LoupEngine, ensure_loup_bundle


def read_bundle_meta(path):
    from bundle import read_bundle

    return read_bundle(path)[0]


NBCL5_CIFS = [
    "/mnt/d/Margraf/FastMACE/fastmace/protocols/data/NbCl5_R2SCAN.cif",
    "/mnt/homes/zwingel/FastMACE/fastmace/protocols/data/NbCl5_R2SCAN.cif",
]

# Forced kernel paths (environment read at engine creation, see include/loupmd.h).
LARGE_VARIANTS = (
    [{"LOUP_SPLINE_VARIANT": v} for v in ("R1", "R1p", "R4", "R4p", "R5")]
    + [{"LOUP_WPRIME_THREADS": "512"}, {"LOUP_REORDER": "1"}, {"LOUP_REORDER": "0"}]
)
SMALL_VARIANTS = [{"LOUP_WPRIME_RCV": "1"}, {"LOUP_WPRIME_RCV": "1", "LOUP_WPRIME_THREADS": "512"}]
VARIANT_KEYS = ("LOUP_SPLINE_VARIANT", "LOUP_WPRIME_THREADS", "LOUP_WPRIME_RCV", "LOUP_REORDER")


STRUCTURE_ROOTS = ["/mnt/d/Margraf/ResearchModule/Structures", "/mnt/homes/zwingel/ResearchModule/Structures"]


def _structure(rel):
    for root in STRUCTURE_ROOTS:
        p = os.path.join(root, rel)
        if os.path.exists(p):
            return ase_read(p)
    raise SystemExit(f"structure {rel} not found under {STRUCTURE_ROOTS}")


def _configs(bases, schedule, seed=20261112):
    rng = np.random.default_rng(seed)
    configs = []
    for name, base in bases:
        for k, (strain, rattle) in enumerate(schedule):
            at = base.copy()
            F = np.eye(3) + strain * np.array([[1.0, 0.1, -0.05], [0.1, 0.8, 0.07], [-0.05, 0.07, 1.2]])
            at.set_cell(at.cell.array @ F.T, scale_atoms=True)
            at.positions += rng.normal(0.0, rattle, size=at.positions.shape)
            configs.append((f"{name}_c{k}", at))
    return configs


def make_organic():
    bases = [("urea_3x3x3", _structure("Organic/urea_cod1008787.cif").repeat((3, 3, 3))),
             ("naphthalene_2x2x2", _structure("Organic/naphthalene_cod2311090.cif").repeat((2, 2, 2))),
             ("paracetamol_2x2x2", _structure("Organic/paracetamol_cod2104364.cif").repeat((2, 2, 2))),
             ("water_216", _structure("Organic/water_216.extxyz"))]
    return _configs(bases, [(0.01, 0.02), (-0.008, 0.04)])


def make_crystals():
    def rep_to(at, length=13.0):
        n = [max(1, int(np.ceil(length / h))) for h in at.cell.cellpar()[:3]]
        return at.repeat(n)

    bases = [(name, rep_to(_structure(f"Benchmark/R2SCAN/{name}.cif"))) for name in ("NaCl", "NaNbO3", "NbCl5")]
    return _configs(bases, [(0.02, 0.05), (-0.015, 0.08)])


def make_structures():
    # SrTiO3 3x3x3 cubic perovskite (135 atoms)
    a_sto = 3.905
    sto = Atoms(
        symbols=["Sr", "Ti", "O", "O", "O"],
        scaled_positions=[
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.0),
            (0.5, 0.0, 0.5),
            (0.0, 0.5, 0.5),
        ],
        cell=[a_sto, a_sto, a_sto],
        pbc=True,
    ).repeat((3, 3, 3))

    # NaCl 2x2x2 conventional (64 atoms)
    nacl = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((2, 2, 2))

    bases = [("NaCl_2x2x2", nacl), ("SrTiO3_3x3x3", sto)]
    cif = next((c for c in NBCL5_CIFS if os.path.exists(c)), None)
    if cif is None:
        raise SystemExit(f"NbCl5 CIF not found (tried {NBCL5_CIFS}); the >= 512-atom structure is required")
    bases.append(("NbCl5_2x2x2", ase_read(cif).repeat((2, 2, 2))))  # 576 atoms

    rng = np.random.default_rng(20261112)
    configs = []
    for name, base in bases:
        for k, (strain, rattle) in enumerate([(0.02, 0.05), (-0.015, 0.08)]):
            at = base.copy()
            F = np.eye(3) + strain * np.array([[1.0, 0.1, -0.05], [0.1, 0.8, 0.07], [-0.05, 0.07, 1.2]])
            at.set_cell(at.cell.array @ F.T, scale_atoms=True)
            at.positions += rng.normal(0.0, rattle, size=at.positions.shape)
            configs.append((f"{name}_c{k}", at))
    return configs


def eval_mace(model_path: str, dtype: str, configs, head=None, device="cuda"):
    kw = {"head": head} if head else {}
    calc = MACECalculator(model_paths=model_path, device=device, default_dtype=dtype, **kw)
    out = []
    for _, at in configs:
        a = at.copy()
        a.calc = calc
        e = float(a.get_potential_energy())
        f = np.asarray(a.get_forces(), dtype=np.float64)
        s = np.asarray(a.get_stress(voigt=False), dtype=np.float64)
        out.append((e, f, s))
    return out


def fd_stress(model_path, configs, device="cuda", h=1e-5):
    """Stress as the float64 MACE energy's central difference under symmetric strain (positions and cell strained,
    so the reciprocal cell and the volume follow). For PolarMACE this is the corrected stress (MACE PR 1718, as
    FastMACE): the installed PolarMACE's autograd stress leaves out the k-space terms' cell dependence."""
    calc = MACECalculator(model_paths=model_path, device=device, default_dtype="float64")
    out = []
    for _, at in configs:
        def energy(eps):
            a = at.copy()
            F = np.eye(3) + eps
            a.set_cell(at.cell.array @ F, scale_atoms=False)
            a.positions = at.positions @ F
            a.calc = calc
            return a.get_potential_energy()

        V = at.get_volume()
        s = np.zeros((3, 3))
        for i in range(3):
            for j in range(i, 3):
                e = np.zeros((3, 3))
                e[i, j] = e[j, i] = h
                d = (energy(e) - energy(-e)) / (2 * h) / V / (1 if i == j else 2)
                s[i, j] = s[j, i] = d
        out.append(s)
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=(
        "/home/fzwin/models/mace-omat-0-medium.model"
        if os.path.exists("/home/fzwin/models/mace-omat-0-medium.model")
        else "/mnt/homes/zwingel/ResearchModule/Models/mace-omat-0-medium.model"))
    ap.add_argument("--head", default=None)
    ap.add_argument("--generic", action="store_true")
    ap.add_argument("--set", default="oxide", choices=("oxide", "organic", "crystals"))
    ap.add_argument("--ref-device", default="cuda", help="device of the MACE references (cpu for models whose "
                    "float64 reference does not fit the GPU)")
    args = ap.parse_args()
    model_path, head = args.model, args.head
    if head is None:  # the exporter's default head of a multi-head model, named explicitly for MACECalculator
        import torch

        heads = list(getattr(torch.load(model_path, map_location="cpu", weights_only=False), "heads", None) or [])
        if len(heads) > 1:
            head = next((h for h in ("omat_pbe", "default", "Default") if h in heads), heads[-1])
            print(f"head: {head}")
    configs = {"oxide": make_structures, "organic": make_organic, "crystals": make_crystals}[args.set]()
    print(f"{args.set} set: " + ", ".join(f"{n} ({len(a)})" for n, a in configs[::2]))
    print("Running MACE float64 reference...")
    ref64 = eval_mace(model_path, "float64", configs, head, args.ref_device)
    print("Running MACE float32 reference...")
    ref32 = eval_mace(model_path, "float32", configs, head, args.ref_device)
    import torch

    polar = type(torch.load(model_path, map_location="cpu", weights_only=False)).__name__ == "PolarMACE"
    # Stress reference per config: MACE float64 autograd, or for PolarMACE the corrected float64 stress (finite
    # differences); the float32 noise level stays the MACE float32-vs-float64 autograd difference (both carry the
    # same local float32 rounding).
    stress_ref = [s for _, _, s in ref64]
    if polar:
        print("PolarMACE: stress reference from float64 energy differences (corrected k-space cell dependence)")
        stress_ref = fd_stress(model_path, configs, args.ref_device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    specialised = {}

    engines = {}
    all_pass = True
    reorder_results = {}

    def run(name, at, refs, variant):
        nonlocal all_pass
        (e64, f64, s64), (e32, f32, s32), s_ref = refs
        zs = tuple(sorted(set(at.get_atomic_numbers().tolist())))
        key = (zs, tuple(sorted(variant.items())))
        if key not in engines:
            for k in VARIANT_KEYS:
                os.environ.pop(k, None)
            os.environ.update(variant)
            bundle = ensure_loup_bundle(model_path, list(zs), head=head, generic=args.generic or None)
            specialised[key] = read_bundle_meta(bundle).get("arch") != "generic"
            engines[key] = LoupEngine(bundle, device_id=0)
            for k in VARIANT_KEYS:
                os.environ.pop(k, None)
        eng = engines[key]
        eng.set_system(at.get_atomic_numbers(), at.get_masses(), skin=0.5)
        e_l, f_l, s_l = eng.evaluate(at.positions, at.cell.array, compute_stress=True)
        if "LOUP_REORDER" in variant:
            reorder_results.setdefault(name, []).append((e_l, f_l, s_l))

        nat = len(at)
        de_l = abs(e_l - e64) * 1000.0 / nat
        de_32 = abs(e32 - e64) * 1000.0 / nat
        df_l = float(np.abs(f_l - f64).max())
        df_32 = float(np.abs(f32 - f64).max())
        ds_l = float(np.abs(s_l - s_ref).max())
        ds_32 = float(np.abs(s32 - s64).max())

        rf = df_l / max(df_32, 1e-5 / 3.0)
        rs = ds_l / max(ds_32, 1e-7 / 3.0)
        ok = (de_l <= max(3.0 * de_32, 0.01)) and (df_l <= max(3.0 * df_32, 1e-5)) and (ds_l <= max(3.0 * ds_32, 1e-7))
        all_pass = all_pass and ok
        tag = ",".join(f"{k[5:].lower()}={v}" for k, v in variant.items()) or "default"
        print(f"{name:18s} {tag:24s} | {de_l:13.5f} | {df_l:13.3e} {rf:6.2f} | {ds_l:14.3e} {rs:6.2f} | {'PASS' if ok else 'FAIL'}")
        if not ok:
            print(f"  ref32: dE/atom={de_32:.5f} meV, dF={df_32:.3e}, dS={ds_32:.3e}")

    print("\n=== LoupMD vs MACE float64 / float32 (3x gate) ===")
    print(f"{'Config':18s} {'Variant':24s} | {'dE/atom (meV)':>13s} | {'dF_max (eV/A)':>13s} {'ratio':>6s} | {'dS_max (eV/A3)':>14s} {'ratio':>6s} | {'Pass'}")
    print("-" * 113)
    for (name, at), r64, r32, sr in zip(configs, ref64, ref32, stress_ref):
        run(name, at, (r64, r32, sr), {})
        zs = tuple(sorted(set(at.get_atomic_numbers().tolist())))
        if specialised.get((zs, ())):  # the forced-variant matrix exists only for the specialised kernels
            for variant in (LARGE_VARIANTS if len(at) >= 512 else SMALL_VARIANTS):
                run(name, at, (r64, r32, sr), variant)

    # The reorder only permutes atoms inside species groups: results must agree to float32 summation-order noise.
    for name, res in reorder_results.items():
        (e_on, f_on, s_on), (e_off, f_off, s_off) = res
        nat = len(f_on)
        de = abs(e_on - e_off) * 1000.0 / nat
        df = float(np.abs(f_on - f_off).max())
        ds = float(np.abs(s_on - s_off).max())
        ok = de < 1e-3 and df < 1e-4 and ds < 1e-6
        all_pass = all_pass and ok
        print(f"{name}: reorder on vs off dE/atom {de:.2e} meV, dF {df:.2e} eV/A, dS {ds:.2e} eV/A3 -> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
