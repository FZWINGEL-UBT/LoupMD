"""CPU-only measurements for the dispersion design note (no GPU, no network).

Usage: python measure_d3_reference.py OUT.json ROOT

Reference: simple-dftd3 (python package dftd3 1.6.0), float64. Structures: the project's own cells, each rattled by
0.05 A (seed 1) so forces do not vanish by symmetry. Reports, per structure:
  A  two-body D3(BJ, PBE) against the pair cutoff: energy, force and pressure differences to the 150 bohr sum
  B  the three-body ATM term against its cutoff and its size relative to the two-body term
  C  the four dampings (PBE) side by side
  D  sensitivity to the coordination-number cutoff
  E  pair and triple counts per atom
"""
import json
import sys
import time

import numpy as np
from ase import Atoms
from ase.io import read
from ase.lattice.hexagonal import Graphite

from dftd3.interface import (DispersionModel, ModifiedRationalDampingParam, ModifiedZeroDampingParam,
                             RationalDampingParam, ZeroDampingParam)

BOHR = 0.529177210903
HARTREE = 27.211386245988
GPA = 160.21766208  # eV/A^3 -> GPa

ROOT = sys.argv[2]  # folder that holds FastMACE/ and ResearchModule/


def structures():
    out = {}
    out["NbCl5 (R2SCAN cell)"] = read(f"{ROOT}/FastMACE/fastmace/protocols/data/NbCl5_R2SCAN.cif")
    out["naphthalene crystal"] = read(f"{ROOT}/ResearchModule/Structures/Organic/naphthalene_cod2311090.cif")
    out["urea crystal"] = read(f"{ROOT}/ResearchModule/Structures/Organic/urea_cod1008787.cif").repeat((2, 2, 2))
    out["water 216"] = read(f"{ROOT}/ResearchModule/Structures/Organic/water_216.extxyz")
    out["graphite 4x4x2"] = Graphite("C", latticeconstant={"a": 2.461, "c": 6.708}, size=(4, 4, 2))
    a = 3.905
    out["SrTiO3 3x3x3"] = Atoms("SrTiO3", scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5),
                                                              (0, 0.5, 0.5)], cell=[a] * 3, pbc=True).repeat((3, 3, 3))
    rng = np.random.default_rng(1)
    for at in out.values():
        at.pbc = True
        at.positions += rng.normal(0.0, 0.05, at.positions.shape)
    return out


def evaluate(atoms, param, disp2=60.0, disp3=40.0, cn=40.0):
    model = DispersionModel(atoms.numbers, atoms.positions / BOHR, np.asarray(atoms.cell) / BOHR,
                            np.array([True, True, True]))
    model.set_realspace_cutoff(disp2=disp2, disp3=disp3, cn=cn)
    t0 = time.perf_counter()
    res = model.get_dispersion(param, grad=True)
    dt = time.perf_counter() - t0
    energy = float(res["energy"]) * HARTREE
    forces = -np.asarray(res["gradient"]) * HARTREE / BOHR
    stress = np.asarray(res["virial"]) * HARTREE / atoms.get_volume()  # eV/A^3
    return energy, forces, stress, dt


def bj(xc="pbe", atm=False):
    return RationalDampingParam(method=xc, atm=atm)


def probe_atm_switch(atoms):
    """Does the atm keyword of the damping parameters switch the three-body term?"""
    e0 = evaluate(atoms, RationalDampingParam(method="pbe", atm=False))[0]
    e1 = evaluate(atoms, RationalDampingParam(method="pbe", atm=True))[0]
    return e0, e1


def main():
    results = {}
    S = structures()
    first = next(iter(S.values()))
    try:
        e0, e1 = probe_atm_switch(first)
        print(f"atm keyword works: E(atm=False) = {e0:.6f} eV, E(atm=True) = {e1:.6f} eV", flush=True)
    except Exception as exc:  # pragma: no cover
        print("atm keyword not accepted:", exc)
        sys.exit(1)
    for name, at in S.items():
        n = len(at)
        rho = n / at.get_volume()
        r = {"atoms": n, "density_per_A3": rho}
        print(f"\n== {name}: {n} atoms, {rho:.4f} atoms/A^3", flush=True)
        # A: two-body BJ PBE against the pair cutoff (cn cutoff fixed at 40 bohr, no ATM)
        ref_e, ref_f, ref_s, _ = evaluate(at, bj(), disp2=150.0)
        print(f"A  D3(BJ,PBE) two-body at 150 bohr: {ref_e / n * 1000:.3f} meV/atom, pressure "
              f"{-np.trace(ref_s) / 3 * GPA:+.4f} GPa (ASE sign: negative stress trace = compression)")
        rows = []
        for cut in (12.0, 20.0, 30.0, 40.0, 60.0, 95.0):
            e, f, s, dt = evaluate(at, bj(), disp2=cut)
            row = {"cutoff_bohr": cut, "cutoff_A": cut * BOHR, "dE_meV_per_atom": (e - ref_e) / n * 1000,
                   "max_dF_meV_per_A": float(np.abs(f - ref_f).max()) * 1000,
                   "dP_GPa": float(-(np.trace(s) - np.trace(ref_s)) / 3 * GPA),
                   "neighbours_per_atom": 4.0 / 3.0 * np.pi * (cut * BOHR) ** 3 * rho, "seconds": dt}
            rows.append(row)
            print(f"   cutoff {cut:5.0f} bohr ({cut * BOHR:5.1f} A): dE {row['dE_meV_per_atom']:+8.3f} meV/atom, "
                  f"max dF {row['max_dF_meV_per_A']:7.4f} meV/A, dP {row['dP_GPa']:+7.4f} GPa, "
                  f"{row['neighbours_per_atom']:8.0f} neighbours/atom, {dt:5.2f} s", flush=True)
        r["pair_cutoff"] = rows
        r["bj_two_body_meV_per_atom"] = ref_e / n * 1000
        # B: three-body term against its cutoff (two-body at 60 bohr)
        e2, f2, s2, _ = evaluate(at, bj(atm=False), disp2=60.0)
        rows = []
        for cut3 in (10.0, 15.0, 20.0, 30.0, 40.0):
            if 4.0 / 3.0 * np.pi * (cut3 * BOHR) ** 3 * rho > 1500 and n > 400:
                continue  # triples grow as n^2: keep the CPU run short
            e, f, s, dt = evaluate(at, bj(atm=True), disp2=60.0, disp3=cut3)
            row = {"cutoff3_bohr": cut3, "E_atm_meV_per_atom": (e - e2) / n * 1000,
                   "max_F_atm_meV_per_A": float(np.abs(f - f2).max()) * 1000,
                   "P_atm_GPa": float(-(np.trace(s) - np.trace(s2)) / 3 * GPA), "seconds": dt}
            rows.append(row)
            print(f"B  ATM cutoff {cut3:4.0f} bohr: E_atm {row['E_atm_meV_per_atom']:+7.3f} meV/atom "
                  f"({100 * (e - e2) / e2:+.2f}% of two-body), max F_atm {row['max_F_atm_meV_per_A']:.3f} meV/A, "
                  f"P_atm {row['P_atm_GPa']:+.4f} GPa, {dt:6.2f} s", flush=True)
        r["atm"] = rows
        # C: dampings with PBE parameters, two-body only, 60 bohr
        rows = {}
        for label, param in (("bj", RationalDampingParam(method="pbe", atm=False)),
                             ("zero", ZeroDampingParam(method="pbe", atm=False)),
                             ("bjm", ModifiedRationalDampingParam(method="pbe", atm=False)),
                             ("zerom", ModifiedZeroDampingParam(method="pbe", atm=False))):
            e, f, s, _ = evaluate(at, param, disp2=60.0)
            rows[label] = {"E_meV_per_atom": e / n * 1000, "rms_F_meV_per_A": float(np.sqrt((f ** 2).mean())) * 1000,
                           "P_GPa": float(-np.trace(s) / 3 * GPA)}
        print("C  dampings (PBE, two-body, 60 bohr): " + "; ".join(
            f"{k}: {v['E_meV_per_atom']:.2f} meV/atom, rms F {v['rms_F_meV_per_A']:.2f} meV/A, P {v['P_GPa']:+.3f} GPa"
            for k, v in rows.items()), flush=True)
        r["dampings"] = rows
        # D: coordination-number cutoff (two-body BJ at 60 bohr)
        rows = []
        for cn in (8.0, 11.34, 15.0, 20.0, 30.0):
            e, f, s, _ = evaluate(at, bj(), disp2=60.0, cn=cn)
            rows.append({"cn_cutoff_bohr": cn, "dE_meV_per_atom": (e - e2) / n * 1000,
                         "max_dF_meV_per_A": float(np.abs(f - f2).max()) * 1000})
        print("D  CN cutoff (reference 40 bohr): " + "; ".join(
            f"{x['cn_cutoff_bohr']:.1f} bohr: dE {x['dE_meV_per_atom']:+.4f} meV/atom, max dF "
            f"{x['max_dF_meV_per_A']:.4f} meV/A" for x in rows), flush=True)
        r["cn_cutoff"] = rows
        results[name] = r
    json.dump(results, open(sys.argv[1], "w"), indent=1)
    print("\nwritten", sys.argv[1])


if __name__ == "__main__":
    main()
