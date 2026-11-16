"""The real-space two-body D3 term (plan task S6-D04, ``src/dispersion/d3_pair.cu``) against simple-dftd3 at the
same settings, for each damping, on NbCl5 and a molecular crystal (paracetamol), and the d3bj-mace preset against
the torch-dftd calculator ``mace_mp(dispersion=True)`` builds. Tolerances of the plan: 1e-6 eV/atom, 1e-5 eV/A,
1e-7 eV/A^3. Needs a GPU and a library with ``loup_d3_evaluate``; no model."""
from __future__ import annotations

import numpy as np
import pytest

from loupmd import _d3
from loupmd import dispersion as d
from loupmd.bench import suites
from loupmd.errors import BackendError, InputError

BOHR = _d3.BOHR
HARTREE = 27.211386245988
TOL_E, TOL_F, TOL_S = 1e-6, 1e-5, 1e-7


def _lib_or_skip():
    try:
        return _d3._lib()
    except BackendError as error:
        pytest.skip(str(error))


def _rattled(atoms, seed=1, sigma=0.05):
    atoms = atoms.copy()
    atoms.pbc = True
    atoms.positions += np.random.default_rng(seed).normal(0.0, sigma, atoms.positions.shape)
    return atoms


def _system(name):
    from ase.io import read

    if name == "nbcl5":
        try:
            return _rattled(read(suites.structure_file("nbcl5")))
        except InputError as error:
            pytest.skip(str(error))
    if name == "paracetamol":
        return _rattled(read(suites._SHIPPED / "paracetamol_cod2104364.cif"))
    if name == "srtio3":
        return _rattled(suites.srtio3(2))
    raise ValueError(name)


def _reference(atoms, record):
    """simple-dftd3 (LGPL, called as an external library) at the record's settings, with the three-body term
    (s9 = 1) when the record has it."""
    dftd3 = pytest.importorskip("dftd3.interface")
    p = record.parameters
    s9 = 1.0 if record.three_body else 0.0
    param = {"bj": lambda: dftd3.RationalDampingParam(s6=p["s6"], s8=p["s8"], a1=p["a1"], a2=p["a2"], s9=s9),
             "bjm": lambda: dftd3.ModifiedRationalDampingParam(s6=p["s6"], s8=p["s8"], a1=p["a1"], a2=p["a2"],
                                                               s9=s9),
             "zero": lambda: dftd3.ZeroDampingParam(s6=p["s6"], s8=p["s8"], rs6=p["rs6"], rs8=p["rs8"],
                                                    alp=p["alpha"], s9=s9),
             "zerom": lambda: dftd3.ModifiedZeroDampingParam(s6=p["s6"], s8=p["s8"], rs6=p["rs6"], rs8=p["rs8"],
                                                             alp=p["alpha"], bet=p["beta"], s9=s9)}[record.damping]()
    model = dftd3.DispersionModel(atoms.numbers, atoms.positions / BOHR, np.asarray(atoms.cell) / BOHR,
                                  np.array([True, True, True]))
    model.set_realspace_cutoff(disp2=record.pair_cutoff_bohr, disp3=record.three_body_cutoff_bohr,
                               cn=record.cn_cutoff_bohr)
    res = model.get_dispersion(param, grad=True)
    return (float(res["energy"]) * HARTREE, -np.asarray(res["gradient"]) * HARTREE / BOHR,
            np.asarray(res["virial"]) * HARTREE / atoms.get_volume())


def _compare(atoms, got, ref):
    e, f, s = ref
    n = len(atoms)
    de = abs(got["energy"] - e) / n
    df = float(np.abs(got["forces"] - f).max())
    ds = float(np.abs(got["stress"] - s).max())
    assert de <= TOL_E and df <= TOL_F and ds <= TOL_S, (de, df, ds)
    return de, df, ds


@pytest.mark.parametrize("system", ["nbcl5", "paracetamol", "srtio3"])
@pytest.mark.parametrize("damping", ["bj", "zero", "bjm", "zerom"])
def test_each_damping_agrees_with_simple_dftd3(system, damping):
    lib = _lib_or_skip()
    atoms = _system(system)
    record = d.resolve(d.Dispersion(damping=damping, long_range="none"), functional="pbe")
    got = _d3.evaluate(atoms, record, lib=lib)
    _compare(atoms, got, _reference(atoms, record))


def test_shorter_cutoffs_agree_with_simple_dftd3():
    """A pair cutoff below the CN cutoff and both shorter than the cell (SrTiO3 2x2x2 is 7.8 A, 14.8 bohr)."""
    lib = _lib_or_skip()
    atoms = _system("srtio3")
    record = d.resolve(d.Dispersion(pair_cutoff_bohr=20.0, cn_cutoff_bohr=30.0, long_range="none"),
                       functional="pbe")
    _compare(atoms, _d3.evaluate(atoms, record, lib=lib), _reference(atoms, record))


def test_d3bj_mace_agrees_with_the_torch_dftd_calculator_of_mace_mp():
    """mace_mp(dispersion=True) adds TorchDFTD3Calculator(damping="bj", xc="pbe", cutoff=40 bohr, cnthr 40 bohr);
    compared at float64 (mace_mp's default float32 would set the tolerance, not the term)."""
    torch = pytest.importorskip("torch")
    calc_mod = pytest.importorskip("torch_dftd.torch_dftd3_calculator")
    lib = _lib_or_skip()
    atoms = _system("paracetamol")
    record = d.resolve("d3bj-mace", functional="PBE")
    ref = calc_mod.TorchDFTD3Calculator(device="cpu", damping="bj", xc="pbe", cutoff=40.0 * BOHR,
                                        cnthr=40.0 * BOHR, dtype=torch.float64)
    ref_atoms = atoms.copy()
    ref_atoms.calc = ref
    from ase.stress import voigt_6_to_full_3x3_stress

    expected = (ref_atoms.get_potential_energy(), ref_atoms.get_forces(),
                voigt_6_to_full_3x3_stress(ref_atoms.get_stress()))
    _compare(atoms, _d3.evaluate(atoms, record, lib=lib), expected)


def test_per_atom_energies_the_energy_only_path_and_wrapping():
    lib = _lib_or_skip()
    atoms = _system("paracetamol")
    record = d.resolve("d3bj", functional="pbe")
    full = _d3.evaluate(atoms, record, per_atom=True, lib=lib)
    assert abs(full["energies"].sum() - full["energy"]) < 1e-9 * max(1.0, abs(full["energy"]))
    only = _d3.evaluate(atoms, record, forces=False, stress=False, lib=lib)
    assert abs(only["energy"] - full["energy"]) < 1e-9 * abs(full["energy"])
    moved = atoms.copy()                       # atoms outside the cell, by whole lattice vectors
    moved.positions += np.random.default_rng(3).integers(-2, 3, (len(atoms), 3)) @ np.asarray(atoms.cell)
    again = _d3.evaluate(moved, record, lib=lib)
    assert abs(again["energy"] - full["energy"]) / len(atoms) < 1e-8
    assert np.abs(again["forces"] - full["forces"]).max() < 1e-6


def test_the_switch_is_smooth_and_its_forces_are_the_gradient():
    """smoothing=w: forces and stress are the derivatives of the switched energy (central differences)."""
    lib = _lib_or_skip()
    atoms = _system("srtio3")
    record = d.resolve(d.Dispersion(pair_cutoff_bohr=20.0, cn_cutoff_bohr=20.0, smoothing=6.0, long_range="none"),
                       functional="pbe")
    got = _d3.evaluate(atoms, record, lib=lib)
    plain = _d3.evaluate(atoms, d.resolve(d.Dispersion(pair_cutoff_bohr=20.0, cn_cutoff_bohr=20.0,
                                                       long_range="none"), functional="pbe"), lib=lib)
    assert got["energy"] > plain["energy"]           # the switch removes attraction near the cutoff
    h = 1e-3
    for atom, axis in ((0, 0), (7, 2), (21, 1)):
        up, down = atoms.copy(), atoms.copy()
        up.positions[atom, axis] += h
        down.positions[atom, axis] -= h
        fd = -(_d3.evaluate(up, record, forces=False, stress=False, lib=lib)["energy"] -
               _d3.evaluate(down, record, forces=False, stress=False, lib=lib)["energy"]) / (2 * h)
        assert abs(fd - got["forces"][atom, axis]) < 2e-5, (atom, axis, fd, got["forces"][atom, axis])
    def fd_stress_error(rec, analytic, eps=1e-3):
        strained = []
        for sign in (1, -1):
            a = atoms.copy()
            a.set_cell(np.asarray(atoms.cell) @ (np.eye(3) + sign * eps * np.diag([1.0, 0.0, 0.0])), scale_atoms=True)
            strained.append(_d3.evaluate(a, rec, forces=False, stress=False, lib=lib)["energy"])
        return abs((strained[0] - strained[1]) / (2 * eps) / atoms.get_volume() - analytic[0, 0])

    # the bound: float32 rounding of the pair energies (1e-7 of the total) over the strain step. The unswitched sum
    # is no control here: straining moves pairs across its hard cutoff, and its energy jumps.
    switched = fd_stress_error(record, got["stress"])
    bound = 1e-7 * abs(got["energy"]) / (2 * 1e-3 * atoms.get_volume())
    print(f"switch: stress xx vs central difference {switched:.1e} eV/A^3 (rounding bound {bound:.1e})")
    assert switched <= bound


# ----------------------------------------------------------------------------------- the three-body term (S6-D08)

@pytest.mark.parametrize("system,damping", [("nbcl5", "bj"), ("paracetamol", "bj"), ("srtio3", "bj"),
                                            ("srtio3", "zero"), ("paracetamol", "zerom")])
def test_the_three_body_term_agrees_with_simple_dftd3(system, damping):
    """d3bj-atm's 20 bohr three-body cutoff; SrTiO3 2x2x2 (14.8 bohr) is shorter than it, so triangles hold two
    images of one atom."""
    lib = _lib_or_skip()
    atoms = _system(system)
    record = d.resolve(d.Dispersion(damping=damping, three_body=True, three_body_cutoff_bohr=20.0,
                                    long_range="none"), functional="pbe")
    got = _d3.evaluate(atoms, record, lib=lib)
    two = _d3.evaluate(atoms, d.resolve(d.Dispersion(damping=damping, long_range="none"), functional="pbe"), lib=lib)
    ref = _reference(atoms, record)
    de, df, ds = _compare(atoms, got, ref)
    share = (got["energy"] - two["energy"]) / abs(two["energy"])
    print(f"{system} {damping}: three-body {1e3 * (got['energy'] - two['energy']) / len(atoms):+.3f} meV/atom "
          f"({100 * share:.1f}% of the pair term); vs simple-dftd3 {de:.1e} eV/atom, {df:.1e} eV/A, {ds:.1e} eV/A^3")
    assert share > 0.0                                    # ATM is repulsive in these crystals (design section 3.2)


def test_the_three_body_term_at_40_bohr_agrees_with_simple_dftd3():
    """d3-reference's cutoffs (60 bohr pairs, 40 bohr three-body and coordination numbers): several chunks of
    neighbours per atom."""
    lib = _lib_or_skip()
    atoms = _system("srtio3")
    record = d.resolve("d3-reference", functional="pbe")
    _compare(atoms, _d3.evaluate(atoms, record, lib=lib), _reference(atoms, record))


def test_the_three_body_term_per_atom_energy_only_and_switched():
    lib = _lib_or_skip()
    atoms = _system("srtio3")
    record = d.resolve(d.Dispersion(three_body=True, three_body_cutoff_bohr=20.0, long_range="none"),
                       functional="pbe")
    full = _d3.evaluate(atoms, record, per_atom=True, lib=lib)
    assert abs(full["energies"].sum() - full["energy"]) < 1e-9 * abs(full["energy"])
    only = _d3.evaluate(atoms, record, forces=False, stress=False, lib=lib)
    assert abs(only["energy"] - full["energy"]) < 1e-9 * abs(full["energy"])
    switched = d.resolve(d.Dispersion(three_body=True, three_body_cutoff_bohr=20.0, pair_cutoff_bohr=20.0,
                                      cn_cutoff_bohr=20.0, smoothing=6.0, long_range="none"), functional="pbe")
    got = _d3.evaluate(atoms, switched, lib=lib)
    h = 1e-3
    for atom, axis in ((0, 0), (7, 2), (21, 1)):
        up, down = atoms.copy(), atoms.copy()
        up.positions[atom, axis] += h
        down.positions[atom, axis] -= h
        fd = -(_d3.evaluate(up, switched, forces=False, stress=False, lib=lib)["energy"] -
               _d3.evaluate(down, switched, forces=False, stress=False, lib=lib)["energy"]) / (2 * h)
        assert abs(fd - got["forces"][atom, axis]) < 2e-5, (atom, axis, fd, got["forces"][atom, axis])


def test_what_is_not_built_is_refused_by_the_library():
    lib = _lib_or_skip()
    atoms = _system("srtio3")
    record = d.resolve("d3bj", functional="pbe")
    atoms.pbc = False
    with pytest.raises(InputError, match="periodic cell"):
        _d3.evaluate(atoms, record, lib=lib)
    atoms.pbc = True
    atoms.numbers[0] = 99
    with pytest.raises(BackendError, match="no reference C6 for atomic number 99"):
        _d3.evaluate(atoms, record, lib=lib)


# ------------------------------------------------------------------------------------- inside the engine's passes

def _engine_pair(sto_bundle, atoms, record, skin=0.5):
    """Two engines on the test model: one with the D3 term, one without."""
    from loupmd import LoupEngine

    plain, with_d3 = LoupEngine(sto_bundle), LoupEngine(sto_bundle)
    for eng in (plain, with_d3):
        eng.set_system(atoms.numbers, atoms.get_masses(), skin=skin)
    try:
        with_d3.set_dispersion(record)
    except BackendError as error:
        pytest.skip(str(error))
    return plain, with_d3


def _oxide(rep=3, seed=5):
    atoms = suites.srtio3(rep)
    atoms.positions += np.random.default_rng(seed).normal(scale=0.03, size=atoms.positions.shape)
    return atoms


def test_the_engine_adds_the_term_to_energy_forces_stress_and_per_atom_energies(sto_bundle):
    lib = _lib_or_skip()
    atoms = _oxide()
    record = d.resolve("d3bj", functional="pbe")
    plain, with_d3 = _engine_pair(sto_bundle, atoms, record)
    alone = _d3.evaluate(atoms, record, per_atom=True, lib=lib)
    cell = atoms.cell.array
    for eng in (plain, with_d3):
        eng.set_state(atoms.positions, None, cell)
    e0, f0, s0 = plain.evaluate(atoms.positions, cell)
    e1, f1, s1 = with_d3.evaluate(atoms.positions, cell)
    spread = np.abs(plain.evaluate(atoms.positions, cell)[1] - f0).max()      # the model's own repeat spread
    n = len(atoms)
    # the gate's energy floor (0.01 meV/atom): float32 rounding of the model's total, about 1e-7 of it, comes near 1e-6
    assert abs(e1 - (e0 + alone["energy"])) / n < 1e-5
    assert np.abs(f1 - (f0 + alone["forces"])).max() < max(4 * spread, 1e-5)
    assert np.abs(s1 - (s0 + alone["stress"])).max() < 1e-7
    _, _, _, a0 = plain.evaluate_per_atom(atoms.positions, cell)
    _, _, _, a1 = with_d3.evaluate_per_atom(atoms.positions, cell)
    assert np.abs(a1 - (a0 + alone["energies"])).max() < 1e-5
    assert abs(with_d3.evaluate_energy(atoms.positions, cell) - plain.evaluate_energy(atoms.positions, cell)
               - alone["energy"]) / n < 1e-6
    with_d3.set_dispersion(None)                                              # removed again
    assert abs(with_d3.evaluate(atoms.positions, cell)[0] - e0) / n < 1e-6


@pytest.mark.parametrize("variable_cell", [False, True], ids=["fixed cell", "deforming cell"])
def test_the_md_step_carries_the_term(sto_bundle, monkeypatch, variable_cell):
    """200 captured NVE steps (several list rebuilds; with LOUP_DEBUG_CELL_RATE the cell grows 0.02% per step on
    the device, so the term must read the device's cell): the energy and forces on the device at the end are the
    model's plus the term alone at the final positions and cell."""
    from gpu_support import maxwell_velocities

    from loupmd import _core

    lib = _lib_or_skip()
    if variable_cell:
        monkeypatch.setenv("LOUP_VARIABLE_CELL", "1")
        monkeypatch.setenv("LOUP_DEBUG_CELL_RATE", "2e-4")
    atoms = _oxide()
    record = d.resolve("d3bj", functional="pbe")
    plain, with_d3 = _engine_pair(sto_bundle, atoms, record)
    with_d3.set_state(atoms.positions, maxwell_velocities(atoms, 600.0, 3), atoms.cell.array)
    stats = with_d3.run(200, _core.integrator(_core.LOUP_NVE, 1.0))
    positions, _, forces, energy = with_d3.get_state()
    cell = with_d3.get_cell()
    monkeypatch.delenv("LOUP_VARIABLE_CELL", raising=False)
    monkeypatch.delenv("LOUP_DEBUG_CELL_RATE", raising=False)
    if variable_cell:
        assert abs(np.linalg.det(cell) / atoms.get_volume() - 1.0002 ** 600) < 1e-6
    final = atoms.copy()
    final.positions, final.cell = positions, cell
    plain.set_state(positions, None, cell)
    e0, f0, _ = plain.evaluate(positions, cell)
    alone = _d3.evaluate(final, record, stress=False, lib=lib)
    print(f"{'deforming' if variable_cell else 'fixed'} cell: {stats.rebuilds} rebuilds, D3 {alone['energy']:.4f} eV")
    assert stats.rebuilds >= 1
    assert abs(energy - (e0 + alone["energy"])) / len(atoms) < 1e-5
    assert np.abs(forces - (f0 + alone["forces"])).max() < 1e-4


def test_lm_md_with_the_three_body_preset_in_npt_reports_its_cost(model_path, tmp_path):
    """lm.md(dispersion="d3bj-atm") under the barostat: the run completes, the report names the term and its measured
    cost per evaluation with the three-body share (PLAN.md S6-D08), and the term moves the pressure."""
    import loupmd

    _lib_or_skip()
    ff = loupmd.load(model_path, trusted=True)
    atoms = _oxide()
    result = loupmd.md(atoms, model=ff, ensemble="NPT", temperature_K=300.0, pressure_GPa=0.0, timestep_fs=1.0,
                       steps=100, seed=2, dispersion="d3bj-atm", output=str(tmp_path / "run"))
    cost = result.metadata["dispersion"]["cost"]
    print(f"d3bj-atm on SrTiO3 3x3x3 ({len(atoms)} atoms): {cost}")
    assert cost["ms_per_evaluation"] > 0.0 and cost["three_body_ms"] > 0.0
    assert "three-body term to 20 bohr" in result.summary or "three-body term" in result.summary
    assert "of which the three-body term" in result.summary
    assert np.isfinite(result.simulation.atoms.get_volume())


# ----------------------------------------------------------------------------- loupmd validate --dispersion (S6-D10)

@pytest.mark.parametrize("preset", list(d.PRESETS))
def test_loupmd_validate_dispersion_passes_for_every_preset(preset, tmp_path, capsys):
    from ase.io import write

    from loupmd.cli import main

    pytest.importorskip("dftd3.interface")
    _lib_or_skip()
    path = tmp_path / "sto.extxyz"
    write(path, _system("srtio3"))
    code = main(["validate", "--dispersion", preset, "--model", "omat0-medium", "--structure", str(path)])
    out = capsys.readouterr().out
    print(out.splitlines()[-2])
    assert code == 0 and "dispersion check PASSED" in out, out


def test_the_dispersion_check_fails_for_a_wrong_parameter_set():
    from dataclasses import replace

    from loupmd.validate import dispersion as check

    pytest.importorskip("dftd3.interface")
    lib = _lib_or_skip()
    record = d.resolve("d3bj", functional="pbe")
    wrong = replace(record, parameters=dict(record.parameters, a2=record.parameters["a2"] + 0.05))
    result = check.check([("srtio3", _system("srtio3"))], record, lib=lib, reference_record=wrong)
    assert not result["passed"]
    assert result["rows"][0]["deviation"]["energy_eV_atom"] > 100 * check.BOUNDS["energy_eV_atom"]
    assert check.check([("srtio3", _system("srtio3"))], record, lib=lib)["passed"]
