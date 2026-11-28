"""Force-shifted fixed-C6 D3(BJ) + analytical K/V tail for distilled ILMLIP/FastLR-SK students.

Provides:
- Exact float64 parameter derivation from ``loupmd/data/d3_tables.npz`` using mean coordination numbers
  over training frames (:func:`fit_fixed_d3bj`).
- Exact float64 reference evaluation (:func:`eval_fixed_d3bj_f64`) for energy, forces, virial, stress,
  and per-atom energies (including the analytical ``K / V`` tail).
- Training label transformation (:func:`transform_labels_for_d3`) matching Option B.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

from loupmd._d3 import BOHR, tables as d3_tables
from loupmd.errors import InputError

HARTREE_TO_EV = 27.211386245988
K1_CN = 16.0
K3_CN = 4.0

# D3(BJ) damping parameters: (s6, s8, a1, a2)
D3BJ_FUNCTIONALS: dict[str, dict[str, float]] = {
    "pbe": {"s6": 1.0, "s8": 0.7875, "a1": 0.4289, "a2": 4.4407},
    "r2scan": {"s6": 1.0, "s8": 0.7898, "a1": 0.4948, "a2": 5.7308},
    # Najibi & Goerigk, J. Chem. Theory Comput. 14, 5725 (2018)
    "wb97m": {"s6": 1.0, "s8": 0.3908, "a1": 0.5660, "a2": 3.1280},
    "b3lyp": {"s6": 1.0, "s8": 1.9889, "a1": 0.3981, "a2": 4.4211},
    "pbe0": {"s6": 1.0, "s8": 1.2177, "a1": 0.4145, "a2": 4.8593},
}

# 32-point Gauss-Legendre nodes/weights on [0, 1] for machine-precision int_R^inf 4*pi*r^2/(r^8+f^8) dr
_GL_X, _GL_W = np.polynomial.legendre.leggauss(32)
_GL_U = 0.5 * (_GL_X + 1.0)
_GL_WT = 0.5 * _GL_W


@dataclass(frozen=True)
class FixedD3BJSpec:
    """Serialized fixed-C6 D3(BJ) specification stored in ``arch=ilmlip`` ``.loup`` bundles.

    All pair tables have shape ``(S, S)`` in Ångström / eV units:
    - ``u(r) = -c6 / (r^6 + f6) - c8 / (r^8 + f8)``
    - ``V(r) = u(r) - e_shift - (r - r_cut) * f_shift``  (for ``r < r_cut``, else 0)
    - ``J_st`` (eV·Å^3): exact uniform-density tail integral so ``E_tail = (1 / (2 V)) sum_{s,t} N_s N_t J_{st}``.
    """

    functional: str
    r_cut: float
    force_shift: bool
    use_tail: bool
    Z: tuple[int, ...]
    c6: np.ndarray       # [S, S] eV * A^6
    c8: np.ndarray       # [S, S] eV * A^8
    f6: np.ndarray       # [S, S] A^6
    f8: np.ndarray       # [S, S] A^8
    e_shift: np.ndarray  # [S, S] eV
    f_shift: np.ndarray  # [S, S] eV / A
    J_st: np.ndarray     # [S, S] eV * A^3
    s6: float
    s8: float
    a1: float
    a2: float

    @property
    def S(self) -> int:
        return len(self.Z)

    def to_bundle_entries(self) -> tuple[dict, dict[str, np.ndarray]]:
        """Return ``(meta_updates, tensor_updates)`` for ``write_bundle``."""
        S = self.S
        # Pack [S*S, 6] float32 pair table for fast shared-memory load in ilmlip_disp.cuh
        pair_f32 = np.stack([
            self.c6.reshape(-1),
            self.c8.reshape(-1),
            self.f6.reshape(-1),
            self.f8.reshape(-1),
            self.e_shift.reshape(-1),
            self.f_shift.reshape(-1),
        ], axis=1).astype(np.float32)
        meta = {
            "has_builtin_disp": 1,
            "disp_method": "d3bj_fixed",
            "disp_functional": self.functional,
            "disp_force_shift": int(bool(self.force_shift)),
            "disp_use_tail": int(bool(self.use_tail)),
            "disp_version": 1,
        }
        tensors = {
            "disp_rcut": np.array([self.r_cut], dtype=np.float64),
            "disp_pair": np.ascontiguousarray(pair_f32, dtype=np.float32),
            "disp_J_st": np.ascontiguousarray(self.J_st.reshape(S, S), dtype=np.float64),
            "disp_params": np.array([self.s6, self.s8, self.a1, self.a2], dtype=np.float64),
        }
        return meta, tensors


def _mean_cn_weights(frames: Sequence[Atoms], Z: Sequence[int], cn_cut_bohr: float = 40.0) -> np.ndarray:
    """Compute mean D3 coordination-number Gaussian weights ``[S, 5]`` across ``frames``."""
    tab = d3_tables()
    nref = tab["nref"]      # [95]
    cnref = tab["cnref"]    # [95, 5]
    rcov = tab["rcov"]      # [95] (already scaled 4/3 * r_cov in bohr, matching torch-dftd/d3_pair.cu)
    S = len(Z)
    z_to_s = {int(z): s for s, z in enumerate(Z)}
    w_sum = np.zeros((S, 5), dtype=np.float64)
    w_cnt = np.zeros(S, dtype=np.int64)
    cn_cut_A = cn_cut_bohr * BOHR

    for at in frames:
        nums = np.asarray(at.numbers, dtype=np.int64)
        N = len(nums)
        cn = np.zeros(N, dtype=np.float64)
        if N > 1:
            # Handle periodic and non-periodic frames
            if any(bool(p) for p in at.pbc) and abs(float(np.linalg.det(at.cell.array))) > 1e-6:
                ii, jj, dd = neighbor_list("ijd", at, cn_cut_A)
            else:
                pos = np.asarray(at.positions, dtype=np.float64)
                diff = pos[:, None, :] - pos[None, :, :]
                dist = np.linalg.norm(diff, axis=-1)
                mask = (dist > 1e-12) & (dist < cn_cut_A)
                ii, jj = np.where(mask)
                dd = dist[ii, jj]
            if len(ii) > 0:
                r_bohr = dd / BOHR
                rco = rcov[nums[ii]] + rcov[nums[jj]]
                contrib = 1.0 / (1.0 + np.exp(-K1_CN * (rco / r_bohr - 1.0)))
                np.add.at(cn, ii, contrib)

        for a in range(N):
            zi = int(nums[a])
            s = z_to_s.get(zi)
            if s is None:
                continue
            nr = int(nref[zi])
            if nr <= 0:
                continue
            refs = cnref[zi, :nr]
            diff_cn = cn[a] - refs
            lw = -K3_CN * (diff_cn ** 2)
            lw -= np.max(lw)
            w = np.exp(lw)
            w /= np.sum(w)
            w_sum[s, :nr] += w
            w_cnt[s] += 1

    w_mean = np.zeros((S, 5), dtype=np.float64)
    for s, zi in enumerate(Z):
        nr = int(nref[zi])
        if w_cnt[s] > 0:
            w_mean[s, :nr] = w_sum[s, :nr] / float(w_cnt[s])
        elif nr > 0:
            # Fallback if species absent from sample frames: uniform over reference CNs
            w_mean[s, :nr] = 1.0 / float(nr)
    return w_mean


def _tail_integral_pair(c6: float, c8: float, f_A: float, r_cut: float, e_shift: float, f_shift: float) -> float:
    """Exact uniform-density tail integral ``J_st`` (eV·Å^3) for one species pair."""
    # 1) int_{r_cut}^inf 4*pi*r^2 * (-c6 / (r^6 + f^6)) dr
    if f_A > 1e-12:
        int6 = -(4.0 * math.pi * c6 / (3.0 * (f_A ** 3))) * math.atan((f_A / r_cut) ** 3)
    else:
        int6 = -(4.0 * math.pi * c6) / (3.0 * (r_cut ** 3))

    # 2) int_{r_cut}^inf 4*pi*r^2 * (-c8 / (r^8 + f^8)) dr via 32-pt Gauss-Legendre on u = r_cut / r in [0, 1]
    ratio = (f_A / r_cut) * _GL_U
    int8 = -(4.0 * math.pi * c8 / (r_cut ** 5)) * float(np.sum(_GL_WT * (_GL_U ** 4) / (1.0 + ratio ** 8)))

    # 3) Compensate energy shift + linear force-shift inside [0, r_cut]:
    #    int_0^R 4*pi*r^2 * [e_shift + (r - R) * f_shift] dr = (4*pi/3) R^3 e_shift - (pi/3) R^4 f_shift
    comp_e = (4.0 * math.pi / 3.0) * (r_cut ** 3) * e_shift
    comp_f = -(math.pi / 3.0) * (r_cut ** 4) * f_shift
    return float(int6 + int8 + comp_e + comp_f)


def fit_fixed_d3bj(
    frames: Sequence[Atoms],
    Z: Sequence[int],
    r_cut: float,
    *,
    functional: str = "pbe",
    parameters: dict[str, float] | None = None,
    force_shift: bool = True,
    use_tail: bool = True,
) -> FixedD3BJSpec:
    """Derive a fixed-C6 D3(BJ) specification from training ``frames`` and ``loupmd/data/d3_tables.npz``."""
    fn_key = functional.lower().strip()
    if parameters is not None:
        p = {k: float(parameters[k]) for k in ("s6", "s8", "a1", "a2")}
    elif fn_key in D3BJ_FUNCTIONALS:
        p = dict(D3BJ_FUNCTIONALS[fn_key])
    else:
        raise InputError(
            f"unknown D3(BJ) functional {functional!r}",
            next_step=f"Pass one of {sorted(D3BJ_FUNCTIONALS)} or explicit parameters={{s6,s8,a1,a2}}.",
        )

    tab = d3_tables()
    c6ref = tab["c6ref"]  # [95, 95, 5, 5] in Hartree * bohr^6
    r2r4 = tab["r2r4"]    # [95] in bohr^2 (dimensionless sqrt(r4/r2)^2)
    Z_t = tuple(int(z) for z in Z)
    S = len(Z_t)
    w_mean = _mean_cn_weights(frames, Z_t)

    c6 = np.zeros((S, S), dtype=np.float64)
    c8 = np.zeros((S, S), dtype=np.float64)
    f6 = np.zeros((S, S), dtype=np.float64)
    f8 = np.zeros((S, S), dtype=np.float64)
    e_shift = np.zeros((S, S), dtype=np.float64)
    f_shift = np.zeros((S, S), dtype=np.float64)
    J_st = np.zeros((S, S), dtype=np.float64)

    s6, s8, a1, a2 = p["s6"], p["s8"], p["a1"], p["a2"]
    rc = float(r_cut)
    rc5, rc6, rc7, rc8 = rc ** 5, rc ** 6, rc ** 7, rc ** 8

    for s in range(S):
        zs = Z_t[s]
        for t in range(S):
            zt = Z_t[t]
            c6_au = float(w_mean[s] @ c6ref[zs, zt] @ w_mean[t])
            qq = 3.0 * float(r2r4[zs]) * float(r2r4[zt])
            f_au = a1 * math.sqrt(qq) + a2
            f_A = f_au * BOHR

            c6_ev = s6 * c6_au * HARTREE_TO_EV * (BOHR ** 6)
            c8_ev = s8 * c6_au * qq * HARTREE_TO_EV * (BOHR ** 8)
            f6_v = f_A ** 6
            f8_v = f_A ** 8

            d6 = rc6 + f6_v
            d8 = rc8 + f8_v
            u_rc = -(c6_ev / d6) - (c8_ev / d8)
            du_rc = (6.0 * c6_ev * rc5) / (d6 * d6) + (8.0 * c8_ev * rc7) / (d8 * d8)

            es = u_rc
            fs = du_rc if force_shift else 0.0

            c6[s, t] = c6_ev
            c8[s, t] = c8_ev
            f6[s, t] = f6_v
            f8[s, t] = f8_v
            e_shift[s, t] = es
            f_shift[s, t] = fs
            J_st[s, t] = _tail_integral_pair(c6_ev, c8_ev, f_A, rc, es, fs) if use_tail else 0.0

    return FixedD3BJSpec(
        functional=fn_key,
        r_cut=rc,
        force_shift=bool(force_shift),
        use_tail=bool(use_tail),
        Z=Z_t,
        c6=c6,
        c8=c8,
        f6=f6,
        f8=f8,
        e_shift=e_shift,
        f_shift=f_shift,
        J_st=J_st,
        s6=s6,
        s8=s8,
        a1=a1,
        a2=a2,
    )


def eval_fixed_d3bj_f64(atoms: Atoms, spec: FixedD3BJSpec) -> dict[str, np.ndarray | float]:
    """Exact float64 reference evaluation of ``spec`` on ``atoms``.

    Returns dict with ``energy`` (eV), ``forces`` ([N, 3] eV/Å), ``virial`` ([6] Voigt xx,xy,xz,yy,yz,zz,
    +dE/dstrain in eV), ``stress`` ([3, 3] eV/Å^3, ASE sign +dE/dstrain / V), ``atom_energies`` ([N] eV),
    and ``e_tail`` (eV).
    """
    nums = np.asarray(atoms.numbers, dtype=np.int64)
    N = len(nums)
    z_to_s = {z: s for s, z in enumerate(spec.Z)}
    sp = np.array([z_to_s[int(z)] for z in nums], dtype=np.int64)

    atom_e = np.zeros(N, dtype=np.float64)
    forces = np.zeros((N, 3), dtype=np.float64)
    virial = np.zeros(6, dtype=np.float64)  # [xx, xy, xz, yy, yz, zz]

    is_per = all(bool(p) for p in atoms.pbc) and abs(float(np.linalg.det(atoms.cell.array))) > 1e-6
    if N > 1:
        if is_per:
            # D_ij = pos[j] - pos[i] + S @ cell -> vector receiver i minus sender j is -D_ij
            ii, jj, DD = neighbor_list("ijD", atoms, spec.r_cut)
            vec = -DD  # pos[i] - pos[j] + shift
        else:
            pos = np.asarray(atoms.positions, dtype=np.float64)
            diff = pos[:, None, :] - pos[None, :, :]
            dist = np.linalg.norm(diff, axis=-1)
            mask = (dist > 1e-12) & (dist < spec.r_cut)
            ii, jj = np.where(mask)
            vec = diff[ii, jj]

        if len(ii) > 0:
            r = np.linalg.norm(vec, axis=1)
            valid = (r > 1e-12) & (r < spec.r_cut)
            ii, jj, vec, r = ii[valid], jj[valid], vec[valid], r[valid]
            si = sp[ii]
            sj = sp[jj]

            c6 = spec.c6[si, sj]
            c8 = spec.c8[si, sj]
            f6 = spec.f6[si, sj]
            f8 = spec.f8[si, sj]
            es = spec.e_shift[si, sj]
            fs = spec.f_shift[si, sj]

            r2 = r * r
            r4 = r2 * r2
            r5 = r4 * r
            r6 = r4 * r2
            r7 = r6 * r
            r8 = r4 * r4
            inv_d6 = 1.0 / (r6 + f6)
            inv_d8 = 1.0 / (r8 + f8)

            u = -(c6 * inv_d6) - (c8 * inv_d8)
            du = (6.0 * c6 * r5) * (inv_d6 * inv_d6) + (8.0 * c8 * r7) * (inv_d8 * inv_d8)

            v_pair = u - es - (r - spec.r_cut) * fs
            dv_pair = du - fs

            # Each undirected pair appears twice in the full neighbour list -> 0.5 for energy & virial,
            # full -dv_pair * (vec / r) on receiver i.
            np.add.at(atom_e, ii, 0.5 * v_pair)
            coef = dv_pair / r
            f_edge = -coef[:, None] * vec
            np.add.at(forces, ii, f_edge)

            # Virial = +0.5 * sum_e (dv_pair / r) * (vec x vec)  (+dE/dstrain)
            vc = 0.5 * coef
            virial[0] += float(np.sum(vc * vec[:, 0] * vec[:, 0]))
            virial[1] += float(np.sum(vc * vec[:, 0] * vec[:, 1]))
            virial[2] += float(np.sum(vc * vec[:, 0] * vec[:, 2]))
            virial[3] += float(np.sum(vc * vec[:, 1] * vec[:, 1]))
            virial[4] += float(np.sum(vc * vec[:, 1] * vec[:, 2]))
            virial[5] += float(np.sum(vc * vec[:, 2] * vec[:, 2]))

    e_tail = 0.0
    vol = abs(float(np.linalg.det(atoms.cell.array))) if is_per else 0.0
    if spec.use_tail and is_per and vol > 1e-12:
        counts = np.bincount(sp, minlength=spec.S).astype(np.float64)
        e_per_sp = 0.5 * (spec.J_st @ counts) / vol
        atom_e += e_per_sp[sp]
        e_tail = float(0.5 * (counts @ spec.J_st @ counts) / vol)
        # d(K/V)/d(eps_aa) = -K/V = -e_tail
        virial[0] -= e_tail
        virial[3] -= e_tail
        virial[5] -= e_tail

    energy = float(np.sum(atom_e))
    stress = np.zeros((3, 3), dtype=np.float64)
    if is_per and vol > 1e-12:
        stress[0, 0] = virial[0] / vol
        stress[0, 1] = stress[1, 0] = virial[1] / vol
        stress[0, 2] = stress[2, 0] = virial[2] / vol
        stress[1, 1] = virial[3] / vol
        stress[1, 2] = stress[2, 1] = virial[4] / vol
        stress[2, 2] = virial[5] / vol

    return {
        "energy": energy,
        "forces": forces,
        "virial": virial,
        "stress": stress,
        "atom_energies": atom_e,
        "e_tail": e_tail,
    }


def transform_labels_for_d3(
    frames: Sequence[Atoms],
    spec: FixedD3BJSpec,
    *,
    labels_have_d3: bool = False,
    ref_d3_fn: Callable[[Atoms], dict] | None = None,
) -> list[Atoms]:
    """Transform ``REF_energy``, ``REF_forces``, and ``REF_stress`` for Option B training.

    - If ``labels_have_d3=False`` (e.g. PBE labels without D3): if ``ref_d3_fn`` is provided, adds the
      small local remainder ``(D3_ref_in - V_builtin)``; if ``ref_d3_fn is None``, labels are returned
      unchanged (exact for saturated-CN crystals/salts where fixed-C6 matches exact D3 inside ``r_cut``).
    - If ``labels_have_d3=True`` (e.g. ωB97M-D3BJ or OFF24 labels): subtracts ``V_builtin + tail`` (or
      ``D3_ref_out + V_builtin`` when ``ref_d3_fn`` is given) so ``student + V_builtin + tail`` reproduces
      the target labels cleanly without double-counting.
    """
    out: list[Atoms] = []
    for at in frames:
        c = at.copy()
        c.calc = None
        bi = eval_fixed_d3bj_f64(c, spec)
        if not labels_have_d3:
            if ref_d3_fn is not None:
                ref = ref_d3_fn(c)
                de = float(ref["energy"]) - float(bi["energy"])
                df = np.asarray(ref["forces"], dtype=np.float64) - bi["forces"]
                ds = np.asarray(ref["stress"], dtype=np.float64) - bi["stress"] if "stress" in ref else None
            else:
                de, df, ds = 0.0, None, None
        else:
            if ref_d3_fn is not None:
                ref = ref_d3_fn(c)
                de = -float(ref["energy"])
                df = -np.asarray(ref["forces"], dtype=np.float64)
                ds = -np.asarray(ref["stress"], dtype=np.float64) if "stress" in ref else None
            else:
                de = -float(bi["energy"])
                df = -bi["forces"]
                ds = -bi["stress"]

        if "REF_energy" in c.info and de != 0.0:
            c.info["REF_energy"] = float(c.info["REF_energy"]) + de
        if "REF_forces" in c.arrays and df is not None:
            c.arrays["REF_forces"] = np.asarray(c.arrays["REF_forces"], dtype=np.float64) + df
        if "REF_stress" in c.info and ds is not None:
            c.info["REF_stress"] = np.asarray(c.info["REF_stress"], dtype=np.float64).reshape(3, 3) + ds
        out.append(c)
    return out
