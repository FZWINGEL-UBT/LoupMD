#!/usr/bin/env python3
"""Generate src/train/mace_ft_sh_cg.cuh: exact spherical-harmonic JVP and sparse Wigner-3j CG tables
for the pure-CUDA foundation MACE fine-tuner (lmax_x in {0, 1, 2}, SH l <= 3)."""
from __future__ import annotations

import math
from pathlib import Path
import sympy as sp
import torch
from e3nn import o3


def conv_paths(lmax_x: int) -> list[tuple[int, int, int]]:
    paths = []
    for l1 in range(lmax_x + 1):
        p1 = (-1) ** l1
        for l2 in range(4):
            p2 = (-1) ** l2
            for l3 in range(abs(l1 - l2), min(l1 + l2, 3) + 1):
                if p1 * p2 == (-1) ** l3:
                    paths.append((l1, l2, l3))
    return sorted(paths, key=lambda p: p[2])


def emit_sh16_jvp() -> list[str]:
    x, y, z, dx, dy, dz = sp.symbols("x y z dx dy dz", real=True)
    S3 = sp.sqrt(3)
    S5 = sp.sqrt(5)
    S56 = sp.sqrt(sp.Rational(5, 6))
    S38 = sp.sqrt(sp.Rational(3, 8))
    C1, C2, C3 = sp.sqrt(3), sp.sqrt(5), sp.sqrt(7)

    s20 = S3 * x * z
    s21 = S3 * x * y
    y2 = y * y
    x2z2 = x * x + z * z
    s22 = y2 - sp.Rational(1, 2) * x2z2
    s23 = S3 * y * z
    s24 = sp.Rational(1, 2) * S3 * (z * z - x * x)
    s30 = S56 * (s20 * z + s24 * x)
    s31 = S5 * s20 * y
    s32 = S38 * (4 * y2 - x2z2) * x
    s33 = sp.Rational(1, 2) * y * (2 * y2 - 3 * x2z2)
    s34 = S38 * z * (4 * y2 - x2z2)
    s35 = S5 * s24 * y
    s36 = S56 * (s24 * z - s20 * x)

    Y = [
        sp.Integer(1), x * C1, y * C1, z * C1,
        s20 * C2, s21 * C2, s22 * C2, s23 * C2, s24 * C2,
        s30 * C3, s31 * C3, s32 * C3, s33 * C3, s34 * C3, s35 * C3, s36 * C3,
    ]
    dY = [sp.expand(sp.diff(f, x) * dx + sp.diff(f, y) * dy + sp.diff(f, z) * dz) for f in Y]
    # Replace constants with float literals cleanly via cse
    reps, exprs = sp.cse(dY, optimizations="basic")

    def flit(val: float) -> str:
        s = f"{val:.9g}"
        if "." not in s and "e" not in s and "E" not in s:
            s += ".0"
        return s + "f"

    def c_expr(e) -> str:
        if isinstance(e, sp.Number) or e.is_number:
            return flit(float(e.evalf(17)))
        if isinstance(e, sp.Symbol):
            return str(e)
        if isinstance(e, sp.Add):
            return "(" + " + ".join(c_expr(a) for a in e.args) + ")"
        if isinstance(e, sp.Mul):
            return "(" + " * ".join(c_expr(a) for a in e.args) + ")"
        if isinstance(e, sp.Pow):
            base, exp = e.args
            if exp == 2:
                b = c_expr(base)
                return f"({b} * {b})"
            if exp == 3:
                b = c_expr(base)
                return f"({b} * {b} * {b})"
            raise ValueError(f"Unexpected pow: {e}")
        raise ValueError(f"Unexpected expr: {type(e)} {e}")

    lines = [
        "// Exact JVP of loup::edge_sh16 w.r.t. unit vector (x, y, z) along tangent (dx, dy, dz).",
        "// Caller projects edge vector tangent v onto unit sphere: du = (v - (v.u)*u) / r.",
        "__device__ __forceinline__ void sh16_jvp(float x, float y, float z, float dx, float dy, float dz, float dY[16]) {",
    ]
    for sym, expr in reps:
        lines.append(f"  const float {sym} = {c_expr(expr)};")
    for i, expr in enumerate(exprs):
        lines.append(f"  dY[{i}] = {c_expr(expr)};")
    lines.append("}")
    return lines


def emit_cg_tables() -> list[str]:
    lines = [
        "struct CGEntry { uint8_t m1, m2, m3; float val; };",
        "struct PathMeta { uint8_t l1, l2, l3, l3_slot; uint16_t cg_off, cg_len; };",
        "",
    ]
    for lx in (0, 1, 2):
        paths = conv_paths(lx)
        cg_entries: list[tuple[int, int, int, float]] = []
        metas: list[tuple[int, int, int, int, int, int]] = []
        l3_cnt = [0, 0, 0, 0]
        for l1, l2, l3 in paths:
            W3 = o3.wigner_3j(l1, l2, l3, dtype=torch.float64)
            off = len(cg_entries)
            for m1 in range(2 * l1 + 1):
                for m2 in range(2 * l2 + 1):
                    for m3 in range(2 * l3 + 1):
                        v = float(W3[m1, m2, m3].item())
                        if abs(v) > 1e-12:
                            cg_entries.append((m1, m2, m3, v))
            metas.append((l1, l2, l3, l3_cnt[l3], off, len(cg_entries) - off))
            l3_cnt[l3] += 1

        lines.append(f"__device__ __constant__ PathMeta kPaths_L{lx}[{len(metas)}] = {{")
        for l1, l2, l3, slot, off, ln in metas:
            lines.append(f"  {{{l1}, {l2}, {l3}, {slot}, {off}, {ln}}},")
        lines.append("};")
        lines.append(f"__device__ __constant__ CGEntry kCG_L{lx}[{len(cg_entries)}] = {{")
        for m1, m2, m3, v in cg_entries:
            s = f"{v:.9g}"
            if "." not in s and "e" not in s and "E" not in s:
                s += ".0"
            lines.append(f"  {{{m1}, {m2}, {m3}, {s}f}},")
        lines.append("};")
        lines.append("")
    return lines


def main():
    out_path = Path(__file__).resolve().parents[2] / "src" / "train" / "mace_ft" / "mace_ft_sh_cg.cuh"
    lines = [
        "#pragma once",
        "// Auto-generated by tools/dev/gen_mace_ft_sh_cg.py. Do not edit.",
        "#include <cstdint>",
        "",
        "namespace loup_ft {",
        "",
        *emit_sh16_jvp(),
        "",
        *emit_cg_tables(),
        "}  // namespace loup_ft",
        "",
    ]
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
