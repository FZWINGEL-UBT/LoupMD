"""Generate src/train/generated_train.cuh: straight-line CUDA device code for the Loup-S training library.

* ``sh16``: e3nn 0.4.4 component-normalised real spherical harmonics through l = 3 of a unit vector and their
  Cartesian gradients with respect to the (unnormalised) edge vector (same expressions as
  tools/dev/gen_fasttrain_basis.py);
* ``poly_*``: the correlation-3 invariant polynomials of the symmetric contraction (terms of
  loupmd/fasttrain/_product_terms.py, i.e. of the model given to tools/dev/gen_fasttrain_product.py) with their
  vector-Jacobian, Jacobian-vector and Hessian-vector products.

Usage: python tools/dev/gen_strain_kernels.py   (no GPU needed)
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'src' / 'train' / 'generated_train.cuh'


def load_terms():
    """TERMS and K from the generated terms module, read as text."""
    ns = {}
    src = (ROOT / 'loupmd' / 'fasttrain' / '_product_terms.py').read_text()
    k = re.search(r'^K = (\d+)$', src, re.M)
    a = src.index('TERMS = [')
    b = src.index(']\n', a)
    exec(src[a:b + 2], ns)      # noqa: S102 - our own generated file
    return ns['TERMS'], int(k.group(1))


# ----------------------------------------------------------------------------- harmonics
x, y, z = sp.symbols('ux uy uz')
s3 = sp.sqrt(3)
sh = {(0, 0): sp.Integer(1), (1, 0): x, (1, 1): y, (1, 2): z}
sh[(2, 0)] = s3 * x * z
sh[(2, 1)] = s3 * x * y
y2, x2z2 = y ** 2, x ** 2 + z ** 2
sh[(2, 2)] = y2 - sp.Rational(1, 2) * x2z2
sh[(2, 3)] = s3 * y * z
sh[(2, 4)] = s3 / 2 * (z ** 2 - x ** 2)
sh[(3, 0)] = sp.sqrt(sp.Rational(5, 6)) * (sh[(2, 0)] * z + sh[(2, 4)] * x)
sh[(3, 1)] = sp.sqrt(5) * sh[(2, 0)] * y
sh[(3, 2)] = sp.sqrt(sp.Rational(3, 8)) * (4 * y2 - x2z2) * x
sh[(3, 3)] = sp.Rational(1, 2) * y * (2 * y2 - 3 * x2z2)
sh[(3, 4)] = sp.sqrt(sp.Rational(3, 8)) * z * (4 * y2 - x2z2)
sh[(3, 5)] = sp.sqrt(5) * sh[(2, 4)] * y
sh[(3, 6)] = sp.sqrt(sp.Rational(5, 6)) * (sh[(2, 4)] * z - sh[(2, 0)] * x)


def ccode(e):
    t = sp.ccode(sp.N(sp.expand(e), 17))
    t = re.sub(r'pow\((u[xyz]), (\d)\)', lambda m: '(' + '*'.join([m.group(1)] * int(m.group(2))) + ')', t)
    return re.sub(r'(\d+\.\d+(e[-+]?\d+)?)', r'T(\1)', t)


def harmonics():
    L = ['template <typename T>',
         '__device__ __forceinline__ void sh16(T ux, T uy, T uz, T rinv, T* __restrict__ Y, T* __restrict__ JY) {',
         '    T p;']
    k = 0
    for l in range(4):
        for m in range(2 * l + 1):
            P = sp.sqrt(2 * l + 1) * sh[(l, m)]
            L.append(f'    p = {ccode(P) if P != 1 else "T(1)"};')
            L.append(f'    Y[{k}] = p;')
            for j, v in enumerate((x, y, z)):
                g = sp.diff(P, v)
                uj = ('ux', 'uy', 'uz')[j]
                gs = ccode(g) if g != 0 else 'T(0)'
                # d/dv P(v/|v|) = (grad P(u) - l P(u) u) / r for P homogeneous of degree l
                L.append(f'    JY[{3 * k + j}] = ({gs} - T({l}) * p * {uj}) * rinv;')
            k += 1
    L.append('}')
    return L


# ----------------------------------------------------------------------------- polynomials
def mono(m):
    return '*'.join(f'x[{j}]' for j in m) if m else 'T(1)'


def deriv(m, j):
    n = m.count(j)
    if n == 0:
        return None
    rest = list(m)
    rest.remove(j)
    return n, tuple(rest)


def lit(v):
    return f'T({float(v)!r})'


def polys(terms, K):
    L = [f'constexpr int POLY_K = {K};', f'constexpr int POLY_TERMS = {len(terms)};', '']
    by_k = defaultdict(list)
    for k, m, v in terms:
        by_k[k].append((tuple(m), v))
    # P_k
    L += ['template <typename T>', '__device__ __forceinline__ void poly_p(const T* x, T* P) {']
    for k in range(K):
        e = ' + '.join(f'{lit(v)} * {mono(m)}' for m, v in by_k[k]) or 'T(0)'
        L.append(f'    P[{k}] = {e};')
    L.append('}')
    # sum_k W_k P_k (fused, no array of P)
    L += ['template <typename T>', '__device__ __forceinline__ T poly_dot(const T* x, const T* W) {', '    T s = T(0);']
    for k in range(K):
        e = ' + '.join(f'{lit(v)} * {mono(m)}' for m, v in by_k[k]) or 'T(0)'
        L.append(f'    s += W[{k}] * ({e});')
    L += ['    return s;', '}']
    # vjp: G_j = sum_k V_k dP_k/dx_j
    gj = defaultdict(list)
    for k, m, v in terms:
        for j in sorted(set(m)):
            n, rest = deriv(tuple(m), j)
            gj[j].append((k, n * v, rest))
    L += ['template <typename T>', '__device__ __forceinline__ void poly_vjp(const T* x, const T* V, T* G) {']
    for j in range(16):
        e = ' + '.join(f'V[{k}] * {lit(c)} * {mono(r)}' for k, c, r in gj[j]) or 'T(0)'
        L.append(f'    G[{j}] = {e};')
    L.append('}')
    # jvp: J_k = sum_j dP_k/dx_j t_j
    L += ['template <typename T>', '__device__ __forceinline__ void poly_jvp(const T* x, const T* t, T* J) {']
    for k in range(K):
        parts = []
        for m, v in by_k[k]:
            for j in sorted(set(m)):
                n, rest = deriv(m, j)
                parts.append(f'{lit(n * v)} * {mono(rest)} * t[{j}]')
        L.append(f'    J[{k}] = {" + ".join(parts) or "T(0)"};')
    L.append('}')
    # hvp: H_i = sum_k V_k sum_j d2P_k/dx_i dx_j t_j
    hi = defaultdict(list)
    for k, m, v in terms:
        m = tuple(m)
        for i in sorted(set(m)):
            ni, rest = deriv(m, i)
            for j in sorted(set(rest)):
                nj, rest2 = deriv(rest, j)
                hi[i].append((k, ni * nj * v, rest2, j))
    L += ['template <typename T>', '__device__ __forceinline__ void poly_hvp(const T* x, const T* V, const T* t, T* H) {']
    for i in range(16):
        e = ' + '.join(f'V[{k}] * {lit(c)} * {mono(r)} * t[{j}]' for k, c, r, j in hi[i]) or 'T(0)'
        L.append(f'    H[{i}] = {e};')
    L.append('}')
    return L


def main():
    terms, K = load_terms()
    L = ['// Generated by tools/dev/gen_strain_kernels.py; do not edit.',
         '#pragma once', '', 'namespace lst {', '']
    L += harmonics() + [''] + polys(terms, K) + ['', '}  // namespace lst', '']
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text('\n'.join(L))
    print(f'wrote {OUT} ({len(terms)} terms, K = {K})')


if __name__ == '__main__':
    main()
