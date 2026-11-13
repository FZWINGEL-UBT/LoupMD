#!/usr/bin/env python3
"""Export any supported MACE family into a generic .loup bundle (arch=generic; src/generic.cu runs it).

Supported: ScaleShiftMACE with RealAgnostic{,Residual}, RealAgnosticDensity{,Residual} and
RealAgnosticResidualNonLinear interaction blocks (MACE-OMAT-0 small/medium, MACE-OFF, MACE-MH-1; MACE-POLAR-1's
local part). Every weight is extracted by probing the float64 e3nn modules with basis inputs, so the bundle holds
plain matrices in the engine's layouts and every normalisation constant (e3nn path weights, 1/avg_num_neighbors,
activation second-moment constants, ScaleShift) is folded in numerically rather than re-derived.

Engine layouts: node features of degree l are [N, 2l+1, C] (component-major, channel fastest). The conv tensor
product of instruction k = (l1, l2, l3) computes per channel c: mid_k[m3] = R_k,c(r) * sum T_k[m1, m2, m3] x_l1[m1]
Y_l2[m2] with T_k = wigner_3j(l1, l2, l3) (generated into the kernels); the e3nn path weight is folded into R_k.

Usage: python export_generic.py --model M --elements 8,22,38 --output out.loup [--head NAME]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

FASTMACE_ROOT = next(
    (p for p in (Path("/mnt/d/Margraf/FastMACE"), Path("/mnt/homes/zwingel/FastMACE")) if p.exists()),
    Path("/mnt/d/Margraf/FastMACE"),
)
sys.path.insert(0, str(FASTMACE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from e3nn import o3  # noqa: E402

from bundle import write_bundle  # noqa: E402

EXPORTER_VERSION = 2
KINDS = {
    "RealAgnosticInteractionBlock": "agnostic",
    "RealAgnosticResidualInteractionBlock": "agnostic_res",
    "RealAgnosticDensityInteractionBlock": "density",
    "RealAgnosticDensityResidualInteractionBlock": "density_res",
    "RealAgnosticResidualNonLinearInteractionBlock": "nonlinear",
}
KIND_CODE = {"agnostic": 0, "agnostic_res": 1, "density": 2, "density_res": 3, "nonlinear": 4}
SH_LMAX = 3


# ----------------------------------------------------------------------------------------------- irreps helpers
def blocks(irreps):
    """[(mul, l, parity, offset)] of an e3nn Irreps in mul_ir layout."""
    out, off = [], 0
    for mul, ir in o3.Irreps(irreps):
        out.append((mul, ir.l, ir.p, off))
        off += mul * ir.dim
    return out


def lmax_of(irreps):
    return max(ir.l for _, ir in o3.Irreps(irreps))


def per_l_mul(irreps):
    """{l: total multiplicity} (all blocks of a degree, in irreps order)."""
    out = {}
    for mul, ir in o3.Irreps(irreps):
        out[ir.l] = out.get(ir.l, 0) + mul
    return out


def single_mul(irreps):
    """The common multiplicity of an irreps whose blocks all share one (e.g. 128x0e+128x1o)."""
    muls = {mul for mul, _ in o3.Irreps(irreps)}
    assert len(muls) == 1, f"{irreps}: mixed multiplicities"
    return muls.pop()


def to_engine(x, irreps):
    """[n, dim] mul_ir -> {l: [n, 2l+1, mul_l]} (blocks of one degree concatenated along channels)."""
    parts = {}
    for mul, l, _, off in blocks(irreps):
        d = 2 * l + 1
        parts.setdefault(l, []).append(x[:, off:off + mul * d].reshape(-1, mul, d).transpose(1, 2))
    return {l: torch.cat(v, dim=2) for l, v in parts.items()}


def from_engine(parts, irreps):
    """Inverse of to_engine."""
    n = next(iter(parts.values())).shape[0]
    used = {l: 0 for l in parts}
    cols = []
    for mul, l, _, _ in blocks(irreps):
        p = parts[l][:, :, used[l]:used[l] + mul]
        used[l] += mul
        cols.append(p.transpose(1, 2).reshape(n, -1))
    return torch.cat(cols, dim=1)


# ----------------------------------------------------------------------------------------------- probes
@torch.no_grad()
def probe_linear(fn, irreps_in, irreps_out):
    """{l: (W [mul_in_l, mul_out_l], bias [mul_out_l] or None)} of an equivariant linear map fn (mul_ir in/out).
    Checked on random inputs afterwards."""
    din = o3.Irreps(irreps_in).dim
    zero = fn(torch.zeros(1, din, dtype=torch.float64))
    base = to_engine(zero, irreps_out)
    mul_in = per_l_mul(irreps_in)
    mul_out = per_l_mul(irreps_out)
    out = {}
    for l, mi in mul_in.items():
        if l not in mul_out:
            continue
        x = {ll: torch.zeros(mi_, 2 * ll + 1, m_, dtype=torch.float64) for ll, m_ in mul_in.items() for mi_ in [mi]}
        for u in range(mi):
            x[l][u, 0, u] = 1.0
        y = to_engine(fn(from_engine(x, irreps_in)), irreps_out)
        W = (y[l][:, 0, :] - base[l][:, 0, :])  # [mi, mo]
        bias = base[l][0, 0, :] if (l == 0 and base[l].abs().max() > 0) else None
        out[l] = (W, bias)
    # verify on random input
    xr = torch.randn(3, din, dtype=torch.float64)
    yr = to_engine(fn(xr), irreps_out)
    xe = to_engine(xr, irreps_in)
    for l, mo in mul_out.items():
        pred = torch.zeros(3, 2 * l + 1, mo, dtype=torch.float64)
        if l in out:
            W, bias = out[l]
            pred = xe[l] @ W
            if bias is not None:
                pred = pred + bias
        err = (pred - yr[l]).abs().max().item()
        assert err < 1e-9 * max(1.0, yr[l].abs().max().item()), f"linear probe mismatch l={l}: {err}"
    return out


def W_of(probe, l):
    return probe[l][0]


@torch.no_grad()
def probe_species_linear(fn, irreps_in, irreps_out, n_elements, cols):
    """Species-dependent linear (FullyConnectedTensorProduct(x, one-hot)): {l: W [S, mul_in_l, mul_out_l]}."""
    per_s = []
    for c in cols:
        attrs = torch.zeros(1, n_elements, dtype=torch.float64)
        attrs[0, c] = 1.0

        def f(x, attrs=attrs):
            return fn(x, attrs.expand(x.shape[0], -1))

        per_s.append(probe_linear(f, irreps_in, irreps_out))
    return {l: torch.stack([p[l][0] for p in per_s]) for l in per_s[0]}


# ----------------------------------------------------------------------------------------------- conv TP
def conv_paths(lmax_x):
    """[(l1, l2, l3)] of the generic conv tensor product for input 0e..lmax_x (alternating parity) with SH l <= 3
    and target 0e+1o+2e+3o, in MACE's instruction order (tp_out_irreps_with_instructions: enumerated over
    (input, SH, output degree), then stably sorted by output irrep, so mid block k is instruction k)."""
    paths = []
    for l1 in range(lmax_x + 1):
        p1 = (-1) ** l1
        for l2 in range(SH_LMAX + 1):
            p2 = (-1) ** l2
            for l3 in range(abs(l1 - l2), min(l1 + l2, SH_LMAX) + 1):
                if p1 * p2 == (-1) ** l3:
                    paths.append((l1, l2, l3))
    return sorted(paths, key=lambda p: p[2])


@torch.no_grad()
def extract_conv(conv_tp, irreps_x, sh_irreps, lmax_x):
    """Per instruction k: (l1, l2, l3, alpha_k, i_out) with T_k = alpha_k * wigner_3j(l1, l2, l3)."""
    ins = list(conv_tp.instructions)
    xb, yb = blocks(irreps_x), blocks(sh_irreps)
    ob = blocks(conv_tp.irreps_out)
    mul = single_mul(irreps_x)
    numel = conv_tp.weight_numel
    assert numel == len(ins) * mul, f"conv_tp weight_numel {numel} != {len(ins)} x {mul}"
    out = []
    for k, i in enumerate(ins):
        assert i.connection_mode == "uvu", i.connection_mode
        l1, l2, l3 = xb[i.i_in1][1], yb[i.i_in2][1], ob[i.i_out][1]
        d1, d2, d3 = 2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1
        x = torch.zeros(d1 * d2, o3.Irreps(irreps_x).dim, dtype=torch.float64)
        y = torch.zeros(d1 * d2, o3.Irreps(sh_irreps).dim, dtype=torch.float64)
        for m1 in range(d1):
            for m2 in range(d2):
                x[m1 * d2 + m2, xb[i.i_in1][3] + 0 * d1 + m1] = 1.0  # channel 0
                y[m1 * d2 + m2, yb[i.i_in2][3] + m2] = 1.0
        w = torch.zeros(d1 * d2, numel, dtype=torch.float64)
        w[:, k * mul + 0] = 1.0
        res = conv_tp(x, y, w)
        T = res[:, ob[i.i_out][3]:ob[i.i_out][3] + d3].reshape(d1, d2, d3)  # channel 0 of the output block
        W3 = o3.wigner_3j(l1, l2, l3, dtype=torch.float64)
        alpha = float((T * W3).sum() / (W3 * W3).sum())
        assert (T - alpha * W3).abs().max() < 1e-6 * max(1.0, T.abs().max().item()), f"path {k} not a 3j multiple"
        out.append((l1, l2, l3, alpha, i.i_out))
    assert [(a, b, c) for a, b, c, _, _ in out] == conv_paths(lmax_x), "instruction order differs from the generator's"
    assert [i_out for *_, i_out in out] == list(range(len(out))), "mid irreps are not in instruction order"
    return out


# ----------------------------------------------------------------------------------------------- radial chains
def _split_net(net):
    """(hidden callable, last layer) of an e3nn FullyConnectedNet or MACE RadialMLP."""
    seq = net.net if hasattr(net, "net") and isinstance(net.net, torch.nn.Sequential) else net
    layers = list(seq.children())
    hidden = torch.nn.Sequential(*layers[:-1])
    return hidden, layers[-1]


@torch.no_grad()
def _last_affine(last, width):
    """(W [out, width], b [out]) of the last layer (probed: e3nn normalisation folded in)."""
    zero = last(torch.zeros(1, width, dtype=torch.float64))[0]
    eye = last(torch.eye(width, dtype=torch.float64)) - zero
    return eye.T.contiguous(), zero


class RadialChain:
    """Layer t's radial chain per ordered species pair (sender s, receiver t): columns [h c (H), c, density, pad],
    with `weight` [Q + 1, H + 2] giving the conv weights (rows k * mul + u, path factors folded) and, as the last
    row, the edge density. `with_derivative` is what radial.spline.fit consumes."""

    def __init__(self, model, layer, cols, dtype, path_scale, kind):
        import copy

        self.dtype = dtype
        self.cols = list(cols)
        self.n_el = len(model.atomic_numbers)
        self.atomic_numbers = model.atomic_numbers
        self.radial = copy.deepcopy(model.radial_embedding).to(dtype)
        self.kind = kind
        hidden, last = _split_net(layer.conv_tp_weights)
        self.hidden = copy.deepcopy(hidden).to(dtype)
        self.density = (copy.deepcopy(layer.density_fn).to(dtype)
                        if kind in ("density", "density_res", "nonlinear") else None)
        if kind == "nonlinear":
            onehot = torch.zeros(len(self.cols), self.n_el, dtype=torch.float64)
            onehot[torch.arange(len(self.cols)), torch.as_tensor(self.cols)] = 1.0
            with torch.no_grad():
                self.src = copy.deepcopy(layer.source_embedding).double()(onehot).to(dtype)
                self.tgt = copy.deepcopy(layer.target_embedding).double()(onehot).to(dtype)
        with torch.no_grad():
            width = self._inputs(torch.tensor([2.0], dtype=dtype), 0, 0)[1].shape[1]
        W, b = _last_affine(copy.deepcopy(last).double(), width)  # [Q, H], [Q]
        Q = W.shape[0]
        scale = torch.as_tensor(path_scale, dtype=torch.float64)
        assert scale.shape[0] == Q, f"{scale.shape[0]} path-scaled rows for {Q} radial outputs"
        W, b = W * scale[:, None], b * scale
        self.H, self.Q = width, Q
        weight = torch.zeros(Q + 1, width + 2, dtype=torch.float64)
        weight[:Q, :width] = W
        weight[:Q, width] = b
        weight[Q, width + 1] = 1.0
        self.weight = weight

    def _inputs(self, r, s, t):
        n = r.shape[0]
        attrs = torch.zeros(2, self.n_el, dtype=self.dtype)
        attrs[0, self.cols[s]] = 1.0
        attrs[1, self.cols[t]] = 1.0
        ei = torch.stack([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])
        feats, cutoff = self.radial(r[:, None], attrs, ei, self.atomic_numbers)
        c = torch.ones_like(r[:, None]) if cutoff is None else cutoff
        inputs = feats
        if self.kind == "nonlinear":
            inputs = torch.cat([feats, self.src[s].expand(n, -1), self.tgt[t].expand(n, -1)], dim=1)
        return inputs, self.hidden(inputs), c

    @property
    def device(self):
        return torch.device("cpu")

    def __call__(self, r, s, t):
        inputs, h, c = self._inputs(r, s, t)
        if self.density is not None:
            dens = torch.tanh(self.density(inputs) ** 2) * c
        else:
            dens = torch.zeros_like(c)
        return torch.cat([h * c, c, dens, torch.zeros_like(c)], dim=1)

    def with_derivative(self, r, s, t):
        return torch.func.jvp(lambda x: self(x, s, t), (r,), (torch.ones_like(r),))


# ----------------------------------------------------------------------------------------------- products
def product_tensors(mc64, mc32, t, cols):
    """(factored weights [S' , segments_w, C], features signature, segments) of products[t] (cuEq SC)."""
    from fastmace.kernels.cuda.sc_factored import factor, fold_weights
    from fastmace.kernels.cuda.sc_generated import merged_terms, shape_of

    sc = mc64.products[t].symmetric_contractions
    sc32 = mc32.products[t].symmetric_contractions
    sh = shape_of(sc32)
    assert sh is not None, f"products[{t}]: not a generated-kernel symmetric contraction"
    terms = merged_terms(sc32.etp.polynomial)
    prog, feats, K = factor(terms, sh[0], sh[1], sh[2])
    raw = torch.einsum("zau,ab->zbu", sc.weight.detach(), sc.projection.detach())
    agnostic = raw.shape[0] == 1
    raw = raw if agnostic else raw[cols]
    w = fold_weights(raw.reshape(raw.shape[0], -1), K, sh[1], sh[3]).reshape(raw.shape[0], sh[1], sh[3])
    sig = product_signature(prog, feats, sh)
    return w.to(torch.float32).numpy(), sig, sh, agnostic


def product_signature(prog, feats, sh):
    h = hashlib.sha256(repr((sh[0], sh[1], sh[2], prog.nodes, [sorted(f.items()) if f else None for f in feats])).encode())
    return h.hexdigest()[:16]


# ----------------------------------------------------------------------------------------------- readouts
@torch.no_grad()
def act_constant(act_module, fn):
    width = o3.Irreps(str(act_module.irreps_in)).dim
    x = torch.linspace(-3, 3, 13, dtype=torch.float64)[:, None].expand(-1, width).contiguous()
    y = act_module(x)[:, :1]
    x = x[:, :1]
    ref = fn(x)
    mask = ref.abs() > 1e-3
    c = (y[mask] / ref[mask])
    assert (c - c[0]).abs().max() < 1e-9, "activation is not a scaled silu/sigmoid"
    return float(c[0])


@torch.no_grad()
def export_readout(ro, irreps_in, head, n_heads, scale):
    """dict: kind 0 linear (w [C]); kind 1 MLP (W1 [C, H], b1 [H], act_c, W2 [H], b2) or 2 bias MLP (+ Wm, bm)."""
    name = type(ro).__name__
    C0 = per_l_mul(irreps_in)[0]
    if name == "LinearReadoutBlock":
        W = W_of(probe_linear(ro.linear, irreps_in, ro.linear.irreps_out), 0)  # [C0, n_out]
        col = head if W.shape[1] > 1 else 0
        return dict(kind=0, w=(W[:, col] * scale))
    if name in ("NonLinearReadoutBlock", "NonLinearBiasReadoutBlock"):
        p1 = probe_linear(ro.linear_1, irreps_in, ro.linear_1.irreps_out)
        W1, b1 = p1[0]
        Hall = W1.shape[1]
        nh = getattr(ro, "num_heads", 1) or 1
        Hh = Hall // nh if nh > 1 else Hall
        sel = slice(head * Hh, (head + 1) * Hh) if nh > 1 else slice(0, Hall)
        W1 = W1[:, sel]
        b1 = (b1[sel] if b1 is not None else torch.zeros(Hh, dtype=torch.float64))
        act = ro.non_linearity
        c = act_constant(act, torch.nn.functional.silu)
        out = dict(kind=1, W1=W1, b1=b1, act_c=c)
        if name == "NonLinearBiasReadoutBlock":
            pm = probe_linear(ro.linear_mid, ro.linear_mid.irreps_in, ro.linear_mid.irreps_out)
            Wm, bm = pm[0]
            out.update(kind=2, Wm=Wm[sel, sel], bm=(bm[sel] if bm is not None else torch.zeros(Hh, dtype=torch.float64)))
        p2 = probe_linear(ro.linear_2, ro.linear_2.irreps_in, ro.linear_2.irreps_out)
        W2, b2 = p2[0]
        col = head if W2.shape[1] > 1 else 0
        out["W2"] = W2[sel, col] * scale
        out["b2"] = float(b2[col] * scale) if b2 is not None else 0.0
        return out
    raise ValueError(f"unsupported readout {name}")


# ----------------------------------------------------------------------------------------------- main export
def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_models(model_path):
    from mace.cli.convert_e3nn_cueq import run as convert_e3nn_cueq

    m64 = torch.load(model_path, map_location="cpu", weights_only=False).double()
    m32 = torch.load(model_path, map_location="cpu", weights_only=False).float()
    return m64, m32, convert_e3nn_cueq(m64), convert_e3nn_cueq(m32)


def export_model(model_path, atomic_nums, out_path, head=None):
    from fastmace.rewrites import radial as rad
    from fastmace.rewrites.radial import spline as rspline

    atomic_nums = sorted({int(z) for z in atomic_nums})
    S = len(atomic_nums)
    m64, m32, mc64, mc32 = load_models(model_path)
    cls = type(m64).__name__
    if cls not in ("ScaleShiftMACE", "MACE", "PolarMACE"):
        raise ValueError(f"unsupported model class {cls}")
    all_z = [int(z) for z in m64.atomic_numbers.tolist()]
    missing = [z for z in atomic_nums if z not in all_z]
    if missing:
        raise ValueError(f"elements {missing} are not in the model")
    cols = [all_z.index(z) for z in atomic_nums]
    n_el = len(all_z)
    heads = list(getattr(m64, "heads", None) or ["default"])
    if head is None:
        head_idx = len(heads) - 1 if len(heads) > 1 else 0
        if len(heads) > 1:
            for pref in ("omat_pbe", "default", "Default"):
                if pref in heads:
                    head_idx = heads.index(pref)
                    break
    else:
        if head not in heads:
            raise ValueError(f"head '{head}' not in {heads}")
        head_idx = heads.index(head)
    r_max = float(m64.r_max)
    ss = getattr(m64, "scale_shift", None)
    scale = float(ss.scale.reshape(-1)[min(head_idx, ss.scale.numel() - 1)]) if ss is not None else 1.0
    shift = float(ss.shift.reshape(-1)[min(head_idx, ss.shift.numel() - 1)]) if ss is not None else 0.0
    e0_all = m64.atomic_energies_fn.atomic_energies.detach().reshape(-1, n_el)
    e0 = e0_all[min(head_idx, e0_all.shape[0] - 1)][cols].numpy().astype(np.float64) + shift

    sh_irreps = o3.Irreps(str(m64.spherical_harmonics.irreps_out))
    assert sh_irreps == o3.Irreps.spherical_harmonics(SH_LMAX), f"SH {sh_irreps}: the engine takes lmax 3"
    pairs = [(s, t) for s in range(S) for t in range(S)]  # ordered (sender, receiver)
    sym_pairs = [(s, t) for s in range(S) for t in range(s, S)]  # kernels_common sym_pair order
    n_layers = len(m64.interactions)
    meta = dict(arch="generic", exporter_version=EXPORTER_VERSION, model_sha256=file_sha256(model_path),
                model_class=cls, S=S, Z=atomic_nums, n_layers=n_layers, head=heads[head_idx],
                sh_permute=int(cls == "PolarMACE"))
    T = {"r_max": np.array([r_max]), "e0": e0}

    # node embedding (species -> node features of layer 0)
    emb = W_of(probe_linear(m64.node_embedding.linear, o3.Irreps(f"{n_el}x0e"), m64.node_embedding.linear.irreps_out), 0)
    T["embedding"] = emb[cols].to(torch.float32).numpy()  # [S, C0]

    readouts = list(m64.readouts)
    for t, (it, pr) in enumerate(zip(m64.interactions, m64.products)):
        kind = KINDS.get(type(it).__name__)
        if kind is None:
            raise ValueError(f"interactions[{t}]: unsupported block {type(it).__name__}")
        p = f"L{t}."
        node_in = o3.Irreps(str(it.node_feats_irreps))
        x_irreps = o3.Irreps(str(it.linear_up.irreps_out))
        hidden_out = o3.Irreps(str(pr.linear.irreps_out))
        lx = lmax_of(x_irreps)
        Cx = single_mul(x_irreps)
        Cin = single_mul(node_in)
        Cout = single_mul(hidden_out)
        target = o3.Irreps(str(it.irreps_out))
        Cm = single_mul(target)
        meta.update({p + "kind": KIND_CODE[kind], p + "lmax_in": lmax_of(node_in), p + "lmax_x": lx,
                     p + "lmax_out": lmax_of(hidden_out), p + "C_in": Cin, p + "C_x": Cx, p + "C_msg": Cm,
                     p + "C_out": Cout})
        # linear_up
        up = probe_linear(it.linear_up, node_in, x_irreps)
        for l in range(lx + 1):
            T[p + f"W_up_{l}"] = W_of(up, l).to(torch.float32).numpy()
        # conv paths and their e3nn factors
        paths = extract_conv(it.conv_tp, x_irreps, sh_irreps, lx)
        meta[p + "n_paths"] = len(paths)
        path_scale = np.repeat([a for _, _, _, a, _ in paths], Cx)
        # mid block k is path k, so W_lin_l3's rows are [paths of degree l3 in path order, channel]
        mid = o3.Irreps(str(it.conv_tp.irreps_out))
        # message linear (linear for agnostic/density, linear_1 for nonlinear) on mid irreps (blocks in mid order)
        lin = it.linear_1 if kind == "nonlinear" else it.linear
        lin_out = o3.Irreps(str(lin.irreps_out))
        plin = probe_linear(lin, mid, lin_out)
        inv_avg = 1.0 / float(it.avg_num_neighbors) if kind in ("agnostic", "agnostic_res") else 1.0
        W_lin64 = {}
        for l3 in range(SH_LMAX + 1):
            W = W_of(plin, l3)  # rows: mid blocks of degree l3 in mid order, channel fastest within each block
            W_lin64[l3] = W * inv_avg
            T[p + f"W_lin_{l3}"] = W_lin64[l3].to(torch.float32).numpy()
        if kind == "nonlinear":
            T[p + "alpha_beta"] = np.array([float(it.alpha), float(it.beta)])
            res = probe_linear(it.linear_res, x_irreps, lin_out)
            for l in range(lx + 1):
                T[p + f"W_res_{l}"] = W_of(res, l).to(torch.float32).numpy()
            l2p = probe_linear(it.linear_2, target, target)
            for l in range(SH_LMAX + 1):
                T[p + f"W_2_{l}"] = W_of(l2p, l).to(torch.float32).numpy()
            g = it.equivariant_nonlin
            meta[p + "gate_C"] = Cm
            T[p + "gate_c"] = np.array(gate_constants(g, lin_out, Cm))
            sc = probe_linear(it.skip_tp, node_in, hidden_out)
            for l in range(lmax_of(hidden_out) + 1):
                if l in sc:
                    T[p + f"W_sc_{l}"] = W_of(sc, l)[None].to(torch.float32).numpy()  # [1, Cin, Cout]
            meta[p + "sc_species"] = 0
        elif kind in ("agnostic", "density"):
            sk = probe_species_linear(it.skip_tp, target, target, n_el, cols)
            for l in range(SH_LMAX + 1):
                T[p + f"W_skip_msg_{l}"] = sk[l].to(torch.float32).numpy()  # [S, Cm, Cm]
        else:
            sk = probe_species_linear(it.skip_tp, node_in, hidden_out, n_el, cols)
            for l in range(lmax_of(hidden_out) + 1):
                if l in sk:
                    T[p + f"W_sc_{l}"] = sk[l].to(torch.float32).numpy()  # [S, Cin, Cout]
            meta[p + "sc_species"] = 1
        meta[p + "has_sc"] = int(any(k.startswith(p + "W_sc_") for k in T))
        # radial spline (all paths + density as the last column)
        ch64 = RadialChain(m64, it, cols, torch.float64, path_scale, kind)
        ch32 = RadialChain(m32, m32.interactions[t], cols, torch.float32, path_scale, kind)
        # The radial chain reads the species only through symmetric terms (Agnesi's covalent radii) unless the block
        # embeds sender and receiver separately (nonlinear): then one table per ordered pair.
        sym = kind != "nonlinear"
        meta[p + "pair_sym"] = int(sym)
        # Layer 0 reads only the species-dependent embedding, so its message is linear in the per-receiver outer sum
        # S0[s, j, (l, m)] = sum over edges from species s of col_j(r) Y_lm (col = [h c, c] of the radial chain):
        # the table holds the H + 2 chain columns (with the edge density last) instead of the 4 C conv weights, and
        # the conv weights, the layer input, the path factors and the message linear fold into C0_l [S (H+1), lin].
        first_hidden = t == 0
        meta[p + "first_hidden"] = int(first_hidden)
        if first_hidden:
            assert lx == 0
            Jc = ch64.H + 2
            # The grid is certified on the conv weights themselves (the outputs that matter); the chain columns are
            # then interpolated on it (the weights are linear in the columns, so only float32 rounding differs).
            spl, cert = rspline.fit(ch64, ch32, ch64.weight, sym_pairs if sym else pairs, rad.R_LO, r_max,
                                    rad.CERTIFY_POINTS, rad.R_PHYSICAL, seed=1)
            spl["coef"] = rspline.coefficients(ch64, torch.eye(Jc, dtype=torch.float64), sym_pairs if sym else pairs,
                                               spl["grid"]).to(torch.float32)
            meta[p + "J"] = Jc
            Wf = ch64.weight[: ch64.Q, : ch64.H + 1]  # path-scaled [Q, H + 1]
            x_sp = emb[cols] @ W_of(up, 0)  # [S, Cx]: layer 0's linear_up of the species embedding
            for l in range(SH_LMAX + 1):
                w3 = float(o3.wigner_3j(0, l, l, dtype=torch.float64)[0, 0, 0])
                blocks_s = [(x_sp[s][:, None] * Wf[l * Cx:(l + 1) * Cx]).T @ W_lin64[l] * w3 for s in range(S)]
                T[p + f"first_C0_{l}"] = torch.cat(blocks_s, dim=0).to(torch.float32).numpy()  # [S (H+1), lin]
        else:
            spl, cert = rspline.fit(ch64, ch32, ch64.weight, sym_pairs if sym else pairs, rad.R_LO, r_max,
                                    rad.CERTIFY_POINTS, rad.R_PHYSICAL, seed=1)
        if cert["worst"] > 1.0:
            raise RuntimeError(f"layer {t}: radial spline certificate {cert['worst']:.3f} > 1")
        grid = spl["grid"]
        coef = spl["coef"]  # [P, I, 4, Q + 1] (first_hidden: [P, I, 4, H + 2])
        I = int(coef.shape[1])
        assert len(grid) <= 8
        T[p + "spline"] = coef.permute(0, 1, 3, 2).contiguous().numpy().astype(np.float32)  # [P, I, cols, 4]
        T[p + "grid"] = np.array([[a, b, n] for a, b, n in grid], np.float64)
        meta[p + "intervals"] = I
        meta[p + "spline_cert_ppm"] = int(cert["worst"] * 1e6)
        # product
        w, sig, sh, agnostic = product_tensors(mc64, mc32, t, cols)
        T[p + "prod_w"] = w
        meta[p + "prod_sig"] = sig
        meta[p + "prod_species"] = int(not agnostic)
        meta[p + "prod_segments_w"] = int(sh[1])
        pl = probe_linear(pr.linear, hidden_out, hidden_out)
        for l in range(lmax_of(hidden_out) + 1):
            T[p + f"W_p_{l}"] = W_of(pl, l).to(torch.float32).numpy()
        # readout of this layer's features
        ro_idx = t if len(readouts) > 1 else (0 if t == n_layers - 1 else None)
        if ro_idx is not None and ro_idx < len(readouts):
            r = export_readout(readouts[ro_idx], hidden_out, head_idx, len(heads), scale)
            meta[p + "readout"] = r["kind"]
            if r["kind"] == 0:
                T[p + "ro_w"] = r["w"].to(torch.float32).numpy()
            else:
                meta[p + "ro_H"] = int(r["W1"].shape[1])
                T[p + "ro_W1"] = r["W1"].to(torch.float32).numpy()
                T[p + "ro_b1"] = r["b1"].to(torch.float32).numpy()
                T[p + "ro_W2"] = r["W2"].to(torch.float32).numpy()
                T[p + "ro_act_c"] = np.array([r["act_c"]])
                if r["kind"] == 2:
                    T[p + "ro_Wm"] = r["Wm"].to(torch.float32).numpy()
                    T[p + "ro_bm"] = r["bm"].to(torch.float32).numpy()
                # the final bias is a constant per atom: fold it into E0
                e0 = e0 + r["b2"]
        else:
            meta[p + "readout"] = -1
        print(f"layer {t}: {kind}, x {x_irreps}, paths {len(paths)}, intervals {I}, "
              f"spline cert {cert['worst']:.3f}, product sig {sig}", flush=True)
    T["e0"] = e0
    meta["has_lr"] = int(cls == "PolarMACE")
    if cls == "PolarMACE":
        import export_polar

        export_polar.export_lr(m64, cols, T, meta, str(m64.products[-1].linear.irreps_out))

    # ZBL pair repulsion (per ordered pair)
    has_zbl = int(bool(getattr(m64, "pair_repulsion", False)))
    meta["has_zbl"] = has_zbl
    if has_zbl:
        zfn = m64.pair_repulsion_fn
        T["zbl_c"] = zfn.c.detach().numpy().astype(np.float64)
        T["zbl_p"] = np.array([float(zfn.p.item())])
        a_exp, a_pref = float(zfn.a_exp.item()), float(zfn.a_prefactor.item())
        cov = zfn.covalent_radii.detach().numpy().astype(np.float64)
        rc, inv_a, pref = [], [], []
        for s, t in pairs:
            zu, zv = float(atomic_nums[s]), float(atomic_nums[t])
            rc.append(cov[atomic_nums[s]] + cov[atomic_nums[t]])
            inv_a.append((zu ** a_exp + zv ** a_exp) / (a_pref * 0.529))
            pref.append(scale * 0.5 * 14.3996 * zu * zv)
        T["zbl_r_cut"] = np.array(rc, np.float32)
        T["zbl_inv_a"] = np.array(inv_a, np.float32)
        T["zbl_pref"] = np.array(pref, np.float32)
    tensors = {k: (v.numpy() if hasattr(v, "numpy") else np.asarray(v)) for k, v in T.items()}
    tensors = {k: (v.astype(np.float64) if v.dtype == np.float64 else v) for k, v in tensors.items()}
    write_bundle(out_path, meta, tensors)
    print(f"Exported {out_path} ({Path(out_path).stat().st_size / 1e6:.1f} MB), head {heads[head_idx]}")
    return meta


@torch.no_grad()
def gate_constants(g, irreps_in, C):
    """[c_scalar_act, c_gate_act]: the gate computes c_s silu(scalars) and (c_g sigmoid(gates)) * gated."""
    irreps_in = o3.Irreps(irreps_in)
    x = torch.zeros(5, irreps_in.dim, dtype=torch.float64)
    vals = torch.linspace(-1.5, 1.5, 5, dtype=torch.float64)
    x[:, 0] = vals  # scalar channel 0
    x[:, C] = vals  # first gate channel (gates follow the C scalars in the sorted 0e block)
    b1 = [b for b in blocks(irreps_in) if b[1] == 1][0]
    x[:, b1[3]] = 1.0  # gated 1o channel 0, m = 0
    y = g(x)
    out_irreps = o3.Irreps(str(g.irreps_out))
    ob1 = [b for b in blocks(out_irreps) if b[1] == 1][0]
    cs = (y[:, 0] / torch.nn.functional.silu(vals))
    cg = (y[:, ob1[3]] / torch.sigmoid(vals))
    mask = torch.nn.functional.silu(vals).abs() > 1e-6
    assert (cs[mask] - cs[mask][0]).abs().max() < 1e-9 and (cg - cg[0]).abs().max() < 1e-9, "unexpected gate"
    return [float(cs[mask][0]), float(cg[0])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--elements", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--head", default=None)
    a = ap.parse_args()
    export_model(a.model, [int(x) for x in a.elements.split(",") if x.strip()], Path(a.output), a.head)


if __name__ == "__main__":
    main()
