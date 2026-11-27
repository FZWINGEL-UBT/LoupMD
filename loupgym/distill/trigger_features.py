"""Extract Stage 0 trigger features from a trusted stock scalar MACE student.

Use a fresh directory for each train, valid and fixed-student-test extxyz. One
energy-only torch forward per geometry; force errors ALWAYS come from the joint
engine STUDENT_forces and REF_forces on that record. This is offline feature
extraction, not an engine timing or a fidelity gate. A .model is a pickle: pass
its SHA-256 from a trusted source, not a digest supplied by an untrusted sender.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import re
import shutil
from importlib.metadata import version
from pathlib import Path

import numpy as np
from ase.io import iread

from .train import MACE_VERSION


_SAMPLER = {'train': 'teacher', 'valid': 'teacher', 'fixed-student-test': 'student'}


def _sha256(value, name):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', value):
        raise ValueError(f'{name}: expected a SHA-256 digest (64 hex digits)')
    return value.lower()


def trusted_bytes(path, sha256):
    """Verify precisely the bytes that will be passed to the pickle loader."""
    expected = _sha256(sha256, '--sha256')
    blob = Path(path).read_bytes()
    if hashlib.sha256(blob).hexdigest() != expected:
        raise ValueError(f'{path}: SHA-256 mismatch; refusing to load checkpoint')
    return blob


def _forward_features(model, data, width):
    # In MACE/ScaleShiftMACE 0.3.16 node_feats concatenates post-product blocks
    # in layer order. linear_2 consumes the *activated* final readout features.
    # radial_embedding gets the exact lengths used for these edges, including
    # periodic images (positions[receiver] - positions[sender] + shifts).
    captured = {}

    def radial_input(_module, args):
        captured['distance'] = args[0][:, 0]

    def linear_input(_module, args):
        captured['hidden'] = args[0]

    with (model.radial_embedding.register_forward_pre_hook(radial_input),
          model.readouts[-1].linear_2.register_forward_pre_hook(linear_input)):
        output = model(data, training=False, compute_force=False)
    return dict(h=output['node_feats'][:, :width], energy=output['node_energy'],
                edge_index=data['edge_index'], **captured)


def _load_student(path, sha256, device):
    """Load only after verification. Neither import nor --help probes torch devices."""
    blob = trusted_bytes(path, sha256)
    if version('mace-torch') != MACE_VERSION:
        raise ValueError(f'extraction requires mace-torch=={MACE_VERSION}')
    # e3nn 0.4.x uses legacy pickle for its own installed package constants.
    os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
    import torch
    from mace import data as mace_data
    from mace.modules.blocks import LinearReadoutBlock, NonLinearReadoutBlock
    from mace.modules.models import MACE, ScaleShiftMACE
    from mace.modules.radial import PolynomialCutoff
    from mace.tools import torch_geometric, torch_tools, utils

    model = torch.load(io.BytesIO(blob), map_location='cpu', weights_only=False)
    del blob
    if (type(model) not in (MACE, ScaleShiftMACE) or len(model.heads) != 1
            or len(model.products) != 2 or len(model.readouts) != 2
            or not isinstance(model.readouts[0], LinearReadoutBlock)
            or not isinstance(model.readouts[1], NonLinearReadoutBlock)):
        raise ValueError('need a stock two-layer, single-head MACE scalar student with both readouts')
    hidden = model.readouts[-1].hidden_irreps
    for irreps in [p.linear.irreps_out for p in model.products] + [hidden]:
        if irreps.dim < 1 or any(ir.l != 0 or ir.p != 1 for _, ir in irreps):
            raise ValueError('expected scalar (0e) post-product and readout features')
    if hidden.dim != 16:
        raise ValueError('expected 16 activated final readout features')
    width = model.products[0].linear.irreps_out.dim
    envelope = model.radial_embedding.cutoff_fn
    if not isinstance(envelope, PolynomialCutoff) or float(envelope.r_max) != float(model.r_max):
        raise ValueError('expected a MACE polynomial envelope at the graph cutoff')
    cutoff, cutoff_p = float(model.r_max), int(envelope.p)
    table = utils.AtomicNumberTable(model.atomic_numbers.tolist())
    dtype = next(model.parameters()).dtype
    head = model.heads[0]
    model.eval().to(device)

    def extract(atoms):
        # Empty keyspec: no teacher/evaluation labels enter the model graph.
        config = mace_data.config_from_atoms(atoms, key_specification=mace_data.KeySpecification(), head_name=head)
        with torch_tools.default_dtype(dtype), torch.no_grad():
            graph = mace_data.AtomicData.from_config(config, z_table=table, cutoff=cutoff, heads=[head])
            batch = torch_geometric.Batch.from_data_list([graph]).to(device)
            values = _forward_features(model, batch.to_dict(), width)
        # Same unpadded single-graph edges in the same order, already on CPU.
        values['edge_index'] = graph.edge_index
        return dict(cutoff=cutoff, cutoff_p=cutoff_p, **{k: v.detach().cpu().numpy() for k, v in values.items()})

    return head, extract


def _finite(value, shape, key):
    a = np.asarray(value)
    if a.dtype.kind not in 'fiu' or a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f'{key}: expected finite numbers of shape {shape}')
    return a


def _frame_fields(atoms, split, sha256, head):
    info = atoms.info
    if info.get('split') != split:
        raise ValueError(f"split: expected {split}, got {info.get('split')!r}")
    if info.get('sampler') != _SAMPLER[split]:
        raise ValueError(f'sampler: {split} requires {_SAMPLER[split]}')
    row = {}
    for key in ('frame_id', 'trajectory_id', 'sampler', 'split', 'regime', 'teacher_head'):
        text = info.get(key)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f'{key}: expected nonempty string')
        row[key] = text
    row.update(teacher_sha256=_sha256(info.get('teacher_sha256'), 'teacher_sha256'),
               prediction_sha256=sha256, prediction_head=head)
    n = len(atoms)
    if n == 0 or not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell).all():
        raise ValueError('expected nonempty finite geometry')
    row['numbers'] = atoms.numbers
    ref = _finite(atoms.arrays.get('REF_forces'), (n, 3), 'REF_forces').astype(np.float64, copy=False)
    row['ref_forces'] = ref
    if split != 'train':
        if _sha256(info.get('prediction_sha256'), 'prediction_sha256') != sha256:
            raise ValueError('prediction_sha256 does not match trusted feature checkpoint')
        if info.get('prediction_head') != head:
            raise ValueError('prediction_head does not match feature checkpoint head')
        pred = _finite(atoms.arrays.get('STUDENT_forces'), (n, 3), 'STUDENT_forces')
        row['force_error'] = _finite(np.linalg.norm(pred - ref, axis=1), (n,), 'force_error')
    return row


def _numeric_fields(values, n):
    h = np.asarray(values['h'])
    if h.ndim != 2 or h.shape[1] == 0:
        raise ValueError('h: expected a nonempty scalar channel dimension')
    edges = np.asarray(values['edge_index'])
    if (edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype.kind not in 'iu'
            or np.any(edges < 0) or np.any(edges >= n)):
        raise ValueError('edge_index: expected valid 2 x E integer sender/receiver indices')
    distances = _finite(values['distance'], (edges.shape[1],), 'distance')
    cutoff = _finite(values['cutoff'], (), 'cutoff')
    p = np.asarray(values['cutoff_p'])
    if cutoff <= 0 or np.any(distances < 0):
        raise ValueError('expected positive cutoff and nonnegative distance')
    if p.shape != () or p.dtype.kind not in 'iu' or p < 1:
        raise ValueError('cutoff_p: expected positive integer polynomial order')
    return dict(h=_finite(h, (n, h.shape[1]), 'h'), hidden=_finite(values['hidden'], (n, 16), 'hidden'),
                energy=_finite(values['energy'], (n,), 'energy'), edge_index=edges,
                distance=distances, cutoff=cutoff, cutoff_p=p)


def write_split(frames, output, split, sha256, head, extract):
    """Stream one split to numbered, pickle-free NPZs; publish only a complete directory."""
    if split not in _SAMPLER:
        raise ValueError(f'unsupported split: {split}')
    sha256 = _sha256(sha256, 'prediction_sha256')
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'refusing to replace {output}')
    partial = output.with_name(output.name + '.partial')
    partial.mkdir(parents=True, exist_ok=False)
    seen, teacher = set(), None
    try:
        for index, atoms in enumerate(frames):
            row = _frame_fields(atoms, split, sha256, head)
            identity = (row['teacher_sha256'], row['teacher_head'])
            if teacher is not None and teacher != identity:
                raise ValueError('teacher identity changed within split')
            teacher = identity
            if row['frame_id'] in seen:
                raise ValueError(f"duplicate frame_id: {row['frame_id']}")
            seen.add(row['frame_id'])
            row.update(_numeric_fields(extract(atoms), len(atoms)))
            np.savez(partial / f'{index:08d}.npz', **row)
        if not seen:
            raise ValueError('empty frame set')
        if output.exists():
            raise FileExistsError(f'refusing to replace {output}')
        partial.rename(output)
    except BaseException:
        shutil.rmtree(partial)
        raise
    return len(seen)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, type=Path, help='trusted physical-energy student.model, not optimizer state')
    p.add_argument('--sha256', required=True, help='expected checkpoint SHA-256 from a trusted source')
    p.add_argument('--device', required=True, help='explicit torch device: cpu, cuda or cuda:N')
    p.add_argument('--split', required=True, choices=_SAMPLER)
    p.add_argument('--frames', required=True, type=Path, help='labelled extxyz for exactly this split')
    p.add_argument('--out', required=True, type=Path, help='new directory for this split (must not exist)')
    args = p.parse_args(argv)
    if not re.fullmatch(r'cpu|cuda(?::\d+)?', args.device):
        p.error('--device must be cpu, cuda or cuda:N')
    head, extract = _load_student(args.model, args.sha256, args.device)
    count = write_split(iread(args.frames, index=':', format='extxyz'), args.out, args.split, args.sha256, head, extract)
    print(f'{args.out}: {count} {args.split} frames (offline features, not engine timing)')


if __name__ == '__main__':
    main()
