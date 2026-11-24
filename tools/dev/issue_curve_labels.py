"""Issue the SrTiO3 corrected learning curve on NFS (CPU only).

stage ROOT NFS_ORIGINALS freezes 388 receipted shards in groups of four. The
old payloads supply geometries, not authorized labels. Run qualify_frozen_cells
on each group under the workstation guard, then issue ROOT FIRST and assemble
ROOT COUNT. Each point is an atomic, independently auditable nested release.
"""
import argparse
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from loupmd.distill import curve, reissue
from loupmd.distill.train import file_digest


RELEASE = 'curve-labels-v2'
GROUP = 4
SHARDS = 388
SHARD_FRAMES = 25
LIBS = {'leeloo': 'c147f32501f36ba57bb10a016b6611ecb3701f0fff5dd89c68ab76e61095d494',
        'coraline': 'a536f453dc147235d40eeacbe0aca49f9115b637fd0dbbcd255d4a8c1e9511ce'}
BUNDLE = 'c115fe6f517f7161e2d24e1a7594de387f1a29d2e7951e72f24d72e24a86a106'
BASE_MANIFEST = '42bf027b2657b7d5fd434f18611da3541ba2a120252b037356c76c9af128cae7'
DIAGNOSTIC_SHA = '088d3cc53c16e3ad7ef717c82cccf3df575e3aff077e887b6738cdab6853e105'
TEMPERATURES = (300, 450, 300, 600, 300, 525, 375)


def checked_bytes(path, pin):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != pin:
        raise ValueError(f'{path}: receipt digest mismatch')
    return raw


def pinned_shard(path, pin):
    if curve._sidecar(path) != pin:
        raise ValueError(f'{path}: sidecar disagrees with frozen receipt')
    return checked_bytes(path, pin)


def new_json(path, data):
    with path.open('x', encoding='utf8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def stage(root, original):
    ext = original / 'curve-extended'
    suffix = original / 'cell-fix-048489b9/missing-299-387'
    catalog = dict(schema='loupmd.curve-inputs.v2', groups={}, shards={}, receipts={})
    def receipt(path, pin):
        catalog['receipts'][str(path.relative_to(original))] = pin
        return json.loads(checked_bytes(path, pin))
    early = receipt(ext/'results/curve-3000/manifest.json',
                    'fa32ab053b61bab21f19998786cc9993effe70d448e82eaaeb2d29ce6c8587a2')['input_sha256']
    old = receipt(ext/'queue/queue.json', '025176caeee5e79d10d288499d60d84b8efe20d9fb84d7d3c05c124584bfeb1b')
    new = receipt(suffix/'queue.json', 'fdf0425ffa0004c7635e769b760228d5babc44dcb99c05846846c39bde375cf8')
    if old['claims'] or new['claims'] or set(map(int, old['receipts'])) != set(range(34, 299)) or set(map(int, new['receipts'])) != set(range(299, SHARDS)):
        raise ValueError('incomplete or unsealed source queue')
    for start in range(0, SHARDS, GROUP):
        chunks = []
        for i in range(start, start+GROUP):
            name = f'part-{i:02d}.xyz'
            if i < 34:
                folder = original/'curve1/shards' if i < 28 else ext/'imported'
                pin = early[name]
            else:
                queue = old if i < 299 else new
                folder = ext/'queue/shards' if i < 299 else suffix/'shards'
                pin = queue['receipts'][str(i)]['sha256']
            if name in early and early[name] != pin:
                raise ValueError(f'{name}: frozen release and receipt disagree')
            path = folder/name
            chunks.append(pinned_shard(path, pin))
            catalog['shards'][str(i)] = dict(source=str(path.relative_to(original)), sha256=pin,
                                             sampler='pre-fix frozen' if i < 299 else 'fixed-cell suffix')
            if i >= 34:
                catalog['shards'][str(i)]['owner'] = queue['receipts'][str(i)]['owner']
        payload = b''.join(chunks)
        dest = root/f'work/group-{start}.xyz'
        digest = sha256(payload).hexdigest()
        if dest.exists():
            checked_bytes(dest, digest)
        else:
            with dest.open('xb') as stream:
                stream.write(payload)
        catalog['groups'][str(start)] = digest
    new_json(root/'catalog.json', catalog)


def no_arithmetic_override(root, worker, env):
    # loupmd fills these location variables automatically on first load. Reject
    # arbitrary kernel search paths or any numerical override; retain the report.
    cache = root.parent/'curve1'/f'cache-{worker}'
    paths = {'LOUP_CACHE_DIR': str(cache), 'LOUP_KERNEL_PATH': str(cache/'lib')}
    return isinstance(env, dict) and all(k in paths and v == paths[k] for k, v in env.items())


def register_exact(root, worker, library, digest, diagnostic):
    """Operator pins a new build after its same-geometry, positive, independent reproduction."""
    if worker not in LIBS or digest in LIBS.values() or file_digest(library) != digest:
        raise ValueError('unknown worker or strict build pin mismatch')
    raw = diagnostic.read_bytes()
    gate = reissue._receipt(raw, 'exact calibration', digest, 2, DIAGNOSTIC_SHA)
    if (gate['bundle_sha256'] != BUNDLE or not no_arithmetic_override(root, worker, gate.get('native_environment'))
            or [r['replica'] for r in gate['rows']] != [364, 365]
            or not all(r[k].get('passed') is True and r[k].get('failed') == []
                       for r in gate['rows'] for k in ('fresh', 'sequential'))
            or not all(reissue.native_paths_agree(r.get('sequential_vs_fresh')) for r in gate['rows'])):
        raise ValueError('strict calibration failed')
    new_json(root/f'exact-{worker}.json', dict(library=str(library.resolve()), library_sha256=digest,
                                               diagnostic=str(diagnostic.resolve()), diagnostic_sha256=sha256(raw).hexdigest()))


def label_policy(root, start, gate):
    lib = gate['library_sha256']
    env = gate.get('native_environment', {})
    rejected = root/f'work/rejected-{start}.json'
    for worker, default in LIBS.items():
        if lib == default:
            if not no_arithmetic_override(root, worker, env) or rejected.exists():
                raise ValueError('only an unchanged, never-rejected default result is eligible')
            return dict(worker=worker, label_mode='default', conv_table='packed8', native_environment=env)
        registry = root/f'exact-{worker}.json'
        if not registry.exists():
            continue
        raw = registry.read_bytes()
        cert = json.loads(raw)
        if lib != cert['library_sha256']:
            continue
        checked_bytes(Path(cert['diagnostic']), cert['diagnostic_sha256'])
        if not no_arithmetic_override(root, worker, gate.get('native_environment')) or not rejected.exists():
            raise ValueError('strict build requires no runtime override and a complete negative default report')
        rejection = rejected.read_bytes()
        old = json.loads(rejection)
        for key in ('source_sha256', 'bundle_sha256', 'teacher_sha256', 'teacher_head', 'frames', 'schema'):
            if old[key] != gate[key]:
                raise ValueError(f'default rejection differs from strict gate: {key}')
        old_worker = next((w for w, h in LIBS.items() if h == old.get('library_sha256')), None)
        if (old.get('passed') is not False or not old_worker
                or not no_arithmetic_override(root, old_worker, old.get('native_environment', {})) or len(old['rows']) != gate['frames']
                or [r['frame_id'] for r in old['rows']] != [r['frame_id'] for r in gate['rows']]):
            raise ValueError('not a rejected default group on these exact frames')
        return dict(worker=worker, label_mode='exact', conv_table='fp32', native_environment=env,
                    default_rejection_sha256=sha256(rejection).hexdigest(), exact_registry_sha256=sha256(raw).hexdigest())
    raise ValueError('unqualified label library')


def issue(root, start):
    out = root/f'issued/group-{start}'
    if out.exists():
        raise FileExistsError(out)
    catalog = json.loads((root/'catalog.json').read_bytes())
    pin = catalog['groups'][str(start)]
    raw = checked_bytes(root/f'work/group-{start}.xyz', pin)
    receipt = (root/f'gates/group-{start}.json').read_bytes()
    lib = json.loads(receipt)['library_sha256']
    gate = reissue._receipt(receipt, f'group-{start}', lib, SHARD_FRAMES*GROUP, pin)
    if gate['bundle_sha256'] != BUNDLE:
        raise ValueError('unexpected label bundle')
    stream = StringIO(raw.decode('utf8'))
    record = dict(source_sha256=pin, gate_sha256=sha256(receipt).hexdigest(), library_sha256=lib,
                  **label_policy(root, start, gate), shards={},
                  historical_passed=sum(row['historical']['passed'] for row in gate['rows']))
    inv = curve._Inventory()
    with TemporaryDirectory(dir=out.parent, prefix=f'.group-{start}-') as scratch:
        temp = Path(scratch)/'publish'; temp.mkdir()
        for j in range(GROUP):
            i = start+j
            rows = gate['rows'][j*SHARD_FRAMES:(j+1)*SHARD_FRAMES]
            dest = temp/f'part-{i:02d}.xyz'
            with dest.open('xb') as w:
                for k, row in enumerate(rows):
                    at, payload = reissue._issue_frame(stream, row, lib, RELEASE)
                    inv.accept(at, 'train', i*SHARD_FRAMES+k)
                    expected = dict(seed=7901, burn_in=1000, stride=1, step=1001, regime='target',
                                    temperature_K=TEMPERATURES[i % len(TEMPERATURES)], integrator='langevin',
                                    sampler_sha256=reissue.TEACHER_SHA256, sampler_head='omat_pbe',
                                    teacher_sha256=reissue.TEACHER_SHA256, teacher_head='omat_pbe')
                    if any(at.info.get(key) != value for key, value in expected.items()):
                        raise ValueError(f'{row["frame_id"]}: inconsistent sampling or teacher metadata')
                    w.write(payload)
            digest = file_digest(dest)
            dest.with_suffix('.sha256').write_text(f'{digest}  {dest.name}\n', encoding='utf8')
            record['shards'][dest.name] = dict(source=catalog['shards'][str(i)], output_sha256=digest)
        if stream.read():
            raise ValueError('extra source bytes')
        new_json(temp/'receipt.json', record)
        temp.rename(out)


def assemble(root, count):
    if count not in curve.LEVELS:
        raise ValueError(f'invalid curve point {count}')
    base = root/'base-labels-v2'
    base_record = json.loads(checked_bytes(base/'manifest.json', BASE_MANIFEST))
    for split, data in base_record['splits'].items():
        checked_bytes(base/f'gates/{split}.json', data['gate_sha256'])
    needed = (count-curve.BASE_FRAMES)//SHARD_FRAMES
    dirs = [root/f'issued/group-{i}' for i in range(0, needed, GROUP)]
    catalog = json.loads((root/'catalog.json').read_bytes())
    evidence = dict(schema='loupmd.curve-release.v2', label_release=RELEASE, base_manifest_sha256=BASE_MANIFEST,
                    input_catalog_sha256=file_digest(root/'catalog.json'), groups={})
    for i, folder in zip(range(0, needed, GROUP), dirs, strict=True):
        receipt = folder/'receipt.json'
        rec = json.loads(receipt.read_bytes())
        if rec['source_sha256'] != catalog['groups'][str(i)] or rec['gate_sha256'] != file_digest(root/f'gates/group-{i}.json'):
            raise ValueError(f'group {i}: receipt mismatch')
        for j in range(i, i+GROUP):
            part = f'part-{j:02d}.xyz'
            if rec['shards'][part]['source'] != catalog['shards'][str(j)] or curve._sidecar(folder/part) != rec['shards'][part]['output_sha256']:
                raise ValueError(f'{part}: release receipt mismatch')
        evidence['groups'][str(i)] = dict(receipt_sha256=file_digest(receipt), gate_sha256=rec['gate_sha256'],
                                          worker=rec['worker'], library_sha256=rec['library_sha256'],
                                          label_mode=rec.get('label_mode', 'default'),
                                          conv_table=rec.get('conv_table', 'packed8'),
                                          native_environment=rec.get('native_environment', {}))
        if rec.get('label_mode') == 'exact':
            evidence['groups'][str(i)].update(default_rejection_sha256=rec['default_rejection_sha256'],
                                               exact_registry_sha256=rec['exact_registry_sha256'])
    dest = root/f'results/curve-{count}'
    if dest.exists():
        raise FileExistsError(dest)
    with TemporaryDirectory(dir=dest.parent, prefix=f'.curve-{count}-') as scratch:
        staged = Path(scratch)/'release'
        pins = {s.replace('-', '_')+'_sha256': v['output_sha256'] for s,v in base_record['splits'].items()}
        pins['base_sha256'] = pins.pop('train_sha256')
        manifest = curve._assemble(argparse.Namespace(output=staged, shards=dirs, count=count, base=base, **pins))
        evidence['train_sha256'] = manifest['train_sha256']
        evidence['split_counts'] = manifest['split_counts']
        new_json(staged/'release.json', evidence)
        staged.rename(dest)
        print(count, manifest['train_sha256'], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(required=True, dest='command')
    for name, arg, typ in [('stage', 'original', Path), ('issue', 'first', int), ('assemble', 'count', int)]:
        cmd = sub.add_parser(name); cmd.add_argument('root', type=Path); cmd.add_argument(arg, type=typ)
    exact = sub.add_parser('register-exact')
    exact.add_argument('root', type=Path)
    exact.add_argument('worker', choices=tuple(LIBS))
    exact.add_argument('library', type=Path)
    exact.add_argument('digest')
    exact.add_argument('diagnostic', type=Path)
    args = vars(p.parse_args()); cmd = args.pop('command')
    {'stage': stage, 'issue': issue, 'assemble': assemble, 'register-exact': register_exact}[cmd](*args.values())


if __name__ == '__main__':
    main()
