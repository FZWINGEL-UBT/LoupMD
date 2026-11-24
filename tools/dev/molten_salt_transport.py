"""Report the 1200 K teacher diffusion reference across independent liquid replicas.

Run with the directory containing ``transport{0,1,2,3}.extxyz``. For each species
report all seed estimates, across-seed SEM and a sensitivity check using the
nonoverlapping halves of each trajectory. D is in A^2/ps; multiply by 1e-4
for cm^2/s. This is Langevin self diffusion at the recorded friction and finite
cell size, not conductivity or a thermodynamic-limit experimental prediction.
"""
from pathlib import Path
import argparse
import json

from ase.io import read
import numpy as np

from loupmd.distill.observable_io import file_hash
from loupmd.distill.salt import transport


def report(directory):
    seeds, files, groups, half = [], {}, {}, {}
    for i in range(4):
        p = directory / f'transport{i}.extxyz'
        a = read(p, index=':')
        first = a[0].info
        if first['temperature_K'] != 1200 or first['friction_per_fs'] != .01 or first['md_seed'] in seeds:
            raise ValueError('incompatible sampling protocol or duplicate seed')
        seeds.append(int(first['md_seed']))
        files[p.name] = file_hash(p)
        full = transport(a)
        blocks = [transport(b) for b in (a[:len(a)//2], a[len(a)//2:])]
        for species, row in full['species'].items():
            groups.setdefault(species, []).append(row['D_A2_ps'])
            half.setdefault(species, []).append([b['species'][species]['D_A2_ps'] for b in blocks])
    return dict(temperature_K=1200, friction_per_fs=.01, md_seeds=seeds, files_sha256=files,
                diffusion_A2_ps={k: dict(per_seed=v, mean=np.mean(v), across_seed_SEM=np.std(v, ddof=1)/np.sqrt(len(v)),
                                         minimum=min(v), maximum=max(v), half_trajectory_estimates=half[k])
                                for k, v in groups.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    print(json.dumps(report(parser.parse_args().directory), indent=2, allow_nan=False))
