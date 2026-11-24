"""Recreate the six disjoint rocksalt seed structures in a NEW output directory."""
import hashlib
from pathlib import Path
import sys

from ase.build import bulk
from ase.io import write
import numpy as np


def main(directory):
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / 'seeds.sha256').open('x') as manifest:
        for i in range(6):
            a = bulk('NaCl', 'rocksalt', a=6, cubic=True).repeat((3, 3, 3))
            a.numbers[a.numbers == 11] = np.random.default_rng(90101+i).permutation(np.repeat([3, 11, 19], 36))
            rho = 1.55
            a.set_cell(np.eye(3)*(a.get_masses().sum()*1.66053906660/rho)**(1/3), scale_atoms=True)
            a.info.update(cation_seed=90101+i, initial_density_g_cm3=rho)
            p = directory / f'seed{i}.extxyz'
            write(p, a)
            manifest.write(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
