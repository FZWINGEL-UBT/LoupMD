# SPDX-License-Identifier: LGPL-2.1-or-later
# Provenance: port of ASE 3.27's ase/optimize/lbfgs.py (LBFGS, use_line_search=False), LGPL-2.1-or-later; see
# NOTICE.md.
"""ASE 3.27's ``LBFGS`` on host arrays.

ASE's step at iteration ``k`` with positions ``x``, gradient ``g`` (minus the forces, zero on fixed atoms) and
the previous ``r0``, ``g0``:

- the history gains ``s = x - r0``, ``y = g - g0``, ``rho = 1 / (y . s)`` (from ``k = 1``) and loses its oldest
  pair once it holds more than ``memory``;
- the two-loop recursion gives ``z ~ H g`` with ``H0 = 1 / alpha``; the direction is ``p = -z``;
- if the largest per-atom length of ``p`` is ``>= maxstep`` the whole step is scaled to it; then times ``damping``.

The history is kept as two arrays of rows, newest last, so each loop of the recursion is a few BLAS calls.
"""
from __future__ import annotations

import numpy as np

MAX_MAXSTEP_A = 1.0       # ASE's limit on maxstep, in A


class LBFGSState:
    """The optimiser's state for ``n`` atoms. ``step(positions, forces)`` returns the next positions."""

    def __init__(self, n_atoms, maxstep_A=0.2, memory=100, damping=1.0, alpha=70.0):
        self.maxstep, self.memory, self.damping, self.H0 = float(maxstep_A), int(memory), float(damping), 1.0 / alpha
        self.s = np.zeros((self.memory, 3 * n_atoms))
        self.y = np.zeros((self.memory, 3 * n_atoms))
        self.rho = np.zeros(self.memory)
        self.count = 0                      # pairs in the history
        self.iteration = 0
        self.r0 = self.g0 = None

    def _append(self, s, y):
        if self.count == self.memory:       # drop the oldest pair
            self.s[:-1] = self.s[1:]
            self.y[:-1] = self.y[1:]
            self.rho[:-1] = self.rho[1:]
            self.count -= 1
        self.s[self.count], self.y[self.count], self.rho[self.count] = s, y, 1.0 / np.dot(y, s)
        self.count += 1

    def step(self, positions, forces):
        x = np.asarray(positions, dtype=float).reshape(-1)
        g = -np.asarray(forces, dtype=float).reshape(-1)
        if self.iteration > 0:
            self._append(x - self.r0, g - self.g0)
        n = self.count
        s, y, rho = self.s, self.y, self.rho
        a = np.empty(n)
        q = g.copy()
        for i in range(n - 1, -1, -1):
            a[i] = rho[i] * np.dot(s[i], q)
            q -= a[i] * y[i]
        z = self.H0 * q
        for i in range(n):
            b = rho[i] * np.dot(y[i], z)
            z += s[i] * (a[i] - b)
        p = -z.reshape(-1, 3)
        longest = np.sqrt((p ** 2).sum(axis=1)).max()
        if longest >= self.maxstep:
            p *= self.maxstep / longest
        self.iteration += 1
        self.r0, self.g0 = x.copy(), g
        return (x + (p * self.damping).reshape(-1)).reshape(-1, 3)
