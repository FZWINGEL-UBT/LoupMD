"""Symmetrised terms of MACE's correlation-3 coupling tensors (shared by the generator and the check)."""
from collections import defaultdict


def symmetrised_terms(U3, U2, U1):
    """[(k, sorted monomial, coefficient)], k over cubic, then quadratic, then linear; and K."""
    K3, K2, K1 = U3.shape[-1], U2.shape[-1], U1.shape[-1]
    acc = defaultdict(float)
    for w, y, i, k in (U3 != 0).nonzero().tolist():
        acc[(k, tuple(sorted((w, y, i))))] += float(U3[w, y, i, k])
    for w, y, k in (U2 != 0).nonzero().tolist():
        acc[(K3 + k, tuple(sorted((w, y))))] += float(U2[w, y, k])
    for w, k in (U1 != 0).nonzero().tolist():
        acc[(K3 + K2 + k, (w,))] += float(U1[w, k])
    terms = sorted((k, m, v) for (k, m), v in acc.items() if abs(v) > 1e-13)
    return terms, K3 + K2 + K1


def model_terms(model):
    out = []
    for p in model.products:
        c = p.symmetric_contractions.contractions[0]
        out.append(symmetrised_terms(c.U_matrix_3.double().cpu(), c.U_matrix_2.double().cpu(),
                                     c.U_matrix_1.double().cpu()))
    return out
