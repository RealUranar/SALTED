"""Folded (VERSION 4) prediction against the unfolded one: pack_model.fold_lambda and prediction.compute_density_folded_structure."""

import numpy as np
import pytest

from salted.pack_model import fold_lambda
from salted.prediction import compute_density_descriptor_structure, compute_density_folded_structure, compute_prediction

SPE, NA, M, NCUT = "X", 3, 4, 5
LMAX = {SPE: 2}
NMAX = {(SPE, 0): 3, (SPE, 1): 4, (SPE, 2): 1}
GENV = {0: False, 1: False, 2: True}  # FEATL above lam 0 and GENV both get exercised
ATOM_IDX, NATOM = {(0, SPE): [0, 1, 2]}, {(0, SPE): NA}


@pytest.mark.parametrize("zeta", [1, 2, 3])
def test_folded_matches_unfolded(zeta):
    rng = np.random.default_rng(zeta)
    pvec, feats, projs, weights, model = {}, {}, {}, [], {}
    for lam in range(LMAX[SPE] + 1):
        d = 2 * lam + 1
        pvec[lam] = rng.normal(size=(1, NA, NCUT) if lam == 0 else (1, NA, d, NCUT))
        V = rng.normal(size=(M * d, M * d - 1))
        F = rng.normal(size=(M * d, NCUT))
        W = rng.normal(size=(NMAX[(SPE, lam)], V.shape[1]))
        projs[(lam, SPE)] = V
        feats[(lam, SPE)] = V.T @ F if zeta == 1 else F  # zeta = 1 as get_feats_projs gives it
        weights.append(W.ravel())
        for name, m in fold_lambda(V, F, W, lam, zeta == 1, GENV[lam]).items():
            model.setdefault(name.lower(), {SPE: {}})[SPE][str(lam)] = m

    psi = compute_density_descriptor_structure(0, 0, [0], ATOM_IDX, NATOM, LMAX, [SPE], zeta, pvec, feats, projs, {SPE: M})
    psi_folded = compute_density_folded_structure(0, 0, ATOM_IDX, NATOM, LMAX, [SPE], zeta, pvec, model)
    args = (0, [[SPE] * NA], np.array([NA]), LMAX)
    ref = compute_prediction(*args, NMAX, [SPE], psi, np.concatenate(weights))
    np.testing.assert_allclose(compute_prediction(*args, NMAX, [SPE], psi_folded, None), ref,
                               rtol=1e-12, atol=1e-12 * abs(ref).max())

    # a basis that is not the one the model was trained with
    with pytest.raises(ValueError):
        compute_prediction(*args, {**NMAX, (SPE, 0): 2}, [SPE], psi_folded, None)
