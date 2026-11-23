"""Unit tests for salted.novelty_selection (kernel-novelty sparse selection)."""

import h5py
import numpy as np
import pytest
import yaml

from salted.novelty_selection import IncrementalCholesky, residual_novelty, select_sparse
from salted.sys_utils import do_fps

ZETA = 2.0


def _data(n=300, f=20, seed=0):
    return np.random.default_rng(seed).normal(size=(n, f))


def _brute_novelty(X, R, zeta=ZETA):
    """Novelty of every row w.r.t. rows R from scratch: 1 - k_aR K_RR^-1 k_Ra / k_aa."""
    kxx = np.einsum("ij,ij->i", X, X) ** zeta
    if len(R) == 0:
        return np.ones(len(X))
    kR = (X @ X[R].T) ** zeta
    return 1 - np.einsum("ij,ij->i", kR, np.linalg.solve(kR[R], kR.T).T) / kxx


def test_cholesky_novelty_equals_rkhs_projector_route():
    X = _data()
    R = np.random.default_rng(1).choice(len(X), 30, replace=False)
    kRR = (X[R] @ X[R].T) ** ZETA
    eva, eve = np.linalg.eigh(kRR)
    V = eve / np.sqrt(eva)
    kxx = np.einsum("ij,ij->i", X, X) ** ZETA
    ref = 1 - np.sum(((X @ X[R].T) ** ZETA @ V) ** 2, axis=1) / kxx
    np.testing.assert_allclose(residual_novelty(X, R, ZETA), ref, rtol=0, atol=1e-10)
    sub = np.array([5, 7, 250, 3])
    np.testing.assert_allclose(residual_novelty(X, R, ZETA, eval_idx=sub), ref[sub], rtol=0, atol=1e-10)


def test_maxnov_equals_brute_force_greedy():
    X = _data(60, 8)
    got = select_sparse(X, np.zeros(60, int), 12, hybrid_fraction=0.0)
    R = []
    for _ in range(12):
        nov = _brute_novelty(X, R)
        nov[R] = -np.inf
        R.append(int(np.argmax(nov)))
    assert got.tolist() == R


def test_totalnov_exact_equals_brute_force_greedy():
    X = _data(50, 6)
    got = select_sparse(X, np.zeros(50, int), 10, hybrid_fraction=1.0, ncand=0, neval=0)
    R = []
    for _ in range(10):
        before = _brute_novelty(X, R).sum()
        gain = np.full(len(X), -np.inf)
        for i in set(range(len(X))) - set(R):
            gain[i] = before - _brute_novelty(X, R + [i]).sum()
        R.append(int(np.argmax(gain)))
    assert got.tolist() == R


def _species_data():
    X = _data(240, 10, seed=3)
    species = np.repeat([0, 1, 2], [40, 150, 50])  # sorted, so ties across species break alike
    X[40:190] *= 0.7  # unequal norms: novelty must be normalised per atom
    return X, species


@pytest.mark.parametrize("f", [0.0, 0.6, 1.0])
def test_shared_budget_counts_and_species(f, capsys):
    X, species = _species_data()
    idx = select_sparse(X, species, 37, hybrid_fraction=f, seed=7)
    assert len(idx) == 37 and len(np.unique(idx)) == 37
    again = select_sparse(X, species, 37, hybrid_fraction=f, seed=7)
    assert idx.tolist() == again.tolist()
    assert "seed=7" in capsys.readouterr().out


def test_shared_maxnov_equals_global_brute_force():
    """The lazy per-species cache must give the global argmax of within-species novelty."""
    X, species = _species_data()
    got = select_sparse(X, species, 25, hybrid_fraction=0.0)
    R = []
    for _ in range(25):
        nov = np.empty(len(X))
        for s in range(3):
            m = np.flatnonzero(species == s)
            nov[m] = _brute_novelty(X[m], [np.flatnonzero(m == r)[0] for r in R if species[r] == s])
        nov[R] = -np.inf
        R.append(int(np.argmax(nov)))
    assert got.tolist() == R


def test_shared_totalnov_exact_equals_global_brute_force():
    X, species = _species_data()
    X, species = X[::4], species[::4]
    got = select_sparse(X, species, 12, hybrid_fraction=1.0, ncand=0, neval=0)
    members = [np.flatnonzero(species == s) for s in range(3)]

    def total(R):
        return sum(_brute_novelty(X[m], [np.flatnonzero(m == r)[0] for r in R if species[r] == s]).sum()
                   for s, m in enumerate(members))

    R = []
    for _ in range(12):
        before = total(R)
        gain = np.full(len(X), -np.inf)
        for i in set(range(len(X))) - set(R):
            gain[i] = before - total(R + [i])
        R.append(int(np.argmax(gain)))
    assert got.tolist() == R


def test_totalnov_subsampled_gain_is_unbiased():
    """Mean over evaluation draws of the estimated gain equals the exact gain."""
    X = _data(80, 6)
    ch = IncrementalCholesky(X, ZETA)
    for p in (0, 11, 23):
        ch.add(p)
    cand = np.array([5, 40, 60])
    exact = ch.estimated_gain(cand, None, 0)
    rng = np.random.default_rng(0)
    draws = [ch.estimated_gain(cand, rng, 20) for _ in range(4000)]
    np.testing.assert_allclose(np.mean(draws, axis=0), exact, rtol=0.03)


def test_invalid_rows_excluded_and_duplicates_fall_back(capsys):
    X = _data(40, 6)
    X[3] = 0.0
    X[4, 2] = np.nan
    X[10:20] = X[10]  # duplicates: novelty runs out
    idx = select_sparse(X[10:20], np.zeros(10, int), 4, hybrid_fraction=0.0)
    assert len(np.unique(idx)) == 4
    assert "WARNING" in capsys.readouterr().out
    idx = select_sparse(X, np.zeros(40, int), 20, hybrid_fraction=0.0)
    assert 3 not in idx and 4 not in idx
    assert "2 of 40 atoms excluded" in capsys.readouterr().out


def test_bad_arguments_raise():
    X, species = _species_data()
    for kw in ({"ncand": -1}, {"neval": 1.5}, {"seed": -1}, {"hybrid_fraction": 2}):
        with pytest.raises(ValueError):
            select_sparse(X, species, 10, **kw)


def test_float32_rows_give_float64_novelty():
    X = _data(200, 10)
    R = np.arange(0, 200, 7)
    np.testing.assert_allclose(residual_novelty(X.astype(np.float32), R, ZETA),
                               residual_novelty(X.astype(np.float32).astype(np.float64), R, ZETA), rtol=0, atol=1e-13)


def test_incremental_cholesky_residual_of_pivot_is_zero():
    X = _data(30, 5)
    ch = IncrementalCholesky(X, ZETA, cap=1)
    for p in (4, 9, 17):
        assert ch.add(p)
    assert ch.d[[4, 9, 17]].tolist() == [0.0, 0.0, 0.0] and ch.L.shape[1] >= 3


WATER = "3\nProperties=species:S:1:pos:R:3\nO 0.0 0.0 {z}\nH 0.757 0.586 {z}\nH -0.757 0.586 {z}\n"


def _run_dir(tmp_path, gpr_extra):
    ntrain, nfeat = 6, 12
    (tmp_path / "geoms.xyz").write_text("".join(WATER.format(z=0.1 * i) for i in range(8)))
    (tmp_path / "basis.yaml").write_text(yaml.safe_dump({"b": {s: {"lmax": 1, "nmax": [2, 1]} for s in "HO"}}))
    inp = {
        "salted": {"saltedname": "t", "saltedpath": "./"},
        "system": {"filename": "./geoms.xyz", "species": ["H", "O"]},
        "qm": {"path2qm": "./", "qmcode": "aims", "dfbasis": "b", "dfbasis_file": "./basis.yaml"},
        "descriptor": {
            f"rep{i}": {"type": "rho", "rcut": 4.0, "sig": 0.3, "nrad": 8, "nang": 6, "neighspe": ["H", "O"]}
            for i in (1, 2)
        },
        "gpr": {"z": 2.0, "Menv": 9, "Ntrain": ntrain, **gpr_extra},
    }
    (tmp_path / "inp.yaml").write_text(yaml.safe_dump(inp))
    conf = [5, 0, 3, 7, 1, 2]
    (tmp_path / "regrdir_t").mkdir()
    np.savetxt(tmp_path / "regrdir_t" / f"training_set_N{ntrain}.txt", conf, fmt="%i")
    (tmp_path / "equirepr_t").mkdir()
    feat = np.random.default_rng(5).normal(size=(ntrain, 3, nfeat))
    feat /= np.linalg.norm(feat, axis=-1, keepdims=True)
    with h5py.File(tmp_path / "equirepr_t" / "FEAT-0.h5", "w") as h5f:
        h5f["descriptor"] = feat
        h5f["configuration_indices"] = np.array(conf)
    return feat.reshape(-1, nfeat), np.tile([1, 0, 0], ntrain)  # species index in ["H", "O"]


def test_build_fps_file_is_byte_identical(tmp_path, monkeypatch):
    """sparse_selection: fps must write what the original code wrote: do_fps on the dense rows, %i savetxt."""
    from salted import sparse_selection

    dense, spe = _run_dir(tmp_path, {"sparse_selection": "fps"})
    monkeypatch.chdir(tmp_path)
    sparse_selection.build()
    fps_idx = np.array(do_fps(dense, 9), int)
    np.savetxt(tmp_path / "expected.txt", np.vstack((fps_idx, spe[fps_idx])).T, fmt="%i")
    assert (tmp_path / "equirepr_t" / "sparse_set_9.txt").read_bytes() == (tmp_path / "expected.txt").read_bytes()


def test_build_default_is_hybrid(tmp_path, monkeypatch):
    """Without sparse keys the file holds select_sparse's hybrid picks (f 0.6, seed 42) and their species."""
    from salted import sparse_selection

    dense, spe = _run_dir(tmp_path, {})
    monkeypatch.chdir(tmp_path)
    sparse_selection.build()
    out = np.loadtxt(tmp_path / "equirepr_t" / "sparse_set_9.txt", dtype=int)
    assert out.shape == (9, 2) and len(np.unique(out[:, 0])) == 9
    assert (spe[out[:, 0]] == out[:, 1]).all()
    assert out[:, 0].tolist() == select_sparse(dense, spe, 9, nspecies=2).tolist()


def test_hybrid_shared_is_totalnov_prefix_then_maxnov_tail():
    X, species = _species_data()
    M, f = 30, 0.6
    n1 = 18
    got = select_sparse(X, species, M, hybrid_fraction=f, ncand=0, neval=0)
    head = select_sparse(X, species, n1, hybrid_fraction=1.0, ncand=0, neval=0)
    assert got[:n1].tolist() == head.tolist()
    R = [[] for _ in range(3)]
    for p in head:
        R[species[p]].append(p)
    for p in got[n1:]:
        best, arg = -np.inf, None
        for s in range(3):
            m = np.flatnonzero(species == s)
            nov = _brute_novelty(X[m], [int(np.searchsorted(m, r)) for r in R[s]])
            nov[[int(np.searchsorted(m, r)) for r in R[s]]] = -np.inf
            i = int(np.argmax(nov))
            if nov[i] > best:
                best, arg = nov[i], m[i]
        assert p == arg
        R[species[p]].append(p)


def test_hybrid_fraction_must_be_in_unit_interval():
    X, species = _species_data()
    for f in (-0.1, 1.5, True):
        with pytest.raises(ValueError):
            select_sparse(X, species, 10, hybrid_fraction=f)
