"""pack_model writes a NORM block and VERSION 7 only for a normalisation other
than unit; read_model reads it back, "unit" when the block is absent."""

import struct

import h5py
import numpy as np
import yaml

from salted import basis, pack_model, read_model
from test_parse_config import MINIMAL_INP, write_inp

NCUT, MENV, NANG = 5, 3, 2


def fake_model(path, normalisation=None):
    """inp.yaml and the files pack_model reads, with random data (the same on every call);
    normalisation None leaves the key out of inp.yaml"""
    data = yaml.safe_load(yaml.safe_dump(MINIMAL_INP))
    for rep in ("rep1", "rep2"):
        data["descriptor"][rep]["nang"] = NANG
    data["descriptor"]["sparsify"] = {"ncut": NCUT, "nsamples": 10}
    if normalisation is not None:
        data["descriptor"]["normalisation"] = normalisation
    data["gpr"]["Menv"] = MENV
    data["qm"]["dfbasis"], data["qm"]["dfbasis_file"] = "mybasis", str(path / "basis.yaml")
    (path / "basis.yaml").write_text("mybasis:\n  H: {lmax: 1, nmax: [3, 2]}\n  O: {lmax: 2, nmax: [4, 3, 2]}\n")
    write_inp(path, data)

    species = data["system"]["species"]
    lmax, nmax = basis.basiset("mybasis", str(path / "basis.yaml"))
    lmax_max = max(lmax.values())
    rng = np.random.default_rng(0)
    for d in ("coefficients/averages", "wigners", "equirepr_test", "regrdir_test/M3_zeta2.0"):
        (path / d).mkdir(parents=True, exist_ok=True)
    for spe in species:
        np.save(path / f"coefficients/averages/averages_{spe}.npy", rng.normal(size=nmax[(spe, 0)]))
    for lam in range(lmax_max + 1):
        np.savetxt(path / f"wigners/wigner_lam-{lam}_lmax1-{NANG}_lmax2-{NANG}.dat", rng.normal(size=7))
        np.save(path / f"equirepr_test/fps{NCUT}-{lam}.npy", rng.permutation(20)[:NCUT])
    weights = []
    with h5py.File(path / f"equirepr_test/projector_M{MENV}_zeta2.0.h5", "w") as hp, \
         h5py.File(path / f"equirepr_test/FEAT_M-{MENV}.h5", "w") as hf:
        for spe in species:
            for lam in range(lmax[spe] + 1):
                rows = MENV * (2 * lam + 1)
                hp[f"projectors/{spe}/{lam}"] = rng.normal(size=(rows, rows - 1))
                hf[f"sparse_descriptors/{spe}/{lam}"] = rng.normal(size=(rows, NCUT))
                weights.append(rng.normal(size=nmax[(spe, lam)] * (rows - 1)))
    np.save(path / "regrdir_test/M3_zeta2.0/weights_N40_reg-6.npy", np.concatenate(weights))


def test_norm_block(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pack_model, "HAS_PYSCF", False)
    files = {}
    for norm in ("unit", "rel_power"):
        fake_model(tmp_path, None if norm == "unit" else norm)
        pack_model.build()
        files[norm] = tmp_path / f"{norm}.salted"
        (tmp_path / "test.salted").rename(files[norm])

    headers, models = {}, {}
    for norm, fname in files.items():
        with open(fname, "rb") as f:
            headers[norm] = read_model.read_header(f)
        models[norm] = read_model.read_salted_model(str(fname))
    (vu, bu), (vr, br) = headers["unit"], headers["rel_power"]
    assert vu == 4 and "NORM" not in bu and models["unit"]["normalisation"] == "unit"
    assert vr == 7 and set(br) == set(bu) | {"NORM"} and models["rel_power"]["normalisation"] == "rel_power"
    raw = files["rel_power"].read_bytes()
    assert raw[br["NORM"]:br["NORM"] + 17] == struct.pack("<ii", 4, 9) + b"rel_power"
    # every other block holds the same data
    for key, value in models["unit"].items():
        if key not in ("version", "normalisation"):
            np.testing.assert_equal(models["rel_power"][key], value)
