"""Unit tests for salted.sys_utils.ParseConfig."""

import pytest
import yaml

from salted.sys_utils import ParseConfig

MINIMAL_INP = {
    "salted": {"saltedname": "test", "saltedpath": "./"},
    "system": {"filename": "./geoms.xyz", "species": ["H", "O"]},
    "qm": {"path2qm": "./", "qmcode": "aims", "dfbasis": "FHI-aims-light"},
    "descriptor": {
        "rep1": {
            "type": "rho",
            "rcut": 4.0,
            "sig": 0.3,
            "nrad": 8,
            "nang": 6,
            "neighspe": ["H", "O"],
        },
        "rep2": {
            "type": "rho",
            "rcut": 4.0,
            "sig": 0.3,
            "nrad": 8,
            "nang": 6,
            "neighspe": ["H", "O"],
        },
    },
    "gpr": {"z": 2.0, "Menv": 100, "Ntrain": 40},
}


WATER_XYZ = """3
Properties=species:S:1:pos:R:3
O 0.0 0.0 0.0
H 0.757 0.586 0.0
H -0.757 0.586 0.0
"""


def write_inp(tmp_path, data):
    (tmp_path / "inp.yaml").write_text(yaml.safe_dump(data))
    # ParseConfig checks that system.filename exists
    (tmp_path / "geoms.xyz").write_text(WATER_XYZ)


def test_parse_minimal_input(tmp_path, monkeypatch):
    write_inp(tmp_path, MINIMAL_INP)
    monkeypatch.chdir(tmp_path)
    inp = ParseConfig().parse_input()
    assert inp.salted.saltedname == "test"
    assert inp.qm.qmcode == "aims"
    assert inp.gpr.Menv == 100


def test_defaults_are_applied(tmp_path, monkeypatch):
    write_inp(tmp_path, MINIMAL_INP)
    monkeypatch.chdir(tmp_path)
    inp = ParseConfig().parse_input()
    assert inp.gpr.regul == pytest.approx(1e-6)
    assert inp.gpr.trainfrac == pytest.approx(1.0)
    assert inp.gpr.trainsel in ("random", "sequential")
    assert inp.descriptor.sparsify.ncut == 0


@pytest.mark.parametrize("value, ok", [(None, True), ("rel_power", True), ("relpower", False)])
def test_descriptor_normalisation(tmp_path, monkeypatch, value, ok):
    """None: the key is left out and defaults to unit"""
    data = yaml.safe_load(yaml.safe_dump(MINIMAL_INP))
    if value is not None:
        data["descriptor"]["normalisation"] = value
    write_inp(tmp_path, data)
    monkeypatch.chdir(tmp_path)
    if ok:
        assert ParseConfig().parse_input().descriptor.normalisation == (value or "unit")
    else:
        with pytest.raises(Exception):
            ParseConfig().parse_input()


def test_missing_required_field_raises(tmp_path, monkeypatch):
    broken = yaml.safe_load(yaml.safe_dump(MINIMAL_INP))
    del broken["gpr"]["Menv"]
    write_inp(tmp_path, broken)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(Exception):
        ParseConfig().parse_input()


def test_invalid_value_raises(tmp_path, monkeypatch):
    broken = yaml.safe_load(yaml.safe_dump(MINIMAL_INP))
    broken["gpr"]["Ntrain"] = -5
    write_inp(tmp_path, broken)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(Exception):
        ParseConfig().parse_input()


def test_missing_inp_yaml_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(Exception):
        ParseConfig().parse_input()


def test_dfbasis_file_defaults_to_none(tmp_path, monkeypatch):
    write_inp(tmp_path, MINIMAL_INP)
    monkeypatch.chdir(tmp_path)
    inp = ParseConfig().parse_input()
    # optional field: unset -> None
    assert inp.qm.dfbasis_file is None


def test_dfbasis_file_set_resolves_to_path(tmp_path, monkeypatch):
    ext = tmp_path / "basis_data.yaml"
    ext.write_text("FHI-aims-light: {}\n")
    data = yaml.safe_load(yaml.safe_dump(MINIMAL_INP))
    data["qm"]["dfbasis_file"] = str(ext)
    write_inp(tmp_path, data)
    monkeypatch.chdir(tmp_path)
    inp = ParseConfig().parse_input()
    assert inp.qm.dfbasis_file == str(ext)


def test_dfbasis_file_nonexistent_path_raises(tmp_path, monkeypatch):
    data = yaml.safe_load(yaml.safe_dump(MINIMAL_INP))
    data["qm"]["dfbasis_file"] = str(tmp_path / "does_not_exist.yaml")
    write_inp(tmp_path, data)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(Exception):
        ParseConfig().parse_input()


@pytest.mark.parametrize("value, ok", [(None, True), ("float32", True), ("float16", False)])
def test_gpr_matrix_dtype(tmp_path, monkeypatch, value, ok):
    """None: the key is left out and defaults to float64"""
    data = yaml.safe_load(yaml.safe_dump(MINIMAL_INP))
    if value is not None:
        data["gpr"]["matrix_dtype"] = value
    write_inp(tmp_path, data)
    monkeypatch.chdir(tmp_path)
    if ok:
        assert ParseConfig().parse_input().gpr.matrix_dtype == (value or "float64")
    else:
        with pytest.raises(Exception):
            ParseConfig().parse_input()


@pytest.mark.parametrize("value, ok", [(None, True), ("jacobi", True), ("super", False)])
def test_gpr_preconditioner(tmp_path, monkeypatch, value, ok):
    """None: the key is left out and defaults to kronc"""
    data = yaml.safe_load(yaml.safe_dump(MINIMAL_INP))
    if value is not None:
        data["gpr"]["preconditioner"] = value
    write_inp(tmp_path, data)
    monkeypatch.chdir(tmp_path)
    if ok:
        assert ParseConfig().parse_input().gpr.preconditioner == (value or "kronc")
    else:
        with pytest.raises(Exception):
            ParseConfig().parse_input()


@pytest.mark.parametrize(
    "key, value, ok, default",
    [
        ("sparse_selection", None, True, "fps"),
        ("sparse_selection", "totalnov", True, "fps"),
        ("sparse_selection", "novelty", False, "fps"),
        ("sparse_selection", "hybrid", True, "fps"),
        ("sparse_budget", None, True, "shared"),
        ("sparse_budget", "per_species", True, "shared"),
        ("sparse_budget", "species", False, "shared"),
        ("sparse_ncand", None, True, 256),
        ("sparse_ncand", 0, True, 256),
        ("sparse_ncand", -1, False, 256),
        ("sparse_neval", None, True, 4096),
        ("sparse_neval", 100, True, 4096),
        ("sparse_neval", 1.5, False, 4096),
        ("sparse_neval", True, False, 4096),
        ("sparse_hybrid_fraction", None, True, 0.8),
        ("sparse_hybrid_fraction", 0.6, True, 0.8),
        ("sparse_hybrid_fraction", 1, True, 0.8),
        ("sparse_hybrid_fraction", 1.5, False, 0.8),
        ("sparse_hybrid_fraction", True, False, 0.8),
        ("salted.seed", None, True, 42),
        ("salted.seed", 0, True, 42),
        ("salted.seed", -1, False, 42),
    ],
)
def test_gpr_sparse_selection_keys(tmp_path, monkeypatch, key, value, ok, default):
    """None: the key is left out and takes its default"""
    section, key = key.split(".") if "." in key else ("gpr", key)
    data = yaml.safe_load(yaml.safe_dump(MINIMAL_INP))
    if value is not None:
        data[section][key] = value
    write_inp(tmp_path, data)
    monkeypatch.chdir(tmp_path)
    if ok:
        assert getattr(getattr(ParseConfig().parse_input(), section), key) == (default if value is None else value)
    else:
        with pytest.raises(Exception):
            ParseConfig().parse_input()
