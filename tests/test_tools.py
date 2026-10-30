from pathlib import Path

import pytest

from cathodepilot import tools
from cathodepilot.archive import Descriptor
from cathodepilot.ingest import load_run

DATA = Path(__file__).resolve().parents[1] / "data" / "runs"
needs_data = pytest.mark.skipif(not DATA.exists(), reason="real run data not present")


def test_hypervolume_2d():
    assert tools.hypervolume_2d([[0, 0]], [1, 1]) == 1.0
    assert tools.hypervolume_2d([[0, 0.5], [0.5, 0]], [1, 1]) == pytest.approx(0.75)
    assert tools.hypervolume_2d([[2, 2]], [1, 1]) == 0.0


def test_pareto_front():
    assert sorted(tools.pareto_front({"a": [1, 3], "b": [2, 2], "c": [3, 3]})) == ["a", "b"]


def test_charge_balance():
    assert tools.charge_balance({"Li": 3, "Mn": 1, "Ni": 1, "Co": 1, "O": 6}, tools.GULP_CHARGES) == 0
    assert tools.charge_balance({"Li": 3, "Mn": 1, "Ni": 1, "Co": 1, "O": 7}, tools.GULP_CHARGES) == -2


def test_bounds_advice_counts_reachable_bins():
    adv = tools.bounds_advice([2.29, 2.52, 2.76, 3.50, 3.93], 0.0, 5.0, 10)
    assert adv["bins_reachable_with_observed_range"] == 4
    assert adv["rejected_out_of_bounds"] == 0


@needs_data
def test_integrity_flags_on_real_data():
    archive = load_run(DATA / "archive-oct08")
    flags = tools.integrity_checks(archive)
    kinds = {f["check"] for f in flags}
    assert "result_columns_look_synthetic" in kinds
    assert "not_charge_balanced" in kinds  # Li3MnNiCoO7 and Li6M6O13

    gulp = load_run(DATA / "moqd-oct15")
    gflags = tools.integrity_checks(gulp)
    assert not any(f["check"] == "gulp_not_converged" for f in gflags)
    assert not any(f["check"] == "not_charge_balanced" for f in gflags)
    assert all(f["check"] == "charges_differ_from_textbook" for f in gflags)  # Nb +2 in the GULP setup


@needs_data
def test_build_archive_on_gulp_run():
    recs = load_run(DATA / "moqd-oct15")
    objs = {r.id: [r.energy_per_atom, -r.density] for r in recs}
    grid, log = tools.build_archive(recs, Descriptor("density", 0, 5), Descriptor("volume_per_atom", 5, 25), objs)
    stats = tools.archive_stats(grid, ref=[0.0, 0.0])
    assert stats["occupied_cells"] >= 3
    assert len(log.get("inserted", [])) == 5
