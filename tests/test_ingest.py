from pathlib import Path

import pytest

from cathodepilot.ingest import load_run

DATA = Path(__file__).resolve().parents[1] / "data" / "runs"


@pytest.mark.skipif(not (DATA / "moqd-oct15").exists(), reason="real run data not present")
def test_gulp_export_parses_real_energies():
    recs = {r.id.split("/")[1]: r for r in load_run(DATA / "moqd-oct15")}
    assert set(recs) == {"1x07", "4x48", "4x52", "5x15", "5x18"}
    r = recs["4x48"]
    assert r.formula == "Li3MnNbNiO6"
    assert r.converged is True
    assert r.energy_ev == pytest.approx(-313.31471958)
    assert r.generation == 4
    assert r.density == pytest.approx(3.4977, abs=0.01)  # GULP reports 3.497668
    assert r.evidence["energy_ev"].line is not None


@pytest.mark.skipif(not (DATA / "archive-oct08").exists(), reason="real run data not present")
def test_run_directory_layout_parses():
    recs = load_run(DATA / "archive-oct08")
    assert len(recs) == 13
    first = recs[0]
    assert first.status == "Optimized"
    assert first.raw_result == [-46.5454, 144.0, -3.8788]
    assert first.energy_ev is None  # no GULP output in this export
    assert {r.chem_system for r in recs} >= {"Co-Li-Mn-Ni-O", "Al-Li-Mn-Ni-O", "Fe-Li-Mn-Ni-O", "Li-Mn-Nb-Ni-O"}
