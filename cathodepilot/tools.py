"""Deterministic analysis tools. The agent calls these; it never computes numbers itself.

Each tool returns plain JSON-able dicts and, where it makes a claim, the evidence
(file + line) that supports it.
"""
from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict

from .archive import Descriptor, MOMEGrid, does_dominate
from .ingest import StructureRecord

# Oxidation states used by the team's GULP potentials (from the .got species table).
# Kept separate from textbook values so the integrity check can point out the difference.
GULP_CHARGES = {"Li": 1, "Ni": 3, "Mn": 4, "Co": 2, "Al": 2, "Fe": 2, "Nb": 2, "O": -2}
TEXTBOOK_CHARGES = {"Li": 1, "Ni": 2, "Mn": 4, "Co": 3, "Al": 3, "Fe": 3, "Nb": 5, "O": -2}


def run_overview(records: list[StructureRecord]) -> dict:
    return {
        "structures": len(records),
        "by_chem_system": dict(Counter(r.chem_system for r in records)),
        "by_formula": dict(Counter(r.formula for r in records)),
        "by_status": dict(Counter(r.status for r in records)),
        "gulp_converged": sum(1 for r in records if r.converged),
        "with_energy": sum(1 for r in records if r.energy_ev is not None),
        "generations": sorted({r.generation for r in records}),
    }


def descriptor_values(records: list[StructureRecord], name: str) -> dict[str, float]:
    getters = {
        "density": lambda r: r.density,
        "volume_per_atom": lambda r: r.volume / r.natoms,
        "energy_per_atom": lambda r: r.energy_per_atom,
    }
    if name not in getters:
        raise ValueError(f"unknown descriptor {name!r}; known: {sorted(getters)}")
    out = {}
    for r in records:
        v = getters[name](r)
        if v is not None and math.isfinite(v):
            out[r.id] = v
    return out


def bounds_advice(values: list[float], lo: float, hi: float, bins: int = 10, pad: float = 0.1) -> dict:
    """How many bins the configured bounds can ever fill, and tighter bounds from the data."""
    if not values:
        return {"error": "no values"}
    vmin, vmax = min(values), max(values)
    width = (hi - lo) / bins
    rejected = [v for v in values if v < lo or v > hi]
    first = max(0, math.floor((max(vmin, lo) - lo) / width))
    last = min(bins - 1, math.floor((min(vmax, hi) - lo) / width))
    span = vmax - vmin or abs(vmax) * 0.1 or 1.0
    s_lo, s_hi = vmin - pad * span, vmax + pad * span
    return {
        "configured": [lo, hi], "observed": [round(vmin, 4), round(vmax, 4)], "n": len(values),
        "bins_reachable_with_observed_range": last - first + 1, "bins": bins,
        "rejected_out_of_bounds": len(rejected),
        "suggested": [round(s_lo, 3), round(s_hi, 3)],
        "note": "Suggested bounds pad the observed range by 10% each side. Values outside bounds are "
                "rejected by the archive, so widen if future candidates may exceed them.",
    }


def charge_balance(composition: dict[str, int], charges: dict[str, int]) -> int | None:
    if any(el not in charges for el in composition):
        return None
    return sum(charges[el] * n for el, n in composition.items())


def integrity_checks(records: list[StructureRecord]) -> list[dict]:
    """Flag results that should not be trusted, with the evidence for each flag."""
    flags: list[dict] = []

    with_result = [r for r in records if len(r.raw_result) >= 3]
    if len(with_result) >= 4:
        col2_is_12n = all(abs(r.raw_result[1] - 12 * r.natoms) < 1e-6 for r in with_result)
        col3_is_ratio = all(abs(r.raw_result[2] - r.raw_result[0] / r.natoms) < 1e-3 for r in with_result)
        sizes = [r.natoms for r in with_result]
        firsts = [r.raw_result[0] for r in with_result]
        scales = _corr(sizes, firsts) if len(set(sizes)) > 1 else None
        if col2_is_12n or col3_is_ratio or (scales is not None and abs(scales) < 0.5):
            flags.append({
                "check": "result_columns_look_synthetic", "severity": "high",
                "finding": (f"In {len(with_result)} output.result files: column 2 == 12 x atom count: {col2_is_12n}; "
                            f"column 3 == column 1 / atoms: {col3_is_ratio}; correlation of column 1 with "
                            f"atom count: {None if scales is None else round(scales, 2)} (a real total energy "
                            "grows with cell size, |r| close to 1)."),
                "advice": "Confirm what wrote these files before using them as energies.",
                "evidence": [r.evidence["raw_result"].__dict__ for r in with_result[:3]],
            })

    for r in records:
        gulp_q = charge_balance(r.composition, GULP_CHARGES)
        text_q = charge_balance(r.composition, TEXTBOOK_CHARGES)
        if gulp_q not in (None, 0):
            flags.append({"check": "not_charge_balanced", "severity": "medium", "structure": r.id,
                          "finding": f"{r.formula} has net charge {gulp_q:+d} with the GULP charges.",
                          "evidence": [{"file": r.files["poscar"], "line": 7}]})
        if gulp_q == 0 and text_q not in (None, 0):
            dopant = next((el for el in r.composition if GULP_CHARGES.get(el) != TEXTBOOK_CHARGES.get(el)
                           and el not in ("Ni",)), None)
            flags.append({"check": "charges_differ_from_textbook", "severity": "info", "structure": r.id,
                          "finding": (f"{r.formula} balances with the GULP charges but has net charge "
                                      f"{text_q:+d} with common oxidation states"
                                      f"{f' (e.g. {dopant}: GULP {GULP_CHARGES[dopant]:+d}, usual {TEXTBOOK_CHARGES[dopant]:+d})' if dopant else ''}."),
                          "advice": "Ask the domain expert whether these charges are intended."})

    dens = [r.density for r in records]
    if len(dens) >= 4:
        med = statistics.median(dens)
        for r in records:
            if r.density < 0.75 * med:
                flags.append({"check": "low_density_outlier", "severity": "medium", "structure": r.id,
                              "finding": f"density {r.density:.2f} g/cm3 vs run median {med:.2f}; likely an open, "
                                         "unphysical framework.",
                              "evidence": [{"file": r.files.get("gulp") or r.files["poscar"]}]})

    for r in records:
        if r.converged is False:
            flags.append({"check": "gulp_not_converged", "severity": "high", "structure": r.id,
                          "finding": f"GULP did not report 'Optimisation achieved' (last Gnorm {r.gnorm}).",
                          "evidence": [r.evidence["failure"].__dict__] if "failure" in r.evidence
                          else [{"file": r.files.get("gulp")}]})
    return flags


def pareto_front(points: dict[str, list[float]]) -> list[str]:
    """Ids of non-dominated points (all objectives minimised)."""
    ids = list(points)
    return [i for i in ids if not any(does_dominate(points[j], points[i]) for j in ids if j != i)]


def hypervolume_2d(points: list[list[float]], ref: list[float]) -> float:
    """Area dominated by a set of 2-objective points (minimisation) up to a reference point."""
    pts = sorted(p for p in points if p[0] < ref[0] and p[1] < ref[1])
    hv, best_y = 0.0, ref[1]
    for x, y in pts:
        if y < best_y:
            hv += (ref[0] - x) * (best_y - y)
            best_y = y
    return hv


def archive_stats(grid: MOMEGrid, ref: list[float]) -> dict:
    """Coverage and MOQD score (sum of per-cell hypervolumes) for a 2-objective archive."""
    per_cell = {f"{c[0]},{c[1]}": hypervolume_2d([e.objectives for e in grid.cells[c].entries], ref)
                for c in grid.occupied()}
    return {"coverage": round(grid.coverage(), 4), "occupied_cells": len(per_cell),
            "moqd_score": round(sum(per_cell.values()), 6), "per_cell_hypervolume": per_cell,
            "reference_point": ref}


def build_archive(records: list[StructureRecord], d0: Descriptor, d1: Descriptor,
                  objectives: dict[str, list[float]], bins: tuple[int, int] = (10, 10)) -> tuple[MOMEGrid, dict]:
    """Insert records into a fresh grid. `objectives` maps record id -> minimised objective vector."""
    v0 = descriptor_values(records, d0.name)
    v1 = descriptor_values(records, d1.name)
    grid: MOMEGrid = MOMEGrid((d0, d1), *bins)
    log = defaultdict(list)
    for r in records:
        if r.id not in objectives or r.id not in v0 or r.id not in v1:
            log["skipped_missing_values"].append(r.id)
            continue
        res = grid.insert(v0[r.id], v1[r.id], objectives[r.id], r.id)
        log[res.reason].append(r.id)
    return grid, dict(log)


def _corr(xs: list[float], ys: list[float]) -> float:
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return 0.0 if sx == 0 or sy == 0 else sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)
