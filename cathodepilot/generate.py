"""Deterministic candidate generation from archive parents.

The agent chooses *which* operators, dopants and cells to target; these functions
make the actual structures. Operators mirror XtalOpt's in spirit:
  strain    ~ permustrain (lattice strain)
  rattle    ~ stripple (atomic displacement)
  permute   ~ permutomic (swap two different cations)
  substitute: dopant swap on one transition-metal site (Co -> Al / Fe / Nb, ...)
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

TM = {"Mn", "Ni", "Co", "Al", "Fe", "Nb", "Ti", "Mg", "Cr", "V", "Zr"}
MIN_DISTANCE_A = 1.3


@dataclass
class Child:
    structure: object
    operator: str
    parent_id: str
    detail: str


def strain(s, rng: random.Random, max_strain: float = 0.15):
    e = np.array([[rng.uniform(-max_strain, max_strain) for _ in range(3)] for _ in range(3)])
    e = (e + e.T) / 2
    out = s.copy()
    out.lattice = out.lattice.__class__(np.dot(out.lattice.matrix, np.eye(3) + e))
    return out, f"symmetric strain, max |e| = {np.abs(e).max():.3f}"


def rattle(s, rng: random.Random, amplitude: float = 0.35):
    out = s.copy()
    for i in range(len(out)):
        out.translate_sites([i], [rng.gauss(0, amplitude) for _ in range(3)], frac_coords=False, to_unit_cell=True)
    return out, f"gaussian rattle, sigma = {amplitude} A"


def permute(s, rng: random.Random):
    cations = [i for i, site in enumerate(s) if site.specie.symbol in TM]
    pairs = [(i, j) for i in cations for j in cations if i < j and s[i].specie != s[j].specie]
    if not pairs:
        return None, "no distinct cation pair"
    i, j = rng.choice(pairs)
    out = s.copy()
    si, sj = out[i].specie, out[j].specie
    out.replace(i, sj)
    out.replace(j, si)
    return out, f"swap sites {i} ({si}) and {j} ({sj})"


def substitute(s, rng: random.Random, from_el: str, to_el: str):
    sites = [i for i, site in enumerate(s) if site.specie.symbol == from_el]
    if not sites:
        return None, f"no {from_el} site"
    i = rng.choice(sites)
    out = s.copy()
    out.replace(i, to_el)
    return out, f"site {i}: {from_el} -> {to_el}"


def is_sane(s) -> tuple[bool, str]:
    if len(s) > 1:
        dm = s.distance_matrix
        np.fill_diagonal(dm, np.inf)
        if dm.min() < MIN_DISTANCE_A:
            return False, f"atoms {dm.min():.2f} A apart"
    try:
        if not s.composition.oxi_state_guesses(max_sites=-50):
            return False, "no charge-balanced oxidation states"
    except Exception:
        pass
    return True, "ok"


OPERATORS = ("strain", "rattle", "permute", "substitute")


def make_child(parent, parent_id: str, rng: random.Random, weights: dict[str, float] | None = None,
               dopants: list[str] | None = None, max_tries: int = 20) -> Child | None:
    weights = weights or {"strain": 0.35, "rattle": 0.35, "permute": 0.2, "substitute": 0.1}
    ops, w = zip(*[(k, v) for k, v in weights.items() if v > 0])
    for _ in range(max_tries):
        op = rng.choices(ops, weights=w)[0]
        if op == "strain":
            out, detail = strain(parent, rng)
        elif op == "rattle":
            out, detail = rattle(parent, rng)
        elif op == "permute":
            out, detail = permute(parent, rng)
        else:
            present = [el.symbol for el in parent.composition.elements if el.symbol in TM]
            targets = [d for d in (dopants or ["Al", "Fe", "Nb"]) if d not in present]
            sources = [el for el in present if el in ("Co",)] or present
            if not targets or not sources:
                continue
            out, detail = substitute(parent, rng, rng.choice(sources), rng.choice(targets))
        if out is None:
            continue
        ok, why = is_sane(out)
        if ok:
            return Child(out, op, parent_id, detail)
    return None
