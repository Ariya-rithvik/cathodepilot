"""CHGNet physics: relaxation, energy, average voltage and energy above hull.

Everything here is a fast machine-learned surrogate (CHGNet, trained on Materials
Project relaxations), not DFT. Numbers are for ranking and steering the search.

Fixes two problems in the team's original voltage daemon:
  * the Li-metal reference is computed once and cached, not per structure;
  * calls are synchronous, so a caller never reads a result before it exists.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"


@dataclass
class RelaxResult:
    energy_ev: float
    energy_per_atom: float
    steps: int
    converged: bool
    final_fmax: float
    relaxed_poscar: str


@dataclass
class Evaluation:
    formula: str
    natoms: int
    lithiated: RelaxResult | None
    delithiated: RelaxResult | None
    voltage_v: float | None
    e_above_hull: float | None
    density: float
    volume_per_atom: float
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@lru_cache(maxsize=1)
def _relaxer():
    import torch
    from chgnet.model import CHGNet, StructOptimizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return StructOptimizer(model=CHGNet.load(verbose=False), use_device=device)


def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def relax(structure, fmax: float = 0.1, steps: int = 300, relax_cell: bool = True) -> tuple[object, RelaxResult]:
    from pymatgen.io.vasp import Poscar

    res = _relaxer().relax(structure.copy(), fmax=fmax, steps=steps, relax_cell=relax_cell, verbose=False)
    final = res["final_structure"]
    traj = res["trajectory"]
    final_fmax = float(max(math.sqrt(sum(c * c for c in f)) for f in traj.forces[-1]))
    e = float(traj.energies[-1])
    return final, RelaxResult(e, e / len(final), len(traj.energies), final_fmax <= fmax * 1.0001,
                              final_fmax, Poscar(final).get_str())


@lru_cache(maxsize=1)
def li_reference_energy() -> float:
    """CHGNet energy per atom of relaxed bcc Li. Computed once, cached on disk."""
    cache = CACHE_DIR / "li_reference.json"
    if cache.exists():
        return json.loads(cache.read_text())["energy_per_atom"]
    from pymatgen.core import Lattice, Structure

    li = Structure(Lattice.cubic(3.51), ["Li", "Li"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    _, r = relax(li, fmax=0.01, steps=500)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"energy_per_atom": r.energy_per_atom, "model": "CHGNet"}))
    return r.energy_per_atom


def average_voltage(lith_energy: float, delith_energy: float, n_li: int) -> float:
    """V = (E_delithiated + n * E_Li_metal - E_lithiated) / n, in volts (energies in eV)."""
    return (delith_energy + n_li * li_reference_energy() - lith_energy) / n_li


def evaluate(structure, hull=None, with_voltage: bool = True) -> Evaluation:
    """Relax a lithiated structure, then compute its full-delithiation voltage and hull distance."""
    try:
        relaxed, lith = relax(structure)
    except Exception as e:  # CHGNet can fail on badly overlapping atoms
        return Evaluation(structure.composition.reduced_formula, len(structure), None, None, None, None,
                          float(structure.density), structure.volume / len(structure), error=f"relax_failed: {e}")
    voltage, delith = None, None
    n_li = sum(1 for s in relaxed.species if s.symbol == "Li")
    if with_voltage and n_li:
        empty = relaxed.copy()
        empty.remove_species(["Li"])
        if len(empty):
            try:
                _, delith = relax(empty)
                voltage = average_voltage(lith.energy_ev, delith.energy_ev, n_li)
            except Exception:
                delith = None
    e_hull = hull.e_above_hull(relaxed.composition, lith.energy_ev) if hull is not None else None
    return Evaluation(relaxed.composition.reduced_formula, len(relaxed), lith, delith, voltage, e_hull,
                      float(relaxed.density), relaxed.volume / len(relaxed))


class CHGNetHull:
    """Convex hull built from Materials Project structures re-relaxed with CHGNet.

    Using CHGNet for both the reference phases and the candidates keeps energies on
    one consistent scale (mixing CHGNet with MP's DFT energies would not be).
    """

    def __init__(self, chemsys: str, mp_api_key: str, max_e_above_hull: float = 0.05):
        self.chemsys = chemsys
        cache = CACHE_DIR / f"hull_{chemsys}.json"
        if cache.exists():
            self.entries_data = json.loads(cache.read_text())
        else:
            self.entries_data = self._build(chemsys, mp_api_key, max_e_above_hull)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(self.entries_data))
        self._pd = self._phase_diagram()

    @staticmethod
    def _build(chemsys: str, key: str, max_e: float) -> list[dict]:
        from mp_api.client import MPRester

        with MPRester(key) as mpr:
            docs = mpr.materials.summary.search(chemsys=_all_subsystems(chemsys),
                                                energy_above_hull=(0, max_e),
                                                fields=["material_id", "structure", "formula_pretty"])
        out = []
        for d in docs:
            if len(d.structure) > 60:
                continue
            try:
                _, r = relax(d.structure, steps=200)
                out.append({"material_id": str(d.material_id), "formula": d.formula_pretty,
                            "composition": d.structure.composition.as_dict(), "energy_ev": r.energy_ev})
            except Exception:
                continue
        return out

    def _phase_diagram(self):
        from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
        from pymatgen.core import Composition

        return PhaseDiagram([PDEntry(Composition(e["composition"]), e["energy_ev"], name=e["material_id"])
                             for e in self.entries_data])

    def e_above_hull(self, composition, energy_ev: float) -> float | None:
        from pymatgen.analysis.phase_diagram import PDEntry

        try:
            return float(self._pd.get_e_above_hull(PDEntry(composition, energy_ev), allow_negative=True))
        except Exception:
            return None

    def known_match(self, composition) -> list[str]:
        red = composition.reduced_formula
        return [e["material_id"] for e in self.entries_data if _reduced(e["composition"]) == red]


def _reduced(comp: dict) -> str:
    from pymatgen.core import Composition

    return Composition(comp).reduced_formula


def _all_subsystems(chemsys: str) -> list[str]:
    from itertools import combinations

    els = sorted(chemsys.split("-"))
    return ["-".join(c) for k in range(1, len(els) + 1) for c in combinations(els, k)]
