"""Quality-diversity search loop and the fair comparison between selection strategies.

Arms (same evaluation budget, same seeds):
  random  - parent drawn uniformly from everything evaluated so far
  pareto  - parent drawn from the global Pareto front (stock XtalOpt-style ranking)
  mome    - parent drawn uniformly over occupied archive cells (the team's method)
  agent   - mome, plus a planner that sets operator weights, dopants and target cells each round

Objectives (minimised): [energy metric, -voltage]. The energy metric is energy above
hull when a hull is available, otherwise CHGNet energy per atom (only comparable
within one composition, so runs without a hull must stay single-composition).
Descriptors: density and mean cation-oxygen bond length.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import physics
from .archive import Descriptor, MOMEGrid
from .generate import make_child
from .tools import hypervolume_2d, pareto_front


def mean_cation_o_bond(s, cutoff: float = 2.6) -> float:
    lengths = []
    for i, site in enumerate(s):
        if site.specie.symbol in ("O", "Li"):
            continue
        for n in s.get_neighbors(site, cutoff):
            if n.specie.symbol == "O":
                lengths.append(n.nn_distance)
    return float(np.mean(lengths)) if lengths else math.nan


@dataclass
class Individual:
    id: str
    structure: object
    eval: physics.Evaluation
    descriptors: tuple[float, float]
    objectives: list[float] | None
    parent: str | None
    operator: str
    detail: str
    round: int


@dataclass
class Plan:
    """What the planner (agent or fixed policy) decides for one round."""
    operator_weights: dict[str, float] = field(default_factory=lambda: {"strain": 0.35, "rattle": 0.35,
                                                                         "permute": 0.2, "substitute": 0.1})
    dopants: list[str] = field(default_factory=lambda: ["Al", "Fe", "Nb"])
    target_cells: list[tuple[int, int]] = field(default_factory=list)
    rationale: str = ""


@dataclass
class RunConfig:
    arm: str
    seed: int
    budget: int = 60
    batch: int = 6
    descriptors: tuple[Descriptor, Descriptor] = (Descriptor("density", 2.0, 5.5),
                                                  Descriptor("cation_o_bond", 1.7, 2.3))
    bins: tuple[int, int] = (10, 10)
    with_voltage: bool = True
    operator_weights: dict[str, float] = field(default_factory=lambda: {"strain": 0.4, "rattle": 0.4,
                                                                         "permute": 0.2, "substitute": 0.0})


class Search:
    def __init__(self, cfg: RunConfig, seeds: list[tuple[str, object]], hull=None,
                 planner: Callable[["Search"], Plan] | None = None, log: Callable[[dict], None] | None = None,
                 seed_cache: dict | None = None):
        self.cfg, self.hull, self.planner = cfg, hull, planner
        self.seed_cache = seed_cache
        self.rng = random.Random(cfg.seed)
        self.grid: MOMEGrid[str] = MOMEGrid(cfg.descriptors, *cfg.bins)
        self.pop: dict[str, Individual] = {}
        self.history: list[dict] = []
        self.plans: list[dict] = []
        self.log = log or (lambda e: None)
        self.round = 0
        self.evaluations = 0
        for sid, s in seeds:
            self._evaluate_and_insert(sid, s, None, "seed", "seed structure")

    # --- objectives / descriptors ------------------------------------------------
    def _objectives(self, ev: physics.Evaluation) -> list[float] | None:
        if ev.lithiated is None:
            return None
        e = ev.e_above_hull if ev.e_above_hull is not None else ev.lithiated.energy_per_atom
        if ev.voltage_v is None or not math.isfinite(ev.voltage_v):
            return None
        return [e, -ev.voltage_v]

    def _duplicate_of(self, relaxed) -> str | None:
        from pymatgen.analysis.structure_matcher import StructureMatcher

        m = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
        for other in self.pop.values():
            if other.objectives and other.structure.composition.reduced_formula == relaxed.composition.reduced_formula \
                    and m.fit(other.structure, relaxed):
                return other.id
        return None

    def _evaluate_and_insert(self, iid, s, parent, op, detail):
        t0 = time.time()
        key = None
        if op == "seed" and self.seed_cache is not None:
            key = f"{iid}|{self.cfg.with_voltage}"
            ev = self.seed_cache.get(key)
        else:
            ev = None
        if ev is None:
            ev = physics.evaluate(s, hull=self.hull, with_voltage=self.cfg.with_voltage)
            if key:
                self.seed_cache[key] = ev
        self.evaluations += 1
        relaxed = s
        if ev.lithiated is not None:
            from pymatgen.core import Structure
            relaxed = Structure.from_str(ev.lithiated.relaxed_poscar, fmt="poscar")
        d = (ev.density, mean_cation_o_bond(relaxed))
        obj = self._objectives(ev)
        dup = self._duplicate_of(relaxed) if obj and op != "seed" else None
        ind = Individual(iid, relaxed, ev, d, obj, parent, op, detail, self.round)
        ind.duplicate_of = dup
        self.pop[iid] = ind
        if dup:
            from .archive import InsertResult
            res = InsertResult(False, self.grid.descriptor_to_cell(*d), f"duplicate of {dup}")
        else:
            res = self.grid.insert(d[0], d[1], obj, iid) if obj else None
        event = {"id": iid, "round": self.round, "operator": op, "detail": detail, "parent": parent,
                 "formula": ev.formula, "density": round(d[0], 4), "cation_o_bond": round(d[1], 4),
                 "objectives": obj, "voltage": ev.voltage_v, "e_above_hull": ev.e_above_hull,
                 "energy_per_atom": ev.lithiated.energy_per_atom if ev.lithiated else None,
                 "cell": list(res.cell) if res and res.cell else None,
                 "accepted": bool(res and res.accepted), "reason": res.reason if res else (ev.error or "no objectives"),
                 "seconds": round(time.time() - t0, 2),
                 "poscar": relaxed.to(fmt="poscar") if obj else None}
        self.history.append(event)
        self.log(event)

    # --- parent selection per arm ---------------------------------------------------
    def _select_parent(self, plan: Plan) -> str | None:
        arm = self.cfg.arm
        scored = [i for i in self.pop.values() if i.objectives and not getattr(i, "duplicate_of", None)]
        if not scored:
            return None
        if arm == "random":
            return self.rng.choice(scored).id
        if arm == "pareto":
            front = pareto_front({i.id: i.objectives for i in scored})
            return self.rng.choice(front)
        if arm == "agent" and plan.target_cells:
            near = [c for c in self.grid.occupied() if any(abs(c[0] - t[0]) + abs(c[1] - t[1]) <= 1
                                                           for t in plan.target_cells)]
            if near and self.rng.random() < 0.7:
                return self.rng.choice(self.grid.cells[self.rng.choice(near)].entries).payload
        return self.grid.select_parent(self.rng)

    def step(self) -> None:
        if self.cfg.arm == "agent" and self.planner:
            plan = self.planner(self)
        else:
            plan = Plan(operator_weights=dict(self.cfg.operator_weights), rationale="fixed policy")
        self.plans.append({"round": self.round, **plan.__dict__})
        for k in range(self.cfg.batch):
            if self.evaluations >= self.cfg.budget:
                break
            pid = self._select_parent(plan)
            if pid is None:
                break
            child = make_child(self.pop[pid].structure, pid, self.rng, plan.operator_weights, plan.dopants)
            if child is None:
                continue
            self._evaluate_and_insert(f"r{self.round:02d}c{k:02d}", child.structure, pid, child.operator, child.detail)
        self.round += 1

    def run(self) -> dict:
        while self.evaluations < self.cfg.budget:
            before = self.evaluations
            self.step()
            if self.evaluations == before:
                break
        return self.metrics()

    # --- metrics ------------------------------------------------------------------
    def metrics(self, ref: list[float] | None = None) -> dict:
        scored = [i for i in self.pop.values() if i.objectives]
        if ref is None:
            ref = [max(i.objectives[0] for i in scored) + 0.05, 0.0] if scored else [0.0, 0.0]
        per_cell = [hypervolume_2d([e.objectives for e in self.grid.cells[c].entries], ref)
                    for c in self.grid.occupied()]
        best_v = max((i.eval.voltage_v for i in scored), default=None)
        return {"arm": self.cfg.arm, "seed": self.cfg.seed, "evaluations": self.evaluations,
                "coverage": round(self.grid.coverage(), 4), "occupied_cells": len(per_cell),
                "moqd_score": round(sum(per_cell), 6), "best_voltage": best_v,
                "failed_or_unscored": sum(1 for i in self.pop.values() if not i.objectives),
                "duplicates": sum(1 for i in self.pop.values() if getattr(i, "duplicate_of", None)),
                "unique_valid": sum(1 for i in self.pop.values() if i.objectives and not getattr(i, "duplicate_of", None)),
                "reference_point": ref}

    def save(self, path: Path, extra: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        archive = [{"cell": list(c), "id": e.payload, "objectives": e.objectives} for c, e in self.grid.members()]
        path.write_text(json.dumps({"archive": archive, "extra": extra or {},
                                    "config": {"arm": self.cfg.arm, "seed": self.cfg.seed,
                                               "budget": self.cfg.budget, "batch": self.cfg.batch,
                                               "descriptors": [d.__dict__ for d in self.cfg.descriptors]},
                                    "metrics": self.metrics(), "history": self.history, "plans": self.plans},
                                   indent=1, default=str))
