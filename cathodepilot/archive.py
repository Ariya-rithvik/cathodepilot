"""Python port of the team's MOME archive (XtalOpt_new, branch quality-diversity).

Mirrors src/search/paretoset.h, src/search/momegrid.h and Search::frontCrowdingDistance /
Search::doesDominate in selection.cpp, so results here match the C++ search:

* objectives are oriented so lower is better in every dimension;
* descriptors pick the cell, objectives decide membership inside it;
* out-of-range or non-finite descriptor values are rejected, never clamped;
* a full cell evicts its least-crowded member;
* parents are drawn uniformly over occupied cells, then uniformly within the cell.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")

PARETO_SET_CAPACITY = 10


def does_dominate(x: list[float], y: list[float]) -> bool:
    any_lt = False
    for xi, yi in zip(x, y):
        if xi > yi:
            return False
        any_lt |= xi < yi
    return any_lt


def front_crowding_distance(front: list[list[float]]) -> list[float]:
    n = len(front)
    if n < 2:
        return [0.0] * n
    dist = [0.0] * n
    for i in range(len(front[0])):
        order = sorted(range(n), key=lambda k: front[k][i])
        dist[order[0]] = dist[order[-1]] = math.inf
        rng = front[order[-1]][i] - front[order[0]][i]
        if rng > 0:
            for j in range(1, n - 1):
                dist[order[j]] += (front[order[j + 1]][i] - front[order[j - 1]][i]) / rng
    return dist


@dataclass
class Entry(Generic[T]):
    objectives: list[float]
    payload: T


class ParetoSet(Generic[T]):
    def __init__(self, capacity: int = PARETO_SET_CAPACITY):
        self.capacity = max(1, capacity)
        self.entries: list[Entry[T]] = []

    def __len__(self) -> int:
        return len(self.entries)

    def insert(self, objectives: list[float], payload: T) -> bool:
        m = len(objectives)
        if m == 0 or not all(math.isfinite(v) for v in objectives):
            return False
        for e in self.entries:
            if len(e.objectives) != m or does_dominate(e.objectives, objectives):
                return False
        self.entries = [e for e in self.entries if not does_dominate(objectives, e.objectives)]
        self.entries.append(Entry(list(objectives), payload))
        if len(self.entries) > self.capacity:
            dist = front_crowding_distance([e.objectives for e in self.entries])
            worst = min(range(len(dist)), key=lambda i: dist[i])  # first minimum, like the C++ loop
            del self.entries[worst]
        return True


@dataclass
class Descriptor:
    name: str
    min: float
    max: float

    def valid(self) -> bool:
        return math.isfinite(self.min) and math.isfinite(self.max) and self.min < self.max


def value_to_bin(value: float, lo: float, hi: float, n_bins: int) -> int | None:
    if not (math.isfinite(value) and math.isfinite(lo) and math.isfinite(hi)) or lo >= hi:
        return None
    if value < lo or value > hi:
        return None
    if value == hi:
        return n_bins - 1
    return min(max(int(math.floor((value - lo) / (hi - lo) * n_bins)), 0), n_bins - 1)


@dataclass
class InsertResult:
    accepted: bool
    cell: tuple[int, int] | None
    reason: str


@dataclass
class MOMEGrid(Generic[T]):
    descriptors: tuple[Descriptor, Descriptor]
    x_bins: int = 10
    y_bins: int = 10
    capacity: int = PARETO_SET_CAPACITY
    cells: dict[tuple[int, int], ParetoSet[T]] = field(default_factory=dict)

    def __post_init__(self):
        self.x_bins, self.y_bins = max(1, self.x_bins), max(1, self.y_bins)

    def descriptor_to_cell(self, d0: float, d1: float) -> tuple[int, int] | None:
        a, b = self.descriptors
        x = value_to_bin(d0, a.min, a.max, self.x_bins)
        y = value_to_bin(d1, b.min, b.max, self.y_bins)
        return None if x is None or y is None else (x, y)

    def insert(self, d0: float, d1: float, objectives: list[float], payload: T) -> InsertResult:
        cell = self.descriptor_to_cell(d0, d1)
        if cell is None:
            return InsertResult(False, None, "descriptor out of bounds or non-finite")
        ps = self.cells.setdefault(cell, ParetoSet(self.capacity))
        ok = ps.insert(objectives, payload)
        if not ps.entries:
            del self.cells[cell]
        return InsertResult(ok, cell, "inserted" if ok else "dominated or invalid objectives")

    def occupied(self) -> list[tuple[int, int]]:
        return sorted(c for c, ps in self.cells.items() if len(ps))

    def coverage(self) -> float:
        return len(self.occupied()) / (self.x_bins * self.y_bins)

    def members(self) -> list[tuple[tuple[int, int], Entry[T]]]:
        return [(c, e) for c in self.occupied() for e in self.cells[c].entries]

    def select_parent(self, rng: random.Random) -> T | None:
        occ = self.occupied()
        if not occ:
            return None
        return rng.choice(self.cells[rng.choice(occ)].entries).payload
