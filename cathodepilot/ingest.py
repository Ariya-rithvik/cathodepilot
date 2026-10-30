"""Parse XtalOpt run output into structure records.

Two layouts are supported, matching the data we have:
  * run directories:  <run>/<GGGGGxIIIII>/{output.POSCAR, structure.state, output.result, *.got}
  * flat exports:     <run>/<G>x<I>.POSCAR + <G>x<I>.got

Every value keeps a pointer to the file and line it came from, so the agent and
UI can cite evidence instead of restating numbers.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

ATOMIC_MASS = {
    "Li": 6.94, "Na": 22.99, "Mg": 24.305, "Al": 26.98, "Ti": 47.87, "V": 50.94, "Cr": 52.00,
    "Mn": 54.94, "Fe": 55.85, "Co": 58.93, "Ni": 58.69, "Cu": 63.55, "Zn": 65.38, "Zr": 91.22,
    "Nb": 92.91, "Mo": 95.95, "W": 183.84, "O": 16.00, "F": 19.00, "P": 30.97, "S": 32.06,
}
AMU_PER_A3_TO_G_PER_CM3 = 1.66054


@dataclass
class Evidence:
    file: str
    line: int | None = None
    text: str = ""


@dataclass
class StructureRecord:
    id: str
    run: str
    generation: int
    index: int
    species: list[str]
    counts: list[int]
    lattice: list[list[float]]
    frac_coords: list[list[float]]
    status: str = "unknown"
    energy_ev: float | None = None
    energy_source: str | None = None
    converged: bool | None = None
    gnorm: float | None = None
    raw_result: list[float] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)
    evidence: dict[str, Evidence] = field(default_factory=dict)

    @property
    def natoms(self) -> int:
        return sum(self.counts)

    @property
    def formula(self) -> str:
        return "".join(f"{s}{c if c != 1 else ''}" for s, c in zip(self.species, self.counts))

    @property
    def composition(self) -> dict[str, int]:
        return dict(zip(self.species, self.counts))

    @property
    def chem_system(self) -> str:
        return "-".join(sorted(self.species))

    @property
    def volume(self) -> float:
        (a1, a2, a3), (b1, b2, b3), (c1, c2, c3) = self.lattice
        return abs(a1 * (b2 * c3 - b3 * c2) - a2 * (b1 * c3 - b3 * c1) + a3 * (b1 * c2 - b2 * c1))

    @property
    def density(self) -> float:
        mass = sum(ATOMIC_MASS[s] * c for s, c in zip(self.species, self.counts))
        return mass * AMU_PER_A3_TO_G_PER_CM3 / self.volume

    @property
    def energy_per_atom(self) -> float | None:
        return None if self.energy_ev is None else self.energy_ev / self.natoms

    def summary(self) -> dict:
        return {
            "id": self.id, "run": self.run, "generation": self.generation, "formula": self.formula,
            "chem_system": self.chem_system, "natoms": self.natoms, "status": self.status,
            "energy_ev": self.energy_ev, "energy_per_atom": self.energy_per_atom,
            "energy_source": self.energy_source, "converged": self.converged,
            "volume_A3": round(self.volume, 3), "density_g_cm3": round(self.density, 4),
            "raw_result": self.raw_result, "files": self.files,
        }


def parse_poscar(path: Path) -> tuple[list[str], list[int], list[list[float]], list[list[float]]]:
    lines = path.read_text().splitlines()
    scale = float(lines[1].split()[0])
    lattice = [[float(x) * scale for x in lines[i].split()[:3]] for i in (2, 3, 4)]
    species = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    start = 8 if lines[7].strip()[0].lower() in "dc" else 9
    cartesian = lines[start - 1].strip()[0].lower() == "c"
    coords = [[float(x) for x in lines[start + i].split()[:3]] for i in range(sum(counts))]
    if cartesian:
        coords = _cart_to_frac(coords, lattice)
    return species, counts, lattice, coords


def _cart_to_frac(cart, lattice):
    import numpy as np

    return (np.array(cart) @ np.linalg.inv(np.array(lattice))).tolist()


_GOT_ENERGY = re.compile(r"Final energy\s*=\s*(-?\d+\.\d+)\s*eV")
_GOT_GNORM = re.compile(r"Final Gnorm\s*=\s*(\d+\.\d+)")
_GOT_CYCLE = re.compile(r"Cycle:\s+\d+\s+Energy:\s+(-?\d+\.\d+)\s+Gnorm:\s+(\d+\.\d+)")


def parse_gulp_output(path: Path) -> dict:
    """Pull the final energy, convergence flag and Gnorm out of a GULP .got file."""
    out: dict = {"converged": False, "energy_ev": None, "gnorm": None, "failure": None, "evidence": {}}
    last_cycle = None
    for n, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if "Optimisation achieved" in line:
            out["converged"] = True
            out["evidence"]["converged"] = Evidence(str(path), n, line.strip())
        elif m := _GOT_ENERGY.search(line):
            out["energy_ev"] = float(m.group(1))
            out["evidence"]["energy_ev"] = Evidence(str(path), n, line.strip())
        elif m := _GOT_GNORM.search(line):
            out["gnorm"] = float(m.group(1))
        elif m := _GOT_CYCLE.search(line):
            last_cycle = (n, float(m.group(1)), float(m.group(2)))
        elif "Too many failed attempts" in line or "Maximum number of function calls" in line:
            out["failure"] = line.strip()
            out["evidence"]["failure"] = Evidence(str(path), n, line.strip())
    if out["gnorm"] is None and last_cycle:
        out["gnorm"] = last_cycle[2]
    return out


_ID = re.compile(r"(\d+)x(\d+)")


def _gen_index(name: str) -> tuple[int, int]:
    m = _ID.search(name)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def load_run(run_dir: str | Path) -> list[StructureRecord]:
    run_dir = Path(run_dir)
    run = run_dir.name
    records: list[StructureRecord] = []

    for sub in sorted(p for p in run_dir.iterdir() if p.is_dir() and _ID.fullmatch(p.name)):
        poscar = sub / "output.POSCAR"
        if not poscar.exists():
            continue
        species, counts, lattice, coords = parse_poscar(poscar)
        gen, idx = _gen_index(sub.name)
        rec = StructureRecord(f"{run}/{sub.name}", run, gen, idx, species, counts, lattice, coords,
                              files={"poscar": str(poscar)})
        state = sub / "structure.state"
        if state.exists():
            rec.files["state"] = str(state)
            m = re.search(r"Status=(\w+)", state.read_text())
            rec.status = m.group(1) if m else "unknown"
        result = sub / "output.result"
        if result.exists():
            rec.files["result"] = str(result)
            rec.raw_result = [float(x) for x in result.read_text().split() if _is_float(x)]
            rec.evidence["raw_result"] = Evidence(str(result), 1, result.read_text().strip())
        for got in sub.glob("*.got"):
            _apply_gulp(rec, got)
        records.append(rec)

    for poscar in sorted(run_dir.glob("*.POSCAR")):
        species, counts, lattice, coords = parse_poscar(poscar)
        gen, idx = _gen_index(poscar.stem)
        rec = StructureRecord(f"{run}/{poscar.stem}", run, gen, idx, species, counts, lattice, coords,
                              files={"poscar": str(poscar)})
        got = poscar.with_suffix(".got")
        if got.exists():
            _apply_gulp(rec, got)
            rec.status = "Optimized" if rec.converged else "Failed"
        records.append(rec)

    return records


def _apply_gulp(rec: StructureRecord, got: Path) -> None:
    g = parse_gulp_output(got)
    rec.files["gulp"] = str(got)
    rec.converged = g["converged"]
    rec.gnorm = g["gnorm"]
    if g["energy_ev"] is not None:
        rec.energy_ev = g["energy_ev"]
        rec.energy_source = "gulp"
    rec.evidence.update(g["evidence"])


def _is_float(s: str) -> bool:
    try:
        return math.isfinite(float(s))
    except ValueError:
        return False
