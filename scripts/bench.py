"""Frozen benchmark: same seeds, same budget, same operators for every arm.

Usage:  .venv/Scripts/python scripts/bench.py --arms random pareto mome --seeds 0 1 2 --budget 60
Results: results/bench/<arm>_s<seed>.json and results/bench/summary.json
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pymatgen.core import Structure  # noqa: E402

from cathodepilot.search import RunConfig, Search  # noqa: E402

# FROZEN for the comparison. Do not change after runs start.
SEED_IDS = ["00001x00001", "00001x00002", "00001x00006", "00001x00009", "00001x00013"]  # Li3MnNiCoO6
FROZEN_WEIGHTS = {"strain": 0.4, "rattle": 0.4, "permute": 0.2, "substitute": 0.0}  # one composition, no hull yet


def seeds():
    return [(sid, Structure.from_file(ROOT / "data/runs/archive-oct08" / sid / "output.POSCAR")) for sid in SEED_IDS]


def run_one(arm, seed, budget, batch, start, cache, planner=None, out=None):
    t0 = time.time()
    cfg = RunConfig(arm=arm, seed=seed, budget=budget, batch=batch, operator_weights=dict(FROZEN_WEIGHTS))
    s = Search(cfg, start, planner=planner, seed_cache=cache,
               log=lambda e: print(f"  {arm} s{seed} {e['id']:>7} {e['operator']:8} cell={e['cell']} "
                                   f"{e['reason']} ({e['seconds']}s)", flush=True))
    m = s.run()
    m["wall_seconds"] = round(time.time() - t0, 1)
    if out:
        s.save(out, extra={"wall_seconds": m["wall_seconds"]})
    print(json.dumps(m), flush=True)
    return s, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["random", "pareto", "mome"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--out", default="results/bench")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    cache: dict = {}
    start = seeds()
    planner = None
    if "agent" in a.arms:
        from cathodepilot.agent import make_planner
        planner = make_planner()
    for seed in a.seeds:
        for arm in a.arms:
            path = out / f"{arm}_s{seed}.json"
            if path.exists():
                print(f"skip {path.name} (exists)", flush=True)
                continue
            run_one(arm, seed, a.budget, a.batch, start, cache, planner if arm == "agent" else None, path)
    summarize(out)


def summarize(out: Path):
    """MOQD score is recomputed with ONE reference point shared by all runs, so arms are comparable."""
    from collections import defaultdict

    from cathodepilot.tools import hypervolume_2d

    docs = [(p, json.loads(p.read_text())) for p in sorted(out.glob("*_s*.json"))]
    all_obj = [a["objectives"] for _, d in docs for a in d.get("archive", [])]
    ref = [max(o[0] for o in all_obj) + 0.05, 0.0] if all_obj else [0.0, 0.0]
    rows = []
    for p, d in docs:
        cells = defaultdict(list)
        for a in d.get("archive", []):
            cells[tuple(a["cell"])].append(a["objectives"])
        r = d["metrics"] | {"file": p.name, "wall_seconds": d.get("extra", {}).get("wall_seconds")}
        r["moqd_score"] = round(sum(hypervolume_2d(v, ref) for v in cells.values()), 6)
        rows.append(r)
    by_arm: dict = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)
    summary = {}
    for arm, rs in by_arm.items():
        summary[arm] = {k: _ms([r[k] for r in rs if r.get(k) is not None])
                        for k in ("coverage", "occupied_cells", "moqd_score", "best_voltage", "unique_valid",
                                  "duplicates", "failed_or_unscored", "wall_seconds")}
        summary[arm]["runs"] = len(rs)
    summary["_reference_point"] = ref
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def _ms(xs):
    if not xs:
        return None
    return {"mean": round(statistics.fmean(xs), 4), "sd": round(statistics.stdev(xs), 4) if len(xs) > 1 else 0.0,
            "n": len(xs), "values": xs}


if __name__ == "__main__":
    main()
