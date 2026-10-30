"""CathodePilot web app. Serves the UI and a read-mostly JSON API over the real run data.

Run:  .venv/Scripts/python -m uvicorn app.server:app --port 8000
Only POST /api/agent/* spends Token Factory credits; repeats are served from the LLM cache.
"""
from __future__ import annotations

import json
import os
import sys
import time
from functools import lru_cache
from pathlib import Path, PurePosixPath, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.responses import FileResponse, PlainTextResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from cathodepilot import ingest, tools  # noqa: E402

RUNS = [ROOT / "data/runs/archive-oct08", ROOT / "data/runs/moqd-oct15"]
BENCH = ROOT / "results/bench"
# Set CATHODEPILOT_PUBLIC=1 on a hosted demo: nothing that spends credits or changes state, no details of the machine.
PUBLIC = os.environ.get("CATHODEPILOT_PUBLIC") == "1"
app = FastAPI(title="CathodePilot")


def _writes_allowed():
    if PUBLIC:
        raise HTTPException(403, "this is the read-only public demo")


@lru_cache(maxsize=1)
def records():
    out = []
    for d in RUNS:
        out += ingest.load_run(d)
    return out


def _rel(p) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def _fix_paths(o):
    if isinstance(o, dict):
        return {k: (_rel(v) if k == "file" and isinstance(v, str) else _fix_paths(v)) for k, v in o.items()}
    if isinstance(o, list):
        return [_fix_paths(x) for x in o]
    return o


def _runs():
    return [(p, json.loads(p.read_text())) for p in sorted(BENCH.glob("*_s*.json"))]


@app.get("/api/overview")
def overview():
    rows = []
    for r in records():
        s = r.summary()
        s["files"] = {k: _rel(v) for k, v in (s.get("files") or {}).items() if v}
        rows.append(s)
    return {"overview": tools.run_overview(records()), "structures": rows}


@app.get("/api/audit")
def audit():
    dens = list(tools.descriptor_values(records(), "density").values())
    return {"checks": _fix_paths(tools.integrity_checks(records())),
            "bounds": {"team_config": tools.bounds_advice(dens, 0.0, 5.0, 10),
                       "ours": tools.bounds_advice(dens, 2.0, 5.5, 10)},
            "density_values": dens}


@app.get("/api/bench")
def bench():
    if not (BENCH / "summary.json").exists():
        return {"summary": None, "runs": []}
    runs = []
    for p, d in _runs():
        curve, cells = [], set()
        for e in d["history"]:
            if e.get("accepted") and e.get("cell"):
                cells.add(tuple(e["cell"]))
            curve.append(len(cells))
        grid: dict = {}
        for a in d["archive"]:
            k = f"{a['cell'][0]},{a['cell'][1]}"
            g = grid.setdefault(k, {"n": 0, "best_v": 0.0})
            g["n"] += 1
            g["best_v"] = max(g["best_v"], -a["objectives"][1])
        hist = d["history"]
        runs.append({"file": p.name, "arm": d["config"]["arm"], "seed": d["config"]["seed"],
                     "metrics": d["metrics"], "curve": curve, "grid": grid, "plans": d.get("plans", []),
                     "descriptors": d["config"]["descriptors"],
                     "ops": {op: {"tried": sum(1 for e in hist if e["operator"] == op),
                                  "accepted": sum(1 for e in hist if e["operator"] == op and e["accepted"])}
                             for op in ("strain", "rattle", "permute", "substitute")}})
    return {"summary": json.loads((BENCH / "summary.json").read_text()), "runs": runs}


@app.get("/api/shortlist")
def shortlist(limit: int = 12):
    seen, rows = set(), []
    for p, d in _runs():
        in_archive = {a["id"] for a in d["archive"]}
        for e in d["history"]:
            if not e.get("objectives") or e["operator"] == "seed" or e["id"] not in in_archive:
                continue
            key = (round(e["energy_per_atom"], 3), round(e["voltage"], 2))
            if key in seen:
                continue
            seen.add(key)
            rows.append({"run": p.stem, "id": e["id"], "formula": e["formula"], "operator": e["operator"],
                         "detail": e["detail"], "parent": e["parent"], "voltage": e["voltage"],
                         "energy_per_atom": e["energy_per_atom"], "density": e["density"],
                         "cation_o_bond": e["cation_o_bond"], "cell": e["cell"],
                         "has_structure": bool(e.get("poscar"))})
    front = set(tools.pareto_front({str(i): [r["energy_per_atom"], -r["voltage"]] for i, r in enumerate(rows)}))
    for i, r in enumerate(rows):
        r["pareto"] = str(i) in front
    rows.sort(key=lambda r: (not r["pareto"], -r["voltage"]))
    return {"rows": rows[:limit], "total_unique": len(rows),
            "caveat": "CHGNet surrogate. Voltage reads about 0.5 V low against literature (LiCoO2 check), so use it "
                      "for ranking only. These are leads for DFT, not discoveries."}


@app.get("/api/structure/{run}/{iid}", response_class=PlainTextResponse)
def structure(run: str, iid: str):
    p = BENCH / f"{Path(run).name}.json"
    if not p.exists():
        raise HTTPException(404)
    for e in json.loads(p.read_text())["history"]:
        if e["id"] == iid and e.get("poscar"):
            return PlainTextResponse(e["poscar"],
                                     headers={"Content-Disposition": f'attachment; filename="{run}_{iid}.vasp"'})
    raise HTTPException(404, "relaxed structure not stored for this run")


@app.get("/api/file", response_class=PlainTextResponse)
def evidence(path: str = Query(...)):
    # Refuse absolute and network paths before touching the disk: resolving \\\\host\\share would make Windows contact that host.
    if PureWindowsPath(path).anchor or PurePosixPath(path).is_absolute():
        raise HTTPException(403, "only run data, reports and diagnostics can be shown")
    p = (ROOT / path).resolve()
    allowed = [ROOT / "data/runs"] if PUBLIC else [ROOT / "data/runs", ROOT / "data/memory/reports",
                                                   ROOT / "data/memory/diagnostics"]
    if not any(a.resolve() in p.parents for a in allowed) or not p.is_file():
        raise HTTPException(403, "only run data, reports and diagnostics can be shown")
    return p.read_text(encoding="utf-8", errors="replace")[:200000]


@app.get("/api/agent/status")
def agent_status():
    from cathodepilot.agent import load_env, usage_summary

    load_env(ROOT / ".env")
    reviews = [{"run": p.stem, **pl["human_review"]} for p, d in _runs() if p.stem.startswith("agent")
               for pl in d.get("plans", []) if "human_review" in pl]
    audit_file = ROOT / "results/agent_audit.json"
    return {"nebius_key": bool(os.environ.get("NEBIUS_API_KEY")),
            "calle_key": bool(os.environ.get("CALLE_API_KEY")),
            "dry_run": os.environ.get("DRY_RUN", "1") == "1",
            "usage": usage_summary(), "human_reviews": reviews,
            "audit_report": json.loads(audit_file.read_text()) if audit_file.exists() else None}


@app.post("/api/agent/audit")
def agent_audit():
    from cathodepilot.agent import audit as run_audit

    _writes_allowed()
    try:
        res = json.loads(json.dumps(run_audit(records()), default=str))
    except Exception as e:  # bad key, spent budget, network: show the reason in the UI
        raise HTTPException(400, f"{type(e).__name__}: {e}")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/agent_audit.json").write_text(json.dumps(res, indent=1))
    return res


@app.get("/api/watch")
def watch_state():
    from cathodepilot import memory

    store = ROOT / "data/memory"
    states = []
    for f in sorted(store.glob("watch_*.json")):
        try:
            states.append(json.loads(f.read_text()))
        except (json.JSONDecodeError, OSError):  # being replaced right now; the next refresh gets it
            continue
    nb = memory.load()
    if PUBLIC:  # a hosted visitor sees progress and outcomes only: no commands, paths, processes, notes or spoken words
        keep = ("name", "updated", "updated_epoch", "interval_s", "finished", "outcome", "minutes_since_progress",
                "stall_minutes", "waiting_for")
        counters = ("kind", "generation", "structures", "optimized", "failed", "in_progress", "progress_events")
        states = [{**{k: x.get(k) for k in keep}, "progress": {k: x.get("progress", {}).get(k) for k in counters},
                   "run": {"alive": x.get("run", {}).get("alive")}, "events": [],
                   "issues": [{"signature": i.get("signature")} for i in x.get("issues", [])]} for x in states]
        nb = {"goal": nb["goal"], "step": nb["step"], "rules": nb["rules"], "notes": [],
              "incidents": [{**{k: i.get(k) for k in ("time", "signature", "action", "approved")},
                             "outcome": _outcome_word(i.get("outcome"))} for i in nb["incidents"]],
              "runs": [{k: r.get(k) for k in ("name", "started", "ended", "outcome", "generation", "incidents")}
                       for r in nb["runs"]]}
        return {"notebook": nb, "watches": states, "system": _light_system(), "reports": [], "read_only": True}
    reports = [{"name": p.stem, "path": _rel(p), "time": time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))}
               for p in sorted((store / "reports").glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)[:20]]
    return {"notebook": _public_notebook(nb), "watches": states, "system": _light_system(), "reports": reports,
            "read_only": False, "notebook_error": memory.notebook_problem(nb)}


def _outcome_word(outcome) -> str:
    """Outcomes can quote what someone said on the phone or local paths; visitors get only the kind."""
    words = ("resolved", "not resolved", "not executed", "cleared", "pending", "recorded", "supervisor told", "waited",
             "run stopped", "done", "failed", "unverified")
    return next((w for w in words if str(outcome).startswith(w)), "other")


_SYSTEM_CACHE: dict = {"at": 0.0, "value": None}


def _light_system() -> dict:
    """The page refreshes every 10 s, so several tabs must not each start nvidia-smi and scan every process."""
    from cathodepilot.watch import system_status

    if time.time() - _SYSTEM_CACHE["at"] > 8:
        _SYSTEM_CACHE.update(at=time.time(), value={k: v for k, v in system_status(top=0).items() if k != "top_memory"})
    return _SYSTEM_CACHE["value"]


def _public_notebook(nb: dict) -> dict:
    """The page does not need the full text of settings files."""
    runs = [{**r, "settings": sorted(r.get("settings") or {})} for r in nb.get("runs", [])]
    return {**nb, "runs": runs}


@app.post("/api/notebook")
def notebook_edit(body: dict):
    from cathodepilot import memory

    _writes_allowed()
    kind = body.get("kind")
    text = str(body.get("text", "")).strip()
    if not text:
        raise HTTPException(400, "text is empty")
    if kind in ("goal", "step"):
        return _public_notebook(memory.set_field(kind, text[:300]))
    if kind in ("always", "never"):
        return _public_notebook(memory.add_rule(kind, text))
    if kind == "note":
        return _public_notebook(memory.add_note(text))
    raise HTTPException(400, "kind must be goal, step, always, never or note")


@app.get("/")
def index():
    return FileResponse(ROOT / "app/static/index.html")


(ROOT / "app/static").mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=ROOT / "app/static"), name="static")
