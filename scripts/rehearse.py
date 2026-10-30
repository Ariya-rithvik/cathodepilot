"""Full dress rehearsal of every credit-spending path against scripts/fake_nemotron.py. Costs nothing.

Run:  .venv/Scripts/python scripts/rehearse.py           (about 2-3 min: the search step runs real CHGNet)
      .venv/Scripts/python scripts/rehearse.py --quick   (skip the CHGNet search step)
Uses a separate LLM cache, so the real credit ledger is untouched. Writes results/rehearsal/.
"""
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PORT = 8799
CACHE = Path(tempfile.gettempdir()) / "cathodepilot-rehearsal-cache"
shutil.rmtree(CACHE, ignore_errors=True)
os.environ.update({"NEBIUS_BASE_URL": f"http://127.0.0.1:{PORT}/v1", "NEBIUS_API_KEY": "fake-key",
                   "CATHODEPILOT_LLM_CACHE": str(CACHE), "TOKEN_BUDGET": "2000000"})
for k in ("AGENT_MODEL", "FAST_MODEL"):
    os.environ.pop(k, None)

import uvicorn  # noqa: E402

from scripts import fake_nemotron  # noqa: E402

RESULTS = []


def step(name):
    def wrap(fn):
        t0 = time.time()
        try:
            detail = fn()
            RESULTS.append((True, name, detail or "", time.time() - t0))
        except Exception as e:  # noqa: BLE001
            RESULTS.append((False, name, f"{type(e).__name__}: {e}", time.time() - t0))
            traceback.print_exc()
        print(f"{'PASS' if RESULTS[-1][0] else 'FAIL'}  {name}  {RESULTS[-1][2]}", flush=True)
        return fn
    return wrap


def main(quick: bool):
    server = uvicorn.Server(uvicorn.Config(fake_nemotron.app, port=PORT, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.1)

    from cathodepilot import agent, ingest

    records = []
    for d in ("data/runs/archive-oct08", "data/runs/moqd-oct15"):
        records += ingest.load_run(ROOT / d)

    @step("check_setup passes and picks Nemotron chat models (not the embedding model)")
    def _():
        import check_setup
        assert check_setup.main() == 0
        u, n = agent.choose_models(fake_nemotron.MODELS)
        assert (u, n) == ("nvidia/fake-Nemotron-Ultra", "nvidia/fake-Nemotron-Nano"), (u, n)
        return f"ultra={u}"

    audit_holder = {}

    @step("audit: tool loop survives an unknown tool and bad arguments, report has no <think> text")
    def _():
        res = agent.audit(records)
        tools_called = [t["tool"] for t in res["trace"]]
        assert "integrity_checks" in tools_called and "teleport" in tools_called, tools_called
        assert any("error" in (t["result"] if isinstance(t["result"], dict) else {}) for t in res["trace"])
        assert "FAKE AUDIT REPORT" in res["report"] and "<think>" not in res["report"]
        assert "4 of 10" in res["report"], res["report"]
        audit_holder["res"] = res
        return f"{len(res['trace'])} tool calls; report: {res['report'][:70]}..."

    @step("cache: running the same audit again makes zero new model calls")
    def _():
        before = agent.usage_summary()["calls"]
        agent.audit(records)
        after = agent.usage_summary()
        assert after["calls"] == before, (before, after["calls"])
        return f"{after['cached_hits']} answers served from cache"

    @step("web app: POST /api/agent/audit works and saves the report")
    def _():
        from fastapi.testclient import TestClient

        from app.server import app
        real = ROOT / "results/agent_audit.json"
        saved = real.read_bytes() if real.exists() else None
        try:
            r = TestClient(app).post("/api/agent/audit")
            assert r.status_code == 200, r.text
            st = TestClient(app).get("/api/agent/status").json()
            assert st["audit_report"]["report"].startswith("FAKE")
        finally:  # never leave the fake report where the real UI shows it
            real.unlink(missing_ok=True)
            if saved is not None:
                real.write_bytes(saved)
        return "report visible on the Agent page (then restored)"

    @step("budget: a spent TOKEN_BUDGET stops calls before they are sent")
    def _():
        os.environ["TOKEN_BUDGET"] = "1"
        os.environ["LLM_CACHE"] = "0"
        n = fake_nemotron.STATE["requests"]
        try:
            agent.audit(records)
            raise AssertionError("no BudgetExceeded")
        except agent.BudgetExceeded:
            pass
        finally:
            os.environ["TOKEN_BUDGET"], os.environ["LLM_CACHE"] = "2000000", "1"
        assert fake_nemotron.STATE["requests"] == n, "a request was sent after the budget was spent"
        return "BudgetExceeded raised, nothing sent"

    @step("bad key: the planner refuses to start (no GPU time wasted)")
    def _():
        os.environ["NEBIUS_API_KEY"] = "wrong"
        try:
            agent.make_planner()
            raise AssertionError("planner started with a bad key")
        except Exception as e:  # noqa: BLE001
            assert "401" in str(e) or "Authentication" in type(e).__name__, e
            return type(e).__name__
        finally:
            os.environ["NEBIUS_API_KEY"] = "fake-key"

    @step("text tool calls (<TOOLCALL> in content) are parsed")
    def _():
        os.environ["FAKE_TEXT_TOOLCALLS"], os.environ["LLM_CACHE"] = "1", "0"
        try:
            res = agent.audit(records)
        finally:
            os.environ["FAKE_TEXT_TOOLCALLS"], os.environ["LLM_CACHE"] = "0", "1"
        assert "integrity_checks" in [t["tool"] for t in res["trace"]]
        return f"{len(res['trace'])} tool calls parsed from text"

    @step("server errors: two HTTP 500s are retried, the call still succeeds")
    def _():
        fake_nemotron.STATE["fail_left"] = 2
        os.environ["LLM_CACHE"] = "0"
        try:
            res = agent.audit(records)
        finally:
            os.environ["LLM_CACHE"] = "1"
        assert res["report"].startswith("FAKE")
        return "retried and recovered"

    @step("run watch: Nemotron reads the evidence, proposes killing only the hung job, human approves, run recovers")
    def _():
        from cathodepilot import memory
        from cathodepilot.watch import WatchConfig, Watcher
        os.environ["CATHODEPILOT_NOTEBOOK"] = str(CACHE / "notebook.json")
        log = CACHE / "stuck.log"
        asked = []
        c = agent.client()
        cfg = WatchConfig(name="rehearsal", cmd=[sys.executable, str(ROOT / "tests/fixtures/stuck_run.py")],
                          log=str(log), progress_regex=r"^generation \d+ done", stall_minutes=0.04,
                          interval_s=0.5, cooldown_minutes=0.02, ram_pct_max=101, gpu_mem_pct_max=101,
                          disk_free_gb_min=0, store=str(CACHE / "store"), job_names=["python"])
        ultra, nano = agent.pick_models(c)
        w = Watcher(cfg, supervisor=lambda **k: asked.append(k) or {"approved": True, "channel": "test"},
                    planner_client=(c, ultra, nano), log=lambda m: None)
        w.begin(why="rehearsal")
        w.start()
        w.run(max_checks=60)
        inc = [i for i in memory.load()["incidents"] if i["signature"] == "stalled_idle"]
        assert w.finished and "search finished" in log.read_text()
        assert inc[0]["detail"]["source"] == "nemotron" and inc[0]["action"] == "kill_stuck_child"
        assert inc[0]["outcome"].startswith("resolved") and asked
        assert Path(inc[0]["detail"]["diagnostics"], "summary.json").exists()
        report = Path(w.report_path).read_text(encoding="utf-8")
        assert "FAKE SUMMARY" in report and "<think>" not in report
        assert "not in the run facts: 987" in report, "an invented number must be flagged in the report"
        warned = " ".join(x["text"] for x in memory.prestart_warnings("rehearsal"))
        assert "What worked: kill_stuck_child" in warned
        return f"asked: {asked[0]['question']!r} -> {inc[0]['outcome']}; report written; next run warned"

    if quick:
        return

    @step("agent search: real CHGNet, rejected plan fixed, ask_human logged, no element swaps, 1-2 calls per round")
    def _():
        from pymatgen.core import Structure

        from cathodepilot.search import RunConfig, Search
        from scripts.bench import FROZEN_WEIGHTS, SEED_IDS
        seeds = [(s, Structure.from_file(ROOT / "data/runs/archive-oct08" / s / "output.POSCAR")) for s in SEED_IDS[:3]]
        before = agent.usage_summary()["calls"]
        planner = agent.make_planner()
        cfg = RunConfig(arm="agent", seed=0, budget=12, batch=3, operator_weights=dict(FROZEN_WEIGHTS))
        s = Search(cfg, seeds, planner=planner)
        m = s.run()
        out = ROOT / "results/rehearsal/agent_s0.json"
        s.save(out)
        plans = [p for p in s.plans if "human_review" not in p]
        reviews = [p for p in s.plans if "human_review" in p]
        assert plans and all(p["operator_weights"].get("substitute", 0) == 0 for p in plans), plans
        assert all(p["rationale"].startswith("round") for p in plans), [p["rationale"] for p in plans]
        assert reviews and reviews[0]["human_review"]["approved"] is False
        assert all(e["operator"] != "substitute" for e in s.history)
        calls = agent.usage_summary()["calls"] - before
        assert calls <= 3 * len(plans), (calls, len(plans))
        assert any(e.get("poscar") for e in s.history if e["operator"] != "seed")
        return (f"{len(plans)} rounds, {calls} model calls, {m['occupied_cells']} cells, "
                f"{m['duplicates']} duplicates -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main("--quick" in sys.argv)
    print("\n" + "=" * 70)
    for ok, name, detail, secs in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {secs:5.1f}s  {name}")
    failed = sum(1 for r in RESULTS if not r[0])
    print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} passed" + ("" if not failed else "  <-- fix before using real credits"))
    from cathodepilot import agent
    print(f"fake model calls: {agent.usage_summary()['calls']} (real credits used: 0)")
    sys.exit(1 if failed else 0)
