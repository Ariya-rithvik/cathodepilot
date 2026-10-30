"""Nemotron research agent on Nebius Token Factory.

The agent reads the run only through tools and changes the search only through a
validated plan. It never writes numbers or structures itself.

  * Ultra (AGENT_MODEL): plans and explains, with tool calls
  * Nano  (FAST_MODEL):  fast short calls (setup check)

Credit guard: every call goes through chat(), which answers repeated requests from a disk
cache and refuses new calls once TOKEN_BUDGET is spent. The planner puts the archive state
in the prompt and stops as soon as a plan is accepted, so a search round is usually one call.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import re
import sys
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from . import tools
from .archive import Descriptor
from .generate import OPERATORS
from .ingest import StructureRecord

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1"
ROOT = Path(__file__).resolve().parents[1]
ALLOWED_DOPANTS = {"Al", "Fe", "Nb", "Co", "Mn", "Ni", "Ti", "Mg"}
NOT_CHAT = ("embed", "reward", "guard", "safety", "retriev", "rerank", "parse", "clip", "cosmos", "groot")

SYSTEM = """You are CathodePilot, a research agent for a quality-diversity crystal-structure search
for low-cobalt Li-ion cathodes (NMC-type, LiNiMn[X]O2 with X = Co, Al, Fe, Nb).
Rules:
- Every number you state must come from a tool result or the state given to you. Cite where it came from.
- You cannot run physics or write structures. You change the search only by calling submit_plan.
- Be concise and specific. Say what you are unsure about.
- CHGNet and GULP are fast surrogates, not DFT; never call a candidate a discovery."""


def load_env(path: str | Path = ROOT / ".env") -> None:
    p = Path(path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if v.strip():
                    os.environ.setdefault(k.strip(), v.strip().strip('"'))


def client():
    from openai import OpenAI

    load_env()
    key = os.environ.get("NEBIUS_API_KEY")
    if not key:
        raise RuntimeError("NEBIUS_API_KEY missing: copy .env.example to .env and add your Token Factory key")
    return OpenAI(base_url=os.environ.get("NEBIUS_BASE_URL") or DEFAULT_BASE_URL, api_key=key, max_retries=3,
                  timeout=180)


# ---------------------------------------------------------------- credit guard
def _cache_dir() -> Path:
    d = Path(os.environ.get("CATHODEPILOT_LLM_CACHE") or ROOT / "data" / "cache" / "llm")
    d.mkdir(parents=True, exist_ok=True)
    return d


def usage_summary() -> dict:
    ledger = _cache_dir() / "ledger.jsonl"
    rows = [json.loads(l) for l in ledger.read_text().splitlines() if l.strip()] if ledger.exists() else []
    live = [r for r in rows if not r.get("cached")]
    pin, pout = sum(r["prompt_tokens"] for r in live), sum(r["completion_tokens"] for r in live)
    price_in, price_out = float(os.environ.get("PRICE_IN_PER_M", "1.0")), float(os.environ.get("PRICE_OUT_PER_M", "3.0"))
    return {"calls": len(live), "cached_hits": len(rows) - len(live), "prompt_tokens": pin, "completion_tokens": pout,
            "seconds": round(sum(r["seconds"] for r in live), 1),
            "budget_tokens": int(os.environ.get("TOKEN_BUDGET", "2000000")),
            "est_cost_usd": round(pin / 1e6 * price_in + pout / 1e6 * price_out, 4),
            "spend_limit_usd": float(os.environ.get("SPEND_LIMIT_USD", "20")),
            "by_label": {lb: sum(1 for r in live if r["label"].split(" ")[0] == lb)
                         for lb in {r["label"].split(" ")[0] for r in live}},
            "by_model": {m: sum(1 for r in live if r["model"] == m) for m in {r["model"] for r in live}}}


class BudgetExceeded(RuntimeError):
    pass


TRANSIENT = ("APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError")


def chat(c, label: str = "", **kw) -> dict:
    """Cached, budgeted chat completion. Returns {"content", "tool_calls", "finish_reason"} (plain dicts)."""
    d = _cache_dir()
    key = hashlib.sha256(json.dumps(kw, sort_keys=True, default=str).encode()).hexdigest()[:24]
    hit = d / f"{key}.json"

    def log(row):
        with (d / "ledger.jsonl").open("a") as f:
            f.write(json.dumps({"label": label, "model": kw.get("model"), **row}) + "\n")

    if hit.exists() and os.environ.get("LLM_CACHE", "1") == "1":
        log({"cached": True, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0})
        return json.loads(hit.read_text())
    u = usage_summary()
    if u["prompt_tokens"] + u["completion_tokens"] >= u["budget_tokens"]:
        raise BudgetExceeded(f"TOKEN_BUDGET reached ({u['budget_tokens']} tokens); raise it in .env to continue")
    # Dollar limit: refuse if this call's worst case (its whole prompt plus every output token allowed) would pass it.
    rate_in, rate_out = float(os.environ.get("PRICE_IN_PER_M", "1.0")), float(os.environ.get("PRICE_OUT_PER_M", "3.0"))
    est_in = len(json.dumps(kw.get("messages", []), default=str) + json.dumps(kw.get("tools", []), default=str)) // 3
    est_out = int(kw.get("max_tokens") or 4096)
    worst = est_in / 1e6 * rate_in + est_out / 1e6 * rate_out
    if u["est_cost_usd"] + worst > u["spend_limit_usd"]:
        raise BudgetExceeded(f"SPEND_LIMIT_USD reached: about ${u['est_cost_usd']:.2f} spent, this call could add "
                             f"${worst:.4f}, limit ${u['spend_limit_usd']:.2f}. No call was sent.")
    t0 = time.time()
    try:
        r = c.chat.completions.create(**kw)
    except Exception as e:
        if type(e).__name__ in TRANSIENT:  # the answer was lost, but the call may still be billed: count the worst case
            log({"cached": False, "prompt_tokens": est_in, "completion_tokens": est_out, "seconds": 0, "lost": True})
        raise
    m = r.choices[0].message
    out = {"content": m.content or "", "finish_reason": r.choices[0].finish_reason,
           "tool_calls": [{"id": tc.id or f"call_{uuid.uuid4().hex[:8]}", "type": "function",
                           "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"}}
                          for tc in (m.tool_calls or [])]}
    if not out["tool_calls"]:
        out["tool_calls"] = _tool_calls_from_text(out["content"])
    if out["finish_reason"] != "length":  # a cut-off answer must not be replayed forever
        hit.write_text(json.dumps(out))
    log({"cached": False, "prompt_tokens": getattr(r.usage, "prompt_tokens", 0) or 0,
         "completion_tokens": getattr(r.usage, "completion_tokens", 0) or 0, "seconds": round(time.time() - t0, 2),
         "finish_reason": out["finish_reason"]})
    return out


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?(?:</think>|\Z)", "", text or "", flags=re.S).strip()


def _tool_calls_from_text(text: str) -> list[dict]:
    """Some Nemotron deployments return tool calls as text: <TOOLCALL>[{"name":..,"arguments":{..}}]</TOOLCALL>."""
    m = re.search(r"<TOOLCALL>\s*(\[.*?\])\s*</TOOLCALL>", strip_thinking(text), re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(1))
    except json.JSONDecodeError:
        return []
    return [{"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function",
             "function": {"name": it.get("name", ""), "arguments": json.dumps(it.get("arguments", {}))}}
            for it in items if isinstance(it, dict)]


def choose_models(ids: list[str]) -> tuple[str | None, str | None]:
    nv = [i for i in ids if ("nemotron" in i.lower() or "nvidia" in i.lower())
          and not any(b in i.lower() for b in NOT_CHAT)]
    ultra = os.environ.get("AGENT_MODEL") or next((i for i in nv if "ultra" in i.lower()), None) \
        or next((i for i in nv if "super" in i.lower()), nv[0] if nv else None)
    nano = os.environ.get("FAST_MODEL") or next((i for i in nv if "nano" in i.lower()), ultra)
    return ultra, nano


def pick_models(c) -> tuple[str, str]:
    ultra, nano = choose_models([m.id for m in c.models.list().data])
    if not ultra:
        raise RuntimeError("No NVIDIA/Nemotron chat model available for this key")
    return ultra, nano


# ---------------------------------------------------------------- plan validation
def validate_plan(plan: dict, grid_bins: tuple[int, int], allow_substitute: bool = True) -> tuple[bool, list[str], dict]:
    errors, clean = [], {}
    w = plan.get("operator_weights") or {}
    if not isinstance(w, dict):
        errors.append("operator_weights must be an object like {\"strain\": 0.4}")
        w = {}
    unknown = set(w) - set(OPERATORS)
    if unknown:
        errors.append(f"unknown operators {sorted(unknown)}; allowed {list(OPERATORS)}")
    vals = {}
    for k, v in w.items():
        if k in OPERATORS:
            try:
                vals[k] = float(v)
            except (TypeError, ValueError):
                errors.append(f"weight for {k} is not a number: {v!r}")
    if any(not math.isfinite(v) or v < 0 for v in vals.values()):
        errors.append("operator weights must be finite and >= 0")
    if sum(v for v in vals.values() if math.isfinite(v)) <= 0:
        errors.append("operator weights must sum to > 0")
    if not allow_substitute and vals.get("substitute", 0) > 0:
        errors.append("substitute must be 0 in this run: there is no convex hull, so energies of different "
                      "compositions cannot be compared")
    clean["operator_weights"] = vals
    dop = plan.get("dopants") or []
    dop = list(dop) if isinstance(dop, (list, tuple)) else [dop]
    bad = [d for d in dop if d not in ALLOWED_DOPANTS]
    if bad:
        errors.append(f"dopants {bad} not allowed; allowed {sorted(ALLOWED_DOPANTS)}")
    clean["dopants"] = [d for d in dop if d in ALLOWED_DOPANTS]
    cells = []
    for c in plan.get("target_cells") or []:
        try:
            x, y = int(c[0]), int(c[1])
        except (TypeError, ValueError, IndexError, KeyError):
            errors.append(f"bad cell {c!r}")
            continue
        if not (0 <= x < grid_bins[0] and 0 <= y < grid_bins[1]):
            errors.append(f"cell {c} outside the {grid_bins[0]}x{grid_bins[1]} grid")
        else:
            cells.append((x, y))
    clean["target_cells"] = cells[:12]
    clean["rationale"] = str(plan.get("rationale", ""))[:1200]
    return not errors, errors, clean


# ---------------------------------------------------------------- tool schemas
def _fn(name, desc, props=None, required=None):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props or {}, "required": required or []}}}


AUDIT_TOOLS = [
    _fn("run_overview", "Counts of structures by chemical system, formula, status and generation."),
    _fn("integrity_checks", "Deterministic checks for untrustworthy results, with evidence (file, line)."),
    _fn("bounds_advice", "How many archive bins a descriptor's configured bounds can reach given the observed values.",
        {"descriptor": {"type": "string", "enum": ["density", "volume_per_atom", "energy_per_atom"]},
         "min": {"type": "number"}, "max": {"type": "number"}, "bins": {"type": "integer"}},
        ["descriptor", "min", "max"]),
    _fn("list_structures", "Per-structure summary rows (formula, energy, density, status, files)."),
]

PLAN_TOOLS = [
    _fn("archive_state", "Fresh archive state: grid occupancy, coverage, MOQD score, best voltage, recent results."),
    _fn("submit_plan", "Submit the plan for the next search round. It is validated before it runs.",
        {"operator_weights": {"type": "object", "description": f"weights for {list(OPERATORS)}",
                              "additionalProperties": {"type": "number"}},
         "dopants": {"type": "array", "items": {"type": "string"}},
         "target_cells": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}},
                          "description": "grid cells [x, y] to explore around (empty or sparse regions)"},
         "rationale": {"type": "string"}},
        ["operator_weights", "rationale"]),
    _fn("ask_human", "Ask the human supervisor before a risky or uncertain step (changing descriptor bounds, "
        "adding dopants, ignoring a failed check, or when tool results conflict). The search waits for the answer.",
        {"question": {"type": "string"}, "proposed_action": {"type": "string"},
         "reason": {"type": "string", "description": "why this needs a human, citing tool results"}},
        ["question", "proposed_action", "reason"]),
]


_STDIN: queue.Queue[str] = queue.Queue()
_stdin_reader: list[threading.Thread] = []


def _start_stdin_reader() -> None:
    """One reader for the whole process, so an unanswered question can never swallow the answer to the next."""
    def read():
        for line in sys.stdin:
            _STDIN.put(line)
    if not _stdin_reader:
        _stdin_reader.append(threading.Thread(target=read, daemon=True))
        _stdin_reader[0].start()


def console_supervisor(question: str, proposed_action: str, reason: str) -> dict:
    """Default supervisor: asks in the terminal. The UI passes its own (approve/reject buttons)."""
    print(f"\n[AGENT ASKS] {question}\n  proposed: {proposed_action}\n  why: {reason}", flush=True)
    while not _STDIN.empty():  # anything typed before this question was not an answer to it
        _STDIN.get_nowait()
    _start_stdin_reader()
    print("  approve within 4 min? [y/n/comment]: ", end="", flush=True)
    try:  # nobody at the keyboard must not freeze the watcher
        a = _STDIN.get(timeout=240).strip()
    except queue.Empty:
        return {"approved": False, "comment": "no answer in the terminal within 4 minutes", "channel": "console"}
    approved = a.lower() in ("y", "yes")
    return {"approved": approved, "comment": "" if a.lower() in ("y", "yes", "n", "no") else a, "channel": "console"}


def _run_tool_loop(c, model: str, messages: list, tool_specs: list, handlers: dict[str, Callable],
                   max_turns: int = 8, on_event: Callable[[dict], None] | None = None, label: str = "",
                   stop: Callable[[], bool] | None = None) -> tuple[str, list]:
    trace = []
    max_tokens = int(os.environ.get("AGENT_MAX_TOKENS", "4096"))
    for turn in range(max_turns):
        msg = chat(c, f"{label} t{turn}", model=model, messages=messages, tools=tool_specs, temperature=0.2,
                   max_tokens=max_tokens)
        # Send back only role/content/tool_calls: servers reject extra fields such as reasoning_content.
        am = {"role": "assistant", "content": strip_thinking(msg["content"])}
        if msg["tool_calls"]:
            am["tool_calls"] = msg["tool_calls"]
        messages.append(am)
        if not msg["tool_calls"]:
            return strip_thinking(msg["content"]), trace
        for call in msg["tool_calls"]:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"]["arguments"] or "{}")
                if not isinstance(args, dict):
                    raise ValueError("arguments must be a JSON object")
            except (json.JSONDecodeError, ValueError) as e:
                args, result = {}, {"error": f"could not parse arguments: {e}"}
            else:
                h = handlers.get(name)
                try:
                    result = h(**args) if h else {"error": f"unknown tool {name}; available {list(handlers)}"}
                except TypeError as e:
                    result = {"error": f"bad arguments for {name}: {e}"}
                except Exception as e:  # a tool failure goes back to the model, it never kills the run
                    result = {"error": f"{type(e).__name__}: {e}"}
            ev = {"tool": name, "args": args, "result": result}
            trace.append(ev)
            if on_event:
                on_event(ev)
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": json.dumps(result, default=str)[:12000]})
        if stop and stop():
            return "", trace
    return "(stopped after max tool turns)", trace


# ---------------------------------------------------------------- audit
def audit(records: list[StructureRecord], on_event=None) -> dict:
    c = client()
    ultra, _ = pick_models(c)
    handlers = {
        "run_overview": lambda: tools.run_overview(records),
        "integrity_checks": lambda: tools.integrity_checks(records),
        "bounds_advice": lambda descriptor, min, max, bins=10: tools.bounds_advice(
            list(tools.descriptor_values(records, descriptor).values()), float(min), float(max), int(bins)),
        "list_structures": lambda: [r.summary() for r in records],
    }
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Audit this run before we use it to steer the search. The team's "
                 "MOME config uses density bounds 0-5 g/cm3 with 10 bins. Report: what the run contains, "
                 "which results should not be trusted and why (with evidence), whether the descriptor "
                 "bounds waste the grid, and what to fix first. Use the tools, then write the report in "
                 "under 300 words."}]
    report, trace = _run_tool_loop(c, ultra, messages, AUDIT_TOOLS, handlers, on_event=on_event, label="audit")
    return {"model": ultra, "report": report, "trace": trace}


# ---------------------------------------------------------------- planner for the search


def make_planner(on_event=None, supervisor: Callable[..., dict] | None = None):
    """Return a planner callable for search.Search(arm='agent').

    supervisor(question, proposed_action, reason) -> {"approved": bool, "comment": str}.
    None = no human in the loop (benchmark runs): ask_human is answered "not approved, use a safe default".
    Network hiccups fall back to the frozen policy for that round (logged); a bad key or a spent budget stops the run.
    """
    from .search import Plan

    c = client()
    ultra, _ = pick_models(c)

    def planner(search) -> Plan:
        submitted: dict = {}
        allow_sub = search.hull is not None

        def archive_state():
            occ = search.grid.occupied()
            recent = search.history[-search.cfg.batch * 2:]
            m = search.metrics()
            return {"round": search.round, "evaluations": search.evaluations, "budget": search.cfg.budget,
                    "grid": list(search.cfg.bins), "descriptors": [asdict(d) for d in search.cfg.descriptors],
                    **{k: m[k] for k in ("coverage", "occupied_cells", "moqd_score", "best_voltage",
                                         "duplicates", "unique_valid")},
                    "occupied": [{"cell": list(c), "n": len(search.grid.cells[c]),
                                  "best_voltage": round(max(-e.objectives[1] for e in search.grid.cells[c].entries), 3)}
                                 for c in occ],
                    "recent": [{k: e[k] for k in ("operator", "detail", "cell", "accepted", "reason", "voltage")}
                               for e in recent],
                    "previous_plans": [{k: p.get(k) for k in ("round", "operator_weights", "target_cells", "rationale")}
                                       for p in search.plans[-2:] if "human_review" not in p],
                    "allowed_operators": [o for o in OPERATORS if allow_sub or o != "substitute"]}

        def ask_human(question, proposed_action, reason):
            if supervisor is None:
                ans = {"approved": False, "comment": "no supervisor in this run; choose a conservative plan"}
            else:
                ans = supervisor(question=question, proposed_action=proposed_action, reason=reason)
            search.plans.append({"round": search.round, "human_review": {"question": question,
                                 "proposed_action": proposed_action, "reason": reason, **ans}})
            return ans

        def submit_plan(**plan):
            ok, errors, clean = validate_plan(plan, search.cfg.bins, allow_substitute=allow_sub)
            if ok:
                submitted.update(clean)
                return {"accepted": True, "plan": clean}
            return {"accepted": False, "errors": errors}

        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": "Plan the next round of the quality-diversity search. Current "
                     f"archive state:\n{json.dumps(archive_state(), default=str)}\n\nCall submit_plan now. Aim to "
                     "fill empty or sparse cells next to occupied ones, to cut duplicate structures (they waste "
                     "budget), and to find high-voltage, low-energy trade-offs. If submit_plan is rejected, fix it. "
                     "For risky or uncertain steps call ask_human first and follow the answer."}]
        fallback = dict(search.cfg.operator_weights)
        try:
            text, trace = _run_tool_loop(c, ultra, messages, PLAN_TOOLS,
                                         {"archive_state": archive_state, "submit_plan": submit_plan,
                                          "ask_human": ask_human},
                                         max_turns=4, on_event=on_event, label=f"plan r{search.round}",
                                         stop=lambda: bool(submitted))
        except Exception as e:
            if type(e).__name__ not in TRANSIENT:
                raise
            return Plan(operator_weights=fallback, rationale=f"[agent unavailable: {type(e).__name__}] frozen policy used")
        if not submitted:
            return Plan(operator_weights=fallback,
                        rationale=f"[no valid plan] frozen policy used. ({strip_thinking(text)[:200]})")
        return Plan(**submitted)

    return planner


def default_descriptors() -> tuple[Descriptor, Descriptor]:
    return Descriptor("density", 2.0, 5.5), Descriptor("cation_o_bond", 1.7, 2.3)
