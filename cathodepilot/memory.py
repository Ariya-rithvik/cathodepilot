"""Lab notebook: the coworker's memory of what we are doing.

Holds the goal, the current step, the team's ALWAYS / NEVER rules, free notes, every incident the
watcher has seen with what was done and whether it worked, and every run with why it was started and
its settings. The agent reads it before every decision. A past incident with the same signature is
offered as the known fix, and prestart_warnings() compares a new run with similar past runs.
Stored as JSON at data/memory/notebook.json (one file, easy to read and to edit by hand). The watcher
and the web app may write at the same time, so every change happens under a small lock file.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT = {
    "goal": "Run the XtalOpt / MOME cathode search and keep it healthy until the budget is spent.",
    "step": "",
    "rules": [
        {"kind": "never", "text": "Never kill a process that is not on the allowlist or not part of our run."},
        {"kind": "never", "text": "Never change mutation settings while a comparison is running."},
        {"kind": "always", "text": "Always ask the supervisor before stopping, restarting or killing anything."},
    ],
    "notes": [],
    "incidents": [],
    "runs": [],
}


def _path() -> Path:
    return Path(os.environ.get("CATHODEPILOT_NOTEBOOK") or ROOT / "data" / "memory" / "notebook.json")


@contextmanager
def _locked(timeout: float = 10.0):
    """Cross-process lock for read-modify-write. A lock left by a crashed writer expires after 30 s."""
    lock = _path().with_suffix(".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd, t0 = None, time.time()
    while fd is None and time.time() - t0 < timeout:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except OSError:  # held by another writer (PermissionError while it is being removed on Windows)
            try:
                if time.time() - lock.stat().st_mtime > 30:
                    lock.unlink()
                    continue
            except OSError:
                pass
            time.sleep(0.03)
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
            try:
                lock.unlink()
            except OSError:
                pass


def _retry_io(fn, tries: int = 40):
    """Windows refuses to replace a file another process is reading; those windows are short."""
    for i in range(tries):
        try:
            return fn()
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(0.05)


def load() -> dict:
    p = _path()
    if not p.exists():
        return json.loads(json.dumps(DEFAULT))
    try:
        nb = json.loads(_retry_io(lambda: p.read_text(encoding="utf-8-sig")))  # PowerShell and Notepad may add a BOM
        if not isinstance(nb, dict):
            raise ValueError("the notebook must be a JSON object")
    except ValueError as e:  # includes bad JSON and bad encoding. Reading never moves anything; see save()
        nb = json.loads(json.dumps(DEFAULT))
        nb["_error"] = f"{p.name} could not be read ({e}). Using defaults until it is fixed."
        return nb
    for k, v in DEFAULT.items():
        nb.setdefault(k, json.loads(json.dumps(v)))
    return nb


def save(nb: dict) -> None:
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    if nb.pop("_error", None) and p.exists():  # about to write over a damaged file: keep it, and keep saying so
        old = p.with_name(f"{p.stem}.corrupt-{int(time.time())}.json")
        _retry_io(lambda: p.replace(old))
        nb["recovered_from"] = old.name
    tmp = p.with_name(f"{p.stem}.{os.getpid()}.{uuid.uuid4().hex[:6]}.tmp")
    tmp.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    _retry_io(lambda: tmp.replace(p))


def notebook_problem(nb: dict) -> str | None:
    if nb.get("_error"):
        return nb["_error"]
    if nb.get("recovered_from"):
        return (f"The notebook was damaged and restarted from defaults. Your rules and history are in "
                f"{nb['recovered_from']}. Merge them back, then remove recovered_from from notebook.json.")
    return None


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def set_field(field: str, value: str) -> dict:
    with _locked():
        nb = load()
        nb[field] = value
        save(nb)
    return nb


def add_rule(kind: str, text: str) -> dict:
    if kind not in ("always", "never"):
        raise ValueError("kind must be 'always' or 'never'")
    with _locked():
        nb = load()
        nb["rules"].append({"kind": kind, "text": text.strip()[:300]})
        save(nb)
    return nb


def add_note(text: str, source: str = "human") -> dict:
    with _locked():
        nb = load()
        nb["notes"].append({"time": now(), "source": source, "text": text.strip()[:600]})
        nb["notes"] = nb["notes"][-200:]
        save(nb)
    return nb


def add_incident(signature: str, summary: str, action: str, approved: bool | None, outcome: str,
                 detail: dict | None = None) -> dict:
    inc = {"id": f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:4]}", "time": now(), "signature": signature,
           "summary": summary[:400], "action": action, "approved": approved, "outcome": outcome[:400],
           "detail": detail or {}}
    with _locked():
        nb = load()
        nb["incidents"].append(inc)
        nb["incidents"] = nb["incidents"][-300:]
        save(nb)
    return inc


def update_incident(inc_id: str, outcome: str) -> None:
    with _locked():
        nb = load()
        for i in nb["incidents"]:
            if i.get("id") == inc_id:
                i["outcome"] = outcome[:400]
                i["verified_at"] = now()
        save(nb)


def known_fixes(signature: str, nb: dict | None = None) -> list[dict]:
    """Past incidents with the same signature where a real action was approved and worked.
    Waiting is never a fix: otherwise one false alarm would teach the watcher to stay silent on real stalls."""
    nb = nb or load()
    return [i for i in nb["incidents"] if i["signature"] == signature and i.get("approved")
            and i["action"] in ("kill_stuck_child", "restart_run", "free_ram")
            and str(i.get("outcome", "")).startswith("resolved")][-3:]


def as_prompt(nb: dict | None = None, incidents: int = 8) -> str:
    nb = nb or load()
    rules = "\n".join(f"- {r['kind'].upper()}: {r['text']}" for r in nb["rules"]) or "- (none)"
    notes = "\n".join(f"- [{n['time']}] {n['text']}" for n in nb["notes"][-6:]) or "- (none)"
    inc = "\n".join(f"- [{i['time']}] {i['signature']}: {i['summary']} -> {i['action']} "
                    f"({'approved' if i['approved'] else 'not approved'}) -> {i['outcome']}"
                    for i in nb["incidents"][-incidents:]) or "- (none)"
    return (f"GOAL: {nb['goal']}\nCURRENT STEP: {nb['step'] or '(not set)'}\nRULES:\n{rules}\n"
            f"RECENT NOTES:\n{notes}\nPAST INCIDENTS:\n{inc}")


# ---------------------------------------------------------------- runs: why, settings, how they ended
def _norm(path: str | None) -> str | None:
    return os.path.normcase(str(Path(path).resolve())) if path else None


_SECRET = re.compile(r"(?i)^(.*(?:password|passwd|secret|token|api[_-]?key)[^=:]*[=:]).*$", re.M)


def read_settings(paths: list[str] | None, max_bytes: int = 20000) -> dict:
    """Snapshot of settings files, so a later run can be compared with this one. Secret values are never kept."""
    out = {}
    for path in (paths or [])[:5]:
        p = Path(path)
        raw = p.read_bytes()
        name = p.name if p.name not in out else f"{p.parent.name}/{p.name}"
        out[name] = {"path": str(p), "sha256": hashlib.sha256(raw).hexdigest()[:16], "bytes": len(raw),
                     "text": _SECRET.sub(r"\1 ***", raw[:max_bytes].decode("utf-8", "replace"))}
    return out


def settings_diff(old: dict | None, new: dict | None, max_lines: int = 12) -> list[str]:
    """Changed lines between two settings snapshots, empty if identical."""
    old, new = old or {}, new or {}
    out = []
    for name in sorted(set(old) | set(new)):
        if name not in new:
            out.append(f"{name}: not given this time")
        elif name not in old:
            out.append(f"{name}: not recorded last time")
        elif old[name]["sha256"] != new[name]["sha256"]:
            found = [line for line in difflib.unified_diff(old[name]["text"].splitlines(), new[name]["text"].splitlines(),
                                                           lineterm="", n=0)
                     if line[:1] in "+-" and line[:3] not in ("+++", "---")]
            out += [f"{name}: {line[:160]}" for line in found] or [f"{name}: changed, in a part not kept in the notebook"]
    return out[:max_lines]


def start_run(name: str, cmd: list[str] | None = None, xtal_dir: str | None = None, log: str | None = None,
              why: str = "", settings: dict | None = None, pid: int | None = None) -> dict:
    run = {"id": f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:4]}", "name": name, "started": now(), "ended": None,
           "why": why[:400], "cmd": " ".join(cmd) if cmd else None, "xtal_dir": _norm(xtal_dir), "log": log,
           "pid": pid, "settings": settings or {}, "outcome": "running"}
    with _locked():
        nb = load()
        nb["runs"].append(run)
        nb["runs"] = nb["runs"][-60:]
        save(nb)
    return run


def finish_run(run_id: str, **fields) -> None:
    with _locked():
        nb = load()
        for r in nb["runs"]:
            if r["id"] == run_id:
                r.update(fields, ended=now())
        save(nb)


def update_run(run_id: str, **fields) -> None:
    with _locked():
        nb = load()
        for r in nb["runs"]:
            if r["id"] == run_id:
                r.update(fields)
        save(nb)


def running_copies(name: str, cmd: list[str] | None = None, xtal_dir: str | None = None) -> list[dict]:
    """Earlier runs of the same thing whose process is still alive: starting another copy would mix their outputs."""
    import psutil

    out = []
    for r in similar_runs(name, cmd, xtal_dir):
        try:
            if r.get("pid") and r.get("create_time") \
                    and abs(psutil.Process(r["pid"]).create_time() - r["create_time"]) < 1:
                out.append(r)
        except psutil.Error:
            pass
    return out


def similar_runs(name: str, cmd: list[str] | None = None, xtal_dir: str | None = None,
                 nb: dict | None = None) -> list[dict]:
    nb = nb or load()
    c, x = " ".join(cmd) if cmd else None, _norm(xtal_dir)
    return [r for r in nb["runs"] if r["name"] == name or (c and r.get("cmd") == c) or (x and r.get("xtal_dir") == x)]


def prestart_warnings(name: str, cmd: list[str] | None = None, xtal_dir: str | None = None,
                      settings: dict | None = None, nb: dict | None = None) -> list[dict]:
    """What similar past runs teach us, before this one starts. Free: no model involved."""
    nb = nb or load()
    out: list[dict] = []
    if notebook_problem(nb):
        out.append({"level": "warn", "text": notebook_problem(nb)})
    past = similar_runs(name, cmd, xtal_dir, nb)
    if past:
        last = past[-1]
        gen = f", last generation {last['generation']}" if last.get("generation") else ""
        if last.get("ended") is None:
            out.append({"level": "warn", "text": f"The last similar run '{last['name']}' (started {last['started']}) "
                        f"never reported an end. The watcher or the computer may have stopped; check what it left "
                        f"behind before starting again."})
        else:
            out.append({"level": "info", "text": f"Last similar run '{last['name']}' started {last['started']}: "
                        f"{last.get('outcome')}{gen}."})
        ids = {r["id"] for r in past}
        by_sig: dict[str, list[dict]] = {}
        for i in nb["incidents"]:
            if i.get("detail", {}).get("run_id") in ids:
                by_sig.setdefault(i["signature"], []).append(i)
        for sig, items in by_sig.items():
            at = next((i["detail"].get("generation") for i in reversed(items) if i["detail"].get("generation")), None)
            where = f", last at generation {at}" if at else ""
            worked = sorted({i["action"] for i in items if i.get("approved")
                             and str(i.get("outcome", "")).startswith("resolved")})
            fix = f" What worked: {', '.join(worked)}." if worked else " Nothing is recorded as having fixed it."
            out.append({"level": "warn", "text": f"{sig.replace('_', ' ')} happened {len(items)} time(s) in similar "
                        f"runs{where}.{fix}"})
        diff = settings_diff(last.get("settings"), settings) if settings is not None and last.get("settings") else None
        if diff:
            out.append({"level": "info", "text": "Settings changed since the last similar run: " + "; ".join(diff)})
        elif diff == [] and by_sig:
            out.append({"level": "warn", "text": "Settings are identical to the run that had these problems, "
                                                 "so the same problems may happen again."})
    for r in nb["rules"]:
        if r["kind"] == "never":
            out.append({"level": "rule", "text": f"Rule: {r['text']}"})
    return out
