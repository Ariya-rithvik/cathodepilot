"""Watch a long run and keep it healthy. Calls the supervisor before any action.

Watch an XtalOpt run that is already going (attach by PID, progress = structures in the run folder):
  .venv/Scripts/python scripts/watch.py --name xtal-oct --xtal-dir D:/xtalopt/run1 --pid 12345 --stall-min 30 \
      --why "NMC111 baseline, 60 generations" --settings D:/xtalopt/run1/xtalopt.state

Launch our benchmark under the watcher (it can then restart it, after a verified backup):
  .venv/Scripts/python scripts/watch.py --name bench --log results/bench/agent_log.txt --progress-regex "cell=" \
      --protect results/bench --stall-min 10 --console -- .venv/Scripts/python scripts/bench.py --arms agent --seeds 0 1 2

Only processes named in --job-names (default: gulp) can ever be stopped as a stuck job. A restart needs a data
folder to back up (--xtal-dir or --protect); relative paths mean relative to the project folder.
Before starting, it prints what similar past runs teach us (free, no model). If there are warnings and --console
is on, it asks before launching. Supervisor: CALL-E phone call when CALLE_API_KEY is set and DRY_RUN=0; --console
asks in this terminal; otherwise nothing that needs approval is executed (it only records and waits).
Nemotron diagnoses only when a problem needs a decision; with no Nebius key the notebook's known fixes and default
rules are used instead. When the run ends, or on Ctrl-C, a report is written to data/memory/reports.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cathodepilot import agent, memory  # noqa: E402
from cathodepilot.watch import FAILURE_REGEX, WatchConfig, Watcher, _slug  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="run")
    ap.add_argument("--pid", type=int)
    ap.add_argument("--xtal-dir")
    ap.add_argument("--log")
    ap.add_argument("--progress-regex", default=r"\S", help="log lines that count as progress; set this for real runs")
    ap.add_argument("--job-names", nargs="*", default=["gulp"],
                    help="only processes with these words in their name can be called a stuck job")
    ap.add_argument("--failure-regex", default=FAILURE_REGEX, help="log lines that count as a failed job")
    ap.add_argument("--failure-alert", type=int, default=5, help="alert after this many new failed jobs")
    ap.add_argument("--stall-min", type=float, default=30)
    ap.add_argument("--interval", type=float, default=60)
    ap.add_argument("--cooldown-min", type=float, default=30)
    ap.add_argument("--killable", nargs="*", default=[], help="process names free_ram may close, e.g. chrome.exe")
    ap.add_argument("--auto-known", action="store_true",
                    help="stop a hung job or restart without asking if the same fix worked twice before")
    ap.add_argument("--console", action="store_true", help="ask for approval in this terminal")
    ap.add_argument("--why", default="", help="why this run was started; kept in the notebook and the report")
    ap.add_argument("--settings", nargs="*", default=[], help="settings files to remember and compare with past runs")
    ap.add_argument("--protect", nargs="*", default=[], help="more files or folders to back up before a restart")
    ap.add_argument("--backup-max-gb", type=float, default=5, help="refuse to restart if the backup would be bigger")
    ap.add_argument("--step", help="set the notebook's current step")
    ap.add_argument("--check", action="store_true", help="only print what past runs say about this one, then exit")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- command to launch and own")
    a = ap.parse_args()
    agent.load_env()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if a.pid and cmd:
        ap.error("give --pid to attach to a running process, or a command to launch, not both")
    if not (a.pid or cmd or a.xtal_dir or a.log or a.check):
        ap.error("nothing to watch: give a command after --, or --pid, --xtal-dir or --log")
    if a.xtal_dir and not Watcher._abs(a.xtal_dir).exists():
        ap.error(f"--xtal-dir does not exist: {a.xtal_dir}")
    if a.pid and not (a.xtal_dir or a.log):
        ap.error("--pid needs --xtal-dir or --log, or stalls cannot be measured")
    if cmd and not a.log:
        a.log = f"data/memory/logs/{_slug(a.name)}.log"
        print(f"run output goes to {a.log}")
    if cmd and a.progress_regex == r"\S":
        print("note: every line of output counts as progress; set --progress-regex to something that means real progress")
    missing = [p for p in a.protect if not Watcher._abs(p).exists()]
    if missing:
        ap.error(f"--protect path does not exist: {missing}")
    if a.step:
        memory.set_field("step", a.step)

    damaged = memory.load().get("_error")
    if damaged and not a.check:
        sys.exit(f"Not started: {damaged} Your rules and past fixes would be missing. Fix the file first.")
    settings = memory.read_settings(a.settings)
    warnings = memory.prestart_warnings(a.name, cmd or None, a.xtal_dir, settings if a.settings else None)
    if warnings:
        print("\nBefore this run, from the notebook:")
        for w in warnings:
            print(f"  [{w['level'].upper()}] {w['text']}")
        print()
    if a.check:
        return
    alive = memory.running_copies(a.name, cmd or None, a.xtal_dir)
    if alive and cmd:
        sys.exit(f"Not started: a similar run is still alive (pid {alive[-1]['pid']}). Two copies would mix their outputs.")
    if a.console and cmd and any(w["level"] == "warn" for w in warnings):
        if input("Start anyway? [y/n]: ").strip().lower() not in ("y", "yes"):
            print("Not started.")
            return

    sup = None
    if os.environ.get("CALLE_API_KEY") and os.environ.get("DRY_RUN", "1") == "0":
        from cathodepilot.phone import phone_supervisor
        sup = phone_supervisor()
        to, allowed = os.environ.get("SUPERVISOR_PHONE", ""), os.environ.get("CALLE_ALLOWED_NUMBERS", "")
        print("supervisor: CALL-E phone")
        if not to or to.replace(" ", "") not in allowed.replace(" ", ""):
            print("WARNING: SUPERVISOR_PHONE is not in CALLE_ALLOWED_NUMBERS, so no call can be placed")
    elif a.console:
        sup = agent.console_supervisor
        print("supervisor: this terminal")
    else:
        print("supervisor: none (actions that need approval will not run)")

    client = None
    if os.environ.get("NEBIUS_API_KEY"):
        try:
            c = agent.client()
            ultra, nano = agent.pick_models(c)
            client = (c, ultra, nano)
            print(f"diagnosis: {ultra}, report summary: {nano} (only when needed)")
        except Exception as e:  # noqa: BLE001  a model outage must not stop the watch
            print(f"diagnosis: notebook + default rules (model unreachable: {type(e).__name__})")
    else:
        print("diagnosis: notebook + default rules (no Nebius key)")

    cfg = WatchConfig(name=a.name, cmd=cmd or None, pid=a.pid, log=a.log, progress_regex=a.progress_regex,
                      failure_regex=a.failure_regex, failure_alert=a.failure_alert, xtal_dir=a.xtal_dir,
                      stall_minutes=a.stall_min, interval_s=a.interval, cooldown_minutes=a.cooldown_min,
                      killable=a.killable, auto_known=a.auto_known, protect_paths=a.protect,
                      backup_max_gb=a.backup_max_gb, job_names=a.job_names,
                      max_calls=int(os.environ.get("CALLE_MAX_CALLS", "4")))
    w = Watcher(cfg, supervisor=sup, planner_client=client)
    w.begin(why=a.why, settings=settings, warnings=warnings)
    try:
        w.start()
    except OSError as e:
        w.outcome = f"could not start the command: {e}"
        w.finish()
        sys.exit(w.outcome)
    try:
        w.run()
    finally:  # Ctrl-C still leaves a report; the run was started in its own group, so it keeps going
        w.finish("watch stopped by the user")
    if w.report_path:
        print(f"\nReport: {w.report_path}")


if __name__ == "__main__":
    main()
