# CathodePilot

An AI research partner for battery-materials search. It audits a crystal-structure search run, tests search strategies in a controlled experiment, lets a Nemotron agent steer the next round with a human approving risky steps, and watches long runs so a hung job does not cost you a night.

Built for the Nebius x NVIDIA Global AI Hackathon, Best Apps and Agents track. MIT licensed.

## What it does

| Page | What you get |
|---|---|
| **Audit** | Deterministic checks on real XtalOpt / GULP runs. Every finding opens the exact file and line. |
| **Experiment** | Random vs plain Pareto vs MOME, same seeds, budget and frozen mutation settings, 3 random seeds each, mean ± sd. |
| **Agent** | Nemotron reads results through tools and proposes a plan. A validator checks every plan before it runs. |
| **Shortlist** | Ranked leads with energy and voltage. These are CHGNet surrogate results, for DFT follow-up, not discoveries. |
| **Run Watch** | A coworker for long runs: detects stalls and failures, saves evidence, phones you before acting, checks the fix, remembers it, warns the next similar run, and writes a report. |

## Where NVIDIA and Nebius are used

- **NVIDIA Nemotron on Nebius Token Factory** does all the reasoning: the audit report, search plans, Run Watch diagnosis and the report summary. Ultra plans, Nano handles short calls. The client is OpenAI-compatible, in [cathodepilot/agent.py](cathodepilot/agent.py).
- **CHGNet on an NVIDIA GPU** relaxes and scores every candidate.
- **Credit guard:** every model call goes through one function with a disk cache and a token budget, and the app shows what has been spent.

## Setup

Python 3.12. For an NVIDIA GPU, install a CUDA build of torch first:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv/Scripts/python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add `NEBIUS_API_KEY`. Keep `.env` out of git. Then check the setup; this makes 2 short model calls:

```bash
.venv/Scripts/python check_setup.py
```

## Run

The app, at http://localhost:8000:

```bash
.venv/Scripts/python -m uvicorn app.server:app --port 8000
```

The experiment. The results in `results/bench` are the ones we measured; finished runs are skipped:

```bash
.venv/Scripts/python scripts/bench.py --arms random pareto mome agent --seeds 0 1 2 --budget 60
```

Tests, and a full rehearsal against a local fake model that spends no credits:

```bash
.venv/Scripts/python -m pytest -q tests
.venv/Scripts/python scripts/rehearse.py
```

For a hosted demo set `CATHODEPILOT_PUBLIC=1`. The app is then read-only and shows no machine details.

## Run Watch

Attach to an XtalOpt run that is already going:

```bash
.venv/Scripts/python scripts/watch.py --name xtal --xtal-dir D:/xtalopt/run1 --pid 12345 --stall-min 30 --why "NMC111 baseline" --console
```

Launch a run under the watcher, which can then also restart it:

```bash
.venv/Scripts/python scripts/watch.py --name bench --log results/bench/agent_log.txt --progress-regex "cell=" --protect results/bench --stall-min 10 --console -- .venv/Scripts/python scripts/bench.py --arms agent --seeds 0 1 2
```

Use `--check` with the same name to see only what past runs say. Approval comes from the terminal with `--console`, or by CALL-E phone call when `CALLE_API_KEY` is set and `DRY_RUN=0`.

Safety rules in the code:

- Only processes named in `--job-names` (default: gulp) can be stopped as a stuck job, and only the jobs named to you on the call.
- A restart needs a data folder to back up. If the hash-checked backup fails, nothing is stopped.
- Experiment files are never deleted.
- If the run recovers or ends while you are being asked, nothing is touched.
- Waiting is never learned as a fix.

Known limits: it watches this computer only, not a remote server. A job started through `start /b` before the watcher first sees it cannot be tracked. Real XtalOpt `structure.state` values other than the ones listed in `watch.py` count as unknown.

## Honest results

From our real runs, 3 seeds each, 60 evaluations: MOME filled 9.0 ± 1.0 cells against 5.3 ± 2.3 for plain Pareto, and it did **not** beat random parent choice (10.3 ± 2.1). 47 to 82% of each budget re-found known structures. CHGNet voltages read about 0.5 V low against the LiCoO2 literature value, so they are for ranking only.
