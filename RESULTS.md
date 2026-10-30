# CathodePilot results record

Recorded 30 October 2026. Raw per-run data: [results/bench](results/bench). Model audit: [results/agent_audit.json](results/agent_audit.json).

## Experiment design

| Setting | Value |
|---|---|
| Starting structures | 5 NMC111 seeds (Li3MnNiCoO6) from our XtalOpt run, the same for every strategy |
| Budget | 60 CHGNet evaluations per run, including the seeds |
| Repeats | 3 random seeds per strategy (0, 1, 2) |
| Archive | 10 x 10 grid: density 2.0-5.5 g/cm3, mean cation-oxygen bond 1.7-2.3 Å |
| Mutations | Frozen before the first run: strain, rattle, permute. No element swaps, because there is no convex hull yet |
| Duplicates | Detected with pymatgen StructureMatcher and rejected |
| Scoring | MOQD score, recomputed with one reference point shared by all runs |
| Hardware | NVIDIA RTX 3050 (6 GB), CHGNet v0.3.0 model |
| Agent model | NVIDIA Nemotron 3 Ultra on Nebius Token Factory |

## Ranking

Ranked by MOQD score, higher is better. Values are mean ± sd over 3 seeds.

| Rank | Strategy | MOQD score | Cells filled (of 100) | Unique valid | Duplicates (of 60) | Best voltage* (V) |
|---|---|---|---|---|---|---|
| 1 | Random parent | 14.08 ± 2.25 | 10.3 ± 2.1 | 32.3 ± 3.8 | 27.7 ± 3.8 | 4.31 ± 0.06 |
| 2 | Nemotron + MOME | 12.77 ± 2.81 | 9.0 ± 1.7 | 29.7 ± 4.2 | 30.3 ± 4.2 | 4.16 ± 0.10 |
| 3 | MOME | 12.62 ± 0.80 | 9.0 ± 1.0 | 25.0 ± 7.0 | 35.0 ± 7.0 | 4.18 ± 0.17 |
| 4 | Plain Pareto | 7.71 ± 4.48 | 5.3 ± 2.3 | 11.0 ± 3.6 | 49.0 ± 3.6 | 4.08 ± 0.24 |

*CHGNet voltage reads about 0.5 V low against the LiCoO2 literature value, so it is for ranking only.

**The order of ranks 1 to 3 is not statistically settled.** With 3 seeds their spreads overlap. Only plain Pareto is clearly behind on cells filled and unique structures.

## Per-seed results

| Strategy | Seed | MOQD score | Cells | Unique valid | Duplicates |
|---|---|---|---|---|---|
| Random | 0 / 1 / 2 | 14.79 / 15.89 / 11.56 | 11 / 12 / 8 | 28 / 35 / 34 | 32 / 25 / 26 |
| Nemotron + MOME | 0 / 1 / 2 | 14.13 / 9.54 / 14.64 | 10 / 7 / 10 | 33 / 25 / 31 | 27 / 35 / 29 |
| MOME | 0 / 1 / 2 | 12.33 / 13.53 / 12.01 | 9 / 10 / 8 | 25 / 32 / 18 | 35 / 28 / 42 |
| Plain Pareto | 0 / 1 / 2 | 5.21 / 12.88 / 5.04 | 4 / 8 / 4 | 8 / 15 / 10 | 52 / 45 / 50 |

## What we can and cannot claim

**Supported by the data**
- MOME beat plain Pareto selection: 1.7 times the cells and 2.3 times the unique structures.
- Duplicates are the biggest waste: 42 to 87% of evaluations across all runs.
- The agent reacted to evidence: on seed 0 it moved from mostly strain to mostly permute as it saw strain produce duplicates.

**Not supported**
- That MOME beats random parent choice. It did not.
- That Nemotron steering improves the search. It had about 5 fewer duplicates than MOME alone, but within the spread, and seed 1 was the weakest of its runs.
- That any candidate is a discovery. These are CHGNet surrogate results for DFT follow-up.

## Audit of our own data

Deterministic checks found 14 items across 18 structures:

| Finding | Count |
|---|---|
| Result files that look synthetic (13 files, reported as one finding) | 1 |
| Compositions not charge-balanced | 4 |
| Low-density outliers | 2 |
| GULP charges that differ from textbook values | 7 |

The team's density bounds of 0-5 g/cm3 reach only 4 of 10 bins on the observed data.

The Nemotron audit found the same main problems, but miscounted: it reported 5 charge-imbalanced structures where the checks found 4. The deterministic checks are the source of truth.

## Model cost

43 live Token Factory calls for the setup check, audit, all agent runs and Run Watch tests, about **$0.24**. This is our own estimate from token counts at assumed prices of $1 in and $3 out per million tokens. The account balance on Token Factory is the authoritative figure.

## Run Watch

Tested on a stand-in process that hangs on a child job, with real Nemotron: it detected the stall, saved evidence, proposed one action, asked for approval in the terminal, stopped only the named job, confirmed the run recovered, and wrote a report.

Not yet tried: a real XtalOpt run and a real CALL-E phone call.

## Limits

- 3 seeds and 60 evaluations are a small sample.
- One seed start was slowed by a heavily loaded laptop, which affects run time, not the search results.
- All structures come from one composition, because there is no Materials Project hull yet.
