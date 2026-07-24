# Reproducing the reported results

All results in the manuscript come from a single production run
(`RUN_TAG = flux_full_20260720_v1`) executed with the code in this repository.
The design is paired-seed deterministic: the same fifty seed identifiers recur
across every condition, so all reported numbers reproduce exactly. The
production run used the driver defaults — `BASE_SEED = 20260425` with
`SUITE_SEED_MODE = "paired"` — so run *i* of every condition uses seed
`20260425 + i`, i.e., seeds 20260426–20260475 for the fifty paired markets.
Running any driver with its defaults therefore reproduces the reported
numbers exactly.

## Analysis families → drivers

| Manuscript exhibit | Driver script | Primary outputs |
|---|---|---|
| Tables 4–6, Figures 2–4; Appendix Tables B1, B3 | `scripts/run_robustness_suite.jl` | `robustness_suite_per_run.csv`, `robustness_suite_summary.csv`, `robustness_suite_paired_treatment_effects.csv`, paradox scorecards |
| Table 7, Figure 5 (endogenous adoption) | `scripts/run_robustness_suite.jl` (adoption preset; per-run telemetry under `adoption/`) | `robustness_suite_per_run.csv` (adoption cells), `adoption/` trajectories |
| Appendix Table B2, Appendix Figure B2 (33-cell capacity grid) | `scripts/run_capacity_identification_sweep.jl` | capacity-identification per-run and summary files |
| Appendix B5 (performed demand & strategic probes), Appendix Figure B3 | `scripts/run_posthoc_strategic_counterfactuals.jl` | post-hoc condition per-run and summary files |
| Baseline ledger diagnostics (Table 4 Panel B; Appendix B4) | `scripts/run_mixed_tier_analysis_full.jl` | per-investment ledger and opportunity-set probes |
| Knightian-weight sensitivity | `scripts/run_knightian_weight_family_sweep.jl` | weight-family sweep summaries |
| Decorrelation / paradox-strength exhibits | `scripts/build_decorrelation_exhibit.jl`, `scripts/build_paradox_exhibits.jl` | derived exhibit tables from the suite CSVs |

Population size, run count, and horizon are set with `N_AGENTS`, `N_RUNS`, and
`N_ROUNDS`; condition subsets with `CONDITIONS=...`; presets with
`SUITE_PRESET=...` (see the header of each driver).

## Run manifests

`manifests/` contains the condition manifests of the production run — one row
per experimental condition with its design, preset, and seed mode — for the
robustness suite (83 fixed-tier conditions), the adoption suite (14
conditions), the 33-cell capacity-identification sweep, and the 9 strategic
post-hoc conditions, together with `provenance_validation.csv`, the post-run
validation that every suite's recorded run tag, commit, and parameter state
matched the launched configuration. The commit hashes recorded there refer to
the internal research repository from which the production run was launched;
the engine code at that commit is mirrored exactly by this repository.

## Supplementary appendices

`docs/supplementary_appendices.docx` contains the full supplementary
appendices accompanying the manuscript: Appendix A (agent trait profile,
domain-resolved AI capability, production design and calibration, the model
formalization, and market regimes) and Appendix B (the complete results
supplement, including the full 83-condition survival battery, the
capacity-identification sweep, the full outcome-family estimates, and the
performative-value and strategic-targeting analyses).

## Tests

```bash
julia --project=. --threads=auto test/runtests.jl
```

The regression suite (1,728 tests) covers configuration and calibration,
market-condition schemas, agent decision paths, opportunity evaluation,
crowding dynamics, the four-dimensional uncertainty accounting, the paradox
scorecard, engine invariants, and I/O round-trips.
