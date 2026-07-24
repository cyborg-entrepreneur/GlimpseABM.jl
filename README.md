# GlimpseABM.jl

Julia implementation of **GlimpseABM**, an agent-based model of AI-augmented
entrepreneurial decision-making under Knightian uncertainty.

The model operationalizes the four dimensions of Knightian uncertainty from
Townsend et al. (2025, *Academy of Management Review*):

1. **Actor ignorance** — information gaps about current states (first-order).
2. **Practical indeterminism** — unpredictable execution outcomes.
3. **Agentic novelty** — genuinely new possibilities created by action.
4. **Competitive recursion** — strategic interdependence among actors.

AI is modeled as a set of capability/cost **tiers** (`none`, `basic`,
`advanced`, `premium`/`frontier_ai`). These are abstract treatment conditions
and adaptive adoption choices — not predictions about any specific system, and
not a claim that the frontier tier is full AGI. The central object of study is
the **paradox of future knowledge**: AI compresses first-order actor ignorance
while shifting uncertainty downstream into the second-order dimensions, so that
correlated, foresight-driven commitment can become self-undermining.

## Requirements

- **Julia ≥ 1.9** (the package targets 1.9; see `Project.toml`).
- The checked-in `Manifest.toml` was resolved under **Julia 1.12.3**. For an
  exact reproduction, use Julia 1.12.x.

## Setup

From the repository root:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

On a Julia version other than 1.12.x, if instantiation reports a manifest
mismatch, resolve dependencies fresh against the compatibility bounds in
`Project.toml`:

```bash
julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
```

## Tests

```bash
julia --project=. --threads=auto test/runtests.jl
```

The regression suite covers configuration and calibration, market-condition
schemas, agent decision paths, opportunity evaluation, crowding dynamics,
confidence-based sizing, the four-dimensional uncertainty accounting, the
paradox scorecard, and I/O round-trips.

## Reproducing the paper

See [REPRODUCING.md](REPRODUCING.md) for the mapping from each manuscript exhibit to its driver script, the production-run manifests (`manifests/`), and the supplementary appendices (`docs/supplementary_appendices.docx`).

## Experiments

All scripts run with `--project=.` and write generated outputs under `results/`
(git-ignored). Population size, run count, and horizon are configurable through
the `N_AGENTS`, `N_RUNS`, and `N_ROUNDS` environment variables.

```bash
# Primary mixed-tier analysis: survival, returns, the four Knightian
# dimensions, and the paradox-of-future-knowledge scorecard.
julia --project=. --threads=auto scripts/run_mixed_tier_analysis_full.jl

# Robustness & sensitivity suite — placebo, mechanism-decomposition,
# alternative-specification, and boundary-condition checks.
julia --project=. --threads=auto scripts/run_robustness_suite.jl --list   # list conditions
julia --project=. --threads=auto scripts/run_robustness_suite.jl          # run the suite

# Sensitivity sweep over the Knightian perception-weight family.
julia --project=. --threads=auto scripts/run_knightian_weight_family_sweep.jl

# Build paradox-centered tables (T1-T2) and figures (F1-F5) from a
# robustness-suite results directory (auto-picks the newest by default).
julia --project=. scripts/build_paradox_exhibits.jl --results results/robustness_<timestamp>
```

## Package layout

```text
src/
  GlimpseABM.jl          # Module entry point and exports
  config.jl              # Model configuration and empirical calibration
  models.jl              # Core data structures and realized returns
  agents.jl              # Agent resources, decisions, learning, outcomes
  information.jl         # AI/human information generation
  knowledge.jl           # Knowledge signals and perception
  innovation.jl          # Innovation and recombination mechanics
  market.jl              # Market environment and opportunities
  market_conditions.jl   # Market-condition schema and dynamics
  uncertainty.jl         # Four-dimensional Knightian uncertainty tracking
  simulation.jl          # Simulation orchestration
  action_keys.jl         # Canonical action/metric key definitions
  numpy_rng.jl           # Optional deterministic MT19937 stream
  utils.jl               # Shared helpers
  io.jl                  # Result serialization (CSV / JLD2 / Arrow)

scripts/
  run_mixed_tier_analysis_full.jl     # Primary mixed-tier analysis
  run_robustness_suite.jl             # Robustness & sensitivity suite
  run_knightian_weight_family_sweep.jl# Perception-weight sensitivity sweep
  build_paradox_exhibits.jl           # Tables and figures from suite output
  _safe_stats.jl, _launch_metadata.jl # Shared script helpers

test/
  runtests.jl            # Full regression suite
```

## Citation

If you use this software in academic research, please cite:

> Townsend, D. M., Hunt, R. A., Rady, J., Manocha, P., & Jin, J-H. (2025).
> Are the Futures Computable? Knightian Uncertainty & Artificial Intelligence.
> *The Academy of Management Review*, 50(2), 415–440.

## License

MIT — see [LICENSE](LICENSE).
