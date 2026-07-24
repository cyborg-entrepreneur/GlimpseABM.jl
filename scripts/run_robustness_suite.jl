#!/usr/bin/env julia
#
# Manuscript robustness suite for the paradox of future knowledge.
#
# This is the curated robustness suite used for manuscript tables.
#
# PRIMARY vs ROBUSTNESS DESIGNS: the paper's primary models are the mixed
# fixed-tier (fixed_mixed) conditions — N=1000, exactly 250 agents per tier
# (none/basic/advanced/premium), tiers locked for the whole run via
# AGENT_AI_MODE="fixed" + fixed_ai_level (see scripts/_fixed_tier_assignment.jl).
# These feed the causal IUT machinery. Emergent-adoption conditions are
# robustness/descriptive checks only and are excluded from the causal IUT
# machinery.
#
# Conditions are organized around four theory-driven categories:
#   1. internal_validity_placebo
#   2. mechanism_decomposition
#   3. alternative_frontier_ai_specification
#   4. boundary_conditions_generalizability
#
# Statistical design is cross-cutting: by default every condition reuses the
# same seed sequence as baseline so within-run treatment effects and
# condition-vs-baseline deltas are paired.
#
# Usage:
#   julia --threads=auto --project=. scripts/run_robustness_suite.jl --list
#   SUITE_PRESET=minimal julia --threads=auto --project=. scripts/run_robustness_suite.jl
#   N_AGENTS=2000 N_RUNS=50 SUITE_PRESET=core julia --threads=auto --project=. scripts/run_robustness_suite.jl
#   CONDITIONS=BASELINE,NO_CROWDING N_AGENTS=64 N_RUNS=2 N_ROUNDS=6 julia --project=. scripts/run_robustness_suite.jl

# Activate the project environment only when run as a standalone script.
# test_robustness_suite_wiring.jl include()s this file inside a module for
# harness access; the Pkg.test sandbox has no Pkg stdlib on the load path, so
# includes must not switch the active environment mid-suite.
if abspath(PROGRAM_FILE) == @__FILE__
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

using GlimpseABM
using Statistics
using Random
using DataFrames
using CSV
using Dates
using Printf

include(joinpath(@__DIR__, "_safe_stats.jl"))
include(joinpath(@__DIR__, "_launch_metadata.jl"))
# Provides AI_TIERS, balanced_tier_assignments, apply_balanced_fixed_tiers!
include(joinpath(@__DIR__, "_fixed_tier_assignment.jl"))
const TIER_RANK = Dict("none" => 1, "basic" => 2, "advanced" => 3, "premium" => 4)
const EMERGENT_UNCERTAINTY_DIMENSIONS = [
    "actor_ignorance",
    "practical_indeterminism",
    "agentic_novelty",
    "competitive_recursion",
]
const PERCEPTION_TELEMETRY_DIMENSIONS = EMERGENT_UNCERTAINTY_DIMENSIONS
const PERCEPTION_TELEMETRY_FIELDS = [
    "bounded_level",
    "normalized_score",
    "raw_score",
    "local_raw",
    "prior",
    "gap_vs_prior",
    "bound_hit",
]
const TIER_PERCEPTION_KEYS = [
    "perceived_$(dim)_$(field)"
    for dim in PERCEPTION_TELEMETRY_DIMENSIONS
    for field in PERCEPTION_TELEMETRY_FIELDS
]
const MARKET_RECURSION_DIAGNOSTIC_KEYS = [
    "market_crowding_pressure",
    "market_opportunity_overlap",
    "market_investment_concentration",
    "market_opportunity_competition",
    "market_ai_herding_intensity",
    "market_ai_action_correlation",
    "market_combo_reuse_pressure",
]
# Outcome families reported together in the final ablation exhibit.  Keep the
# legacy survival-only paired file for downstream compatibility, and emit this
# long-form panel as its multivariate companion.  Scales are display units:
# thousands of dollars, counts per founder, and percentage points.
const PAIRED_OUTCOME_SPECS = [
    (
        outcome="terminal_net_worth",
        outcome_label="Terminal net worth",
        outcome_family="economic",
        source_column=:mean_net_worth,
        scale=1.0e-3,
        unit="thousand_usd_per_founder",
    ),
    (
        outcome="knowledge_combinations",
        outcome_label="New knowledge combinations",
        outcome_family="innovation",
        source_column=:combinations_per_agent,
        scale=1.0,
        unit="combinations_per_founder",
    ),
    (
        outcome="five_year_survival",
        outcome_label="Five-year survival",
        outcome_family="survival",
        source_column=:survival_rate,
        scale=100.0,
        unit="percentage_points",
    ),
]
const PARADOX_SCORECARD_KEYS = [
    "actor_ignorance_level",
    "all_agent_actor_ignorance_level",
    "survivor_actor_ignorance_level",
    "actor_ignorance_survivorship_gap",
    "actor_ignorance_observations",
    "first_order_knowledge_gain_vs_none",
    "first_order_knowledge_delta_band",
    "first_order_knowledge_effect_size",
    "visible_opportunity_gain_vs_none",
    "info_quality_gain_vs_none",
    "practical_indeterminism_level",
    "all_agent_practical_indeterminism_level",
    "survivor_practical_indeterminism_level",
    "practical_indeterminism_survivorship_gap",
    "practical_indeterminism_observations",
    "practical_indeterminism_delta_vs_none",
    "practical_indeterminism_delta_band",
    "practical_indeterminism_effect_size",
    "agentic_novelty_level",
    "all_agent_agentic_novelty_level",
    "survivor_agentic_novelty_level",
    "agentic_novelty_survivorship_gap",
    "agentic_novelty_observations",
    "agentic_novelty_delta_vs_none",
    "agentic_novelty_delta_band",
    "agentic_novelty_effect_size",
    "competitive_recursion_level",
    "all_agent_competitive_recursion_level",
    "survivor_competitive_recursion_level",
    "competitive_recursion_survivorship_gap",
    "competitive_recursion_observations",
    "competitive_recursion_delta_vs_none",
    "competitive_recursion_delta_band",
    "competitive_recursion_effect_size",
    "perceived_actor_ignorance_level",
    "perceived_actor_ignorance_delta_vs_none",
    "perceived_actor_ignorance_gap_vs_prior",
    "perceived_practical_indeterminism_level",
    "perceived_practical_indeterminism_delta_vs_none",
    "perceived_practical_indeterminism_gap_vs_prior",
    "perceived_agentic_novelty_level",
    "perceived_agentic_novelty_delta_vs_none",
    "perceived_agentic_novelty_gap_vs_prior",
    "perceived_competitive_recursion_level",
    "perceived_competitive_recursion_delta_vs_none",
    "perceived_competitive_recursion_gap_vs_prior",
    "epistemic_horizon_pressure",
    "epistemic_horizon_delta_vs_none",
    "epistemic_horizon_delta_band",
    "epistemic_horizon_effect_size",
    "knowledge_horizon_tradeoff",
    "confidence_abs_gap_delta_vs_none",
    "realized_multiple_std_delta_vs_none",
    "niches_per_agent_delta_vs_none",
    "combinations_per_agent_delta_vs_none",
    "market_crowding_pressure",
    "market_opportunity_overlap",
    "market_investment_concentration",
    "market_opportunity_competition",
    "market_ai_herding_intensity",
    "market_ai_action_correlation",
    "market_combo_reuse_pressure",
    "survival_effect_pp_vs_none",
    "first_order_evidence_count",
    "second_order_evidence_count",
    "paradox_alignment_score",
]

const N_AGENTS = parse(Int, get(ENV, "N_AGENTS", "1000"))
const N_ROUNDS = parse(Int, get(ENV, "N_ROUNDS", "60"))
const N_RUNS = parse(Int, get(ENV, "N_RUNS", "25"))
const BASE_SEED = parse(Int, get(ENV, "BASE_SEED", "20260425"))
const SUITE_PRESET = lowercase(get(ENV, "SUITE_PRESET", "core"))
# Venture-panel dump (supplementary tail/crowding analysis): when on, BASELINE runs
# emit a per-agent CSV with design factors, outcomes, realized multiples, latent
# returns, and opportunity capacity. Off by default; canonical runs are unaffected.
const VENTURE_PANEL = lowercase(get(ENV, "VENTURE_PANEL", "0")) in ("1", "true", "on", "yes")

# Option-B heavy-tail comparison: a venture-like right tail on opportunity
# returns. Kept off by default because the canonical model locates the heavy
# tail in niche/market size; available for comparison via OPTION_B_TAIL=on.
const OPTION_B_TAIL = lowercase(get(ENV, "OPTION_B_TAIL", "off")) != "off"
const HEAVY_TAIL_RETURNS = lowercase(get(ENV, "HEAVY_TAIL_RETURNS", "off")) in ("1", "true", "on", "yes")
const B_SIGMA_MULT  = parse(Float64, get(ENV, "B_SIGMA_MULT", "4.0"))
const B_SIGMA_CAP   = parse(Float64, get(ENV, "B_SIGMA_CAP",  "3.0"))
const B_RANGE_MULT  = parse(Float64, get(ENV, "B_RANGE_MULT", "250.0"))
const B_CLAMP_MAX   = parse(Float64, get(ENV, "B_CLAMP_MAX",  "2000.0"))
const B_OPS         = parse(Float64, get(ENV, "B_OPS",        "0.70"))
# Canonical niche-size tail: realistic opportunity returns with the
# venture-like heavy tail represented as niche/market size
# (NICHE_SIZE_LOG_SIGMA). Capacity/crowding telemetry and return-capture ratios
# support supplemental mechanism checks; survival is a downstream outcome.
# Calibrated default: sigma=1.5, ops=0.85.
const NICHE_CANONICAL = lowercase(get(ENV, "NICHE_CANONICAL", "on")) != "off"
const NICHE_SIGMA = parse(Float64, get(ENV, "NICHE_SIGMA", "1.5"))
const NICHE_OPS   = parse(Float64, get(ENV, "NICHE_OPS",   "0.85"))
const DERIVED_GATE = parse(Float64, get(ENV, "DERIVED_GATE", "0.55"))  # recombinant knowledge maturity gate
const NICHE_MODERATE_SIGMA = parse(Float64, get(ENV, "NICHE_MODERATE_SIGMA", string(NICHE_SIGMA * 0.5)))
const SUITE_SEED_MODE = lowercase(get(ENV, "SUITE_SEED_MODE", "paired"))
const CONDITION_FILTER = strip(get(ENV, "CONDITIONS", ""))
const OUTPUT_DIR = get(ENV, "OUTPUT_DIR",
    joinpath(@__DIR__, "..", "results", "robustness_run_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"))

struct RobustnessCondition
    name::String
    category::String
    description::String
    theoretical_role::String
    design::String
    preset::String
    apply!::Function
end

function replace_ai_level!(
    config::EmergentConfig,
    tier::String;
    cost::Union{Float64,Nothing}=nothing,
    cost_type::Union{String,Nothing}=nothing,
    info_quality::Union{Float64,Nothing}=nothing,
    info_breadth::Union{Float64,Nothing}=nothing,
    per_use_cost::Union{Float64,Nothing}=nothing,
)
    old = config.AI_LEVELS[tier]
    config.AI_LEVELS[tier] = GlimpseABM.AILevelConfig(
        isnothing(cost) ? old.cost : cost,
        isnothing(cost_type) ? old.cost_type : cost_type,
        isnothing(info_quality) ? old.info_quality : info_quality,
        isnothing(info_breadth) ? old.info_breadth : info_breadth,
        isnothing(per_use_cost) ? old.per_use_cost : per_use_cost,
    )
end

function copy_domain_capabilities!(config::EmergentConfig, target::String, source::String)
    config.AI_DOMAIN_CAPABILITIES[target] = deepcopy(config.AI_DOMAIN_CAPABILITIES[source])
end

function set_domain_capabilities!(
    config::EmergentConfig,
    tier::String;
    accuracy::Float64,
    hallucination_rate::Float64,
    bias::Float64 = 0.0,
)
    config.AI_DOMAIN_CAPABILITIES[tier] = Dict(
        domain => GlimpseABM.AIDomainCapability(accuracy, hallucination_rate, bias)
        for domain in keys(config.AI_DOMAIN_CAPABILITIES[tier])
    )
    return nothing
end

function replace_producer_weights(
    weights::KnightianProducerWeights;
    actor=weights.actor_ignorance,
    practical=weights.practical_indeterminism,
    novelty=weights.agentic_novelty,
    recursion=weights.competitive_recursion,
)
    return KnightianProducerWeights(
        actor_ignorance=actor,
        practical_indeterminism=practical,
        agentic_novelty=novelty,
        competitive_recursion=recursion,
    )
end

finite_mean(values)::Float64 = begin
    vals = Float64[]
    for value in values
        value isa Number || continue
        v = Float64(value)
        isfinite(v) && push!(vals, v)
    end
    isempty(vals) ? 0.0 : mean(vals)
end

function venture_metric_mean(events, index::Int)::Union{Missing,Float64}
    values = Float64[]
    for event in events
        index <= length(event) || continue
        value = Float64(event[index])
        isfinite(value) && push!(values, value)
    end
    return isempty(values) ? missing : mean(values)
end

function venture_effective_saturation_mean(events)::Union{Missing,Float64}
    values = Float64[]
    for event in events
        length(event) >= 17 || continue
        base_capacity = Float64(event[8])
        raw_saturation = Float64(event[11])
        effective_capacity = Float64(event[17])
        if isfinite(base_capacity) && isfinite(raw_saturation) &&
           isfinite(effective_capacity) && effective_capacity > 0.0
            push!(values, raw_saturation * base_capacity / effective_capacity)
        end
    end
    return isempty(values) ? missing : mean(values)
end

function venture_ratio_mean(events, numerator_index::Int, denominator_index::Int)::Union{Missing,Float64}
    values = Float64[]
    for event in events
        max(numerator_index, denominator_index) <= length(event) || continue
        numerator = Float64(event[numerator_index])
        denominator = Float64(event[denominator_index])
        if isfinite(numerator) && isfinite(denominator) && denominator > 0.0
            push!(values, numerator / denominator)
        end
    end
    return isempty(values) ? missing : mean(values)
end

function history_mean(sim::EmergentSimulation, key::String)::Float64
    return finite_mean(get(h, key, nothing) for h in sim.history)
end

# Per-run TOTAL over the round history. Counts/sums semantics: a round with
# zero events contributes 0 (zero events is data, not missingness) — this is
# the right aggregator for pivot counts, pivot capital totals, and innovation
# attempts. Only all-NaN-capable telemetry MEANS (e.g. pivot_recovery_rate,
# NaN on zero-pivot rounds) go through finite_mean instead.
function history_total(sim::EmergentSimulation, key::String)::Float64
    total = 0.0
    for h in sim.history
        value = get(h, key, nothing)
        value isa Number || continue
        v = Float64(value)
        isfinite(v) && (total += v)
    end
    return total
end

function emergent_dimension_default(dim::String)::Float64
    return dim == "competitive_recursion" ? 0.0 : 0.5
end

# Per-tier emergent levels are `missing` (CSV: empty cell; pandas: NaN) when the
# tier has no aggregated observations for that dimension — substituting the
# prior (0.5, or 0.0 for competitive_recursion) would let empty cells
# masquerade as data. Observation/count fields stay numeric so the schema and
# the "this cell was empty" evidence are stable.
function emergent_summary_fields(prefix::String, tier_emergent::AbstractDict)::Dict{String,Any}
    fields = Dict{String,Any}()
    for dim in EMERGENT_UNCERTAINTY_DIMENSIONS
        obs = Float64(get(tier_emergent, "$(dim)_observations", 0.0))
        fields["$(prefix)_emergent_$(dim)"] = obs > 0 ?
            Float64(get(tier_emergent, dim, emergent_dimension_default(dim))) :
            missing
        fields["$(prefix)_emergent_$(dim)_observations"] = obs
    end
    fields["$(prefix)_emergent_agentic_novelty_relevant_observations"] =
        Float64(get(tier_emergent, "agentic_novelty_relevant_observations", 0.0))
    fields["$(prefix)_emergent_agent_count"] = Float64(get(tier_emergent, "n_agents", 0.0))
    fields["$(prefix)_emergent_alive_count"] = Float64(get(tier_emergent, "n_alive", 0.0))
    fields["$(prefix)_emergent_failed_count"] = Float64(get(tier_emergent, "n_failed", 0.0))
    return fields
end

noop!(config::EmergentConfig) = nothing
no_crowding!(config::EmergentConfig) = (
    config.DISABLE_COMPETITION_DYNAMICS = true;
    config.COMPETITION_SCALE_FACTOR = 0.0;
    config.CROWDING_STRENGTH_LAMBDA = 0.0;
    nothing
)
no_return_dilution!(config::EmergentConfig) = (config.CROWDING_STRENGTH_LAMBDA = 0.0; nothing)
low_crowding!(config::EmergentConfig) = (
    config.COMPETITION_SCALE_FACTOR = 0.5;
    config.CROWDING_STRENGTH_LAMBDA = 0.75;
    nothing
)
high_crowding!(config::EmergentConfig) = (
    config.COMPETITION_SCALE_FACTOR = 2.0;
    config.CROWDING_STRENGTH_LAMBDA = 3.0;
    nothing
)
free_ai!(config::EmergentConfig) = (config.AI_COST_INTENSITY = 0.0; nothing)
double_ai_cost!(config::EmergentConfig) = (config.AI_COST_INTENSITY = 2.0; nothing)
# Performative-demand post-hoc test: an opportunity's effective capacity (its
# implicit customer demand) grows with the LAGGED collective dollar commitment
# the niche attracted, making value/demand endogenous to prior collective action
# rather than a fixed exogenous draw. COMMON = the enlarged demand is shared by
# all founders in the niche; APPROPRIABLE = the uplift accrues to each founder in
# proportion to their own commitment share (category ownership). Dose-response
# over ψ ∈ {0.25, 0.5, 1.0}. All are no-ops when ELASTICITY stays 0 (baseline).
performative_common_025!(config::EmergentConfig)      = (config.PERFORMATIVE_DEMAND_ELASTICITY = 0.25; config.PERFORMATIVE_DEMAND_APPROPRIABLE = false; nothing)
performative_common_050!(config::EmergentConfig)      = (config.PERFORMATIVE_DEMAND_ELASTICITY = 0.50; config.PERFORMATIVE_DEMAND_APPROPRIABLE = false; nothing)
performative_common_100!(config::EmergentConfig)      = (config.PERFORMATIVE_DEMAND_ELASTICITY = 1.00; config.PERFORMATIVE_DEMAND_APPROPRIABLE = false; nothing)
performative_appropriable_025!(config::EmergentConfig) = (config.PERFORMATIVE_DEMAND_ELASTICITY = 0.25; config.PERFORMATIVE_DEMAND_APPROPRIABLE = true; nothing)
performative_appropriable_050!(config::EmergentConfig) = (config.PERFORMATIVE_DEMAND_ELASTICITY = 0.50; config.PERFORMATIVE_DEMAND_APPROPRIABLE = true; nothing)
performative_appropriable_100!(config::EmergentConfig) = (config.PERFORMATIVE_DEMAND_ELASTICITY = 1.00; config.PERFORMATIVE_DEMAND_APPROPRIABLE = true; nothing)
# Idiosyncratic return-noise ablation/sweep (σ default 0.38; provenance and
# threshold-asymmetry documented at RETURN_NOISE_SCALE in config.jl). σ=0
# removes the execution-risk layer entirely; 0.19/0.57 bracket the default at
# 0.5×/1.5×. If the frontier trap vanishes at σ=0, the headline depends on
# noise-under-threshold classification, not convergence — these rows exist so
# that question is answered by data, not assumption.
no_return_noise!(config::EmergentConfig) = (config.RETURN_NOISE_SCALE = 0.0; nothing)
return_noise_low!(config::EmergentConfig) = (config.RETURN_NOISE_SCALE = 0.19; nothing)
return_noise_high!(config::EmergentConfig) = (config.RETURN_NOISE_SCALE = 0.57; nothing)
novelty_generalization_stress_on!(config::EmergentConfig) = (config.NOVELTY_NOISE_INVERSION_FACTOR = 0.4; nothing)
premium_exec_5x!(config::EmergentConfig) = (config.AI_EXECUTION_SUCCESS_MULTIPLIERS["premium"] = 5.0; nothing)
# Tail/cost robustness setters. Return-tail fields are retained for OPTION_B_TAIL
# comparison runs; under the current canonical model the same tail cells also
# move the niche-size capacity tail so labels match the live mechanism.
frontier_exec_1_5x!(config::EmergentConfig) = (config.AI_EXECUTION_SUCCESS_MULTIPLIERS["premium"] = 1.5; nothing)
frontier_exec_2x!(config::EmergentConfig)   = (config.AI_EXECUTION_SUCCESS_MULTIPLIERS["premium"] = 2.0; nothing)
frontier_exec_3x!(config::EmergentConfig)   = (config.AI_EXECUTION_SUCCESS_MULTIPLIERS["premium"] = 3.0; nothing)
function truncated_tail!(config::EmergentConfig)
    config.LOG_SIGMA_MULT = 1.0
    config.LOG_SIGMA_CAP = 1.0
    config.RETURN_RANGE_MAX_MULT = 1.0
    config.RETURN_CLAMP_MAX = 25.0
    if NICHE_CANONICAL
        config.NICHE_SIZE_LOG_SIGMA = 0.0
    end
    return nothing
end
function moderate_tail!(config::EmergentConfig)
    if NICHE_CANONICAL
        # Niche-size variant: realistic return tail (mirrors truncated_tail!) + a moderate
        # niche-size tail, so the truncated/moderate/baseline sweep varies ONLY niche sigma
        # (0.0 / NICHE_MODERATE_SIGMA / NICHE_SIGMA) and nothing else.
        config.LOG_SIGMA_MULT = 1.0
        config.LOG_SIGMA_CAP = 1.0
        config.RETURN_RANGE_MAX_MULT = 1.0
        config.RETURN_CLAMP_MAX = 25.0
        config.NICHE_SIZE_LOG_SIGMA = NICHE_MODERATE_SIGMA
    else
        # Option-B comparison path: a moderate latent-return tail.
        config.LOG_SIGMA_MULT = 2.5
        config.LOG_SIGMA_CAP = 2.0
        config.RETURN_RANGE_MAX_MULT = 50.0
        config.RETURN_CLAMP_MAX = 500.0
    end
    return nothing
end
ops_cost_060!(config::EmergentConfig) = (config.OPS_COST_INTENSITY = 0.60; nothing)
ops_cost_075!(config::EmergentConfig) = (config.OPS_COST_INTENSITY = 0.75; nothing)
ops_cost_100!(config::EmergentConfig) = (config.OPS_COST_INTENSITY = 1.00; nothing)
difficulty_scaled_ai_cost!(config::EmergentConfig) = (config.AI_DIFFICULTY_COST_SCALING = 1.0; nothing)
ai_complementarity_on!(config::EmergentConfig) = (config.AI_COMPLEMENTARITY_STRENGTH = 1.0; nothing)

# ── Addendum cells: unaided-floor sweep, bias-free AI, wealth-scaled compute,
#    and endogenous-adoption learning variants ───────────────────────────────
# NO_AI_BIAS isolates the small systematic bias term while leaving the
# hallucination and overconfidence dials unchanged.
no_ai_bias!(config::EmergentConfig) = (config.AI_BIAS_INTENSITY = 0.0; nothing)
# Rich-get-richer marginal compute: visibility budget scales with
# capital/initial_equity (clamped [0.25, 4.0]), uniformly across tiers; the
# extra analyses are billed per use, so compute is bought at the margin.
wealth_scaled_compute!(config::EmergentConfig) = (config.WEALTH_COMPUTE_SCALING = 1.0; nothing)

# Previous payoff specification: blend 0.30 × the market-wide action-category
# HHI into every opportunity's local I/K load. Pure outstanding-capital load is
# now canonical because it has the cleaner productive-capacity interpretation;
# this named cell preserves the prior equation as robustness.
crowding_action_hhi_blend!(config::EmergentConfig) =
    (config.CROWDING_INDEX_BLEND = 0.30; nothing)
# Remove the created-opportunity scoring affinities (any-created ×1.1,
# own-creation ×1.2 → both 1.0) so creation-channel convergence is tested
# without the hard-coded attraction to founder-created niches.
no_created_bonus!(config::EmergentConfig) = (config.CREATED_OPP_AFFINITY = 1.0;
    config.OWN_CREATION_AFFINITY = 1.0; nothing)
# Isolate per-opportunity founder/component fit without bundling it with the S2
# comparative-advantage strategy rule.
opportunity_component_fit_on!(config::EmergentConfig) =
    (config.ENABLE_OPPORTUNITY_COMPONENTS = true; nothing)
# Remove only background/public opportunity replenishment. Exploration niches
# and innovation-spawned opportunities remain live, making the evolving option
# set endogenous after initialization.
endogenous_opportunities_only!(config::EmergentConfig) =
    (config.ENABLE_BACKGROUND_OPPORTUNITY_REPLENISHMENT = false; nothing)
# The historical exploration mechanism opened Uniform{1,2,3} related niches per
# successful creation event. These cells expose and bracket that multiplicity;
# FIXED_TWO is mean-preserving and removes only the draw's variance.
niche_multiplicity_one!(config::EmergentConfig) = (
    config.NICHE_OPPORTUNITIES_PER_DISCOVERY_MIN = 1;
    config.NICHE_OPPORTUNITIES_PER_DISCOVERY_MAX = 1;
    nothing
)
niche_multiplicity_fixed_two!(config::EmergentConfig) = (
    config.NICHE_OPPORTUNITIES_PER_DISCOVERY_MIN = 2;
    config.NICHE_OPPORTUNITIES_PER_DISCOVERY_MAX = 2;
    nothing
)
niche_multiplicity_three!(config::EmergentConfig) = (
    config.NICHE_OPPORTUNITIES_PER_DISCOVERY_MIN = 3;
    config.NICHE_OPPORTUNITIES_PER_DISCOVERY_MAX = 3;
    nothing
)
# Unaided-floor sweep: the no-AI tier's published characteristics (quality
# 0.25 / breadth 0.20) are the one anchor the alternative-frontier sweeps
# never moved. These cells bracket it in both directions, so the trap's
# magnitude can be shown (or shown not) to depend on how weak unaided search
# is assumed to be.
human_baseline_weak!(config::EmergentConfig) =
    (replace_ai_level!(config, "none"; info_quality=0.15, info_breadth=0.12); nothing)
human_baseline_strong!(config::EmergentConfig) =
    (replace_ai_level!(config, "none"; info_quality=0.35, info_breadth=0.30); nothing)
# Endogenous-adoption learning-cadence variants (defaults: review every 3
# rounds after a 12-round maturity freeze). Fast = monthly reviews after half
# a maturity cycle; slow = biannual reviews after 1.5 cycles. These bound how
# much the adoption equilibrium depends on the assumed sorting speed.
adopt_fast_learning!(config::EmergentConfig) = (
    config.AI_TIER_REVIEW_INTERVAL = 1;
    config.AI_TIER_INITIAL_FREEZE_ROUNDS = 6;
    nothing
)
adopt_slow_learning!(config::EmergentConfig) = (
    config.AI_TIER_REVIEW_INTERVAL = 6;
    config.AI_TIER_INITIAL_FREEZE_ROUNDS = 18;
    nothing
)

function no_hallucination_overconfidence!(config::EmergentConfig)
    config.HALLUCINATION_INTENSITY = 0.0
    config.OVERCONFIDENCE_INTENSITY = 0.0
    return nothing
end

function cash_only_survival!(config::EmergentConfig)
    # Counterfactual: cash-on-hand survival rule (deployed capital does not
    # count toward solvency). Measures the illiquidity contribution to the
    # frontier trap relative to the net-worth default.
    config.SURVIVAL_COUNTS_INFLIGHT = false
    return nothing
end

function no_ai_herding_recursion!(config::EmergentConfig)
    base = config.KNIGHTIAN_PRODUCER_WEIGHTS.competitive_recursion
    config.KNIGHTIAN_PRODUCER_WEIGHTS = replace_producer_weights(
        config.KNIGHTIAN_PRODUCER_WEIGHTS;
        recursion=CompetitiveRecursionProducerWeights(
            crowd_weight=base.crowd_weight,
            volatility_weight=base.volatility_weight,
            ai_herd_weight=0.0,
            reuse_weight=base.reuse_weight,
            invest_hhi_weight=base.invest_hhi_weight,
            knowledge_overlap_weight=base.knowledge_overlap_weight,
            ai_action_correlation_excess_weight=0.0,
            population_scale_base=base.population_scale_base,
            population_scale_alive_weight=base.population_scale_alive_weight,
            no_ai_herding_weight=0.0,
        ),
    )
    config.UNCERTAINTY_AI_HERDING_WEIGHT = 0.0
    config.RECURSION_WEIGHTS = Dict(
        "crowd_weight" => base.crowd_weight,
        "volatility_weight" => base.volatility_weight,
        "ai_herd_weight" => 0.0,
        "reuse_weight" => base.reuse_weight,
    )
    return nothing
end

function no_information_advantage_free!(config::EmergentConfig)
    config.AI_COST_INTENSITY = 0.0
    none_cfg = config.AI_LEVELS["none"]
    for tier in ["basic", "advanced", "premium"]
        replace_ai_level!(config, tier;
            info_quality=none_cfg.info_quality,
            info_breadth=none_cfg.info_breadth)
        copy_domain_capabilities!(config, tier, "none")
    end
    return nothing
end

function sham_ai_labels_free!(config::EmergentConfig)
    config.AI_COST_INTENSITY = 0.0
    config.SHAM_AI_LABELS_USE_NONE = true
    none_cfg = config.AI_LEVELS["none"]
    for tier in ["basic", "advanced", "premium"]
        replace_ai_level!(config, tier;
            cost=0.0,
            cost_type="none",
            info_quality=none_cfg.info_quality,
            info_breadth=none_cfg.info_breadth,
            per_use_cost=0.0)
        copy_domain_capabilities!(config, tier, "none")
        config.AI_QUALITY_BOOST[tier] = config.AI_QUALITY_BOOST["none"]
        config.AI_INFORMATION_QUALITY_BOOSTS[tier] = config.AI_INFORMATION_QUALITY_BOOSTS["none"]
        config.AI_EXECUTION_SUCCESS_MULTIPLIERS[tier] = config.AI_EXECUTION_SUCCESS_MULTIPLIERS["none"]
    end
    return nothing
end

function premium_quality_plus50!(config::EmergentConfig)
    config.AI_QUALITY_BOOST["premium"] = 0.50
    config.AI_INFORMATION_QUALITY_BOOSTS["premium"] = 0.50
    return nothing
end

function frontier_coverage_max!(config::EmergentConfig)
    replace_ai_level!(config, "premium"; info_quality=1.0, info_breadth=1.0)
    return nothing
end

function frontier_reasoning_no_error!(config::EmergentConfig)
    set_domain_capabilities!(config, "premium";
        accuracy=0.995,
        hallucination_rate=0.0,
        bias=0.0)
    return nothing
end

function frontier_public_omniscience!(config::EmergentConfig)
    frontier_coverage_max!(config)
    frontier_reasoning_no_error!(config)
    config.AI_QUALITY_BOOST["premium"] = 0.50
    config.AI_INFORMATION_QUALITY_BOOSTS["premium"] = 0.50
    return nothing
end

function all_favorable_to_premium!(config::EmergentConfig)
    config.AI_COST_INTENSITY = 0.0
    config.HALLUCINATION_INTENSITY = 0.0
    config.OVERCONFIDENCE_INTENSITY = 0.0
    config.DISABLE_COMPETITION_DYNAMICS = true
    config.COMPETITION_SCALE_FACTOR = 0.0
    config.CROWDING_STRENGTH_LAMBDA = 0.0
    config.NOVELTY_NOISE_INVERSION_FACTOR = 0.0
    config.AI_EXECUTION_SUCCESS_MULTIPLIERS["premium"] = 10.0
    config.AI_QUALITY_BOOST["premium"] = 0.50
    config.AI_INFORMATION_QUALITY_BOOSTS["premium"] = 0.50
    return nothing
end

function strategic_anticipation_on!(config::EmergentConfig)
    config.STRATEGIC_ANTICIPATION_ENABLED = true
    config.STRATEGIC_ANTICIPATION_STRENGTH = 0.75
    return nothing
end

function differentiated_strategy_on!(config::EmergentConfig)
    config.STRATEGIC_ANTICIPATION_ENABLED = true
    config.STRATEGIC_ANTICIPATION_STRENGTH = 0.75
    config.STRATEGIC_DIVERSIFICATION_ENABLED = true
    config.STRATEGIC_DIVERSIFICATION_STRENGTH = 1.0
    config.STRATEGIC_DIVERSIFICATION_TOP_K = 5
    config.STRATEGIC_DIVERSIFICATION_SCORE_BAND = 0.12
    return nothing
end

function token_pricing_only!(config::EmergentConfig)
    config.AI_COST_MODEL = "token"
    return nothing
end

# Subscription-era pricing is the robustness cell when "token" is the default
# pricing model.
function subscription_era!(config::EmergentConfig)
    config.AI_COST_MODEL = "hybrid"
    return nothing
end

# ── Strategy ladder ────────────────────────────────────────────────────────
# Strategy strengths stay at conservative defaults (1.0); the conditions vary
# only the mode and tier scope. FRONTIER_* cells test whether frontier agents
# can strategically attenuate convergence, and ALL_* cells test whether
# population-wide sophistication relocates or reduces crowding.
function _strategy_ladder!(config::EmergentConfig, mode::String, tiers::Vector{String})
    config.STRATEGY_MODE = mode
    config.STRATEGY_TIERS = tiers
    return nothing
end

const ALL_STRATEGY_TIERS = ["none", "basic", "advanced", "premium"]

frontier_strategy_consensus!(config::EmergentConfig) =
    _strategy_ladder!(config, "consensus_discounting", ["premium"])
frontier_strategy_comparative!(config::EmergentConfig) =
    _strategy_ladder!(config, "comparative_advantage", ["premium"])
frontier_strategy_complement!(config::EmergentConfig) =
    _strategy_ladder!(config, "complement_seeking", ["premium"])
frontier_strategy_composite!(config::EmergentConfig) =
    _strategy_ladder!(config, "agi_native", ["premium"])
all_strategy_consensus!(config::EmergentConfig) =
    _strategy_ladder!(config, "consensus_discounting", ALL_STRATEGY_TIERS)
all_strategy_comparative!(config::EmergentConfig) =
    _strategy_ladder!(config, "comparative_advantage", ALL_STRATEGY_TIERS)
all_strategy_complement!(config::EmergentConfig) =
    _strategy_ladder!(config, "complement_seeking", ALL_STRATEGY_TIERS)
all_strategy_composite!(config::EmergentConfig) =
    _strategy_ladder!(config, "agi_native", ALL_STRATEGY_TIERS)
# S2 with the per-opportunity knowledge-component edge: private-edge re-ranking
# uses the agent's component-knowledge overlap with each opportunity's required
# components. Relies on the recombinant engine (DERIVED_GATE ~0.55).
all_strategy_comparative_components!(config::EmergentConfig) =
    (_strategy_ladder!(config, "comparative_advantage", ALL_STRATEGY_TIERS);
     config.ENABLE_OPPORTUNITY_COMPONENTS = true; nothing)

# ── Open-action extension ──────────────────────────────────────────────────
# A1 = abandonment option (per-round pivot review at an age-dependent haircut,
# default-off); A2 = Hayekian redirection of innovation sector selection away
# from perceived-crowded sectors (default-off). The conditions vary only these
# flags, and the maximal cell additionally activates the strategy ladder.
# PIVOT_DETERIORATION_GAIN = 2.0 in every pivot-enabled cell, calibrated by
# scripts/probe_pivot_liveness.jl to make the trigger region reachable while
# preserving bit-identical default runs.
function open_action_pivot!(config::EmergentConfig)
    config.ENABLE_PIVOT = true
    config.PIVOT_DETERIORATION_GAIN = 2.0
    return nothing
end
open_action_directed_creation!(config::EmergentConfig) =
    (config.ENABLE_DIRECTED_CREATION = true; nothing)
function open_action_both!(config::EmergentConfig)
    config.ENABLE_PIVOT = true
    config.PIVOT_DETERIORATION_GAIN = 2.0
    config.ENABLE_DIRECTED_CREATION = true
    return nothing
end
function open_action_agi_native_market!(config::EmergentConfig)
    # Pivot gain 2.0 inherited via open_action_both!.
    open_action_both!(config)
    _strategy_ladder!(config, "agi_native", ALL_STRATEGY_TIERS)
    return nothing
end

# ── Dose-response sweeps ───────────────────────────────────────────────────
# These cells sweep strategy and pivot dials to their maximum effective values,
# where "effective" is bounded by the use-site clamps:
#   S1 STRATEGY_CONSENSUS_DISCOUNT — clamped to [0,1] at the use site
#     (strategy_shaded_return, src/strategy.jl); the DEFAULT 1.0 is ALREADY
#     the ceiling, so S1 has no dose headroom: any raw value above 1.0 is
#     clamp-neutralized to the default behavior. The two DOSE_S1_* cells
#     verify this saturation in vivo (paired seeds ⇒ both should reproduce
#     FRONTIER_STRATEGY_CONSENSUS bit-for-bit).
#   S2 STRATEGY_EDGE_WEIGHT — clamped to [0,2] (strategy_score_multiplier);
#     2.0 is the maximum effective dose. The overall multiplier clamp
#     [0.25, 2.0] additionally binds when |edge − mean edge| > 0.5.
#   S3 STRATEGY_COMPLEMENT_SHIFT — the S3a action-shift path clamps strength
#     to [0,2] (complement_shift); the S3b score-softening path clamps to
#     [0,1] and is therefore already saturated at the default. 2.0 doubles
#     S3a while leaving S3b at its ceiling — the max effective S3 dose.
#   PIVOT_DETERIORATION_GAIN — validated ≥ 0, unbounded above; the trigger
#     d_eff = clamp(deterioration × conviction × gain, 0, 1) saturates at 1.
#     Gains 4.0 / 8.0 map progressively more of the observed deterioration
#     range into the trigger region for production-scale populations.
function _max_strategy_strengths!(config::EmergentConfig)
    # Max-effective doses given the use-site clamps documented above. S1 is
    # saturated at the default and is set explicitly.
    config.STRATEGY_CONSENSUS_DISCOUNT = 1.0
    config.STRATEGY_EDGE_WEIGHT = 2.0
    config.STRATEGY_COMPLEMENT_SHIFT = 2.0
    return nothing
end
function dose_s1_frontier_2x!(config::EmergentConfig)
    frontier_strategy_consensus!(config)
    config.STRATEGY_CONSENSUS_DISCOUNT = 2.0  # clamp-neutralized to 1.0 at use
    return nothing
end
function dose_s1_frontier_maxx!(config::EmergentConfig)
    frontier_strategy_consensus!(config)
    config.STRATEGY_CONSENSUS_DISCOUNT = 1.0  # the [0,1] ceiling (= default)
    return nothing
end
function dose_s2_frontier_2x!(config::EmergentConfig)
    frontier_strategy_comparative!(config)
    config.STRATEGY_EDGE_WEIGHT = 2.0  # the [0,2] clamp ceiling
    return nothing
end
function dose_composite_frontier_max!(config::EmergentConfig)
    frontier_strategy_composite!(config)
    _max_strategy_strengths!(config)
    return nothing
end
function dose_composite_all_max!(config::EmergentConfig)
    all_strategy_composite!(config)
    _max_strategy_strengths!(config)
    return nothing
end
function _dose_pivot_gain!(config::EmergentConfig, gain::Float64)
    config.ENABLE_PIVOT = true
    config.PIVOT_DETERIORATION_GAIN = gain
    return nothing
end
dose_pivot_gain_4x!(config::EmergentConfig) = _dose_pivot_gain!(config, 4.0)
dose_pivot_gain_8x!(config::EmergentConfig) = _dose_pivot_gain!(config, 8.0)
function dose_open_composite_max!(config::EmergentConfig)
    # Maximal-dose open-action cell: pivot at gain 8.0, directed creation at
    # its default mixture strength, and population-wide agi_native at the
    # max-effective strategy strengths.
    _dose_pivot_gain!(config, 8.0)
    config.ENABLE_DIRECTED_CREATION = true
    _strategy_ladder!(config, "agi_native", ALL_STRATEGY_TIERS)
    _max_strategy_strengths!(config)
    return nothing
end

# ── Emergence mechanism extension ──────────────────────────────────────────
# Two mechanism dials separate signal commonality and decision mixing.
# AI_ERROR_CORRELATION (rho) decouples signal COMMONALITY from signal QUALITY:
# the continuous estimate error becomes
# sqrt(rho)*eps_common(opp, round, tier) + sqrt(1-rho)*eps_idio with total
# variance preserved, so estimate accuracy is invariant in rho BY
# CONSTRUCTION (variance algebra at the config field, src/config.jl).
# DECISION_TEMPERATURE (T) multiplies the calibrated softmax temperature at
# the single action-selection chokepoint (make_decision!, src/agents.jl).
# Both defaults (rho = 0, T = 1) are bit-identical and pinned by
# test/test_emergence_audit.jl.
ai_errors_corr_50!(config::EmergentConfig) = (config.AI_ERROR_CORRELATION = 0.5; nothing)
shared_ai_errors!(config::EmergentConfig) = (config.AI_ERROR_CORRELATION = 1.0; nothing)
decision_noise_high!(config::EmergentConfig) = (config.DECISION_TEMPERATURE = 2.0; nothing)
decision_noise_low!(config::EmergentConfig) = (config.DECISION_TEMPERATURE = 0.5; nothing)

function market_slack_high_capacity!(config::EmergentConfig)
    config.OPPORTUNITY_BASE_CAPACITY *= 2.0
    config.CROWDING_CAPACITY_RATIO_K = 2.25
    config.CROWDING_STRENGTH_LAMBDA = 0.75
    return nothing
end

# One-factor capacity-mean cells: unlike MARKET_SLACK_HIGH_CAPACITY and
# MARKET_DENSE_LOW_CAPACITY, these hold κ, λ, γ, tail dispersion, and every
# other mechanism fixed. They identify whether the dollar capacity scale alone
# manufactures the tier contrast.
capacity_mean_half!(config::EmergentConfig) =
    (config.OPPORTUNITY_BASE_CAPACITY *= 0.5; nothing)
capacity_mean_double!(config::EmergentConfig) =
    (config.OPPORTUNITY_BASE_CAPACITY *= 2.0; nothing)
# Rational-response robustness: agents fully price the currently visible local
# I/K penalty when ranking opportunities. They still cannot observe rivals'
# simultaneous commitments, so any residual crowding is an equilibrium timing
# effect rather than blindness to the existing load.
capacity_aware_selection!(config::EmergentConfig) =
    (config.DECISION_CROWDING_AVERSION_WEIGHT = 1.0; nothing)

function market_dense_low_capacity!(config::EmergentConfig)
    config.OPPORTUNITY_BASE_CAPACITY *= 0.60
    config.CROWDING_CAPACITY_RATIO_K = 1.10
    config.CROWDING_STRENGTH_LAMBDA = 2.25
    return nothing
end

function opportunity_rich_environment!(config::EmergentConfig)
    config.BASE_OPPORTUNITIES = 10
    config.DISCOVERY_PROBABILITY = 0.28
    return nothing
end

function opportunity_sparse_environment!(config::EmergentConfig)
    config.BASE_OPPORTUNITIES = 3
    config.DISCOVERY_PROBABILITY = 0.12
    return nothing
end

# Sector-composition robustness. SECTOR_WEIGHTS drives the
# per-agent home-sector draw (_sample_sector_weighted, agents.jl); the canonical
# baseline is the NVCA-weighted 60/15/15/10 (tech/service/manufacturing/retail).
# These two cells bracket that weighting to test whether the headline deficit is
# an artifact of the tech-heavy population. Sector is assigned independently of
# AI tier, so each tier carries the same sector mix in every cell.
function equal_sectors!(config::EmergentConfig)
    config.SECTOR_WEIGHTS = Dict(
        "tech" => 0.25, "service" => 0.25,
        "manufacturing" => 0.25, "retail" => 0.25,
    )
    return nothing
end

function services_heavy!(config::EmergentConfig)
    # Opposite extreme from the tech-heavy baseline: a low-competition,
    # lighter-tail population dominated by services and retail.
    config.SECTOR_WEIGHTS = Dict(
        "service" => 0.45, "retail" => 0.25,
        "manufacturing" => 0.20, "tech" => 0.10,
    )
    return nothing
end

function build_config(condition::RobustnessCondition, seed::Int)
    config = EmergentConfig(
        N_AGENTS = N_AGENTS,
        N_ROUNDS = N_ROUNDS,
        RANDOM_SEED = seed,
        AGENT_AI_MODE = condition.design == "emergent" ? "emergent" : "fixed",
    )
    if OPTION_B_TAIL
        # Option-B heavy-tail baseline applied to EVERY condition BEFORE its mutator,
        # so TRUNCATED_TAIL / MODERATE_TAIL / OPS_COST_* can override these fields.
        config.LOG_SIGMA_MULT = B_SIGMA_MULT
        config.LOG_SIGMA_CAP = B_SIGMA_CAP
        config.RETURN_RANGE_MAX_MULT = B_RANGE_MULT
        config.RETURN_CLAMP_MAX = B_CLAMP_MAX
        config.OPS_COST_INTENSITY = B_OPS
    end
    if HEAVY_TAIL_RETURNS
        # Realized returns AND perceived estimates track the full latent tail (bounded
        # by RETURN_CLAMP_MAX): an uncrowded unicorn can pay a unicorn-scale realized
        # return, and agents can perceive past ~25x. Crowding still dilutes. The estimate
        # ceiling rises to RETURN_CLAMP_MAX; realized_return's gated caps lift via the flag.
        config.HEAVY_TAIL_RETURNS = true
        config.OPPORTUNITY_RETURN_RANGE = (config.OPPORTUNITY_RETURN_RANGE[1], config.RETURN_CLAMP_MAX)
    end
    if NICHE_CANONICAL
        # Canonical model: realistic opportunity returns with the venture-like
        # heavy tail represented as niche/market size, applied before the
        # condition mutator so cells can override.
        config.NICHE_SIZE_LOG_SIGMA = NICHE_SIGMA
        config.OPS_COST_INTENSITY = NICHE_OPS
        # The recombinant-knowledge gate lets above-average recombinations mint
        # derived components and support cumulative knowledge growth. Applied
        # before the condition mutator so cells can override.
        config.DERIVED_KNOWLEDGE_QUALITY_GATE = DERIVED_GATE
    end
    condition.apply!(config)
    return config
end

function effective_config_rows(conditions::Vector{RobustnessCondition}, seed::Int)
    rows = NamedTuple[]
    for condition in conditions
        cfg = build_config(condition, seed)
        tail_model = cfg.NICHE_SIZE_LOG_SIGMA > 0.0 ? "niche_capacity_lognormal" :
            (cfg.HEAVY_TAIL_RETURNS ? "heavy_realized_returns" :
             (cfg.RETURN_RANGE_MAX_MULT > 1.0 || cfg.RETURN_CLAMP_MAX > 25.0 ?
              "latent_return_tail_only" : "truncated_return_defaults"))
        push!(rows, (
            condition = condition.name,
            category = condition.category,
            design = condition.design,
            preset = condition.preset,
            option_b_tail_env = OPTION_B_TAIL,
            heavy_tail_returns_env = HEAVY_TAIL_RETURNS,
            niche_canonical_env = NICHE_CANONICAL,
            tail_model = tail_model,
            log_sigma_mult = cfg.LOG_SIGMA_MULT,
            log_sigma_cap = cfg.LOG_SIGMA_CAP,
            return_range_max_mult = cfg.RETURN_RANGE_MAX_MULT,
            return_clamp_max = cfg.RETURN_CLAMP_MAX,
            heavy_tail_returns = cfg.HEAVY_TAIL_RETURNS,
            niche_size_log_sigma = cfg.NICHE_SIZE_LOG_SIGMA,
            opportunity_base_capacity = cfg.OPPORTUNITY_BASE_CAPACITY,
            opportunity_capacity_variance = cfg.OPPORTUNITY_CAPACITY_VARIANCE,
            crowding_index_blend = cfg.CROWDING_INDEX_BLEND,
            performative_demand_elasticity =
                cfg.PERFORMATIVE_DEMAND_ELASTICITY,
            performative_demand_appropriable =
                cfg.PERFORMATIVE_DEMAND_APPROPRIABLE,
            decision_crowding_aversion_weight =
                cfg.DECISION_CROWDING_AVERSION_WEIGHT,
            ops_cost_intensity = cfg.OPS_COST_INTENSITY,
            crowding_capacity_ratio_k = cfg.CROWDING_CAPACITY_RATIO_K,
            crowding_strength_lambda = cfg.CROWDING_STRENGTH_LAMBDA,
            crowding_convexity_gamma = cfg.CROWDING_CONVEXITY_GAMMA,
            background_opportunity_replenishment =
                cfg.ENABLE_BACKGROUND_OPPORTUNITY_REPLENISHMENT,
            niche_opportunities_per_discovery_min =
                cfg.NICHE_OPPORTUNITIES_PER_DISCOVERY_MIN,
            niche_opportunities_per_discovery_max =
                cfg.NICHE_OPPORTUNITIES_PER_DISCOVERY_MAX,
            opportunity_component_fit = cfg.ENABLE_OPPORTUNITY_COMPONENTS,
        ))
    end
    return DataFrame(rows)
end

function build_simulation(condition::RobustnessCondition, seed::Int)
    rng = MersenneTwister(seed)
    config = build_config(condition, seed)
    initial_dist = condition.design == "emergent" ?
        Dict("none" => 1.0, "basic" => 0.0, "advanced" => 0.0, "premium" => 0.0) :
        Dict(t => 0.25 for t in AI_TIERS)
    sim = EmergentSimulation(config=config, initial_tier_distribution=initial_dist, seed=seed)
    if condition.design != "emergent"
        apply_balanced_fixed_tiers!(sim, balanced_tier_assignments(N_AGENTS, rng))
    end
    return sim
end

function stable_condition_offset(name::String)
    hash_value = UInt32(2166136261)
    for byte in codeunits(name)
        hash_value = (hash_value ⊻ UInt32(byte)) * UInt32(16777619)
    end
    return Int(hash_value % UInt32(100_000))
end

function condition_seed(condition::RobustnessCondition, run_idx::Int)
    if SUITE_SEED_MODE == "paired"
        return BASE_SEED + run_idx
    elseif SUITE_SEED_MODE == "independent"
        return BASE_SEED + 1000 * stable_condition_offset(condition.name) + run_idx
    end
    error("Unknown SUITE_SEED_MODE=$SUITE_SEED_MODE. Use paired or independent.")
end

function run_one(condition::RobustnessCondition, run_idx::Int)
    seed = condition_seed(condition, run_idx)
    sim = build_simulation(condition, seed)
    for round in 1:N_ROUNDS
        GlimpseABM.step!(sim, round)
    end
    if condition.design == "emergent"
        write_adoption_telemetry(condition, sim, run_idx, seed)
    end
    if VENTURE_PANEL && condition.name in ("BASELINE", "SHAM_AI_LABELS_FREE",
            "NO_INFORMATION_ADVANTAGE_FREE", "NO_CROWDING", "SHARED_AI_ERRORS",
            "FRONTIER_REASONING_NO_ERROR", "CROWDING_ACTION_HHI_BLEND",
            "NO_CREATED_BONUS", "OPPORTUNITY_COMPONENT_FIT_ON",
            "ENDOGENOUS_OPPORTUNITIES_ONLY", "NICHE_MULTIPLICITY_ONE",
            "NICHE_MULTIPLICITY_FIXED_TWO", "NICHE_MULTIPLICITY_THREE",
            "CAPACITY_MEAN_HALF", "CAPACITY_MEAN_DOUBLE",
            "CAPACITY_AWARE_SELECTION",
            "PERFORMATIVE_COMMON_025", "PERFORMATIVE_COMMON_050",
            "PERFORMATIVE_COMMON_100", "PERFORMATIVE_APPROPRIABLE_025",
            "PERFORMATIVE_APPROPRIABLE_050", "PERFORMATIVE_APPROPRIABLE_100",
            "WEALTH_SCALED_COMPUTE", "FRONTIER_QUALITY_PLUS50")
        # Panel for the value-capture + creation-convergence analysis: baseline (the
        # central result), the two placebos (the trap must flatten without a real AI
        # advantage), the crowding knockout, the error-structure cells, and the
        # supplemental capacity, creation, and payoff-specification cells whose
        # saturation, created-niche, and self-created metrics feed the mechanism
        # tables. FRONTIER_QUALITY_PLUS50 joins the set for
        # the innovation-success outcome family (Tier A): it is the
        # quality-boost counterfactual for the innovate channel.
        write_venture_panel(condition, sim, run_idx, seed)
        # Tier A companions: per-innovation and end-of-run niche panels for the
        # innovation-success + financial-performance outcome families.
        write_innovation_panel(condition, sim, run_idx, seed)
        write_niche_panel(condition, sim, run_idx, seed)
        write_opportunity_event_panels(condition, sim, run_idx, seed)
    end
    return summarize_simulation(condition, sim, run_idx, seed)
end

# ── Adoption telemetry ─────────────────────────────────────────────────────
# Per-run files under OUTPUT_DIR/adoption/, one pair per (condition, run):
# unique filenames mean no cross-thread locking and a partial suite still
# leaves complete per-run records. Trajectories come from the round history's
# ai_*_count fields (population tier mix per round — the manuscript's
# adoption-shares exhibit). Agent rows use ai_tier_history, which records
# REVIEW-time choices (not per-round tenure — see choose_ai_level), so
# tier_switches counts review-to-review changes; per-round population tenure
# lives in the trajectory file instead.
function write_adoption_telemetry(
    condition::RobustnessCondition,
    sim::EmergentSimulation,
    run_idx::Int,
    seed::Int,
)
    adoption_dir = joinpath(OUTPUT_DIR, "adoption")
    isdir(adoption_dir) || mkpath(adoption_dir)
    run_tag = lpad(run_idx, 3, '0')

    traj_rows = Dict{Symbol,Any}[]
    for (round_i, h) in enumerate(sim.history)
        push!(traj_rows, Dict{Symbol,Any}(
            :condition => condition.name,
            :run_idx => run_idx,
            :seed => seed,
            :round => round_i,
            :ai_none_count => get(h, "ai_none_count", missing),
            :ai_basic_count => get(h, "ai_basic_count", missing),
            :ai_advanced_count => get(h, "ai_advanced_count", missing),
            :ai_premium_count => get(h, "ai_premium_count", missing),
            :n_alive => get(h, "n_alive", missing),
        ))
    end
    if !isempty(traj_rows)
        CSV.write(joinpath(adoption_dir,
            "$(condition.name)_run$(run_tag)_trajectory.csv"), DataFrame(traj_rows))
    end

    agent_rows = Dict{Symbol,Any}[]
    for agent in sim.agents
        hist = agent.ai_tier_history
        n_reviews = length(hist)
        switches = n_reviews < 2 ? 0 :
            count(i -> hist[i] != hist[i-1], 2:n_reviews)
        push!(agent_rows, Dict{Symbol,Any}(
            :condition => condition.name,
            :run_idx => run_idx,
            :seed => seed,
            :agent_id => agent.id,
            :final_tier => GlimpseABM.get_ai_level(agent),
            :survived => agent.alive,
            :n_tier_reviews => n_reviews,
            :tier_switches => switches,
            :reviews_none => count(==("none"), hist),
            :reviews_basic => count(==("basic"), hist),
            :reviews_advanced => count(==("advanced"), hist),
            :reviews_premium => count(==("premium"), hist),
            :initial_equity => agent.resources.performance.initial_equity,
            :final_capital => agent.resources.capital,
            :ai_trust => Float64(get(agent.traits, "ai_trust", NaN)),
            :uncertainty_tolerance =>
                Float64(get(agent.traits, "uncertainty_tolerance", NaN)),
        ))
    end
    if !isempty(agent_rows)
        CSV.write(joinpath(adoption_dir,
            "$(condition.name)_run$(run_tag)_agents.csv"), DataFrame(agent_rows))
    end
    return nothing
end

# Per-agent venture panel for supplementary tail/crowding analysis. One row per
# founder: randomized design factors, realized outcomes, realized-multiple
# metrics, return-capture ratios, and opportunity-capacity telemetry. Capacity
# is the canonical niche/market-size tail variable. Sourced from
# agent.venture_ledger, which is append-only.
function write_venture_panel(
    condition::RobustnessCondition,
    sim::EmergentSimulation,
    run_idx::Int,
    seed::Int,
)
    panel_dir = joinpath(OUTPUT_DIR, "venture_panel")
    isdir(panel_dir) || mkpath(panel_dir)
    run_tag = lpad(run_idx, 3, '0')

    rows = Dict{Symbol,Any}[]
    rows_raw = Dict{Symbol,Any}[]   # per-investment ledger for the upstream paradox (sight/convergence/yield)
    for agent in sim.agents
        led = agent.venture_ledger      # realized/capture + entry→maturity capital-load trace
        agent_tier = GlimpseABM.get_ai_level(agent)
        for t in led
            saturation_entry = t[10]
            saturation_maturity = t[11]
            action_hhi_maturity = t[12]
            effective_saturation_maturity =
                isfinite(t[17]) && t[17] > 0.0 ?
                saturation_maturity * t[8] / t[17] : saturation_maturity
            effective_load = effective_saturation_maturity +
                sim.config.CROWDING_INDEX_BLEND * action_hhi_maturity
            post_commitment_rival_ratio = isfinite(t[8]) && t[8] > 0.0 ?
                t[14] / t[8] : NaN
            push!(rows_raw, Dict{Symbol,Any}(
                :condition => condition.name, :run_idx => run_idx, :seed => seed,
                :agent_id => agent.id, :tier => agent_tier, :survived => agent.alive,
                :realized => t[1], :crowding => t[2], :ai_estimate => t[3], :latent => t[4],
                :amount => t[5], :round => t[6], :is_created => t[7], :capacity => t[8],
                :self_created => length(t) >= 9 ? t[9] : 0.0,
                :capacity_saturation_at_entry => saturation_entry,
                :capacity_saturation_at_maturity => saturation_maturity,
                :effective_capacity_saturation_at_maturity =>
                    effective_saturation_maturity,
                :capacity_saturation_change => t[13],
                :post_commitment_rival_capital => t[14],
                :post_commitment_rival_capacity_ratio => post_commitment_rival_ratio,
                :action_hhi_at_maturity => action_hhi_maturity,
                :effective_capacity_load => effective_load,
                :crowding_return_multiplier => t[15],
                :realized_latent_capture => t[16],
                :performative_effective_capacity => t[17],
                :performative_lagged_commitment => t[18],
                :performative_commitment_share => t[19],
                :performative_uplift_fraction => t[20]))
        end
        rms = Float64[t[1] for t in led]
        lats = Float64[t[4] for t in led]
        caps = Float64[t[8] for t in led if isfinite(t[8]) && t[8] > 0.0]
        n_inv = length(led)
        max_rm = isempty(rms) ? 0.0 : maximum(rms)
        max_lat = isempty(lats) ? 0.0 : maximum(lats)
        max_cap = isempty(caps) ? 0.0 : maximum(caps)
        best_crowd = NaN; best_est = NaN; best_lat = NaN; best_amt = NaN; best_cap = NaN
        if !isempty(led)
            bi = argmax(rms)
            best_crowd = led[bi][2]; best_est = led[bi][3]
            best_lat = led[bi][4]; best_amt = led[bi][5]
            best_cap = led[bi][8]
        end
        # Return-capture ratio + created-niche convergence (t[7]==1 => created).
        # Capacity columns below carry the canonical niche-size tail directly.
        vcs = Float64[t[1] / t[4] for t in led if t[4] > 0.0]
        cre = filter(t -> t[7] == 1.0, led); endo = filter(t -> t[7] != 1.0, led)
        cap_vc(v) = (w = Float64[t[1] / t[4] for t in v if t[4] > 0.0]; isempty(w) ? NaN : sum(w) / length(w))
        mean_capacity(v) = (w = Float64[t[8] for t in v if isfinite(t[8]) && t[8] > 0.0]; isempty(w) ? NaN : sum(w) / length(w))
        ie = agent.resources.performance.initial_equity
        fc = agent.resources.capital
        # Tier A additions (2026-07-07): innovation-channel outcomes + per-founder
        # balance-sheet position. Read-only views of existing agent state —
        # PerformanceTracker cumulative flows, uncertainty_metrics innovation
        # vectors, and active_investments in-flight capital. No engine change.
        perf = agent.resources.performance
        innovate_deployed = get(perf.deployed_by_action, "innovate", 0.0)
        innovate_returned = get(perf.returned_by_action, "innovate", 0.0)
        invest_deployed = get(perf.deployed_by_action, "invest", 0.0)
        invest_returned = get(perf.returned_by_action, "invest", 0.0)
        explore_spent = get(perf.deployed_by_action, "explore", 0.0)
        overall_deployed = get(perf.deployed_by_action, "overall", 0.0)
        overall_returned = get(perf.returned_by_action, "overall", 0.0)
        in_flight = isempty(agent.active_investments) ? 0.0 :
            sum(Float64(inv["amount"]) for inv in agent.active_investments)
        qual_v = agent.uncertainty_metrics.innovation_qualities
        nov_v = agent.uncertainty_metrics.innovation_novelties
        scar_v = agent.uncertainty_metrics.innovation_scarcities
        push!(rows, Dict{Symbol,Any}(
            :condition => condition.name,
            :run_idx => run_idx,
            :seed => seed,
            :agent_id => agent.id,
            # exogenous design factors (randomly assigned)
            :tier => GlimpseABM.get_ai_level(agent),
            :primary_sector => agent.primary_sector,
            :initial_equity => ie,
            :uncertainty_tolerance => agent.uncertainty_tolerance,
            :innovativeness => agent.innovativeness,
            :competence => agent.competence,
            :ai_trust => agent.ai_trust,
            # realized outcomes
            :survived => agent.alive,
            :survival_rounds => agent.survival_rounds,
            :final_capital => fc,
            :wealth_multiple => ie > 0.0 ? fc / ie : NaN,
            :total_invested => agent.total_invested,
            :total_returned => agent.total_returned,
            :investment_success_count => agent.investment_success_count,
            :investment_failure_count => agent.investment_failure_count,
            # Realized investment multiple tail.
            :n_investments => n_inv,
            :max_realized_multiple => max_rm,
            :mean_realized_multiple => isempty(rms) ? 0.0 : sum(rms) / length(rms),
            :n_hit_10x => count(>=(10.0), rms),
            :n_hit_50x => count(>=(50.0), rms),
            :n_hit_100x => count(>=(100.0), rms),
            # Selection on latent returns and canonical market-size capacity.
            :max_latent_opportunity => max_lat,
            :mean_latent => isempty(lats) ? NaN : sum(lats) / length(lats),
            :n_bet_latent_50x => count(>=(50.0), lats),
            :n_bet_latent_100x => count(>=(100.0), lats),
            :n_bet_latent_1000x => count(>=(1000.0), lats),
            :max_opportunity_capacity => max_cap,
            :mean_opportunity_capacity => isempty(caps) ? NaN : sum(caps) / length(caps),
            # two-stage mechanism: the best winner's latent value, crowding, AI estimate
            :best_inv_realized_over_latent => (isnan(best_lat) || best_lat <= 0.0) ? NaN : max_rm / best_lat,
            :best_inv_latent => best_lat,
            :best_inv_crowding => best_crowd,
            :best_inv_ai_estimate => best_est,
            :best_inv_amount => best_amt,
            :best_inv_capacity => best_cap,
            :mean_crowding => isempty(led) ? NaN : sum(t[2] for t in led) / length(led),
            :mean_capacity_saturation_at_entry => venture_metric_mean(led, 10),
            :mean_capacity_saturation_at_maturity => venture_metric_mean(led, 11),
            :mean_effective_capacity_saturation_at_maturity =>
                venture_effective_saturation_mean(led),
            :mean_capacity_saturation_change => venture_metric_mean(led, 13),
            :mean_post_commitment_rival_capital => venture_metric_mean(led, 14),
            :mean_post_commitment_rival_capacity_ratio => venture_ratio_mean(led, 14, 8),
            :mean_crowding_return_multiplier => venture_metric_mean(led, 15),
            :mean_realized_latent_capture => venture_metric_mean(led, 16),
            :mean_performative_effective_capacity => venture_metric_mean(led, 17),
            :mean_performative_lagged_commitment => venture_metric_mean(led, 18),
            :mean_performative_commitment_share => venture_metric_mean(led, 19),
            :mean_performative_uplift_fraction => venture_metric_mean(led, 20),
            :post_commitment_rival_incidence => isempty(led) ? missing :
                count(t -> isfinite(t[14]) && t[14] > 0.0, led) / length(led),
            # Return-capture efficiency + canonical capacity tail + creation re-convergence.
            :return_capture_mean => isempty(vcs) ? NaN : sum(vcs) / length(vcs),
            :value_capture_mean => isempty(vcs) ? NaN : sum(vcs) / length(vcs),
            :frac_created_niches => n_inv > 0 ? length(cre) / n_inv : NaN,
            :crowd_created => isempty(cre) ? NaN : sum(t[2] for t in cre) / length(cre),
            :crowd_endowed => isempty(endo) ? NaN : sum(t[2] for t in endo) / length(endo),
            :vc_created => cap_vc(cre),
            :vc_endowed => cap_vc(endo),
            :capacity_created => mean_capacity(cre),
            :capacity_endowed => mean_capacity(endo),
            # ── Tier A: innovation-channel outcomes ─────────────────────────
            # innovation_count counts ATTEMPTS (including no-viable-combination
            # attempts that created no Innovation object); hit rate is the
            # per-founder success share of attempts. Quality/novelty/scarcity
            # vectors cover created innovations only (engine path).
            :innovation_count => agent.innovation_count,
            :innovation_success_count => agent.innovation_success_count,
            :innovation_failure_count => agent.innovation_failure_count,
            :innovation_hit_rate => agent.innovation_count > 0 ?
                agent.innovation_success_count / agent.innovation_count : NaN,
            :mean_innovation_quality => isempty(qual_v) ? NaN : sum(qual_v) / length(qual_v),
            :max_innovation_quality => isempty(qual_v) ? NaN : maximum(qual_v),
            :mean_innovation_novelty => isempty(nov_v) ? NaN : sum(nov_v) / length(nov_v),
            :max_innovation_novelty => isempty(nov_v) ? NaN : maximum(nov_v),
            :mean_innovation_scarcity => isempty(scar_v) ? NaN : sum(scar_v) / length(scar_v),
            :niches_discovered => agent.uncertainty_metrics.niches_discovered,
            :new_combinations_created => agent.uncertainty_metrics.new_combinations_created,
            :derivative_adoptions => agent.uncertainty_metrics.derivative_adoptions,
            # ── Tier A: financial performance ───────────────────────────────
            # Channel-split CUMULATIVE capital flows from the PerformanceTracker
            # (unlike the round-level history ROIC, numerator and denominator
            # windows match here). ROIC is NaN when the channel deployed nothing.
            # net_worth = cash + in-flight (deployed, not-yet-matured) capital —
            # the net-worth solvency rule's wealth concept; wealth_multiple
            # above remains the cash-only multiple.
            :innovate_deployed => innovate_deployed,
            :innovate_returned => innovate_returned,
            :innovate_roic => innovate_deployed > 0.0 ?
                (innovate_returned - innovate_deployed) / innovate_deployed : NaN,
            :invest_deployed => invest_deployed,
            :invest_returned => invest_returned,
            :invest_roic => invest_deployed > 0.0 ?
                (invest_returned - invest_deployed) / invest_deployed : NaN,
            :explore_spent => explore_spent,
            :overall_roic => overall_deployed > 0.0 ?
                (overall_returned - overall_deployed) / overall_deployed : NaN,
            :in_flight_capital => in_flight,
            :net_worth => fc + in_flight,
            :net_worth_multiple => ie > 0.0 ? (fc + in_flight) / ie : NaN,
            :ai_usage_count => agent.ai_usage_count,
        ))
    end
    if !isempty(rows)
        CSV.write(joinpath(panel_dir,
            "$(condition.name)_run$(run_tag)_venture.csv"), DataFrame(rows))
    end
    if !isempty(rows_raw)
        CSV.write(joinpath(panel_dir,
            "$(condition.name)_run$(run_tag)_ledger.csv"), DataFrame(rows_raw))
    end
    return nothing
end

# Per-innovation panel (Tier A, 2026-07-07): one row per Innovation object
# created during the run. Innovation attempts that failed to assemble a viable
# knowledge combination create no Innovation object — they appear only in the
# venture panel's innovation_count/innovation_failure_count, so this panel's
# per-creator row count is ≤ that agent's innovation_count while its success
# rows match innovation_success_count exactly.
#
# success/market_impact are set at evaluation time by
# evaluate_innovation_success! on the engine's stored object. cash_multiple is
# NOT stored on the engine's copy (only on the spawn-side duplicate built in
# simulation.jl step!), so R&D spend and realized cash are joined from the
# creator's PerformanceTracker roi_events instead: an agent takes at most one
# action per round, so (creator_id, round, action="innovate") uniquely keys one
# deployment/return pair, and realized_cash / rd_spend reproduces the
# evaluate_innovation_success! cash multiple exactly
# (innovation_return = rd_spend × cash_multiple in _execute_innovate!).
function write_innovation_panel(
    condition::RobustnessCondition,
    sim::EmergentSimulation,
    run_idx::Int,
    seed::Int,
)
    panel_dir = joinpath(OUTPUT_DIR, "venture_panel")
    isdir(panel_dir) || mkpath(panel_dir)
    run_tag = lpad(run_idx, 3, '0')

    # (agent_id, round) → innovate-channel deployment / return totals from the
    # round-stamped roi_events log.
    spend_by_agent_round = Dict{Tuple{Int,Int},Float64}()
    cash_by_agent_round = Dict{Tuple{Int,Int},Float64}()
    for agent in sim.agents
        for ev in agent.resources.performance.roi_events
            get(ev, "action", "") == "innovate" || continue
            rnd = get(ev, "round", nothing)
            rnd isa Integer || continue
            key = (agent.id, Int(rnd))
            amt = Float64(get(ev, "amount", 0.0))
            ev_type = get(ev, "type", "")
            if ev_type == "deployment"
                spend_by_agent_round[key] = get(spend_by_agent_round, key, 0.0) + amt
            elseif ev_type == "return"
                cash_by_agent_round[key] = get(cash_by_agent_round, key, 0.0) + amt
            end
        end
    end

    agent_by_id = Dict(a.id => a for a in sim.agents)
    rows = Dict{Symbol,Any}[]
    for innov in values(sim.innovation_engine.innovations)
        creator = get(agent_by_id, innov.creator_id, nothing)
        key = (innov.creator_id, innov.round_created)
        rd_spend = get(spend_by_agent_round, key, NaN)
        realized = get(cash_by_agent_round, key, NaN)
        push!(rows, Dict{Symbol,Any}(
            :condition => condition.name,
            :run_idx => run_idx,
            :seed => seed,
            :innovation_id => innov.id,
            :creator_id => innov.creator_id,
            # creator_tier is the design's treatment assignment (fixed tier in
            # fixed_mixed); ai_level_used is the tier the agent actually used
            # for THIS attempt (identical under fixed designs, informative in
            # emergent-adoption designs).
            :creator_tier => isnothing(creator) ? missing : GlimpseABM.get_ai_level(creator),
            :creator_survived => isnothing(creator) ? missing : creator.alive,
            :round_created => innov.round_created,
            :type => innov.type,
            :quality => innov.quality,
            :novelty => innov.novelty,
            :scarcity => something(innov.scarcity, NaN),
            :success => innov.success === nothing ? missing : Bool(innov.success),
            :market_impact => something(innov.market_impact, NaN),
            :ai_level_used => innov.ai_level_used,
            :ai_assisted => innov.ai_assisted,
            :is_new_combination => innov.is_new_combination,
            :sector => something(innov.sector, "unsectored"),
            :n_components => length(innov.knowledge_components),
            :combination_signature => something(innov.combination_signature, ""),
            :rd_spend => rd_spend,
            :realized_cash => realized,
            :realized_cash_multiple => (isfinite(rd_spend) && rd_spend > 0.0 && isfinite(realized)) ?
                realized / rd_spend : NaN,
        ))
    end
    if !isempty(rows)
        CSV.write(joinpath(panel_dir,
            "$(condition.name)_run$(run_tag)_innovations.csv"), DataFrame(rows))
    end
    return nothing
end

# End-of-run opportunity panel (Tier A, 2026-07-07): one row per opportunity
# still on the market at the final round, flagged created (founder niche /
# innovation spawn — the is_created rule mirrors agents.jl venture-ledger
# tagging) vs endowed, with full creation lineage (origin_innovation_id,
# created_by) for innovation→commercialization attribution.
#
# SURVIVORSHIP CAVEAT: market maintenance culls opportunities mid-run
# (zero-demand at age>10; low-competition at age>20 when competition dynamics
# are on; age>80 low-competition in clear_old_opportunities!), so this panel
# reflects opportunities that stayed active, not all opportunities ever
# created. The append-only *_opportunity_flows.csv and *_commitments.csv files
# preserve the full stock-flow history; use THIS panel only for end-state niche
# structure and treat measures computed from it as conditional on niche survival.
function write_niche_panel(
    condition::RobustnessCondition,
    sim::EmergentSimulation,
    run_idx::Int,
    seed::Int,
)
    panel_dir = joinpath(OUTPUT_DIR, "venture_panel")
    isdir(panel_dir) || mkpath(panel_dir)
    run_tag = lpad(run_idx, 3, '0')

    agent_by_id = Dict(a.id => a for a in sim.agents)
    rows = Dict{Symbol,Any}[]
    for opp in sim.market.opportunities
        is_created = startswith(opp.id, "niche_") || startswith(opp.id, "spawn_") ||
            opp.origin_innovation_id !== nothing
        creator = opp.created_by === nothing ? nothing :
            get(agent_by_id, opp.created_by, nothing)
        rr = opp.realized_returns
        push!(rows, Dict{Symbol,Any}(
            :condition => condition.name,
            :run_idx => run_idx,
            :seed => seed,
            :opp_id => opp.id,
            :is_created => is_created,
            :origin_innovation_id => something(opp.origin_innovation_id, ""),
            :created_by => opp.created_by === nothing ? missing : opp.created_by,
            :creator_tier => creator === nothing ? missing : GlimpseABM.get_ai_level(creator),
            :sector => something(opp.sector, "unsectored"),
            :discovery_round => opp.discovery_round === nothing ? missing : opp.discovery_round,
            :age => opp.age,
            :capacity => opp.capacity,
            :total_invested => opp.total_invested,
            :saturation => opp.capacity > 0.0 ? opp.total_invested / opp.capacity : NaN,
            :n_capital_events => length(opp.capital_history),
            :n_realized => length(rr),
            :mean_realized_return => isempty(rr) ? NaN : sum(rr) / length(rr),
            :max_realized_return => isempty(rr) ? NaN : maximum(rr),
            :latent_return_potential => opp.latent_return_potential,
            :latent_failure_potential => something(opp.latent_failure_potential, NaN),
            :novelty_score => opp.novelty_score,
            :component_scarcity => opp.component_scarcity,
            :competition => opp.competition,
            :market_impact => opp.market_impact,
            :truly_unknown => opp.truly_unknown,
        ))
    end
    if !isempty(rows)
        CSV.write(joinpath(panel_dir,
            "$(condition.name)_run$(run_tag)_niches.csv"), DataFrame(rows))
    end
    return nothing
end

# Complete stock-flow and commitment ledgers. Unlike the end-of-run niche
# panel, these survive opportunity culling. The commitment sequence is local to
# an opportunity and therefore preserves later same-round rival commitments.
function write_opportunity_event_panels(
    condition::RobustnessCondition,
    sim::EmergentSimulation,
    run_idx::Int,
    seed::Int,
)
    panel_dir = joinpath(OUTPUT_DIR, "venture_panel")
    isdir(panel_dir) || mkpath(panel_dir)
    run_tag = lpad(run_idx, 3, '0')

    flow_rows = Dict{Symbol,Any}[]
    for event in sim.market.opportunity_flow_events
        push!(flow_rows, Dict{Symbol,Any}(
            :condition => condition.name,
            :run_idx => run_idx,
            :seed => seed,
            :round => event.round,
            :event_type => event.event_type,
            :source => event.source,
            :opportunity_id => event.opportunity_id,
            :creator_id => something(event.creator_id, missing),
            :origin_innovation_id => something(event.origin_innovation_id, ""),
            :sector => event.sector,
            :reason => event.reason,
            :capacity => event.capacity,
            :outstanding_capital => event.outstanding_capital,
        ))
    end
    flow_df = isempty(flow_rows) ? DataFrame(
        condition=String[], run_idx=Int[], seed=Int[], round=Int[],
        event_type=String[], source=String[], opportunity_id=String[],
        creator_id=Union{Missing,Int}[], origin_innovation_id=String[],
        sector=String[], reason=String[], capacity=Float64[],
        outstanding_capital=Float64[],
    ) : DataFrame(flow_rows)
    CSV.write(joinpath(panel_dir,
        "$(condition.name)_run$(run_tag)_opportunity_flows.csv"), flow_df)

    commitment_events = GlimpseABM.OpportunityCommitmentEvent[]
    for events in values(sim.market.opportunity_commitment_events)
        append!(commitment_events, events)
    end
    sort!(commitment_events; by=event ->
        (event.round, event.opportunity_id, event.sequence, event.agent_id))
    commitment_rows = Dict{Symbol,Any}[]
    for event in commitment_events
        push!(commitment_rows, Dict{Symbol,Any}(
            :condition => condition.name,
            :run_idx => run_idx,
            :seed => seed,
            :round => event.round,
            :opportunity_id => event.opportunity_id,
            :opportunity_sequence => event.sequence,
            :agent_id => event.agent_id,
            :tier => event.tier,
            :amount => event.amount,
            :capacity => event.capacity,
            :outstanding_after => event.outstanding_after,
            :capacity_saturation_at_entry =>
                event.entry_investment_capacity_ratio,
            :latent_return_potential => event.latent_return_potential,
            :instrument_estimated_return =>
                event.instrument_estimated_return,
            :decision_estimated_return => event.decision_estimated_return,
            :is_created => event.is_created,
            :self_created => event.self_created,
        ))
    end
    commitment_df = isempty(commitment_rows) ? DataFrame(
        condition=String[], run_idx=Int[], seed=Int[], round=Int[],
        opportunity_id=String[], opportunity_sequence=Int[], agent_id=Int[],
        tier=String[], amount=Float64[], capacity=Float64[],
        outstanding_after=Float64[], capacity_saturation_at_entry=Float64[],
        latent_return_potential=Float64[], instrument_estimated_return=Float64[],
        decision_estimated_return=Float64[], is_created=Bool[],
        self_created=Bool[],
    ) : DataFrame(commitment_rows)
    CSV.write(joinpath(panel_dir,
        "$(condition.name)_run$(run_tag)_commitments.csv"), commitment_df)
    return nothing
end

function tier_for_agent(agent::EmergentAgent, design::String)
    if design == "emergent"
        return GlimpseABM.get_ai_level(agent)
    end
    return isnothing(agent.fixed_ai_level) ? GlimpseABM.get_ai_level(agent) : agent.fixed_ai_level
end

function summarize_simulation(condition::RobustnessCondition, sim::EmergentSimulation, run_idx::Int, seed::Int)
    rows = Dict{Symbol,Any}[]
    all_emergent_by_tier = GlimpseABM.aggregate_emergent_uncertainty_by_tier(
        sim.agents;
        include_dead=true,
    )
    survivor_emergent_by_tier = GlimpseABM.aggregate_emergent_uncertainty_by_tier(
        sim.agents;
        include_dead=false,
    )
    market_telemetry = Dict(
        key => history_mean(sim, key)
        for key in MARKET_RECURSION_DIAGNOSTIC_KEYS
    )
    # ── Open-action (P6/P7) per-run telemetry ──────────────────────────────
    # Population-level end-of-run measures for the pivot and directed-creation
    # mechanisms. Schema is identical across conditions: with ENABLE_PIVOT /
    # ENABLE_DIRECTED_CREATION off these counts/totals are
    # genuinely zero (zero events is data, not missingness); only the pivot
    # recovery RATE — NaN on zero-pivot rounds in compile_round_stats — is
    # aggregated through finite_mean so all-NaN runs do not deflate to a fake
    # 0-rate mean (finite_mean's 0.0 fallback only fires when there were no
    # pivots at all, where the rate column is read alongside pivot_count=0).
    pivot_count_run_total = history_total(sim, "pivot_count")
    open_action_telemetry = Dict{String,Float64}(
        "pivot_count_total" => pivot_count_run_total,
        "pivot_capital_recovered_total" => history_total(sim, "pivot_capital_recovered"),
        "pivot_capital_committed_total" => history_total(sim, "pivot_capital_committed"),
        "pivot_recovery_rate_mean" =>
            finite_mean(get(h, "pivot_recovery_rate", nothing) for h in sim.history),
        "innovation_attempts_total" => history_total(sim, "innovation_attempts"),
    )
    # Directed-creation supply measures from end-of-run market state (same
    # pattern as the global_sectors scan below: computed from sim state, not
    # round history). Spawned opportunities are those created from an
    # innovation (origin_innovation_id set by spawn_opportunity_from_innovation!).
    spawned_opps = [o for o in sim.market.opportunities if !isnothing(o.origin_innovation_id)]
    spawn_sector_counts = Dict{String,Int}()
    for o in spawned_opps
        sec = isnothing(o.sector) ? "unsectored" : o.sector
        spawn_sector_counts[sec] = get(spawn_sector_counts, sec, 0) + 1
    end
    n_spawned = length(spawned_opps)
    spawn_sector_hhi = n_spawned > 0 ?
        sum((c / n_spawned)^2 for c in values(spawn_sector_counts)) : 0.0
    open_action_telemetry["spawned_opportunity_count"] = Float64(n_spawned)
    open_action_telemetry["spawned_opportunity_survivor_count"] = Float64(n_spawned)
    open_action_telemetry["spawn_sector_hhi"] = spawn_sector_hhi
    # Dispersion of spawn supply across sectors: number of distinct sectors
    # receiving at least one spawned opportunity (0 when nothing spawned).
    open_action_telemetry["spawn_sector_dispersion"] = Float64(length(spawn_sector_counts))
    # Supply-elasticity input: mean perceived crowding x spawn count.
    # strategy.jl's perceived_crowding reads
    # perception.practical_indeterminism.crowding_pressure, which is exactly
    # the per-round history key "market_crowding_pressure" — so its run mean
    # IS the perceived-crowding regressor, scaled by realized spawn supply.
    mean_perceived_crowding_run = history_mean(sim, "market_crowding_pressure")
    open_action_telemetry["mean_perceived_crowding"] = mean_perceived_crowding_run
    open_action_telemetry["supply_elasticity_input"] =
        mean_perceived_crowding_run * Float64(n_spawned)
    # Complete opportunity stock-flow counts from the append-only event ledger.
    # These totals include opportunities subsequently culled; the legacy
    # spawned_opportunity_count above remains the end-of-run survivor stock for
    # backward compatibility.
    for (key, value) in GlimpseABM.opportunity_flow_counts(sim.market)
        open_action_telemetry[key] = value
    end
    open_action_telemetry["exploration_niche_events_total"] = Float64(sum(
        a.uncertainty_metrics.niches_discovered for a in sim.agents;
        init=0,
    ))
    # Number of distinct sectors invested in across the whole population — the
    # category count for normalizing per-tier sector HHI into [0,1] overlap.
    global_sectors = Set{String}()
    for a in sim.agents
        union!(global_sectors, keys(a.sector_investment_counts))
    end
    n_sectors_global = max(length(global_sectors), 1)
    for tier in AI_TIERS
        agents = [a for a in sim.agents if tier_for_agent(a, condition.design) == tier]
        n_tier = length(agents)
        alive = [a for a in agents if a.alive]
        tier_all_emergent = get(all_emergent_by_tier, tier, Dict{String,Float64}())
        tier_survivor_emergent = get(survivor_emergent_by_tier, tier, Dict{String,Float64}())
        all_emergent_fields = emergent_summary_fields("all_agent", tier_all_emergent)
        survivor_emergent_fields = emergent_summary_fields("survivor", tier_survivor_emergent)

        all_competition = Float64[]
        action_counts = Dict("invest" => 0, "innovate" => 0, "explore" => 0, "maintain" => 0)
        for agent in agents
            append!(all_competition, agent.uncertainty_metrics.competition_levels)
            for action in agent.action_history
                if haskey(action_counts, action)
                    action_counts[action] += 1
                end
            end
        end
        total_actions = sum(values(action_counts))
        total_invested = isempty(agents) ? 0.0 : sum(a.total_invested for a in agents)
        total_returned = isempty(agents) ? 0.0 : sum(a.total_returned for a in agents)
        final_capitals = [GlimpseABM.get_capital(a) for a in agents]
        # Liquidity diagnostics: in-flight (deployed, not-yet-matured) capital,
        # net worth (cash + in-flight), and failure-reason breakdown. Separates
        # genuine insolvency from illiquidity in the survival result.
        in_flight_capitals = [isempty(a.active_investments) ? 0.0 :
            sum(inv["amount"] for inv in a.active_investments) for a in agents]
        net_worths = [GlimpseABM.get_capital(a) + ifc for (a, ifc) in zip(agents, in_flight_capitals)]
        n_liquidity_fail = count(a -> !a.alive && a.failure_reason == "liquidity_failure", agents)
        n_equity_fail = count(a -> !a.alive && a.failure_reason == "equity_failure", agents)
        # Per-tier sector-concentration HHI: pool this tier's cumulative
        # per-sector investment counts, then HHI = Σ(share²). Higher = this tier
        # converges its investments into fewer sectors (direct correlated-
        # commitment evidence, disentangled from the diluted population HHI).
        tier_sector_counts = Dict{String,Int}()
        for a in agents
            for (sec, c) in a.sector_investment_counts
                tier_sector_counts[sec] = get(tier_sector_counts, sec, 0) + c
            end
        end
        tier_sector_total = sum(values(tier_sector_counts); init=0)
        tier_sector_hhi = tier_sector_total > 0 ?
            sum((c / tier_sector_total)^2 for c in values(tier_sector_counts)) : 0.0
        min_hhi = 1.0 / n_sectors_global
        tier_sector_overlap = (tier_sector_total > 0 && n_sectors_global > 1) ?
            clamp((tier_sector_hhi - min_hhi) / (1.0 - min_hhi), 0.0, 1.0) : 0.0
        confidence_outcome = GlimpseABM.confidence_outcome_stats(agents)
        tier_total_niches = isempty(agents) ? 0 : sum(a.uncertainty_metrics.niches_discovered for a in agents)
        tier_combinations = isempty(agents) ? 0 : sum(a.uncertainty_metrics.new_combinations_created for a in agents)
        tier_venture_events = [
            event for agent in agents for event in agent.venture_ledger
        ]
        n_matured_investments = length(tier_venture_events)
        rival_incidence = n_matured_investments == 0 ? missing :
            count(event -> isfinite(event[14]) && event[14] > 0.0,
                  tier_venture_events) / n_matured_investments
        perception_telemetry = Dict(
            key => history_mean(sim, "mean_$(key)_$(tier)")
            for key in TIER_PERCEPTION_KEYS
        )

        # Per-tier statistics for a tier with zero agents are `missing`
        # (CSV: empty cell; pandas: NaN), never a 0.0/prior sentinel. The row
        # itself and the n_agents=0 count are kept so the schema is stable.
        tstat(x) = n_tier == 0 ? missing : x
        row = Dict{Symbol,Any}(
            :condition => condition.name,
            :category => condition.category,
            :description => condition.description,
            :theoretical_role => condition.theoretical_role,
            :design => condition.design,
            :run_idx => run_idx,
            :seed => seed,
            :tier => tier,
            :tier_label => tier_display_label(tier),
            :tier_rank => TIER_RANK[tier],
            :n_agents => n_tier,
            :tier_share => n_tier / max(length(sim.agents), 1),
            :survival_rate => n_tier == 0 ? missing : length(alive) / n_tier,
            :mean_final_capital => isempty(final_capitals) ? missing : mean(final_capitals),
            :mean_in_flight_capital => isempty(in_flight_capitals) ? missing : mean(in_flight_capitals),
            :mean_net_worth => isempty(net_worths) ? missing : mean(net_worths),
            :liquidity_failure_rate => n_tier == 0 ? missing : n_liquidity_fail / n_tier,
            :equity_failure_rate => n_tier == 0 ? missing : n_equity_fail / n_tier,
            :tier_sector_invest_hhi => tier_sector_total > 0 ? tier_sector_hhi : missing,
            :tier_sector_overlap => tier_sector_total > 0 ? tier_sector_overlap : missing,
            :roi => total_invested > 0 ? (total_returned - total_invested) / total_invested : missing,
            :matured_investment_count => n_matured_investments,
            :mean_capacity_saturation_at_entry =>
                venture_metric_mean(tier_venture_events, 10),
            :mean_capacity_saturation_at_maturity =>
                venture_metric_mean(tier_venture_events, 11),
            :mean_effective_capacity_saturation_at_maturity =>
                venture_effective_saturation_mean(tier_venture_events),
            :mean_capacity_saturation_change =>
                venture_metric_mean(tier_venture_events, 13),
            :mean_post_commitment_rival_capital =>
                venture_metric_mean(tier_venture_events, 14),
            :mean_post_commitment_rival_capacity_ratio =>
                venture_ratio_mean(tier_venture_events, 14, 8),
            :post_commitment_rival_incidence => rival_incidence,
            :mean_crowding_return_multiplier =>
                venture_metric_mean(tier_venture_events, 15),
            :mean_realized_latent_capture =>
                venture_metric_mean(tier_venture_events, 16),
            :mean_performative_effective_capacity =>
                venture_metric_mean(tier_venture_events, 17),
            :mean_performative_lagged_commitment =>
                venture_metric_mean(tier_venture_events, 18),
            :mean_performative_commitment_share =>
                venture_metric_mean(tier_venture_events, 19),
            :mean_performative_uplift_fraction =>
                venture_metric_mean(tier_venture_events, 20),
            :mean_competition => isempty(all_competition) ? missing : mean(all_competition),
            :max_competition => isempty(all_competition) ? missing : maximum(all_competition),
            :innovations_per_agent => n_tier == 0 ? missing : sum(a.innovation_count for a in agents) / n_tier,
            :niches_per_agent => n_tier == 0 ? missing : tier_total_niches / n_tier,
            :combinations_per_agent => n_tier == 0 ? missing : tier_combinations / n_tier,
            :invest_share => total_actions > 0 ? action_counts["invest"] / total_actions : missing,
            :innovate_share => total_actions > 0 ? action_counts["innovate"] / total_actions : missing,
            :explore_share => total_actions > 0 ? action_counts["explore"] / total_actions : missing,
            :maintain_share => total_actions > 0 ? action_counts["maintain"] / total_actions : missing,
            :emergent_actor_ignorance => all_emergent_fields["all_agent_emergent_actor_ignorance"],
            :emergent_practical_indeterminism => all_emergent_fields["all_agent_emergent_practical_indeterminism"],
            :emergent_agentic_novelty => all_emergent_fields["all_agent_emergent_agentic_novelty"],
            :emergent_competitive_recursion => all_emergent_fields["all_agent_emergent_competitive_recursion"],
            :mean_visible_opportunities => tstat(history_mean(sim, "mean_visible_opportunities_$(tier)")),
            :mean_info_quality_used => tstat(history_mean(sim, "mean_info_quality_used_$(tier)")),
            :visible_hhi => tstat(history_mean(sim, "visible_hhi_$(tier)")),
        )
        # all/survivor emergent fields carry their own per-dimension missingness
        # (gated on observation counts); market telemetry and open-action
        # (P6/P7) telemetry are population-level (repeated across tier rows,
        # zeros where the open-action flags are off — genuine zero counts).
        # Perception telemetry and confidence-outcome stats are per-tier agent
        # statistics, so they are missing for an empty tier.
        for source in (all_emergent_fields, survivor_emergent_fields, market_telemetry,
                       open_action_telemetry)
            for (key, value) in source
                row[Symbol(key)] = value
            end
        end
        for source in (perception_telemetry, confidence_outcome)
            for (key, value) in source
                row[Symbol(key)] = tstat(value)
            end
        end
        push!(rows, row)
    end
    return rows
end

function all_conditions()
    RobustnessCondition[
        RobustnessCondition(
            "BASELINE", "baseline",
            "Canonical mixed fixed-tier design.",
            "Reference estimate for all paired condition-vs-baseline comparisons.",
            "fixed_mixed", "minimal", noop!),

        RobustnessCondition(
            "SHAM_AI_LABELS_FREE", "internal_validity_placebo",
            "Strict placebo: non-none labels are retained for grouping but all non-none tiers use no-AI behavior with no AI costs.",
            "If tier labels alone generate effects, the model is biased; effects should collapse toward zero.",
            "fixed_mixed", "minimal", sham_ai_labels_free!),

        RobustnessCondition(
            "NO_INFORMATION_ADVANTAGE_FREE", "internal_validity_placebo",
            "Paid tiers receive human-level information and domain capabilities, with AI costs removed.",
            "Separates information-processing advantage from residual tier-label behavior.",
            "fixed_mixed", "core", no_information_advantage_free!),

        RobustnessCondition(
            "FREE_AI", "internal_validity_placebo",
            "AI subscription and usage costs are zero.",
            "Tests whether the frontier_ai result is a pricing artifact rather than a future-knowledge mechanism.",
            "fixed_mixed", "minimal", free_ai!),

        RobustnessCondition(
            "NO_AI_ERRORS", "internal_validity_placebo",
            "Remove hallucination and overconfidence penalties.",
            "Tests whether the paradox is reducible to AI mistakes rather than endogenous market response.",
            "fixed_mixed", "minimal", no_hallucination_overconfidence!),

        RobustnessCondition(
            "NO_CROWDING", "mechanism_decomposition",
            "Disable opportunity-level competition dynamics and convex crowding return dilution.",
            "Refutes the second-order crowding channel if frontier_ai penalties persist unchanged.",
            "fixed_mixed", "minimal", no_crowding!),

        RobustnessCondition(
            "NO_RETURN_DILUTION", "mechanism_decomposition",
            "Set only the convex capital-saturation return-dilution term to zero; competition and recursion signals remain active.",
            "Distinguishes economic crowding losses from informational/behavioral convergence.",
            "fixed_mixed", "core", no_return_dilution!),

        RobustnessCondition(
            "PERFORMATIVE_COMMON_025", "performativity",
            "Performative demand (ψ=0.25, common): effective niche capacity grows with lagged collective dollar commitment; the enlarged demand is shared by all founders in the niche.",
            "Tests whether the frontier deficit is an artifact of objective/fixed value by making demand endogenously performed by prior collective action; survival of the deficit shows the mechanism runs on capture under convergence, not on value being pre-given.",
            "fixed_mixed", "expanded", performative_common_025!),
        RobustnessCondition(
            "PERFORMATIVE_COMMON_050", "performativity",
            "Performative demand (ψ=0.50, common): effective niche capacity grows with lagged collective dollar commitment; the enlarged demand is shared by all founders in the niche.",
            "Dose-response for performed demand as a common good.",
            "fixed_mixed", "expanded", performative_common_050!),
        RobustnessCondition(
            "PERFORMATIVE_COMMON_100", "performativity",
            "Performative demand (ψ=1.00, common): effective niche capacity grows with lagged collective dollar commitment; the enlarged demand is shared by all founders in the niche.",
            "Upper-dose performed-demand common-good test; if the deficit only closes here, escape requires implausibly elastic performed demand.",
            "fixed_mixed", "expanded", performative_common_100!),
        RobustnessCondition(
            "PERFORMATIVE_APPROPRIABLE_025", "performativity",
            "Performative demand (ψ=0.25, appropriable): the demand uplift accrues to each founder in proportion to their own commitment share (category ownership).",
            "Tests whether performativity escapes the trap only when the performed demand is privately appropriable — the same appropriability condition the execution-moat results identify.",
            "fixed_mixed", "expanded", performative_appropriable_025!),
        RobustnessCondition(
            "PERFORMATIVE_APPROPRIABLE_050", "performativity",
            "Performative demand (ψ=0.50, appropriable): the demand uplift accrues to each founder in proportion to their own commitment share (category ownership).",
            "Dose-response for appropriable performed demand.",
            "fixed_mixed", "expanded", performative_appropriable_050!),
        RobustnessCondition(
            "PERFORMATIVE_APPROPRIABLE_100", "performativity",
            "Performative demand (ψ=1.00, appropriable): the demand uplift accrues to each founder in proportion to their own commitment share (category ownership).",
            "Upper-dose appropriable performed-demand test.",
            "fixed_mixed", "expanded", performative_appropriable_100!),

        RobustnessCondition(
            "CROWDING_ACTION_HHI_BLEND", "mechanism_decomposition",
            "Previous payoff equation: add 0.30 times the market-wide action-category HHI to each opportunity's outstanding-capital load.",
            "Tests whether the pure I/K canonical result depends on excluding a market-wide action-concentration term from the local capacity constraint.",
            "fixed_mixed", "expanded", crowding_action_hhi_blend!),

        RobustnessCondition(
            "NO_CREATED_BONUS", "mechanism_decomposition",
            "Remove the created-opportunity scoring affinities (any-created x1.1, own-creation x1.2 -> both 1.0).",
            "Tests whether creation-channel convergence depends on the hard-coded scoring attraction to founder-created niches.",
            "fixed_mixed", "expanded", no_created_bonus!),

        RobustnessCondition(
            "OPPORTUNITY_COMPONENT_FIT_ON", "mechanism_decomposition",
            "Attach knowledge-component requirements to opportunities so founder-specific component overlap creates within-sector fit, with no strategy-rule change.",
            "Isolates whether finer founder-opportunity fit disperses selection or attenuates the frontier deficit without bundling comparative-advantage strategy.",
            "fixed_mixed", "expanded", opportunity_component_fit_on!),

        RobustnessCondition(
            "ENDOGENOUS_OPPORTUNITIES_ONLY", "mechanism_decomposition",
            "Disable background public-opportunity replenishment after initialization; exploration and successful innovation remain the only sources of new opportunities.",
            "Tests whether the result and open-futures dynamics depend on an exogenous continuing flow of market opportunities.",
            "fixed_mixed", "expanded", endogenous_opportunities_only!),

        RobustnessCondition(
            "NICHE_MULTIPLICITY_ONE", "mechanism_decomposition",
            "Each successful exploration-created niche event opens exactly one opportunity instead of a uniform one-to-three draw.",
            "Lower-bound test for whether arbitrary niche multiplicity manufactures the opportunity-creation channel.",
            "fixed_mixed", "expanded", niche_multiplicity_one!),

        RobustnessCondition(
            "NICHE_MULTIPLICITY_FIXED_TWO", "mechanism_decomposition",
            "Each successful exploration-created niche event opens exactly two opportunities, preserving the baseline draw's mean while removing its variance.",
            "Tests whether stochastic multiplicity rather than endogenous creation drives the result.",
            "fixed_mixed", "expanded", niche_multiplicity_fixed_two!),

        RobustnessCondition(
            "NICHE_MULTIPLICITY_THREE", "mechanism_decomposition",
            "Each successful exploration-created niche event opens exactly three opportunities.",
            "Upper-bound dose test for the opportunity-creation multiplicity assumption.",
            "fixed_mixed", "expanded", niche_multiplicity_three!),

        RobustnessCondition(
            "NO_AI_HERDING_RECURSION", "mechanism_decomposition",
            "Suppress AI herding and AI action-correlation contributions to uncertainty recursion.",
            "Tests whether the paradox depends on shared AI signal convergence specifically.",
            "fixed_mixed", "core", no_ai_herding_recursion!),

        RobustnessCondition(
            "CASH_ONLY_SURVIVAL", "mechanism_decomposition",
            "Legacy cash-on-hand survival rule: in-flight investment capital does not count toward solvency (counterfactual to the net-worth default).",
            "Isolates how much of the frontier survival gap is illiquidity (deployed-but-maturing capital) versus genuine insolvency.",
            "fixed_mixed", "core", cash_only_survival!),

        RobustnessCondition(
            "LOW_CROWDING", "mechanism_decomposition",
            "Reduced opportunity-level competition accumulation and convex crowding strength.",
            "Sensitivity check for the market convergence/crowding mechanism.",
            "fixed_mixed", "core", low_crowding!),

        RobustnessCondition(
            "HIGH_CROWDING", "mechanism_decomposition",
            "Increased opportunity-level competition accumulation and convex crowding strength.",
            "Stress test for whether second-order costs intensify under denser convergence.",
            "fixed_mixed", "core", high_crowding!),

        RobustnessCondition(
            "AI_COST_2X", "mechanism_decomposition",
            "Double AI costs; bounds sensitivity to pricing assumptions.",
            "Tests whether cost pressure amplifies but does not solely create the paradox.",
            "fixed_mixed", "core", double_ai_cost!),

        RobustnessCondition(
            "NO_RETURN_NOISE", "mechanism_decomposition",
            "Remove idiosyncratic execution noise on realized returns (RETURN_NOISE_SCALE=0); the opportunity-level Pareto draw remains.",
            "Tests whether the frontier trap depends on noise-under-threshold success classification rather than convergence; also removes the only channel decorrelating co-investor outcomes.",
            "fixed_mixed", "core", no_return_noise!),

        RobustnessCondition(
            "RETURN_NOISE_LOW", "mechanism_decomposition",
            "Halve idiosyncratic return noise (sigma=0.19, 0.5x default).",
            "Lower bracket of the execution-noise sensitivity sweep pending the venture-dispersion calibration pass.",
            "fixed_mixed", "core", return_noise_low!),

        RobustnessCondition(
            "RETURN_NOISE_HIGH", "mechanism_decomposition",
            "Increase idiosyncratic return noise by half (sigma=0.57, 1.5x default).",
            "Upper bracket of the execution-noise sensitivity sweep pending the venture-dispersion calibration pass.",
            "fixed_mixed", "core", return_noise_high!),

        RobustnessCondition(
            "NOVELTY_GENERALIZATION_STRESS_ON", "mechanism_decomposition",
            "Enable an explicit AI generalization stress on novel, data-poor, stealth, and endogenous opportunities.",
            "Tests a stronger agentic-novelty interpretation as a refutation/stress cell, not a baseline assumption.",
            "fixed_mixed", "expanded", novelty_generalization_stress_on!),

        RobustnessCondition(
            "FRONTIER_COVERAGE_MAX", "alternative_frontier_ai_specification",
            "Frontier AI sees maximal public opportunity coverage and breadth.",
            "Operationalizes AGI as near-complete access to public, data-rich opportunity evidence.",
            "fixed_mixed", "minimal", frontier_coverage_max!),

        RobustnessCondition(
            "FRONTIER_REASONING_NO_ERROR", "alternative_frontier_ai_specification",
            "Frontier AI has near-perfect domain reasoning with zero hallucination in all AI domains.",
            "Operationalizes AGI as reliable reasoning/calibration rather than merely more visible opportunities.",
            "fixed_mixed", "minimal", frontier_reasoning_no_error!),

        RobustnessCondition(
            "FRONTIER_PUBLIC_OMNISCIENCE", "alternative_frontier_ai_specification",
            "Frontier AI receives maximal public coverage, near-perfect domain reasoning, and large quality boosts.",
            "Near-omniscience counterfactual for public-information access.",
            "fixed_mixed", "minimal", frontier_public_omniscience!),

        RobustnessCondition(
            "FRONTIER_EXEC_5X", "alternative_frontier_ai_specification",
            "Frontier AI receives 5x investment execution success multiplier.",
            "Tests whether execution dominance can overcome second-order future-knowledge costs.",
            "fixed_mixed", "minimal", premium_exec_5x!),

        # --- Option-B comparison cells: defensibility dose-response, tail
        # heaviness, and cost-calibration invariance. ---
        RobustnessCondition(
            "FRONTIER_EXEC_1_5X", "alternative_frontier_ai_specification",
            "Frontier AI receives 1.5x investment execution success multiplier.",
            "Defensibility dose-response (P19): how strong must an isolating mechanism be to escape the trap?",
            "fixed_mixed", "expanded", frontier_exec_1_5x!),
        RobustnessCondition(
            "FRONTIER_EXEC_2X", "alternative_frontier_ai_specification",
            "Frontier AI receives 2x investment execution success multiplier.",
            "Defensibility dose-response (P19).",
            "fixed_mixed", "expanded", frontier_exec_2x!),
        RobustnessCondition(
            "FRONTIER_EXEC_3X", "alternative_frontier_ai_specification",
            "Frontier AI receives 3x investment execution success multiplier.",
            "Defensibility dose-response (P19).",
            "fixed_mixed", "expanded", frontier_exec_3x!),
        RobustnessCondition(
            "TRUNCATED_TAIL", "opportunity_distribution",
            "Conservative robustness: remove the canonical niche-size tail and retain truncated opportunity-return caps.",
            "P20: the no-size-tail distribution is the conservative case; the trap should persist.",
            "fixed_mixed", "expanded", truncated_tail!),
        RobustnessCondition(
            "MODERATE_TAIL", "opportunity_distribution",
            "Intermediate canonical tail: niche-size log-sigma halfway between truncated and baseline; return-tail settings retained for option-B comparisons.",
            "P20: tail dose-response; the trap is robust across market-size tail weight.",
            "fixed_mixed", "expanded", moderate_tail!),
        RobustnessCondition(
            "OPS_COST_060", "cost_calibration_invariance",
            "Operating-cost level 0.60 (lower; higher aggregate survival).",
            "P21: the frontier deficit is invariant to where in the venture-class band the cost anchor sits.",
            "fixed_mixed", "expanded", ops_cost_060!),
        RobustnessCondition(
            "OPS_COST_075", "cost_calibration_invariance",
            "Operating-cost level 0.75 (between the low-cost stress and the calibrated 0.85 baseline).",
            "P21 cost-calibration invariance.",
            "fixed_mixed", "expanded", ops_cost_075!),
        RobustnessCondition(
            "OPS_COST_100", "cost_calibration_invariance",
            "Operating-cost level 1.00 (default; lower aggregate survival).",
            "P21 cost-calibration invariance.",
            "fixed_mixed", "expanded", ops_cost_100!),

        RobustnessCondition(
            "FRONTIER_QUALITY_PLUS50", "alternative_frontier_ai_specification",
            "Frontier AI receives +50pp innovation quality and information-quality boost.",
            "Tests an AGI-as-output-quality interpretation rather than an AGI-as-market-convergence interpretation.",
            "fixed_mixed", "minimal", premium_quality_plus50!),

        RobustnessCondition(
            "ALL_FAVORABLE_TO_FRONTIER", "alternative_frontier_ai_specification",
            "Extreme pro-frontier cell: no AI cost/error/crowding/novelty penalty plus execution and quality boosts.",
            "Sanity check that the model can reverse under an intentionally pro-frontier world.",
            "fixed_mixed", "minimal", all_favorable_to_premium!),

        RobustnessCondition(
            "STRATEGIC_ANTICIPATION_ON", "strategy_operationalization",
            "Enable AI-assisted anticipation of crowded shared signals.",
            "Independent-implementation robustness check for the canonical S1 strategy-ladder operationalization.",
            "fixed_mixed", "core", strategic_anticipation_on!),

        RobustnessCondition(
            "STRATEGIC_DIVERSIFICATION_ON", "strategy_operationalization",
            "Enable anticipation plus equilibrium-aware diversification across near-top opportunities.",
            "Independent-implementation robustness check for the canonical S1 strategy-ladder operationalization.",
            "fixed_mixed", "core", differentiated_strategy_on!),

        RobustnessCondition(
            "AI_COMPLEMENTARITY_ON", "alternative_frontier_ai_specification",
            "AI accuracy depends on agent traits and sector knowledge.",
            "Operationalizes AGI as a complement to entrepreneurial judgment rather than a substitute.",
            "fixed_mixed", "core", ai_complementarity_on!),

        # Pure usage-based pricing is the baseline (AI_COST_MODEL="token" struct
        # default), while subscription-era pricing is tested as an alternative.
        RobustnessCondition(
            "SUBSCRIPTION_ERA", "alternative_frontier_ai_specification",
            "Subscription-era pricing: monthly seat fees (\$400 advanced / \$3,500 frontier) plus metered usage, replacing the pure usage-based default.",
            "Tests whether the frontier result depends on the marginal-compute pricing default rather than a subscription-plus-usage pricing architecture.",
            "fixed_mixed", "core", subscription_era!),

        RobustnessCondition(
            "DIFFICULTY_SCALED_AI_COST", "alternative_frontier_ai_specification",
            "AI call costs scale with opportunity complexity.",
            "Tests whether frontier AGI is differentially exposed to expensive reasoning on hard opportunities.",
            "fixed_mixed", "core", difficulty_scaled_ai_cost!),

        RobustnessCondition(
            "MARKET_SLACK_HIGH_CAPACITY", "boundary_conditions_generalizability",
            "Opportunity capacity is high and crowding is weaker.",
            "Boundary condition: paradox should weaken when market slack absorbs convergent investment.",
            "fixed_mixed", "core", market_slack_high_capacity!),

        RobustnessCondition(
            "CAPACITY_MEAN_HALF", "boundary_conditions_generalizability",
            "Halve mean opportunity capacity while holding the oversubscription threshold, penalty strength, convexity, and capacity-tail dispersion fixed.",
            "One-factor identification: tests whether the dollar capacity scale alone creates the tier contrast.",
            "fixed_mixed", "expanded", capacity_mean_half!),

        RobustnessCondition(
            "CAPACITY_MEAN_DOUBLE", "boundary_conditions_generalizability",
            "Double mean opportunity capacity while holding the oversubscription threshold, penalty strength, convexity, and capacity-tail dispersion fixed.",
            "One-factor identification: tests whether the dollar capacity scale alone creates the tier contrast.",
            "fixed_mixed", "expanded", capacity_mean_double!),

        RobustnessCondition(
            "CAPACITY_AWARE_SELECTION", "boundary_conditions_generalizability",
            "Agents fully discount the currently observable opportunity-specific I/K penalty when ranking opportunities; contemporaneous rival commitments remain unobserved.",
            "Tests whether capacity losses require strategically naive founders rather than simultaneous convergence on opportunities that looked unsaturated at commitment.",
            "fixed_mixed", "expanded", capacity_aware_selection!),

        RobustnessCondition(
            "MARKET_DENSE_LOW_CAPACITY", "boundary_conditions_generalizability",
            "Opportunity capacity is low and crowding is stronger.",
            "Boundary condition: paradox should strengthen when opportunity spaces saturate quickly.",
            "fixed_mixed", "core", market_dense_low_capacity!),

        RobustnessCondition(
            "OPPORTUNITY_RICH_ENVIRONMENT", "boundary_conditions_generalizability",
            "More initial opportunities and higher discovery probability.",
            "Boundary condition: broad opportunity supply should reduce convergence pressure.",
            "fixed_mixed", "core", opportunity_rich_environment!),

        RobustnessCondition(
            "OPPORTUNITY_SPARSE_ENVIRONMENT", "boundary_conditions_generalizability",
            "Fewer initial opportunities and lower discovery probability.",
            "Boundary condition: scarce opportunity supply should amplify convergence pressure.",
            "fixed_mixed", "core", opportunity_sparse_environment!),

        RobustnessCondition(
            "EQUAL_SECTORS", "boundary_conditions_generalizability",
            "Even 25/25/25/25 split across the four sectors, replacing the NVCA-weighted 60% tech baseline.",
            "Sector-composition robustness: if the deficit is an artifact of the tech-heavy weighting, a balanced population should eliminate it; we predict it attenuates but persists, because crowding operates within every sector.",
            "fixed_mixed", "expanded", equal_sectors!),

        RobustnessCondition(
            "SERVICES_HEAVY", "boundary_conditions_generalizability",
            "Low-tech population dominated by services and retail (service 45 / retail 25 / manufacturing 20 / tech 10).",
            "Opposite extreme from the tech-heavy baseline: the high-competition, heavy-tail tech sector is where convergence bites hardest, so a services-heavy mix should show the smallest deficit; a deficit that survives here shows the paradox is not a tech-sector artifact.",
            "fixed_mixed", "expanded", services_heavy!),

        RobustnessCondition(
            "EMERGENT_ADOPTION_BASELINE", "boundary_conditions_generalizability",
            "Agents start without AI and choose tiers endogenously.",
            "Generalizability check against fixed equal-tier assignment.",
            "emergent", "minimal", noop!),

        # ── Strategy ladder ────────────────────────────────────────────────
        RobustnessCondition(
            "FRONTIER_STRATEGY_CONSENSUS", "agi_strategy_ladder",
            "Frontier-tier agents anticipate consensus congestion (S1): expected returns of consensus-legible opportunities shaded before commitment.",
            "Congestion channel: tests whether the frontier deficit shrinks when frontier users discount opportunities that many rivals' instruments also rank highly.",
            "fixed_mixed", "expanded", frontier_strategy_consensus!),

        RobustnessCondition(
            "FRONTIER_STRATEGY_COMPARATIVE", "agi_strategy_ladder",
            "Frontier-tier agents re-rank by private edge (S2): sector familiarity, knowledge overlap, and execution traits weight the AI's landscape map.",
            "Heterogeneity channel: tests whether founder-market fit adds dispersive information beyond the AI landscape map.",
            "fixed_mixed", "expanded", frontier_strategy_comparative!),

        RobustnessCondition(
            "FRONTIER_STRATEGY_COMPLEMENT", "agi_strategy_ladder",
            "Frontier-tier agents seek complements (S3): explore/innovate effort shifts toward opportunities where their own instrument reports low confidence.",
            "Trap attenuation, codifiability-frontier channel (Hayek/effectuation): building where shared models add least should reduce convergent entry.",
            "fixed_mixed", "expanded", frontier_strategy_complement!),

        RobustnessCondition(
            "FRONTIER_STRATEGY_COMPOSITE", "agi_strategy_ladder",
            "Frontier-tier agents run S1+S2+S3 jointly (agi_native).",
            "Frontier adaptivity composite: tests whether S1+S2+S3 together recover relative advantage beyond the FREE_AI bound.",
            "fixed_mixed", "expanded", frontier_strategy_composite!),

        RobustnessCondition(
            "ALL_STRATEGY_CONSENSUS", "agi_strategy_ladder",
            "Every agent anticipates consensus congestion (S1), regardless of tier.",
            "Reflexivity relocation test: when contrarianism is widespread, does crowding relocate into formerly neglected opportunities or decline?",
            "fixed_mixed", "expanded", all_strategy_consensus!),

        RobustnessCondition(
            "ALL_STRATEGY_COMPARATIVE", "agi_strategy_ladder",
            "Every agent re-ranks by private edge (S2), regardless of tier.",
            "Reflexivity relocation test: symmetric founder-market-fit sorting should disperse entry along heterogeneous edges; P5 predicts the survival value of the strategy still rises with tier.",
            "fixed_mixed", "expanded", all_strategy_comparative!),

        RobustnessCondition(
            "ALL_STRATEGY_COMPARATIVE_COMPONENTS", "agi_strategy_ladder",
            "Every agent re-ranks by private edge (S2) using PER-OPPORTUNITY knowledge-component overlap (ENABLE_OPPORTUNITY_COMPONENTS), not the coarse sector-familiarity fallback.",
            "Tests whether within-sector founder-market fit (recombinant-knowledge overlap, gate ~0.55) adds dispersion beyond sector-familiarity S2.",
            "fixed_mixed", "expanded", all_strategy_comparative_components!),

        RobustnessCondition(
            "ALL_STRATEGY_COMPLEMENT", "agi_strategy_ladder",
            "Every agent seeks complements (S3), regardless of tier.",
            "Reflexivity relocation test: if everyone targets oracle-blind opportunities, second-order crowding should emerge in low-legibility niches (P4's signature).",
            "fixed_mixed", "expanded", all_strategy_complement!),

        RobustnessCondition(
            "ALL_STRATEGY_COMPOSITE", "agi_strategy_ladder",
            "Every agent runs S1+S2+S3 jointly (agi_native), regardless of tier.",
            "Population-wide strategy test: does the paradox of future knowledge survive symmetric AGI-native strategy, or does reflexivity attenuate under symmetric sophistication?",
            "fixed_mixed", "expanded", all_strategy_composite!),

        # ── Open-action extension ─────────────────────────────────────────
        RobustnessCondition(
            "OPEN_ACTION_PIVOT", "open_action_space",
            "Abandonment option (A1): agents review in-flight investments each round and may liquidate at an age-dependent haircut (0.40 → 0.75 of committed capital), redeploying the recovery. Deterioration gain 2.0 per liveness probe (scripts/probe_pivot_liveness.jl).",
            "Pivot-channel test: measures whether escape from crowded commitments attenuates the trap and whether pivots re-correlate onto consensus second-best opportunities.",
            "fixed_mixed", "expanded", open_action_pivot!),

        RobustnessCondition(
            "OPEN_ACTION_DIRECTED_CREATION", "open_action_space",
            "Hayekian redirection (A2): innovating agents bias new-combination sector selection away from sectors they perceive as crowded toward sparse ones; creation volume is unchanged.",
            "Directed-creation test: measures whether steering new combinations toward sparse sectors raises supply elasticity and attenuates convergence pressure.",
            "fixed_mixed", "expanded", open_action_directed_creation!),

        RobustnessCondition(
            "OPEN_ACTION_BOTH", "open_action_space",
            "Both open-action channels (A1 pivot + A2 directed creation) active for all agents. Pivot deterioration gain 2.0 per liveness probe.",
            "P6 + P7 jointly: the full open-action space without strategic sophistication — does opening exit AND entry topology together leave the trap standing where each channel alone does?",
            "fixed_mixed", "expanded", open_action_both!),

        RobustnessCondition(
            "OPEN_ACTION_AGI_NATIVE_MARKET", "open_action_space",
            "Maximal AGI-robustness cell: open action space (pivot + directed creation, pivot gain 2.0 per liveness probe) with every agent running the composite AGI-native strategy.",
            "Joint open-action and strategy test: measures whether the paradox survives both open action spaces and population-wide strategic sophistication.",
            "fixed_mixed", "expanded", open_action_agi_native_market!),

        # ── Dose-response sweeps ──────────────────────────────────────────
        # Each cell pushes one dial to its maximum effective value, bounded by
        # the use-site clamps described near the apply! functions.
        RobustnessCondition(
            "DOSE_S1_FRONTIER_2X", "dose_response",
            "Dose probe: S1 raw consensus discount 2.0, frontier-only. The use-site clamp [0,1] (strategy_shaded_return) neutralizes this to an effective 1.0 — i.e. the documented default. Under paired seeds this cell should be bit-identical to FRONTIER_STRATEGY_CONSENSUS; any divergence would flag a saturation mismatch.",
            "Dose-response: verifies that the S1 default is already the maximum effective dose under the use-site clamp.",
            "fixed_mixed", "expanded", dose_s1_frontier_2x!),

        RobustnessCondition(
            "DOSE_S1_FRONTIER_MAXX", "dose_response",
            "Dose ceiling: S1 consensus discount 1.0, frontier-only — the maximum value NOT clamp-neutralized (use-site clamp [0,1]), which equals the conservative default already run. Together with the 2X probe this pins the S1 dose-response curve as saturated from the start.",
            "Dose-response: bounds the S1 attenuation curve at the maximum expressible anticipatory-congestion discount.",
            "fixed_mixed", "expanded", dose_s1_frontier_maxx!),

        RobustnessCondition(
            "DOSE_S2_FRONTIER_2X", "dose_response",
            "Dose: S2 edge weight 2.0, frontier-only — the use-site clamp ceiling [0,2] (strategy_score_multiplier), i.e. the maximum effective S2 dose, doubling the founder-market-fit re-ranking gradient vs the default 1.0. (The overall [0.25,2.0] multiplier clamp additionally binds when |edge − mean edge| > 0.5.)",
            "Dose-response: bounds the S2 attenuation curve at its private-edge re-ranking ceiling.",
            "fixed_mixed", "expanded", dose_s2_frontier_2x!),

        RobustnessCondition(
            "DOSE_COMPOSITE_FRONTIER_MAX", "dose_response",
            "Maximal-dose composite (agi_native), frontier-only: consensus discount 1.0 (S1 already saturated at its [0,1] clamp ceiling), edge weight 2.0 ([0,2] ceiling), complement shift 2.0 (S3a [0,2] ceiling; the S3b softening path clamps strength to [0,1] and stays at its saturated default). The strongest frontier-only strategy treatment the implementation can express.",
            "Dose-response: tests the strongest frontier-only composite strategy treatment expressible by the implementation.",
            "fixed_mixed", "expanded", dose_composite_frontier_max!),

        RobustnessCondition(
            "DOSE_COMPOSITE_ALL_MAX", "dose_response",
            "Maximal-dose composite (agi_native) for every tier: population-wide S1+S2+S3 at max-effective strengths (discount 1.0 / edge weight 2.0 / complement shift 2.0 — the use-site clamp ceilings).",
            "Dose-response analogue of P4 at the dial ceilings: does symmetric maximal sophistication retire the trap, relocate it, or leave it standing where defaults did neither? Bounds the population-wide attenuation curve.",
            "fixed_mixed", "expanded", dose_composite_all_max!),

        RobustnessCondition(
            "DOSE_PIVOT_GAIN_4X", "dose_response",
            "Abandonment option at deterioration gain 4.0. Gain is validated >= 0 and unbounded above; the trigger d_eff = clamp(deterioration x conviction x gain, 0, 1).",
            "Dose-response for the pivot channel: locates how higher pivot sensitivity changes exit behavior and trap attenuation at production scale.",
            "fixed_mixed", "expanded", dose_pivot_gain_4x!),

        RobustnessCondition(
            "DOSE_PIVOT_GAIN_8X", "dose_response",
            "Abandonment option at deterioration gain 8.0; d_eff saturates (=1) whenever deterioration x conviction >= 0.125, an aggressive upper anchor for the liveness curve.",
            "Dose-response upper anchor for the pivot channel: tests whether making the trigger region highly reachable changes trap attenuation; pivot re-correlation onto consensus second-bests is reported either way.",
            "fixed_mixed", "expanded", dose_pivot_gain_8x!),

        RobustnessCondition(
            "DOSE_OPEN_COMPOSITE_MAX", "dose_response",
            "Maximal-dose open-action cell: pivot at gain 8.0 + directed creation at default mixture strength + population-wide agi_native at max-effective strengths (discount 1.0 / edge 2.0 / shift 2.0).",
            "Dose-response ceiling for the joint open-action and strategy treatment.",
            "fixed_mixed", "expanded", dose_open_composite_max!),

        # ── Emergence mechanism extension ─────────────────────────────────
        RobustnessCondition(
            "AI_ERRORS_CORR_50", "emergence_audit",
            "Half-shared AI estimate errors (AI_ERROR_CORRELATION rho=0.5): the continuous return-estimate error is sqrt(0.5)*eps_common(opp, round, tier) + sqrt(0.5)*eps_idio, total variance preserved — estimate accuracy is unchanged by construction; only cross-agent error commonality within a tier rises.",
            "Quality-vs-commonality decomposition: at fixed accuracy, tests whether the frontier treatment effect changes with shared signal error.",
            "fixed_mixed", "expanded", ai_errors_corr_50!),

        RobustnessCondition(
            "SHARED_AI_ERRORS", "emergence_audit",
            "Fully shared AI estimate errors (AI_ERROR_CORRELATION rho=1.0): every same-tier agent's instrument draws THE SAME standard-normal error per (opportunity, round, tier) — the shared-foundation-model world with correlated blind spots — at exactly the per-agent error variance (accuracy) of the rho=0 default.",
            "Quality-vs-commonality endpoint: tests whether fully shared signal error deepens convergence at unchanged per-agent accuracy.",
            "fixed_mixed", "expanded", shared_ai_errors!),

        RobustnessCondition(
            "DECISION_NOISE_HIGH", "emergence_audit",
            "Decision temperature doubled (DECISION_TEMPERATURE T=2.0): action-selection softmax utilities divided by 2x the calibrated temperature, flattening every agent's invest/innovate/explore/maintain mix toward uniform.",
            "Decision-mixing test: raising T flattens action probabilities and measures whether mixed action selection attenuates convergent entry.",
            "fixed_mixed", "expanded", decision_noise_high!),

        RobustnessCondition(
            "DECISION_NOISE_LOW", "emergence_audit",
            "Decision temperature halved (DECISION_TEMPERATURE T=0.5): action-selection softmax sharpened toward argmax — agents act nearly deterministically on their utilities.",
            "Deterministic-action bracket: lowering T sharpens action probabilities and pairs with DECISION_NOISE_HIGH to bound decision-mixing effects.",
            "fixed_mixed", "expanded", decision_noise_low!),

        # ── Endogenous-adoption family ────────────────────────────────────
        # Design is "emergent": all founders start at no-AI and choose tiers
        # from observed ROI, posted prices, and peer signals.
        # Causal IUT machinery still excludes these (self-selection); the
        # family's claims are about the equilibrium: what tier mix the market
        # converges to, and which mechanism levers move that mix. Run with
        # SUITE_PRESET=adoption (includes BASELINE for paired comparisons).
        RobustnessCondition(
            "ADOPT_BASELINE", "endogenous_adoption",
            "Endogenous tier choice under the baseline economy: founders start unaided and pick tiers from experienced ROI, posted prices, and peer signals.",
            "The adoption equilibrium itself: if the market sorts away from frontier AI while mid tiers persist, the equilibrium prices the trap that the fixed-tier design identifies causally. Anchor cell for the family.",
            "emergent", "adoption", noop!),

        RobustnessCondition(
            "ADOPT_FREE_AI", "endogenous_adoption",
            "Endogenous tier choice with all AI subscription and usage costs at zero.",
            "Separates price-avoidance from trap-avoidance in the adoption equilibrium: if founders shun frontier AI only because it is expensive, free frontier AI should sweep the market; if the trap is epistemic, adoption rises and the population pays in survival instead of fees.",
            "emergent", "adoption", free_ai!),

        RobustnessCondition(
            "ADOPT_SUBSCRIPTION_ERA", "endogenous_adoption",
            "Endogenous tier choice under subscription-era pricing (seat fees plus metered usage) in place of the usage-based default.",
            "Adoption-side mirror of SUBSCRIPTION_ERA: does the contractual seat-fee architecture, with its high trial gate, change the equilibrium mix relative to buying intelligence at the margin?",
            "emergent", "adoption", subscription_era!),

        RobustnessCondition(
            "ADOPT_ERROR_FREE", "endogenous_adoption",
            "Endogenous tier choice with hallucination and overconfidence removed from all AI tiers.",
            "Adoption-side mirror of NO_AI_ERRORS: tests whether error-free instruments change tier demand through experienced consequences rather than instrument quality alone.",
            "emergent", "adoption", no_hallucination_overconfidence!),

        RobustnessCondition(
            "ADOPT_NO_CROWDING", "endogenous_adoption",
            "Endogenous tier choice with competition dynamics and convex crowding dilution disabled.",
            "Adoption-side mirror of NO_CROWDING: with the market-punishment channel removed, frontier ROI improves mechanically — does adoption recover toward the high tiers? Locates how much of the equilibrium's frontier-avoidance is learned crowding cost.",
            "emergent", "adoption", no_crowding!),

        RobustnessCondition(
            "ADOPT_COST_2X", "endogenous_adoption",
            "Endogenous tier choice with all AI costs doubled.",
            "Price-elasticity arm of the adoption family: bounds how the equilibrium mix responds to the most contested assumption (pricing) in the direction opposite ADOPT_FREE_AI.",
            "emergent", "adoption", double_ai_cost!),

        RobustnessCondition(
            "ADOPT_STRATEGIES_ALL", "endogenous_adoption",
            "Endogenous tier choice with the full strategic repertoire (consensus discounting, comparative advantage, complement-seeking) active for every founder.",
            "Tests whether strategic sophistication changes what the market learns to buy: if strategies cannot rescue frontier survival (P1/P8), sophisticated founders should not adopt frontier AI at higher rates either.",
            "emergent", "adoption", all_strategy_composite!),

        RobustnessCondition(
            "ADOPT_FAST_LEARNING", "endogenous_adoption",
            "Endogenous tier choice with monthly tier reviews after a 6-round initial freeze (default: every 3 rounds after 12).",
            "Sorting-speed bound: a faster-learning market should reach the same equilibrium mix sooner if the equilibrium reflects experienced consequences; a different mix would mean the equilibrium is cadence-dependent.",
            "emergent", "adoption", adopt_fast_learning!),

        RobustnessCondition(
            "ADOPT_SLOW_LEARNING", "endogenous_adoption",
            "Endogenous tier choice with tier reviews every 6 rounds after an 18-round initial freeze.",
            "Sorting-speed bound in the slow direction; together with ADOPT_FAST_LEARNING brackets the learning-cadence assumption.",
            "emergent", "adoption", adopt_slow_learning!),

        # ── Fixed-tier supplemental addendum ───────────────────────────────
        RobustnessCondition(
            "HUMAN_BASELINE_WEAK", "supplemental_addendum",
            "Unaided founders' published characteristics lowered to quality 0.15 / breadth 0.12 (default 0.25 / 0.20); AI tiers unchanged.",
            "Lower bracket of the unaided-search anchor: shows whether the frontier trap's magnitude depends on how weak the no-AI comparison group is assumed to be.",
            "fixed_mixed", "adoption", human_baseline_weak!),

        RobustnessCondition(
            "HUMAN_BASELINE_STRONG", "supplemental_addendum",
            "Unaided founders' published characteristics raised to quality 0.35 / breadth 0.30; AI tiers unchanged.",
            "Upper bracket of the unaided-search anchor: a stronger unaided floor narrows the first-order information gap — if the trap shrinks proportionally, it is information-gap-driven; if not, the anchor was not doing hidden work.",
            "fixed_mixed", "adoption", human_baseline_strong!),

        RobustnessCondition(
            "NO_AI_BIAS", "supplemental_addendum",
            "Systematic bias term removed from all AI estimates (AI_BIAS_INTENSITY=0); hallucination and overconfidence unchanged.",
            "Isolates the systematic-bias channel while leaving hallucination and overconfidence unchanged.",
            "fixed_mixed", "adoption", no_ai_bias!),

        RobustnessCondition(
            "WEALTH_SCALED_COMPUTE", "supplemental_addendum",
            "Analysis breadth scales with founder resources (visibility budget x clamp(capital/initial equity, 0.25, 4.0)), uniformly across tiers; extra analyses billed per use.",
            "The rich-get-richer marginal-compute dynamic as a direct experiment rather than a design exclusion: if early winners buying more analysis could invert the frontier trap, this cell is where it would show.",
            "fixed_mixed", "adoption", wealth_scaled_compute!),
    ]
end

function filter_conditions(
    conditions::AbstractVector{RobustnessCondition},
    condition_filter::AbstractString,
)::Vector{RobustnessCondition}
    normalized_filter = strip(condition_filter)
    isempty(normalized_filter) && return collect(conditions)
    requested = Set(strip.(split(normalized_filter, ",")))
    known = Set(condition.name for condition in conditions)
    missing = setdiff(requested, known)
    if !isempty(missing)
        error("Unknown CONDITIONS entries: $(join(sort!(collect(missing)), ", "))")
    end
    # Every non-baseline cell is analyzed as a seed-paired contrast against the
    # baseline produced by this same job. Make treatment-only filters complete
    # by construction instead of silently writing an empty delta artifact.
    any(name -> name != "BASELINE", requested) && push!(requested, "BASELINE")
    return [condition for condition in conditions if condition.name in requested]
end

function selected_conditions()
    conditions = all_conditions()
    if !isempty(CONDITION_FILTER)
        return filter_conditions(conditions, CONDITION_FILTER)
    end

    if SUITE_PRESET == "minimal"
        return [c for c in conditions if c.preset == "minimal"]
    elseif SUITE_PRESET == "core"
        return [c for c in conditions if c.preset in ("minimal", "core")]
    elseif SUITE_PRESET == "expanded"
        # The manuscript's expanded suite excludes the adoption-addendum preset;
        # the addendum runs as its own job.
        return [c for c in conditions if c.preset != "adoption"]
    elseif SUITE_PRESET == "adoption"
        # Endogenous-adoption family plus fixed-tier supplemental addendum.
        # BASELINE is included so paired within-suite treatment effects and the
        # permutation placebo have their anchor without depending on a
        # different run's outputs.
        return [c for c in conditions if c.preset == "adoption" || c.name == "BASELINE"]
    else
        error("Unknown SUITE_PRESET=$SUITE_PRESET. Use minimal, core, expanded, or adoption.")
    end
end

function print_manifest(conditions::Vector{RobustnessCondition})
    println("Theory robustness/refutation suite manifest")
    println("Preset: $SUITE_PRESET")
    println("N_AGENTS=$N_AGENTS  N_ROUNDS=$N_ROUNDS  N_RUNS=$N_RUNS  SEED_MODE=$SUITE_SEED_MODE")
    println()
    for (i, c) in enumerate(conditions)
        @printf("%2d. %-34s %-42s %s\n", i, c.name, c.category, c.description)
    end
end

function scorecard_emergent_key(stats::AbstractDict, dim::String)::String
    all_agent_key = "all_agent_emergent_$(dim)"
    return haskey(stats, all_agent_key) ? all_agent_key : "emergent_$(dim)"
end

# Missing-aware numeric lookup: a tier with no aggregated data (key absent or
# non-numeric) yields `missing`, which propagates through deltas/effect sizes
# instead of substituting a prior or 0.0 sentinel.
function scorecard_stat(stats::AbstractDict, key::String)::Union{Missing,Float64}
    value = get(stats, key, missing)
    return value isa Number ? Float64(value) : missing
end

function scorecard_emergent_level(stats::AbstractDict, dim::String)::Union{Missing,Float64}
    return scorecard_stat(stats, scorecard_emergent_key(stats, dim))
end

function scorecard_survivor_emergent_level(stats::AbstractDict, dim::String)::Union{Missing,Float64}
    level = scorecard_stat(stats, "survivor_emergent_$(dim)")
    return ismissing(level) ? scorecard_emergent_level(stats, dim) : level
end

function scorecard_emergent_observations(stats::AbstractDict, dim::String)::Float64
    all_agent_key = "all_agent_emergent_$(dim)_observations"
    fallback_key = "emergent_$(dim)_observations"
    return Float64(get(stats, all_agent_key, get(stats, fallback_key, 0.0)))
end

function metric_delta_band(
    stats::AbstractDict,
    baseline::AbstractDict,
    key::String;
    min_band::Float64 = 0.005,
)::Float64
    n = max(1.0, Float64(get(stats, "n_runs", 1)))
    n0 = max(1.0, Float64(get(baseline, "n_runs", 1)))
    sd = Float64(get(stats, key * "_run_std", 0.0))
    sd0 = Float64(get(baseline, key * "_run_std", 0.0))
    se = sqrt(sd^2 / n + sd0^2 / n0)
    return max(min_band, se)
end

function scorecard_metric_delta_band(
    stats::AbstractDict,
    baseline::AbstractDict,
    dim::String;
    min_band::Float64 = 0.005,
)::Float64
    key = scorecard_emergent_key(stats, dim)
    return metric_delta_band(stats, baseline, key; min_band=min_band)
end

function signed_effect_size(delta::Union{Missing,Float64}, band::Float64)::Union{Missing,Float64}
    return delta / max(band, eps(Float64))
end

function horizon_pressure(stats::AbstractDict)::Union{Missing,Float64}
    # `mean` over a vector containing missing is itself missing: horizon
    # pressure is undefined unless all three components have data.
    return mean(Union{Missing,Float64}[
        scorecard_emergent_level(stats, "practical_indeterminism"),
        scorecard_emergent_level(stats, "agentic_novelty"),
        scorecard_emergent_level(stats, "competitive_recursion"),
    ])
end

function horizon_delta_band(stats::AbstractDict, baseline::AbstractDict)::Float64
    component_keys = [
        scorecard_emergent_key(stats, "practical_indeterminism"),
        scorecard_emergent_key(stats, "agentic_novelty"),
        scorecard_emergent_key(stats, "competitive_recursion"),
    ]
    bands = [
        metric_delta_band(stats, baseline, key; min_band=0.005)
        for key in component_keys
    ]
    return max(0.005, sqrt(sum(b^2 for b in bands)) / length(bands))
end

function dominant_second_order_channel(
    practical_delta::Union{Missing,Float64},
    novelty_delta::Union{Missing,Float64},
    recursion_delta::Union{Missing,Float64},
)::String
    channels = [
        ("practical_indeterminism", practical_delta),
        ("agentic_novelty", novelty_delta),
        ("competitive_recursion", recursion_delta),
    ]
    # missing deltas (tier with no data) cannot be the dominant channel
    positive = filter(pair -> coalesce(pair[2] > 0.0, false), channels)
    isempty(positive) && return "none"
    idx = argmax([Float64(pair[2]) for pair in positive])
    return positive[idx][1]
end

function perceived_dimension_level(stats::AbstractDict, dim::String)::Union{Missing,Float64}
    return scorecard_stat(stats, "perceived_$(dim)_bounded_level")
end

function perceived_dimension_gap_vs_prior(stats::AbstractDict, dim::String)::Union{Missing,Float64}
    return scorecard_stat(stats, "perceived_$(dim)_gap_vs_prior")
end

function build_paradox_mechanism_scorecard(summary::Dict)::Dict{String,Dict{String,Any}}
    stats = summary["summary_stats"]
    baseline = stats["none"]
    baseline_horizon = horizon_pressure(baseline)
    decision_diag = get(summary, "decision_diagnostics_summary", Dict{String,Any}())
    market_diag(key::String)::Float64 = Float64(get(decision_diag, "mean_" * key, 0.0))
    scorecard = Dict{String,Dict{String,Any}}()

    for tier in AI_TIERS
        s = stats[tier]
        actor_level = scorecard_emergent_level(s, "actor_ignorance")
        actor_level_baseline = scorecard_emergent_level(baseline, "actor_ignorance")
        practical_level = scorecard_emergent_level(s, "practical_indeterminism")
        practical_level_baseline = scorecard_emergent_level(baseline, "practical_indeterminism")
        novelty_level = scorecard_emergent_level(s, "agentic_novelty")
        novelty_level_baseline = scorecard_emergent_level(baseline, "agentic_novelty")
        recursion_level = scorecard_emergent_level(s, "competitive_recursion")
        recursion_level_baseline = scorecard_emergent_level(baseline, "competitive_recursion")

        actor_gain = actor_level_baseline - actor_level
        visible_gain = scorecard_stat(s, "mean_visible_opportunities") -
            scorecard_stat(baseline, "mean_visible_opportunities")
        info_gain = scorecard_stat(s, "mean_info_quality_used") -
            scorecard_stat(baseline, "mean_info_quality_used")

        practical_delta = practical_level - practical_level_baseline
        novelty_delta = novelty_level - novelty_level_baseline
        recursion_delta = recursion_level - recursion_level_baseline

        actor_band = scorecard_metric_delta_band(s, baseline, "actor_ignorance"; min_band=0.005)
        visible_band = metric_delta_band(s, baseline, "mean_visible_opportunities"; min_band=1.0)
        info_band = metric_delta_band(s, baseline, "mean_info_quality_used"; min_band=0.005)
        practical_band = scorecard_metric_delta_band(s, baseline, "practical_indeterminism"; min_band=0.005)
        novelty_band = scorecard_metric_delta_band(s, baseline, "agentic_novelty"; min_band=0.005)
        recursion_band = scorecard_metric_delta_band(s, baseline, "competitive_recursion"; min_band=0.005)

        perceived_actor = perceived_dimension_level(s, "actor_ignorance")
        perceived_practical = perceived_dimension_level(s, "practical_indeterminism")
        perceived_novelty = perceived_dimension_level(s, "agentic_novelty")
        perceived_recursion = perceived_dimension_level(s, "competitive_recursion")

        horizon = horizon_pressure(s)
        horizon_delta = horizon - baseline_horizon
        horizon_band = horizon_delta_band(s, baseline)
        # A missing comparison (tier with no data on that channel) counts as
        # "no evidence", never as evidence.
        first_order_observed = coalesce(actor_gain > actor_band, false) ||
            coalesce(visible_gain > visible_band, false) ||
            coalesce(info_gain > info_band, false)
        horizon_recession_observed = coalesce(horizon_delta > horizon_band, false)
        paradox_logic_supported = first_order_observed && horizon_recession_observed
        first_order_evidence_count = sum([
            coalesce(actor_gain > actor_band, false),
            coalesce(visible_gain > visible_band, false),
            coalesce(info_gain > info_band, false),
        ])
        second_order_evidence_count = sum([
            coalesce(practical_delta > practical_band, false),
            coalesce(novelty_delta > novelty_band, false),
            coalesce(recursion_delta > recursion_band, false),
        ])
        alignment_score = paradox_logic_supported ?
            (first_order_evidence_count / 3.0) * max(0.0, horizon_delta - horizon_band) :
            0.0

        scorecard[tier] = Dict{String,Any}(
            "tier" => tier,
            "actor_ignorance_level" => actor_level,
            "all_agent_actor_ignorance_level" => actor_level,
            "survivor_actor_ignorance_level" => scorecard_survivor_emergent_level(s, "actor_ignorance"),
            "actor_ignorance_survivorship_gap" =>
                scorecard_survivor_emergent_level(s, "actor_ignorance") - actor_level,
            "actor_ignorance_observations" => scorecard_emergent_observations(s, "actor_ignorance"),
            "first_order_knowledge_gain_vs_none" => actor_gain,
            "first_order_knowledge_delta_band" => actor_band,
            "first_order_knowledge_effect_size" => signed_effect_size(actor_gain, actor_band),
            "visible_opportunity_gain_vs_none" => visible_gain,
            "info_quality_gain_vs_none" => info_gain,
            "practical_indeterminism_level" => practical_level,
            "all_agent_practical_indeterminism_level" => practical_level,
            "survivor_practical_indeterminism_level" =>
                scorecard_survivor_emergent_level(s, "practical_indeterminism"),
            "practical_indeterminism_survivorship_gap" =>
                scorecard_survivor_emergent_level(s, "practical_indeterminism") - practical_level,
            "practical_indeterminism_observations" =>
                scorecard_emergent_observations(s, "practical_indeterminism"),
            "practical_indeterminism_delta_vs_none" => practical_delta,
            "practical_indeterminism_delta_band" => practical_band,
            "practical_indeterminism_effect_size" => signed_effect_size(practical_delta, practical_band),
            "agentic_novelty_level" => novelty_level,
            "all_agent_agentic_novelty_level" => novelty_level,
            "survivor_agentic_novelty_level" => scorecard_survivor_emergent_level(s, "agentic_novelty"),
            "agentic_novelty_survivorship_gap" =>
                scorecard_survivor_emergent_level(s, "agentic_novelty") - novelty_level,
            "agentic_novelty_observations" => scorecard_emergent_observations(s, "agentic_novelty"),
            "agentic_novelty_delta_vs_none" => novelty_delta,
            "agentic_novelty_delta_band" => novelty_band,
            "agentic_novelty_effect_size" => signed_effect_size(novelty_delta, novelty_band),
            "competitive_recursion_level" => recursion_level,
            "all_agent_competitive_recursion_level" => recursion_level,
            "survivor_competitive_recursion_level" =>
                scorecard_survivor_emergent_level(s, "competitive_recursion"),
            "competitive_recursion_survivorship_gap" =>
                scorecard_survivor_emergent_level(s, "competitive_recursion") - recursion_level,
            "competitive_recursion_observations" =>
                scorecard_emergent_observations(s, "competitive_recursion"),
            "competitive_recursion_delta_vs_none" => recursion_delta,
            "competitive_recursion_delta_band" => recursion_band,
            "competitive_recursion_effect_size" => signed_effect_size(recursion_delta, recursion_band),
            "perceived_actor_ignorance_level" => perceived_actor,
            "perceived_actor_ignorance_delta_vs_none" =>
                perceived_actor - perceived_dimension_level(baseline, "actor_ignorance"),
            "perceived_actor_ignorance_gap_vs_prior" =>
                perceived_dimension_gap_vs_prior(s, "actor_ignorance"),
            "perceived_practical_indeterminism_level" => perceived_practical,
            "perceived_practical_indeterminism_delta_vs_none" =>
                perceived_practical - perceived_dimension_level(baseline, "practical_indeterminism"),
            "perceived_practical_indeterminism_gap_vs_prior" =>
                perceived_dimension_gap_vs_prior(s, "practical_indeterminism"),
            "perceived_agentic_novelty_level" => perceived_novelty,
            "perceived_agentic_novelty_delta_vs_none" =>
                perceived_novelty - perceived_dimension_level(baseline, "agentic_novelty"),
            "perceived_agentic_novelty_gap_vs_prior" =>
                perceived_dimension_gap_vs_prior(s, "agentic_novelty"),
            "perceived_competitive_recursion_level" => perceived_recursion,
            "perceived_competitive_recursion_delta_vs_none" =>
                perceived_recursion - perceived_dimension_level(baseline, "competitive_recursion"),
            "perceived_competitive_recursion_gap_vs_prior" =>
                perceived_dimension_gap_vs_prior(s, "competitive_recursion"),
            "epistemic_horizon_pressure" => horizon,
            "epistemic_horizon_delta_vs_none" => horizon_delta,
            "epistemic_horizon_delta_band" => horizon_band,
            "epistemic_horizon_effect_size" => signed_effect_size(horizon_delta, horizon_band),
            "knowledge_horizon_tradeoff" => horizon_delta - actor_gain,
            "confidence_abs_gap_delta_vs_none" =>
                scorecard_stat(s, "confidence_outcome_abs_gap_mean") -
                scorecard_stat(baseline, "confidence_outcome_abs_gap_mean"),
            "realized_multiple_std_delta_vs_none" =>
                scorecard_stat(s, "confidence_outcome_realized_multiple_std") -
                scorecard_stat(baseline, "confidence_outcome_realized_multiple_std"),
            "niches_per_agent_delta_vs_none" =>
                scorecard_stat(s, "mean_niches_per_agent") -
                scorecard_stat(baseline, "mean_niches_per_agent"),
            "combinations_per_agent_delta_vs_none" =>
                scorecard_stat(s, "mean_combinations_per_agent") -
                scorecard_stat(baseline, "mean_combinations_per_agent"),
            "market_crowding_pressure" => market_diag("market_crowding_pressure"),
            "market_opportunity_overlap" => market_diag("market_opportunity_overlap"),
            "market_investment_concentration" => market_diag("market_investment_concentration"),
            "market_opportunity_competition" => market_diag("market_opportunity_competition"),
            "market_ai_herding_intensity" => market_diag("market_ai_herding_intensity"),
            "market_ai_action_correlation" => market_diag("market_ai_action_correlation"),
            "market_combo_reuse_pressure" => market_diag("market_combo_reuse_pressure"),
            "survival_effect_pp_vs_none" =>
                100.0 * (scorecard_stat(s, "mean_survival_rate") -
                         scorecard_stat(baseline, "mean_survival_rate")),
            "first_order_evidence_count" => first_order_evidence_count,
            "second_order_evidence_count" => second_order_evidence_count,
            "first_order_benefit_observed" => first_order_observed,
            "horizon_recession_observed" => horizon_recession_observed,
            "paradox_logic_supported" => paradox_logic_supported,
            "dominant_second_order_channel" => dominant_second_order_channel(
                practical_delta, novelty_delta, recursion_delta
            ),
            "paradox_alignment_score" => alignment_score,
        )
    end

    return scorecard
end

# Condition×tier summary mean over the non-missing per-run values; a cell with
# zero non-missing values is itself `missing` (empty in CSV), not 0.0.
skipmissing_mean(values) = begin
    vals = collect(skipmissing(values))
    isempty(vals) ? missing : mean(vals)
end

skipmissing_std(values) = begin
    vals = collect(skipmissing(values))
    length(vals) > 1 ? std(vals) : 0.0
end

function summarize_rows(per_run_df::DataFrame)
    rows = NamedTuple[]
    for group in groupby(per_run_df, [:condition, :category, :description, :theoretical_role, :design, :tier, :tier_rank])
        condition = first(group.condition)
        tier = first(group.tier)
        condition_rows = per_run_df[per_run_df.condition .== condition, :]
        none_rows = condition_rows[condition_rows.tier .== "none", :]
        none_survival = nrow(none_rows) == 0 ? missing : skipmissing_mean(none_rows.survival_rate)
        survival_mean = skipmissing_mean(group.survival_rate)
        te_vs_none_pp = (ismissing(none_survival) || ismissing(survival_mean)) ?
            missing : (survival_mean - none_survival) * 100
        push!(rows, (
            condition = condition,
            category = first(group.category),
            description = first(group.description),
            theoretical_role = first(group.theoretical_role),
            design = first(group.design),
            tier = tier,
            tier_label = tier_display_label(tier),
            tier_rank = first(group.tier_rank),
            n_runs = length(unique(group.run_idx)),
            mean_tier_share = mean(group.tier_share),
            survival_mean = survival_mean,
            survival_sd = skipmissing_std(group.survival_rate),
            te_vs_none_pp = te_vs_none_pp,
            mean_final_capital = skipmissing_mean(group.mean_final_capital),
            mean_in_flight_capital = skipmissing_mean(group.mean_in_flight_capital),
            mean_net_worth = skipmissing_mean(group.mean_net_worth),
            roi_mean = skipmissing_mean(group.roi),
            matured_investment_count_mean =
                skipmissing_mean(group.matured_investment_count),
            mean_capacity_saturation_at_entry =
                skipmissing_mean(group.mean_capacity_saturation_at_entry),
            mean_capacity_saturation_at_maturity =
                skipmissing_mean(group.mean_capacity_saturation_at_maturity),
            mean_effective_capacity_saturation_at_maturity =
                skipmissing_mean(
                    group.mean_effective_capacity_saturation_at_maturity),
            mean_capacity_saturation_change =
                skipmissing_mean(group.mean_capacity_saturation_change),
            mean_post_commitment_rival_capital =
                skipmissing_mean(group.mean_post_commitment_rival_capital),
            mean_post_commitment_rival_capacity_ratio =
                skipmissing_mean(group.mean_post_commitment_rival_capacity_ratio),
            post_commitment_rival_incidence =
                skipmissing_mean(group.post_commitment_rival_incidence),
            mean_crowding_return_multiplier =
                skipmissing_mean(group.mean_crowding_return_multiplier),
            mean_realized_latent_capture =
                skipmissing_mean(group.mean_realized_latent_capture),
            mean_performative_effective_capacity =
                skipmissing_mean(group.mean_performative_effective_capacity),
            mean_performative_lagged_commitment =
                skipmissing_mean(group.mean_performative_lagged_commitment),
            mean_performative_commitment_share =
                skipmissing_mean(group.mean_performative_commitment_share),
            mean_performative_uplift_fraction =
                skipmissing_mean(group.mean_performative_uplift_fraction),
            exploration_niche_events_total_mean =
                skipmissing_mean(group.exploration_niche_events_total),
            exploration_opportunities_created_total_mean =
                skipmissing_mean(group.exploration_opportunities_created_total),
            innovation_opportunities_spawned_total_mean =
                skipmissing_mean(group.innovation_opportunities_spawned_total),
            background_opportunities_created_total_mean =
                skipmissing_mean(group.background_opportunities_created_total),
            opportunities_publicized_total_mean =
                skipmissing_mean(group.opportunities_publicized_total),
            opportunities_culled_total_mean =
                skipmissing_mean(group.opportunities_culled_total),
            opportunities_created_total_mean =
                skipmissing_mean(group.opportunities_created_total),
            opportunity_ending_stock_mean =
                skipmissing_mean(group.opportunity_ending_stock),
            opportunity_stock_flow_residual_mean =
                skipmissing_mean(group.opportunity_stock_flow_residual),
            opportunity_commitments_total_mean =
                skipmissing_mean(group.opportunity_commitments_total),
            mean_competition = skipmissing_mean(group.mean_competition),
            max_competition = skipmissing_mean(group.max_competition),
            innovations_per_agent = skipmissing_mean(group.innovations_per_agent),
            niches_per_agent = skipmissing_mean(group.niches_per_agent),
            combinations_per_agent = skipmissing_mean(group.combinations_per_agent),
            invest_share = skipmissing_mean(group.invest_share),
            innovate_share = skipmissing_mean(group.innovate_share),
            explore_share = skipmissing_mean(group.explore_share),
            maintain_share = skipmissing_mean(group.maintain_share),
        ))
    end
    return DataFrame(rows)
end

function baseline_permutation_placebo(per_run_df::DataFrame; n_iter::Int=1000)
    baseline = per_run_df[per_run_df.condition .== "BASELINE", :]
    none = collect(skipmissing(baseline[baseline.tier .== "none", :survival_rate]))
    premium = collect(skipmissing(baseline[baseline.tier .== "premium", :survival_rate]))
    if length(none) < 2 || length(premium) < 2
        return nothing
    end
    actual = mean(premium) - mean(none)
    pool = vcat(none, premium)
    n = length(none)
    rng = MersenneTwister(12345)
    rows = NamedTuple[]
    for i in 1:n_iter
        shuffled = shuffle(rng, pool)
        fake_none = shuffled[1:n]
        fake_premium = shuffled[n+1:end]
        push!(rows, (iteration=i, null_te_pp=(mean(fake_premium) - mean(fake_none)) * 100))
    end
    null_df = DataFrame(rows)
    nulls = null_df.null_te_pp
    actual_pp = actual * 100
    placebo_summary = DataFrame([(
        actual_te_pp = actual_pp,
        null_mean_pp = mean(nulls),
        null_sd_pp = std(nulls),
        null_p025_pp = quantile(nulls, 0.025),
        null_p975_pp = quantile(nulls, 0.975),
        p_two_sided = mean(abs.(nulls) .>= abs(actual_pp)),
    )])
    return (null_df=null_df, summary=placebo_summary)
end

function per_run_treatment_effect_panel(per_run_df::DataFrame)
    rows = NamedTuple[]
    for condition_df in groupby(per_run_df, [:condition, :category, :description, :theoretical_role, :design])
        for run_idx in unique(condition_df.run_idx)
            run_df = condition_df[condition_df.run_idx .== run_idx, :]
            none_rows = run_df[run_df.tier .== "none", :]
            nrow(none_rows) == 0 && continue
            none_survival = first(none_rows.survival_rate)
            # Empty cells are missing (no agents in the tier this run): the
            # paired contrast is undefined, so skip rather than fabricate 0.0.
            ismissing(none_survival) && continue
            for tier in ("basic", "advanced", "premium")
                tier_rows = run_df[run_df.tier .== tier, :]
                nrow(tier_rows) == 0 && continue
                ismissing(first(tier_rows.survival_rate)) && continue
                push!(rows, (
                    condition = first(condition_df.condition),
                    category = first(condition_df.category),
                    description = first(condition_df.description),
                    theoretical_role = first(condition_df.theoretical_role),
                    design = first(condition_df.design),
                    run_idx = run_idx,
                    tier = tier,
                    tier_label = tier_display_label(tier),
                    te_pp = (first(tier_rows.survival_rate) - none_survival) * 100,
                ))
            end
        end
    end
    return DataFrame(rows)
end

function paired_treatment_effects(per_run_df::DataFrame)
    panel = per_run_treatment_effect_panel(per_run_df)
    rows = NamedTuple[]
    for group in groupby(panel, [:condition, :category, :description, :theoretical_role, :design, :tier])
        values = Float64.(group.te_pp)
        n = length(values)
        mean_pp = n == 0 ? NaN : mean(values)
        sd_pp = n < 2 ? 0.0 : std(values)
        se_pp = n == 0 ? NaN : safe_se(sd_pp, n)
        push!(rows, (
            condition = first(group.condition),
            category = first(group.category),
            description = first(group.description),
            theoretical_role = first(group.theoretical_role),
            design = first(group.design),
            tier = first(group.tier),
            tier_label = tier_display_label(first(group.tier)),
            n_runs = n,
            mean_te_pp = mean_pp,
            sd_te_pp = sd_pp,
            se_te_pp = se_pp,
            ci95_low_pp = mean_pp - 1.96 * se_pp,
            ci95_high_pp = mean_pp + 1.96 * se_pp,
        ))
    end
    return DataFrame(rows)
end

"""
Return a long-form, within-run Frontier/AI-tier minus No-AI effect panel for
the economic, innovation, and survival outcome families used in Figure 4.

An outcome contrast is omitted when either tier's cell is missing or nonfinite;
it is never replaced with zero.  This outcome-specific filtering matters for
endogenous-adoption cells where a tier can be empty in an otherwise valid run.
"""
function per_run_outcome_effect_panel(
    per_run_df::DataFrame;
    specs=PAIRED_OUTCOME_SPECS,
)
    rows = NamedTuple[]
    grouping = [:condition, :category, :description, :theoretical_role, :design]
    for condition_df in groupby(per_run_df, grouping)
        for run_idx in unique(condition_df.run_idx)
            run_df = condition_df[condition_df.run_idx .== run_idx, :]
            none_rows = run_df[run_df.tier .== "none", :]
            nrow(none_rows) == 1 || continue
            for tier in ("basic", "advanced", "premium")
                tier_rows = run_df[run_df.tier .== tier, :]
                nrow(tier_rows) == 1 || continue
                for spec in specs
                    none_value = none_rows[1, spec.source_column]
                    tier_value = tier_rows[1, spec.source_column]
                    (ismissing(none_value) || ismissing(tier_value)) && continue
                    none_float = Float64(none_value)
                    tier_float = Float64(tier_value)
                    (isfinite(none_float) && isfinite(tier_float)) || continue
                    push!(rows, (
                        condition = first(condition_df.condition),
                        category = first(condition_df.category),
                        description = first(condition_df.description),
                        theoretical_role = first(condition_df.theoretical_role),
                        design = first(condition_df.design),
                        run_idx = run_idx,
                        tier = tier,
                        tier_label = tier_display_label(tier),
                        outcome = spec.outcome,
                        outcome_label = spec.outcome_label,
                        outcome_family = spec.outcome_family,
                        source_column = String(spec.source_column),
                        unit = spec.unit,
                        effect = (tier_float - none_float) * spec.scale,
                    ))
                end
            end
        end
    end
    return DataFrame(rows)
end

"""Summarize the multivariate paired panel with 95% normal intervals."""
function paired_outcome_effects(
    per_run_df::DataFrame;
    specs=PAIRED_OUTCOME_SPECS,
)
    panel = per_run_outcome_effect_panel(per_run_df; specs=specs)
    rows = NamedTuple[]
    grouping = [
        :condition, :category, :description, :theoretical_role, :design,
        :tier, :tier_label, :outcome, :outcome_label, :outcome_family,
        :source_column, :unit,
    ]
    for group in groupby(panel, grouping)
        effects = Float64.(group.effect)
        n = length(effects)
        mean_effect = mean(effects)
        sd_effect = n < 2 ? 0.0 : std(effects)
        se_effect = safe_se(sd_effect, n)
        push!(rows, (
            condition = first(group.condition),
            category = first(group.category),
            description = first(group.description),
            theoretical_role = first(group.theoretical_role),
            design = first(group.design),
            tier = first(group.tier),
            tier_label = first(group.tier_label),
            outcome = first(group.outcome),
            outcome_label = first(group.outcome_label),
            outcome_family = first(group.outcome_family),
            source_column = first(group.source_column),
            unit = first(group.unit),
            n_runs = n,
            mean_effect = mean_effect,
            sd_effect = sd_effect,
            se_effect = se_effect,
            ci95_low = mean_effect - 1.96 * se_effect,
            ci95_high = mean_effect + 1.96 * se_effect,
        ))
    end
    return DataFrame(rows)
end

function condition_delta_vs_baseline(per_run_df::DataFrame)
    panel = per_run_treatment_effect_panel(per_run_df)
    baseline = panel[panel.condition .== "BASELINE", [:run_idx, :tier, :te_pp]]
    rename!(baseline, :te_pp => :baseline_te_pp)
    rows = NamedTuple[]
    for group in groupby(panel[panel.condition .!= "BASELINE", :],
                         [:condition, :category, :description, :theoretical_role, :design, :tier])
        joined = innerjoin(group, baseline, on=[:run_idx, :tier])
        deltas = joined.te_pp .- joined.baseline_te_pp
        n = length(deltas)
        n == 0 && continue
        mean_delta = mean(deltas)
        sd_delta = n < 2 ? 0.0 : std(deltas)
        se_delta = safe_se(sd_delta, n)
        push!(rows, (
            condition = first(group.condition),
            category = first(group.category),
            description = first(group.description),
            theoretical_role = first(group.theoretical_role),
            design = first(group.design),
            tier = first(group.tier),
            tier_label = tier_display_label(first(group.tier)),
            n_paired_runs = n,
            mean_te_pp = mean(joined.te_pp),
            baseline_te_pp = mean(joined.baseline_te_pp),
            delta_vs_baseline_pp = mean_delta,
            sd_delta_pp = sd_delta,
            se_delta_pp = se_delta,
            delta_ci95_low_pp = mean_delta - 1.96 * se_delta,
            delta_ci95_high_pp = mean_delta + 1.96 * se_delta,
        ))
    end
    if isempty(rows)
        # Stable schema even with no non-BASELINE contrasts (e.g. a
        # CONDITIONS=BASELINE-only run): emit a typed, zero-row frame so the
        # CSV still carries a header row downstream consumers can parse.
        return DataFrame(
            condition=String[], category=String[], description=String[],
            theoretical_role=String[], design=String[], tier=String[],
            tier_label=String[], n_paired_runs=Int[], mean_te_pp=Float64[],
            baseline_te_pp=Float64[], delta_vs_baseline_pp=Float64[],
            sd_delta_pp=Float64[], se_delta_pp=Float64[],
            delta_ci95_low_pp=Float64[], delta_ci95_high_pp=Float64[],
        )
    end
    return DataFrame(rows)
end

function condition_summary_for_scorecard(condition_df::DataFrame)::Dict{String,Any}
    stats = Dict{String,Dict{String,Any}}()
    for tier_df in groupby(condition_df, :tier)
        tier = String(first(tier_df.tier))
        tier_stats = Dict{String,Any}(
            "n_runs" => length(unique(tier_df.run_idx)),
        )
        for col in names(tier_df)
            vals = Float64[]
            for value in tier_df[!, col]
                value isa Number || continue
                v = Float64(value)
                isfinite(v) && push!(vals, v)
            end
            isempty(vals) && continue
            key = String(col)
            tier_stats[key] = mean(vals)
            tier_stats[key * "_run_std"] = safe_std(vals)
        end
        # A tier with zero non-missing observations gets NO sentinel value: the
        # key stays absent and downstream scorecard lookups resolve to missing.
        haskey(tier_stats, "survival_rate") &&
            (tier_stats["mean_survival_rate"] = tier_stats["survival_rate"])
        haskey(tier_stats, "niches_per_agent") &&
            (tier_stats["mean_niches_per_agent"] = tier_stats["niches_per_agent"])
        haskey(tier_stats, "combinations_per_agent") &&
            (tier_stats["mean_combinations_per_agent"] = tier_stats["combinations_per_agent"])
        stats[tier] = tier_stats
    end

    for tier in AI_TIERS
        if !haskey(stats, tier)
            stats[tier] = Dict{String,Any}("n_runs" => 0)
        end
    end

    decision_diag = Dict{String,Any}()
    for key in MARKET_RECURSION_DIAGNOSTIC_KEYS
        if key in names(condition_df)
            vals = Float64[
                Float64(v) for v in condition_df[!, key]
                if v isa Number && isfinite(Float64(v))
            ]
            decision_diag["mean_" * key] = safe_mean(vals)
            decision_diag["std_" * key] = safe_std(vals)
        end
    end

    return Dict{String,Any}(
        "summary_stats" => stats,
        "decision_diagnostics_summary" => decision_diag,
    )
end

# Keys build_paradox_mechanism_scorecard emits that are deliberately NOT part
# of the hand-maintained PARADOX_SCORECARD_KEYS metric list (identity/boolean/
# categorical columns appended separately by the row writers).
const SCORECARD_NON_METRIC_KEYS = Set([
    "tier",
    "first_order_benefit_observed",
    "horizon_recession_observed",
    "paradox_logic_supported",
    "dominant_second_order_channel",
])

# PARADOX_SCORECARD_KEYS duplicates the keys hand-typed in
# build_paradox_mechanism_scorecard; if the two drift apart the scorecard CSV
# silently filled the gap with 0.0 sentinels. Fail loudly instead.
function assert_scorecard_keys_in_sync(scorecard::AbstractDict)
    expected = Set(PARADOX_SCORECARD_KEYS)
    for (tier, s) in scorecard
        actual = Set(String(k) for k in keys(s) if !(String(k) in SCORECARD_NON_METRIC_KEYS))
        missing_keys = sort!(collect(setdiff(expected, actual)))
        extra_keys = sort!(collect(setdiff(actual, expected)))
        if !isempty(missing_keys) || !isempty(extra_keys)
            error("PARADOX_SCORECARD_KEYS out of sync with build_paradox_mechanism_scorecard " *
                  "(tier=$tier). Listed but not built: [$(join(missing_keys, ", "))]. " *
                  "Built but not listed: [$(join(extra_keys, ", "))].")
        end
    end
    return nothing
end

function paradox_scorecard_rows(per_run_df::DataFrame)
    rows = Dict{Symbol,Any}[]
    for condition_df in groupby(per_run_df, [:condition, :category, :description, :theoretical_role, :design])
        summary = condition_summary_for_scorecard(DataFrame(condition_df))
        scorecard = build_paradox_mechanism_scorecard(summary)
        assert_scorecard_keys_in_sync(scorecard)
        for tier in AI_TIERS
            s = scorecard[tier]
            row = Dict{Symbol,Any}(
                :condition => first(condition_df.condition),
                :category => first(condition_df.category),
                :description => first(condition_df.description),
                :theoretical_role => first(condition_df.theoretical_role),
                :design => first(condition_df.design),
                :tier => tier,
                :tier_label => tier_display_label(tier),
                :tier_rank => TIER_RANK[tier],
            )
            for key in PARADOX_SCORECARD_KEYS
                haskey(s, key) || error("scorecard key missing: $key — " *
                    "PARADOX_SCORECARD_KEYS out of sync with build_paradox_mechanism_scorecard")
                row[Symbol(key)] = s[key]
            end
            row[:first_order_benefit_observed] = get(s, "first_order_benefit_observed", false)
            row[:horizon_recession_observed] = get(s, "horizon_recession_observed", false)
            row[:paradox_logic_supported] = get(s, "paradox_logic_supported", false)
            row[:dominant_second_order_channel] = get(s, "dominant_second_order_channel", "none")
            push!(rows, row)
        end
    end
    df = DataFrame(rows)
    ordered = vcat(
        [:condition, :category, :description, :theoretical_role, :design, :tier, :tier_label, :tier_rank],
        Symbol.(PARADOX_SCORECARD_KEYS),
        [
            :first_order_benefit_observed,
            :horizon_recession_observed,
            :paradox_logic_supported,
            :dominant_second_order_channel,
        ],
    )
    select!(df, [col for col in ordered if col in propertynames(df)])
    return df
end

function paradox_scorecard_delta_vs_baseline(scorecard_df::DataFrame)
    baseline = scorecard_df[scorecard_df.condition .== "BASELINE", :]
    rows = Dict{Symbol,Any}[]

    for row in eachrow(scorecard_df)
        row.condition == "BASELINE" && continue
        base_rows = baseline[baseline.tier .== row.tier, :]
        nrow(base_rows) == 0 && continue
        base = first(eachrow(base_rows))

        for key in PARADOX_SCORECARD_KEYS
            metric = Symbol(key)
            hasproperty(row, metric) && hasproperty(base, metric) || continue
            value = row[metric]
            baseline_value = base[metric]
            (value isa Number && baseline_value isa Number) || continue
            v = Float64(value)
            b = Float64(baseline_value)
            isfinite(v) && isfinite(b) || continue
            push!(rows, Dict{Symbol,Any}(
                :condition => row.condition,
                :category => row.category,
                :description => row.description,
                :theoretical_role => row.theoretical_role,
                :design => row.design,
                :tier => row.tier,
                :tier_label => row.tier_label,
                :tier_rank => row.tier_rank,
                :metric => key,
                :condition_value => v,
                :baseline_value => b,
                :delta_vs_baseline => v - b,
            ))
        end
    end

    return DataFrame(rows)
end

function main()
    conditions = selected_conditions()
    if "--help" in ARGS || "--list" in ARGS || get(ENV, "DRY_RUN", "0") == "1"
        print_manifest(conditions)
        # Config preflight builds every selected condition before launch so
        # script/source mismatches fail early with one clear message.
        for condition in conditions
            try
                build_config(condition, BASE_SEED)
            catch e
                error("DRY_RUN config preflight FAILED for condition " *
                      "$(condition.name): $(sprint(showerror, e)). " *
                      "Deployed src/ is likely out of sync with this script — " *
                      "rsync src/ scripts/ test/ together, then re-run setup_arc.sh.")
            end
        end
        println("\nConfig preflight: all $(length(conditions)) condition configs build cleanly.")
        return
    end

    print_manifest(conditions)
    mkpath(OUTPUT_DIR)
    write_run_provenance!(
        OUTPUT_DIR;
        script_name=basename(@__FILE__),
        parameters=Dict(
            "BASE_SEED" => BASE_SEED,
            "B_CLAMP_MAX" => B_CLAMP_MAX,
            "B_OPS" => B_OPS,
            "B_RANGE_MULT" => B_RANGE_MULT,
            "B_SIGMA_CAP" => B_SIGMA_CAP,
            "B_SIGMA_MULT" => B_SIGMA_MULT,
            "CONDITIONS" => CONDITION_FILTER == "" ? "selected_by_preset" : CONDITION_FILTER,
            "DERIVED_GATE" => DERIVED_GATE,
            "HEAVY_TAIL_RETURNS" => HEAVY_TAIL_RETURNS,
            "N_AGENTS" => N_AGENTS,
            "NICHE_CANONICAL" => NICHE_CANONICAL,
            "NICHE_MODERATE_SIGMA" => NICHE_MODERATE_SIGMA,
            "NICHE_OPS" => NICHE_OPS,
            "NICHE_SIGMA" => NICHE_SIGMA,
            "N_ROUNDS" => N_ROUNDS,
            "N_RUNS" => N_RUNS,
            "OPTION_B_TAIL" => OPTION_B_TAIL,
            "OUTPUT_DIR" => OUTPUT_DIR,
            "RUN_TAG" => get(ENV, "RUN_TAG", ""),
            "SUBMISSION_GIT_COMMIT" =>
                get(ENV, "SUBMISSION_GIT_COMMIT", ""),
            "SUITE_PRESET" => SUITE_PRESET,
            "SUITE_SEED_MODE" => SUITE_SEED_MODE,
            "VENTURE_PANEL" => VENTURE_PANEL,
        ),
        notes=Dict(
            "suite" => "manuscript robustness suite",
        ),
    )
    per_cond_dir = joinpath(OUTPUT_DIR, "per_condition")
    mkpath(per_cond_dir)
    CSV.write(joinpath(OUTPUT_DIR, "robustness_suite_manifest.csv"),
        DataFrame([(name=c.name, category=c.category, description=c.description,
                    theoretical_role=c.theoretical_role, design=c.design,
                    preset=c.preset, seed_mode=SUITE_SEED_MODE) for c in conditions]))
    CSV.write(joinpath(OUTPUT_DIR, "robustness_suite_effective_config.csv"),
        effective_config_rows(conditions, BASE_SEED))

    # ── Flattened (condition × run) task scheduling ─────────────────────────
    # Every (condition, run_idx) pair is one independent task in a single
    # threaded loop. Tasks are built condition-major (all runs of condition 1,
    # then condition 2, ...) so `results` reassembles into deterministic row
    # order: task index ti = (ci-1)*N_RUNS + run_idx.
    n_conditions = length(conditions)
    tasks = [(ci, condition, run_idx)
             for (ci, condition) in enumerate(conditions)
             for run_idx in 1:N_RUNS]
    n_tasks = length(tasks)
    results = Vector{Any}(undef, n_tasks)
    # Per-condition completion tracking: an atomic remaining-run counter per
    # condition. The task that decrements a condition's counter to zero owns
    # that condition's completion: it writes the incremental per_condition CSV
    # checkpoint (serialized through one global lock — writes are rare) and
    # prints the completion line. NOTE: timing semantics changed slightly —
    # conditions now run interleaved, so the reported time is elapsed since
    # suite start, not time spent inside the condition.
    remaining = [Threads.Atomic{Int}(N_RUNS) for _ in conditions]
    completed_conditions = Threads.Atomic{Int}(0)
    write_lock = ReentrantLock()
    t0 = time()

    @printf("Running %d tasks (%d conditions x %d runs) on %d threads\n",
        n_tasks, n_conditions, N_RUNS, Threads.nthreads())
    flush(stdout)
    # :greedy schedules tasks dynamically (work-stealing of the iteration
    # space) — verified to parse and run on local Julia 1.12.3; ARC runs
    # 1.11.3 where :greedy also exists (added in 1.11).
    Threads.@threads :greedy for ti in 1:n_tasks
        ci, condition, run_idx = tasks[ti]
        results[ti] = run_one(condition, run_idx)
        if Threads.atomic_sub!(remaining[ci], 1) == 1
            # Last run of this condition just finished: every slot in this
            # condition's results slice is filled (each was index-assigned
            # before its atomic decrement).
            done = Threads.atomic_add!(completed_conditions, 1) + 1
            cond_rows = reduce(vcat, results[(ci-1)*N_RUNS+1:ci*N_RUNS])
            lock(write_lock) do
                CSV.write(joinpath(per_cond_dir, "$(condition.name).csv"),
                    DataFrame(cond_rows))
            end
            @printf("[%d/%d] Completed %s in %.1fs since suite start (%d/%d conditions done)\n",
                ci, n_conditions, condition.name, time() - t0, done, n_conditions)
            flush(stdout)
        end
    end
    # One collection after the full task loop.
    GC.gc()

    # Reassemble per-run rows grouped by condition in the original condition
    # order.
    all_rows = Dict{Symbol,Any}[]
    for ci in 1:n_conditions
        append!(all_rows, reduce(vcat, results[(ci-1)*N_RUNS+1:ci*N_RUNS]))
    end

    # Write the raw per-run results IMMEDIATELY after the simulation loop,
    # BEFORE any aggregation frame is built: an analysis-layer failure after a
    # multi-day ARC run must never discard the raw results. (The manifest was
    # already written before the loop.)
    per_run_df = DataFrame(all_rows)
    CSV.write(joinpath(OUTPUT_DIR, "robustness_suite_per_run.csv"), per_run_df)
    println("[checkpoint] raw results written: robustness_suite_per_run.csv " *
            "($(nrow(per_run_df)) rows) — before aggregation")
    flush(stdout)

    summary_df = summarize_rows(per_run_df)
    paired_df = paired_treatment_effects(per_run_df)
    paired_outcomes_df = paired_outcome_effects(per_run_df)
    delta_df = condition_delta_vs_baseline(per_run_df)
    scorecard_df = paradox_scorecard_rows(per_run_df)
    scorecard_delta_df = paradox_scorecard_delta_vs_baseline(scorecard_df)

    CSV.write(joinpath(OUTPUT_DIR, "robustness_suite_summary.csv"), summary_df)
    CSV.write(joinpath(OUTPUT_DIR, "robustness_suite_paired_treatment_effects.csv"), paired_df)
    CSV.write(joinpath(OUTPUT_DIR, "robustness_suite_paired_outcome_effects.csv"), paired_outcomes_df)
    CSV.write(joinpath(OUTPUT_DIR, "robustness_suite_delta_vs_baseline.csv"), delta_df)
    CSV.write(joinpath(OUTPUT_DIR, "robustness_suite_paradox_scorecard.csv"), scorecard_df)
    CSV.write(joinpath(OUTPUT_DIR, "robustness_suite_paradox_scorecard_delta_vs_baseline.csv"), scorecard_delta_df)

    placebo = baseline_permutation_placebo(per_run_df)
    if !isnothing(placebo)
        CSV.write(joinpath(OUTPUT_DIR, "baseline_permutation_null.csv"), placebo.null_df)
        CSV.write(joinpath(OUTPUT_DIR, "baseline_permutation_summary.csv"), placebo.summary)
    end

    println()
    println("="^80)
    println("Robustness suite complete")
    println("Output: $OUTPUT_DIR")
    println("="^80)
    show(paired_df[!, [:condition, :category, :tier_label, :mean_te_pp, :ci95_low_pp, :ci95_high_pp]], allrows=true)
    println()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
