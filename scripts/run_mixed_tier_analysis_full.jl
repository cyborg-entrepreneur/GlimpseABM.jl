#!/usr/bin/env julia
"""
Mixed-Population Fixed-Tier Experiment - Full Analysis Version

Collects comprehensive metrics for Table 3-style visualizations:
- Survival rates and trajectories
- Innovation activity share
- Exploration activity share
- Survivor wealth distributions
- Niche creation
- Innovation volume and success rates
- Knowledge recombination quality

Usage:
    julia --threads=auto --project=. scripts/run_mixed_tier_analysis_full.jl
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GlimpseABM
using Statistics
using Random
using DataFrames
using CSV
using Dates
using Printf

include(joinpath(@__DIR__, "_launch_metadata.jl"))

# ============================================================================
# EXPERIMENT PARAMETERS
# ============================================================================

# N_AGENTS / N_RUNS / N_ROUNDS are readable from env so N-sensitivity runs
# (N=1000, 5000) need no forking of the script. Defaults match the
# paper-headline configuration.
const N_AGENTS = parse(Int, get(ENV, "N_AGENTS", "2000"))
const N_ROUNDS = parse(Int, get(ENV, "N_ROUNDS", "60"))
const N_RUNS   = parse(Int, get(ENV, "N_RUNS",   "50"))
const AGENTS_PER_TIER = N_AGENTS ÷ 4
const AI_TIERS = ["none", "basic", "advanced", "premium"]
const EMERGENT_UNCERTAINTY_DIMENSIONS = [
    "actor_ignorance",
    "practical_indeterminism",
    "agentic_novelty",
    "competitive_recursion",
]
const BASE_SEED = parse(Int, get(ENV, "BASE_SEED", "20260425"))

# Tag the output dir with N for sensitivity runs (e.g. mixed_tier_full_n5000_...)
const _OUT_TAG = N_AGENTS == 2000 ? "" : "_n$(N_AGENTS)"
const OUTPUT_DIR = joinpath(@__DIR__, "..", "results", "mixed_tier_full$(_OUT_TAG)_$(Dates.format(now(), "yyyymmdd_HHMMSS"))")

const CONFIDENCE_OUTCOME_KEYS = [
    "confidence_outcome_observed_agents",
    "confidence_outcome_agent_coverage",
    "confidence_outcome_observations",
    "confidence_outcome_weighted_gap_mean",
    "confidence_outcome_weighted_gap_std",
    "confidence_outcome_weighted_abs_gap_mean",
    "confidence_outcome_raw_gap_mean",
    "confidence_outcome_abs_gap_mean",
    "confidence_outcome_positive_gap_rate",
    "confidence_outcome_negative_gap_rate",
    "confidence_outcome_decision_confidence_mean",
    "confidence_outcome_score_mean",
    "confidence_outcome_investment_gap_mean",
    "confidence_outcome_investment_observations",
    "confidence_outcome_innovate_explore_gap_mean",
    "confidence_outcome_innovate_explore_observations",
    "confidence_outcome_ai_gap_mean",
    "confidence_outcome_ai_observations",
    "confidence_outcome_non_ai_gap_mean",
    "confidence_outcome_non_ai_observations",
    "confidence_outcome_realized_multiple_mean",
    "confidence_outcome_realized_multiple_std",
    "confidence_outcome_realized_multiple_p50",
    "confidence_outcome_realized_multiple_p90",
    "confidence_outcome_realized_multiple_p95",
]

const TIER_KNIGHTIAN_KEYS = vcat(
    ["emergent_$(dim)" for dim in EMERGENT_UNCERTAINTY_DIMENSIONS],
    ["all_agent_emergent_$(dim)" for dim in EMERGENT_UNCERTAINTY_DIMENSIONS],
    ["survivor_emergent_$(dim)" for dim in EMERGENT_UNCERTAINTY_DIMENSIONS],
    ["all_agent_emergent_$(dim)_observations" for dim in EMERGENT_UNCERTAINTY_DIMENSIONS],
    ["survivor_emergent_$(dim)_observations" for dim in EMERGENT_UNCERTAINTY_DIMENSIONS],
    [
        "all_agent_emergent_agentic_novelty_relevant_observations",
        "survivor_emergent_agentic_novelty_relevant_observations",
        "all_agent_emergent_agent_count",
        "all_agent_emergent_alive_count",
        "all_agent_emergent_failed_count",
        "survivor_emergent_agent_count",
        "survivor_emergent_alive_count",
        "survivor_emergent_failed_count",
    ],
    [
    "mean_visible_opportunities",
    "mean_info_quality_used",
    "visible_hhi",
    ],
)

const PERCEPTION_TELEMETRY_DIMENSIONS = [
    "actor_ignorance",
    "practical_indeterminism",
    "agentic_novelty",
    "competitive_recursion",
]

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

function telemetry_keys_from_history(history, prefixes)::Vector{String}
    keys_seen = Set{String}()
    for h in history
        for key in keys(h)
            s = String(key)
            any(prefix -> startswith(s, prefix), prefixes) && push!(keys_seen, s)
        end
    end
    return sort(collect(keys_seen))
end

function telemetry_keys_from_results(all_results, prefixes)::Vector{String}
    keys_seen = Set{String}()
    for r in all_results
        get(r, "status", "") == "completed" || continue
        diag = get(r, "decision_diagnostics", Dict())
        diag isa AbstractDict || continue
        for key in keys(diag)
            s = String(key)
            any(prefix -> startswith(s, prefix), prefixes) && push!(keys_seen, s)
        end
    end
    return sort(collect(keys_seen))
end

knightian_component_keys_from_history(history)::Vector{String} =
    telemetry_keys_from_history(history, ("knightian_",))

knightian_component_keys_from_results(all_results)::Vector{String} =
    telemetry_keys_from_results(all_results, ("knightian_",))

perception_component_keys_from_history(history)::Vector{String} =
    telemetry_keys_from_history(history, ("mean_perception_component_",))

perception_component_keys_from_results(all_results)::Vector{String} =
    telemetry_keys_from_results(all_results, ("mean_perception_component_",))

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

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

finite_mean(values)::Float64 = begin
    vals = Float64[]
    for value in values
        value isa Number || continue
        v = Float64(value)
        isfinite(v) && push!(vals, v)
    end
    isempty(vals) ? 0.0 : mean(vals)
end

function history_mean(sim::EmergentSimulation, key::String)::Float64
    return finite_mean(get(h, key, nothing) for h in sim.history)
end

function emergent_dimension_default(dim::String)::Float64
    return dim == "competitive_recursion" ? 0.0 : 0.5
end

function emergent_summary_fields(prefix::String, tier_emergent::AbstractDict)::Dict{String,Float64}
    fields = Dict{String,Float64}()
    for dim in EMERGENT_UNCERTAINTY_DIMENSIONS
        fields["$(prefix)_emergent_$(dim)"] =
            Float64(get(tier_emergent, dim, emergent_dimension_default(dim)))
        fields["$(prefix)_emergent_$(dim)_observations"] =
            Float64(get(tier_emergent, "$(dim)_observations", 0.0))
    end
    fields["$(prefix)_emergent_agentic_novelty_relevant_observations"] =
        Float64(get(tier_emergent, "agentic_novelty_relevant_observations", 0.0))
    fields["$(prefix)_emergent_agent_count"] = Float64(get(tier_emergent, "n_agents", 0.0))
    fields["$(prefix)_emergent_alive_count"] = Float64(get(tier_emergent, "n_alive", 0.0))
    fields["$(prefix)_emergent_failed_count"] = Float64(get(tier_emergent, "n_failed", 0.0))
    return fields
end

function create_tier_assignments(n_agents::Int, rng::AbstractRNG)
    assignments = String[]
    agents_per_tier = n_agents ÷ length(AI_TIERS)
    for tier in AI_TIERS
        append!(assignments, fill(tier, agents_per_tier))
    end
    remainder = n_agents - length(assignments)
    for i in 1:remainder
        push!(assignments, AI_TIERS[mod1(i, length(AI_TIERS))])
    end
    shuffle!(rng, assignments)
    return assignments
end

function create_config(; seed::Int=42)
    # Inherit BLS-calibrated defaults: SURVIVAL_THRESHOLD=$2M,
    # INITIAL_CAPITAL_RANGE=($2.5M, $10M), heterogeneous capital + threshold.
    # Fixed-tier configuration, so mixed-tier survival is directly comparable
    # to the single-population baseline (mean 0.540, BLS 50-55%).
    EmergentConfig(
        N_AGENTS=N_AGENTS,
        N_ROUNDS=N_ROUNDS,
        RANDOM_SEED=seed,
        AGENT_AI_MODE="fixed",
    )
end

"""
Run a single mixed-tier simulation with comprehensive metric collection.
"""
function run_single_mixed_simulation(run_idx::Int, seed::Int)
    rng = Random.MersenneTwister(seed)
    config = create_config(seed=seed)
    tier_assignments = create_tier_assignments(N_AGENTS, rng)

    initial_dist = Dict("none" => 0.25, "basic" => 0.25, "advanced" => 0.25, "premium" => 0.25)
    sim = GlimpseABM.EmergentSimulation(
        config=config,
        initial_tier_distribution=initial_dist,
        seed=seed
    )

    # Override with fixed tier assignments.
    # When overriding tier after construction, we must reset subscriptions
    # to match the new tier. The constructor started a subscription for each
    # agent's originally-sampled tier; leaving those in place while switching
    # fixed_ai_level corrupts billing (agent pays for tier X while using tier Y).
    for (i, agent) in enumerate(sim.agents)
        old_tier = agent.fixed_ai_level
        new_tier = tier_assignments[i]
        agent.fixed_ai_level = new_tier
        agent.current_ai_level = new_tier
        # Cancel all prior subscriptions; ensure_subscription_schedule! will
        # also cancel non-matching tiers internally, but this call pattern is
        # explicit and safer.
        for tier in collect(keys(agent.subscription_accounts))
            GlimpseABM.cancel_subscription_schedule!(agent, tier)
        end
        if new_tier != "none"
            GlimpseABM.ensure_subscription_schedule!(agent, new_tier)
        end
    end

    # Per-round survival trajectories per tier
    trajectories = Dict{String, Vector{Float64}}(tier => Float64[] for tier in AI_TIERS)

    # Time-resolved saturation tracking — investments are released on maturity,
    # so end-of-run snapshot misses peak crowding. Sample saturation each round.
    sat_per_round = Dict{Int, Vector{Float64}}()  # round -> all live saturations

    # Run simulation, tracking survival + saturation each round
    for round in 1:N_ROUNDS
        GlimpseABM.step!(sim, round)
        for tier in AI_TIERS
            tier_agents = filter(a -> a.fixed_ai_level == tier, sim.agents)
            alive_count = count(a -> a.alive, tier_agents)
            push!(trajectories[tier], alive_count / length(tier_agents))
        end
        round_sats = Float64[]
        for o in sim.market.opportunities
            if o.capacity > 0 && o.total_invested > 0
                push!(round_sats, o.total_invested / o.capacity)
            end
        end
        sat_per_round[round] = round_sats
    end

    # End-of-run aggregation. Action shares are computed from agent.action_history
    # over ALL agents (alive + dead), not just survivors at the end. Using the
    # full action history avoids survivor bias: counting only agents alive after
    # each step would exclude actions taken by agents that died that round,
    # skewing shares toward survivors.
    tier_stats = Dict{String, Dict{String, Any}}()
    all_emergent_by_tier = GlimpseABM.aggregate_emergent_uncertainty_by_tier(
        sim.agents;
        include_dead=true,
    )
    survivor_emergent_by_tier = GlimpseABM.aggregate_emergent_uncertainty_by_tier(
        sim.agents;
        include_dead=false,
    )

    for tier in AI_TIERS
        tier_agents = filter(a -> a.fixed_ai_level == tier, sim.agents)
        alive = filter(a -> a.alive, tier_agents)
        tier_all_emergent = get(all_emergent_by_tier, tier, Dict{String,Float64}())
        tier_survivor_emergent = get(survivor_emergent_by_tier, tier, Dict{String,Float64}())
        all_emergent_fields = emergent_summary_fields("all_agent", tier_all_emergent)
        survivor_emergent_fields = emergent_summary_fields("survivor", tier_survivor_emergent)

        survivor_capitals = [a.resources.capital for a in alive]

        # Action counts from full action_history (no survivorship bias)
        action_counts = Dict("invest" => 0, "innovate" => 0, "explore" => 0, "maintain" => 0)
        for agent in tier_agents
            for act in agent.action_history
                haskey(action_counts, act) && (action_counts[act] += 1)
            end
        end
        total_actions = sum(values(action_counts))
        innovate_share = total_actions > 0 ? action_counts["innovate"] / total_actions : 0.0
        explore_share = total_actions > 0 ? action_counts["explore"] / total_actions : 0.0

        # Innovation metrics — agent struct fields
        tier_innovation_count = sum(a.innovation_count for a in tier_agents)
        tier_innovation_successes = sum(a.innovation_success_count for a in tier_agents)
        tier_innovation_success_rate = tier_innovation_count > 0 ?
            tier_innovation_successes / tier_innovation_count : 0.0

        # Niches discovered — read from agent uncertainty metrics
        tier_total_niches = sum(a.uncertainty_metrics.niches_discovered for a in tier_agents)
        tier_combinations = sum(a.uncertainty_metrics.new_combinations_created for a in tier_agents)

        confidence_outcome = GlimpseABM.confidence_outcome_stats(tier_agents)
        mean_visible_opps = history_mean(sim, "mean_visible_opportunities_$tier")
        mean_info_quality = history_mean(sim, "mean_info_quality_used_$tier")
        visible_hhi = history_mean(sim, "visible_hhi_$tier")
        perception_telemetry = Dict(
            key => history_mean(sim, "mean_$(key)_$(tier)")
            for key in TIER_PERCEPTION_KEYS
        )

        tier_stats[tier] = merge(Dict(
            "total" => length(tier_agents),
            "survived" => length(alive),
            "failed" => length(tier_agents) - length(alive),
            "survival_rate" => length(alive) / length(tier_agents),
            "trajectory" => trajectories[tier],

            "mean_survivor_capital" => isempty(survivor_capitals) ? 0.0 : mean(survivor_capitals),
            "median_survivor_capital" => isempty(survivor_capitals) ? 0.0 : median(survivor_capitals),
            "p50_capital" => isempty(survivor_capitals) ? 0.0 : quantile(survivor_capitals, 0.5),
            "p90_capital" => isempty(survivor_capitals) ? 0.0 : quantile(survivor_capitals, 0.9),
            "p95_capital" => isempty(survivor_capitals) ? 0.0 : quantile(survivor_capitals, 0.95),

            "innovate_share" => innovate_share,
            "explore_share" => explore_share,
            "action_counts" => action_counts,

            "innovations_per_agent" => tier_innovation_count / length(tier_agents),
            "innovation_successes_per_agent" => tier_innovation_successes / length(tier_agents),
            "innovation_success_rate" => tier_innovation_success_rate,
            "total_niches" => tier_total_niches,
            "niches_per_agent" => tier_total_niches / length(tier_agents),
            "combinations_per_agent" => tier_combinations / length(tier_agents),
            # Backward-compatible aliases now use all assigned agents; explicit
            # survivor_* columns preserve the older survivor-only view.
            "emergent_actor_ignorance" => all_emergent_fields["all_agent_emergent_actor_ignorance"],
            "emergent_practical_indeterminism" => all_emergent_fields["all_agent_emergent_practical_indeterminism"],
            "emergent_agentic_novelty" => all_emergent_fields["all_agent_emergent_agentic_novelty"],
            "emergent_competitive_recursion" => all_emergent_fields["all_agent_emergent_competitive_recursion"],
            "mean_visible_opportunities" => mean_visible_opps,
            "mean_info_quality_used" => mean_info_quality,
            "visible_hhi" => visible_hhi,
        ), all_emergent_fields, survivor_emergent_fields, confidence_outcome, perception_telemetry)
    end

    # Capacity-saturation diagnostics — time-resolved across all rounds.
    # End-of-run snapshot misses peak crowding because investments are released
    # on maturity (agents.jl:1316). Aggregate over the full per-round panel.
    K_sat = hasproperty(sim.config, :CROWDING_CAPACITY_RATIO_K) ?
        sim.config.CROWDING_CAPACITY_RATIO_K : 1.5
    all_sats = Float64[]
    for r in 1:N_ROUNDS
        append!(all_sats, get(sat_per_round, r, Float64[]))
    end
    # Per-round peak saturation (max in each round), then aggregate
    round_maxes = Float64[]
    for r in 1:N_ROUNDS
        rs = get(sat_per_round, r, Float64[])
        push!(round_maxes, isempty(rs) ? 0.0 : maximum(rs))
    end
    # Per-round frac above K, then aggregate
    round_fracs_K = Float64[]
    for r in 1:N_ROUNDS
        rs = get(sat_per_round, r, Float64[])
        push!(round_fracs_K, isempty(rs) ? 0.0 :
              count(s -> s > K_sat, rs) / length(rs))
    end
    sat_diag = if isempty(all_sats)
        Dict("n_obs" => 0, "max_overall" => 0.0, "p50" => 0.0, "p75" => 0.0,
             "p90" => 0.0, "p95" => 0.0, "p99" => 0.0,
             "frac_above_K" => 0.0, "frac_above_2K" => 0.0,
             "max_round_peak" => 0.0, "mean_round_peak" => 0.0,
             "max_frac_above_K" => 0.0, "mean_frac_above_K_per_round" => 0.0)
    else
        Dict(
            "n_obs" => length(all_sats),
            "max_overall" => maximum(all_sats),
            "p50" => quantile(all_sats, 0.5),
            "p75" => quantile(all_sats, 0.75),
            "p90" => quantile(all_sats, 0.9),
            "p95" => quantile(all_sats, 0.95),
            "p99" => quantile(all_sats, 0.99),
            "frac_above_K" => count(s -> s > K_sat, all_sats) / length(all_sats),
            "frac_above_2K" => count(s -> s > 2 * K_sat, all_sats) / length(all_sats),
            "max_round_peak" => maximum(round_maxes),
            "mean_round_peak" => mean(round_maxes),
            "max_frac_above_K" => maximum(round_fracs_K),
            "mean_frac_above_K_per_round" => mean(round_fracs_K),
        )
    end

    decision_diag_keys = vcat([
        "mean_action_probability_entropy",
        "mean_visible_opportunities",
        "utility_floor_rate",
        "utility_ceiling_rate",
        "roi_clamp_hit_rate",
        "mean_raw_mean_roi",
        "mean_return_evidence",
        "confidence_saturation_rate",
        "visible_hhi_none",
        "visible_hhi_basic",
        "visible_hhi_advanced",
        "visible_hhi_premium",
        "mean_visible_opportunities_none",
        "mean_visible_opportunities_basic",
        "mean_visible_opportunities_advanced",
        "mean_visible_opportunities_premium",
        "mean_info_quality_used_none",
        "mean_info_quality_used_basic",
        "mean_info_quality_used_advanced",
        "mean_info_quality_used_premium",
        "total_ai_analysis_cost",
        "total_ai_decision_analysis_cost",
        "total_ai_action_analysis_cost",
        "total_ai_analysis_cost_none",
        "total_ai_analysis_cost_basic",
        "total_ai_analysis_cost_advanced",
        "total_ai_analysis_cost_premium",
    ], MARKET_RECURSION_DIAGNOSTIC_KEYS,
        knightian_component_keys_from_history(sim.history),
        perception_component_keys_from_history(sim.history))
    decision_diag = Dict{String,Float64}()
    for key in decision_diag_keys
        vals = Float64[Float64(get(h, key, 0.0)) for h in sim.history if haskey(h, key)]
        decision_diag[key] = isempty(vals) ? 0.0 : mean(vals)
    end

    return Dict(
        "run_idx" => run_idx,
        "seed" => seed,
        "tier_stats" => tier_stats,
        "saturation" => sat_diag,
        "decision_diagnostics" => decision_diag,
        "status" => "completed"
    )
end

"""
Aggregate results and compute summary statistics.
"""
function paired_treatment_effects(all_results::AbstractVector{<:AbstractDict})
    safe_mean(v) = isempty(v) ? 0.0 : mean(v)
    safe_std(v) = length(v) < 2 ? 0.0 : std(v)

    paired = Dict{String,Dict{String,Any}}()
    for tier in AI_TIERS
        diffs = Float64[]
        for result in all_results
            get(result, "status", "") == "completed" || continue
            tier_stats = get(result, "tier_stats", Dict{String,Any}())
            haskey(tier_stats, "none") && haskey(tier_stats, tier) || continue

            none_rate = Float64(tier_stats["none"]["survival_rate"])
            tier_rate = Float64(tier_stats[tier]["survival_rate"])
            push!(diffs, (tier_rate - none_rate) * 100.0)
        end

        n_runs = length(diffs)
        mean_pp = safe_mean(diffs)
        sd_pp = safe_std(diffs)
        se_pp = n_runs > 0 ? sd_pp / sqrt(n_runs) : 0.0
        paired[tier] = Dict{String,Any}(
            "mean_pp" => mean_pp,
            "sd_pp" => sd_pp,
            "se_pp" => se_pp,
            "ci_lo_pp" => mean_pp - 1.96 * se_pp,
            "ci_hi_pp" => mean_pp + 1.96 * se_pp,
            "n_runs" => n_runs,
        )
    end

    return paired
end

function scorecard_emergent_key(stats::AbstractDict, dim::String)::String
    all_agent_key = "all_agent_emergent_$(dim)"
    return haskey(stats, all_agent_key) ? all_agent_key : "emergent_$(dim)"
end

function scorecard_emergent_level(stats::AbstractDict, dim::String)::Float64
    key = scorecard_emergent_key(stats, dim)
    return Float64(get(stats, key, emergent_dimension_default(dim)))
end

function scorecard_survivor_emergent_level(stats::AbstractDict, dim::String)::Float64
    key = "survivor_emergent_$(dim)"
    return Float64(get(stats, key, scorecard_emergent_level(stats, dim)))
end

function scorecard_emergent_observations(stats::AbstractDict, dim::String)::Float64
    all_agent_key = "all_agent_emergent_$(dim)_observations"
    legacy_key = "emergent_$(dim)_observations"
    return Float64(get(stats, all_agent_key, get(stats, legacy_key, 0.0)))
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

function perceived_dimension_level(stats::AbstractDict, dim::String)::Float64
    return Float64(get(stats, "perceived_$(dim)_bounded_level", 0.0))
end

function perceived_dimension_gap_vs_prior(stats::AbstractDict, dim::String)::Float64
    return Float64(get(stats, "perceived_$(dim)_gap_vs_prior", 0.0))
end

function horizon_pressure(stats::AbstractDict)::Float64
    return mean([
        scorecard_emergent_level(stats, "practical_indeterminism"),
        scorecard_emergent_level(stats, "agentic_novelty"),
        scorecard_emergent_level(stats, "competitive_recursion"),
    ])
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

function signed_effect_size(delta::Float64, band::Float64)::Float64
    return delta / max(band, eps(Float64))
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

function dominant_second_order_channel(practical_delta::Float64, novelty_delta::Float64, recursion_delta::Float64)::String
    channels = [
        ("practical_indeterminism", practical_delta),
        ("agentic_novelty", novelty_delta),
        ("competitive_recursion", recursion_delta),
    ]
    positive = filter(pair -> pair[2] > 0.0, channels)
    isempty(positive) && return "none"
    idx = argmax([pair[2] for pair in positive])
    return positive[idx][1]
end

"""
Build the mechanism scorecard for the paradox of future knowledge.

The scorecard is diagnostic only. It compares each tier to the no-AI baseline
along the theory chain: actor ignorance reduction as first-order knowledge gain,
then practical indeterminism, agentic novelty, and competitive recursion as
second-order horizon pressure.
"""
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
        visible_gain = Float64(get(s, "mean_visible_opportunities", 0.0)) -
            Float64(get(baseline, "mean_visible_opportunities", 0.0))
        info_gain = Float64(get(s, "mean_info_quality_used", 0.0)) -
            Float64(get(baseline, "mean_info_quality_used", 0.0))

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
        first_order_observed = actor_gain > actor_band ||
            visible_gain > visible_band ||
            info_gain > info_band
        horizon_recession_observed = horizon_delta > horizon_band
        paradox_logic_supported = first_order_observed && horizon_recession_observed
        first_order_evidence_count = sum([
            actor_gain > actor_band,
            visible_gain > visible_band,
            info_gain > info_band,
        ])
        second_order_evidence_count = sum([
            practical_delta > practical_band,
            novelty_delta > novelty_band,
            recursion_delta > recursion_band,
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
                Float64(get(s, "confidence_outcome_abs_gap_mean", 0.0)) -
                Float64(get(baseline, "confidence_outcome_abs_gap_mean", 0.0)),
            "realized_multiple_std_delta_vs_none" =>
                Float64(get(s, "confidence_outcome_realized_multiple_std", 0.0)) -
                Float64(get(baseline, "confidence_outcome_realized_multiple_std", 0.0)),
            "niches_per_agent_delta_vs_none" =>
                Float64(get(s, "mean_niches_per_agent", 0.0)) -
                Float64(get(baseline, "mean_niches_per_agent", 0.0)),
            "combinations_per_agent_delta_vs_none" =>
                Float64(get(s, "mean_combinations_per_agent", 0.0)) -
                Float64(get(baseline, "mean_combinations_per_agent", 0.0)),
            "market_crowding_pressure" => market_diag("market_crowding_pressure"),
            "market_opportunity_overlap" => market_diag("market_opportunity_overlap"),
            "market_investment_concentration" => market_diag("market_investment_concentration"),
            "market_opportunity_competition" => market_diag("market_opportunity_competition"),
            "market_ai_herding_intensity" => market_diag("market_ai_herding_intensity"),
            "market_ai_action_correlation" => market_diag("market_ai_action_correlation"),
            "market_combo_reuse_pressure" => market_diag("market_combo_reuse_pressure"),
            "survival_effect_pp_vs_none" =>
                100.0 * (Float64(get(s, "mean_survival_rate", 0.0)) -
                         Float64(get(baseline, "mean_survival_rate", 0.0))),
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

function aggregate_results(all_results::AbstractVector{<:AbstractDict})
    tier_data = Dict{String, Dict{String, Vector{Float64}}}()
    for tier in AI_TIERS
        tier_data[tier] = Dict(
            "survival_rate" => Float64[],
            "mean_capital" => Float64[],
            "p50_capital" => Float64[],
            "p90_capital" => Float64[],
            "p95_capital" => Float64[],
            "innovate_share" => Float64[],
            "explore_share" => Float64[],
            "innovations_per_agent" => Float64[],
            "innovation_successes_per_agent" => Float64[],
            "innovation_success_rate" => Float64[],
            "niches_per_agent" => Float64[],
            "combinations_per_agent" => Float64[],
        )
        for key in CONFIDENCE_OUTCOME_KEYS
            tier_data[tier][key] = Float64[]
        end
        for key in TIER_KNIGHTIAN_KEYS
            tier_data[tier][key] = Float64[]
        end
        for key in TIER_PERCEPTION_KEYS
            tier_data[tier][key] = Float64[]
        end
    end

    # Collect trajectories
    all_trajectories = Dict{String, Vector{Vector{Float64}}}(tier => Vector{Float64}[] for tier in AI_TIERS)

    successful_runs = 0
    for result in all_results
        if result["status"] == "completed"
            successful_runs += 1
            for tier in AI_TIERS
                stats = result["tier_stats"][tier]
                push!(tier_data[tier]["survival_rate"], stats["survival_rate"])
                push!(tier_data[tier]["mean_capital"], stats["mean_survivor_capital"])
                push!(tier_data[tier]["p50_capital"], stats["p50_capital"])
                push!(tier_data[tier]["p90_capital"], stats["p90_capital"])
                push!(tier_data[tier]["p95_capital"], stats["p95_capital"])
                push!(tier_data[tier]["innovate_share"], stats["innovate_share"])
                push!(tier_data[tier]["explore_share"], stats["explore_share"])
                push!(tier_data[tier]["innovations_per_agent"], stats["innovations_per_agent"])
                push!(tier_data[tier]["innovation_successes_per_agent"], stats["innovation_successes_per_agent"])
                push!(tier_data[tier]["innovation_success_rate"], stats["innovation_success_rate"])
                push!(tier_data[tier]["niches_per_agent"], stats["niches_per_agent"])
                push!(tier_data[tier]["combinations_per_agent"], stats["combinations_per_agent"])
                for key in CONFIDENCE_OUTCOME_KEYS
                    push!(tier_data[tier][key], Float64(get(stats, key, 0.0)))
                end
                for key in TIER_KNIGHTIAN_KEYS
                    push!(tier_data[tier][key], Float64(get(stats, key, 0.0)))
                end
                for key in TIER_PERCEPTION_KEYS
                    push!(tier_data[tier][key], Float64(get(stats, key, 0.0)))
                end
                push!(all_trajectories[tier], stats["trajectory"])
            end
        end
    end

    # Compute mean trajectories
    mean_trajectories = Dict{String, Vector{Float64}}()
    for tier in AI_TIERS
        if !isempty(all_trajectories[tier])
            n_rounds = length(all_trajectories[tier][1])
            mean_traj = zeros(n_rounds)
            for traj in all_trajectories[tier]
                mean_traj .+= traj
            end
            mean_traj ./= length(all_trajectories[tier])
            mean_trajectories[tier] = mean_traj
        end
    end

    # Compute summary statistics. Every aggregate is guarded against
    # empty / single-sample inputs so calibration sweeps and zero-success
    # batches do not poison the saved CSVs with NaN. mean/std on a 0-length
    # vector returns NaN; std on a 1-length vector also returns NaN (sample
    # variance needs n>=2). safe_mean/safe_std collapse both edge cases to
    # 0.0 so downstream consumers see a valid Float64.
    safe_mean(v) = isempty(v) ? 0.0 : mean(v)
    safe_std(v)  = length(v) < 2 ? 0.0 : std(v)
    summary_stats = Dict{String, Dict{String, Any}}()
    for tier in AI_TIERS
        d = tier_data[tier]
        tier_summary = Dict(
            "mean_survival_rate" => safe_mean(d["survival_rate"]),
            "std_survival_rate" => safe_std(d["survival_rate"]),
            "mean_p50_capital" => safe_mean(d["p50_capital"]),
            "mean_p90_capital" => safe_mean(d["p90_capital"]),
            "mean_p95_capital" => safe_mean(d["p95_capital"]),
            "mean_innovate_share" => safe_mean(d["innovate_share"]),
            "mean_explore_share" => safe_mean(d["explore_share"]),
            "mean_innovations_per_agent" => safe_mean(d["innovations_per_agent"]),
            "mean_innovation_successes_per_agent" => safe_mean(d["innovation_successes_per_agent"]),
            "mean_innovation_success_rate" => safe_mean(d["innovation_success_rate"]),
            "mean_niches_per_agent" => safe_mean(d["niches_per_agent"]),
            "mean_combinations_per_agent" => safe_mean(d["combinations_per_agent"]),
            "n_runs" => length(d["survival_rate"]),
            "trajectory" => get(mean_trajectories, tier, Float64[])
        )
        for key in CONFIDENCE_OUTCOME_KEYS
            tier_summary[key] = safe_mean(d[key])
            tier_summary[key * "_run_std"] = safe_std(d[key])
        end
        for key in TIER_KNIGHTIAN_KEYS
            tier_summary[key] = safe_mean(d[key])
            tier_summary[key * "_run_std"] = safe_std(d[key])
        end
        for key in TIER_PERCEPTION_KEYS
            tier_summary[key] = safe_mean(d[key])
            tier_summary[key * "_run_std"] = safe_std(d[key])
        end
        summary_stats[tier] = tier_summary
    end

    # Aggregate saturation diagnostics across runs (time-resolved)
    sat_keys = ["max_overall", "p50", "p75", "p90", "p95", "p99",
                "frac_above_K", "frac_above_2K", "n_obs",
                "max_round_peak", "mean_round_peak",
                "max_frac_above_K", "mean_frac_above_K_per_round"]
    sat_summary = Dict{String, Float64}()
    for key in sat_keys
        vals = Float64[]
        for r in all_results
            r["status"] == "completed" || continue
            haskey(r, "saturation") || continue
            push!(vals, Float64(r["saturation"][key]))
        end
        sat_summary["mean_" * key] = safe_mean(vals)
        sat_summary["std_" * key] = safe_std(vals)
    end

    decision_diag_keys = vcat([
        "mean_action_probability_entropy",
        "mean_visible_opportunities",
        "utility_floor_rate",
        "utility_ceiling_rate",
        "roi_clamp_hit_rate",
        "mean_raw_mean_roi",
        "mean_return_evidence",
        "confidence_saturation_rate",
        "visible_hhi_none",
        "visible_hhi_basic",
        "visible_hhi_advanced",
        "visible_hhi_premium",
        "mean_visible_opportunities_none",
        "mean_visible_opportunities_basic",
        "mean_visible_opportunities_advanced",
        "mean_visible_opportunities_premium",
        "mean_info_quality_used_none",
        "mean_info_quality_used_basic",
        "mean_info_quality_used_advanced",
        "mean_info_quality_used_premium",
        "total_ai_analysis_cost",
        "total_ai_decision_analysis_cost",
        "total_ai_action_analysis_cost",
        "total_ai_analysis_cost_none",
        "total_ai_analysis_cost_basic",
        "total_ai_analysis_cost_advanced",
        "total_ai_analysis_cost_premium",
    ], MARKET_RECURSION_DIAGNOSTIC_KEYS,
        knightian_component_keys_from_results(all_results),
        perception_component_keys_from_results(all_results))
    decision_diag_summary = Dict{String,Float64}()
    for key in decision_diag_keys
        vals = Float64[]
        for r in all_results
            r["status"] == "completed" || continue
            haskey(r, "decision_diagnostics") || continue
            push!(vals, Float64(get(r["decision_diagnostics"], key, 0.0)))
        end
        decision_diag_summary["mean_" * key] = safe_mean(vals)
        decision_diag_summary["std_" * key] = safe_std(vals)
    end

    summary = Dict(
        "summary_stats" => summary_stats,
        "tier_data" => tier_data,
        "mean_trajectories" => mean_trajectories,
        "paired_treatment_effects" => paired_treatment_effects(all_results),
        "saturation_summary" => sat_summary,
        "decision_diagnostics_summary" => decision_diag_summary,
        "successful_runs" => successful_runs
    )
    summary["paradox_mechanism_scorecard"] = build_paradox_mechanism_scorecard(summary)
    return summary
end

"""
Save comprehensive results to CSV files.
"""
function save_results(summary::Dict, all_results::AbstractVector{<:AbstractDict}, output_dir::String)
    mkpath(output_dir)
    stats = summary["summary_stats"]

    # 1. Summary statistics
    summary_rows = []
    for tier in AI_TIERS
        s = stats[tier]
        row = Dict{Symbol,Any}(
            :tier => tier,
            :tier_label => tier_display_label(tier),
            :mean_survival_rate => s["mean_survival_rate"],
            :std_survival_rate => s["std_survival_rate"],
            :mean_p50_capital => s["mean_p50_capital"],
            :mean_p90_capital => s["mean_p90_capital"],
            :mean_p95_capital => s["mean_p95_capital"],
            :mean_innovate_share => s["mean_innovate_share"],
            :mean_explore_share => s["mean_explore_share"],
            :mean_innovations_per_agent => s["mean_innovations_per_agent"],
            :mean_innovation_successes_per_agent => s["mean_innovation_successes_per_agent"],
            :mean_innovation_success_rate => s["mean_innovation_success_rate"],
            :mean_niches_per_agent => s["mean_niches_per_agent"],
            :mean_combinations_per_agent => s["mean_combinations_per_agent"],
            :n_runs => s["n_runs"],
        )
        for key in CONFIDENCE_OUTCOME_KEYS
            row[Symbol(key)] = get(s, key, 0.0)
            row[Symbol(key * "_run_std")] = get(s, key * "_run_std", 0.0)
        end
        for key in TIER_KNIGHTIAN_KEYS
            row[Symbol(key)] = get(s, key, 0.0)
            row[Symbol(key * "_run_std")] = get(s, key * "_run_std", 0.0)
        end
        for key in TIER_PERCEPTION_KEYS
            row[Symbol(key)] = get(s, key, 0.0)
            row[Symbol(key * "_run_std")] = get(s, key * "_run_std", 0.0)
        end
        push!(summary_rows, row)
    end
    summary_df = DataFrame(summary_rows)
    summary_base_cols = [
        :tier,
        :tier_label,
        :mean_survival_rate,
        :std_survival_rate,
        :mean_p50_capital,
        :mean_p90_capital,
        :mean_p95_capital,
        :mean_innovate_share,
        :mean_explore_share,
        :mean_innovations_per_agent,
        :mean_innovation_successes_per_agent,
        :mean_innovation_success_rate,
        :mean_niches_per_agent,
        :mean_combinations_per_agent,
        :n_runs,
    ]
    summary_conf_cols = Symbol.(CONFIDENCE_OUTCOME_KEYS)
    summary_conf_std_cols = Symbol.(key * "_run_std" for key in CONFIDENCE_OUTCOME_KEYS)
    summary_knightian_cols = Symbol.(TIER_KNIGHTIAN_KEYS)
    summary_knightian_std_cols = Symbol.(key * "_run_std" for key in TIER_KNIGHTIAN_KEYS)
    summary_perception_cols = Symbol.(TIER_PERCEPTION_KEYS)
    summary_perception_std_cols = Symbol.(key * "_run_std" for key in TIER_PERCEPTION_KEYS)
    select!(summary_df, vcat(
        summary_base_cols,
        summary_knightian_cols,
        summary_knightian_std_cols,
        summary_perception_cols,
        summary_perception_std_cols,
        summary_conf_cols,
        summary_conf_std_cols,
    ))
    CSV.write(joinpath(output_dir, "summary_stats.csv"), summary_df)

    if haskey(summary, "paradox_mechanism_scorecard")
        scorecard = summary["paradox_mechanism_scorecard"]
        score_rows = []
        for tier in AI_TIERS
            row = Dict{Symbol,Any}(
                :tier => tier,
                :tier_label => tier_display_label(tier),
            )
            s = scorecard[tier]
            for key in PARADOX_SCORECARD_KEYS
                row[Symbol(key)] = get(s, key, 0.0)
            end
            row[:first_order_benefit_observed] = get(s, "first_order_benefit_observed", false)
            row[:horizon_recession_observed] = get(s, "horizon_recession_observed", false)
            row[:paradox_logic_supported] = get(s, "paradox_logic_supported", false)
            row[:dominant_second_order_channel] = get(s, "dominant_second_order_channel", "none")
            push!(score_rows, row)
        end
        score_df = DataFrame(score_rows)
        select!(score_df, vcat(
            [:tier, :tier_label],
            Symbol.(PARADOX_SCORECARD_KEYS),
            [
                :first_order_benefit_observed,
                :horizon_recession_observed,
                :paradox_logic_supported,
                :dominant_second_order_channel,
            ],
        ))
        CSV.write(joinpath(output_dir, "paradox_mechanism_scorecard.csv"), score_df)
    end

    # 2. Per-run detailed data
    run_rows = []
    for result in all_results
        if result["status"] == "completed"
            for tier in AI_TIERS
                s = result["tier_stats"][tier]
                row = Dict{Symbol,Any}(
                    :run_idx => result["run_idx"],
                    :tier => tier,
                    :tier_label => tier_display_label(tier),
                    :survived => s["survived"],
                    :failed => s["failed"],
                    :total => s["total"],
                    :survival_rate => s["survival_rate"],
                    :mean_capital => s["mean_survivor_capital"],
                    :p50_capital => s["p50_capital"],
                    :p90_capital => s["p90_capital"],
                    :p95_capital => s["p95_capital"],
                    :innovate_share => s["innovate_share"],
                    :explore_share => s["explore_share"],
                    :innovations_per_agent => s["innovations_per_agent"],
                    :innovation_successes_per_agent => s["innovation_successes_per_agent"],
                    :innovation_success_rate => s["innovation_success_rate"],
                    :niches_per_agent => s["niches_per_agent"],
                    :combinations_per_agent => s["combinations_per_agent"],
                )
                for key in CONFIDENCE_OUTCOME_KEYS
                    row[Symbol(key)] = get(s, key, 0.0)
                end
                for key in TIER_KNIGHTIAN_KEYS
                    row[Symbol(key)] = get(s, key, 0.0)
                end
                for key in TIER_PERCEPTION_KEYS
                    row[Symbol(key)] = get(s, key, 0.0)
                end
                push!(run_rows, row)
            end
        end
    end
    run_df = DataFrame(run_rows)
    run_base_cols = [
        :run_idx,
        :tier,
        :tier_label,
        :survived,
        :failed,
        :total,
        :survival_rate,
        :mean_capital,
        :p50_capital,
        :p90_capital,
        :p95_capital,
        :innovate_share,
        :explore_share,
        :innovations_per_agent,
        :innovation_successes_per_agent,
        :innovation_success_rate,
        :niches_per_agent,
        :combinations_per_agent,
    ]
    select!(run_df, vcat(
        run_base_cols,
        Symbol.(TIER_KNIGHTIAN_KEYS),
        Symbol.(TIER_PERCEPTION_KEYS),
        Symbol.(CONFIDENCE_OUTCOME_KEYS),
    ))
    CSV.write(joinpath(output_dir, "per_run_data.csv"), run_df)

    # 3. Survival trajectories. Guard against empty trajectory dict
    # (zero successful runs) and missing tier keys (partial-batch case).
    trajectories = summary["mean_trajectories"]
    if !isempty(trajectories) &&
       all(haskey(trajectories, t) && !isempty(trajectories[t]) for t in AI_TIERS)
        traj_df = DataFrame(
            round=1:N_ROUNDS,
            none=trajectories["none"],
            basic=trajectories["basic"],
            advanced=trajectories["advanced"],
            premium=trajectories["premium"],
            frontier_ai=trajectories["premium"],
        )
        CSV.write(joinpath(output_dir, "survival_trajectories.csv"), traj_df)
    end

    # 4. Treatment effects. Guard the std_error denominator so a
    # zero-success batch yields 0.0 instead of Inf (sqrt(0)) and a single-
    # success batch yields std/1 = 0.0 (we already collapsed std to 0.0
    # for n<2 in aggregate_results).
    # Paired run-level contrasts: mixed-tier runs share the same
    # market seed across tiers; the inferential object is therefore the
    # within-run tier-minus-none contrast, not each tier's marginal survival SE.
    baseline = stats["none"]["mean_survival_rate"]
    paired_effects = get(summary, "paired_treatment_effects", paired_treatment_effects(all_results))
    effects = [(
        tier=tier,
        tier_label=tier_display_label(tier),
        survival_rate=stats[tier]["mean_survival_rate"],
        treatment_effect=(stats[tier]["mean_survival_rate"] - baseline) * 100,
        std_error=get(paired_effects[tier], "se_pp", 0.0),
        paired_treatment_effect=get(paired_effects[tier], "mean_pp", 0.0),
        paired_std_error=get(paired_effects[tier], "se_pp", 0.0),
        paired_sd=get(paired_effects[tier], "sd_pp", 0.0),
        paired_ci_lo=get(paired_effects[tier], "ci_lo_pp", 0.0),
        paired_ci_hi=get(paired_effects[tier], "ci_hi_pp", 0.0),
        paired_n_runs=get(paired_effects[tier], "n_runs", 0),
        unpaired_tier_std_error=stats[tier]["n_runs"] > 0 ?
            stats[tier]["std_survival_rate"] / sqrt(stats[tier]["n_runs"]) :
            0.0,
    ) for tier in AI_TIERS]
    CSV.write(joinpath(output_dir, "treatment_effects.csv"), DataFrame(effects))

    # 5. Saturation diagnostics — single-row summary
    if haskey(summary, "saturation_summary")
        sat = summary["saturation_summary"]
        sat_row = [(
            mean_n_obs = get(sat, "mean_n_obs", 0.0),
            mean_max_overall = get(sat, "mean_max_overall", 0.0),
            mean_p50 = get(sat, "mean_p50", 0.0),
            mean_p75 = get(sat, "mean_p75", 0.0),
            mean_p90 = get(sat, "mean_p90", 0.0),
            mean_p95 = get(sat, "mean_p95", 0.0),
            mean_p99 = get(sat, "mean_p99", 0.0),
            mean_frac_above_K = get(sat, "mean_frac_above_K", 0.0),
            mean_frac_above_2K = get(sat, "mean_frac_above_2K", 0.0),
            mean_max_round_peak = get(sat, "mean_max_round_peak", 0.0),
            mean_mean_round_peak = get(sat, "mean_mean_round_peak", 0.0),
            mean_max_frac_above_K = get(sat, "mean_max_frac_above_K", 0.0),
            mean_mean_frac_above_K_per_round = get(sat, "mean_mean_frac_above_K_per_round", 0.0),
        )]
        CSV.write(joinpath(output_dir, "saturation_diagnostics.csv"), DataFrame(sat_row))
    end

    # 6. Per-run saturation (for distribution plots)
    sat_rows = []
    for r in all_results
        r["status"] == "completed" || continue
        haskey(r, "saturation") || continue
        s = r["saturation"]
        push!(sat_rows, (
            run_idx = r["run_idx"],
            n_obs = s["n_obs"],
            sat_max_overall = s["max_overall"],
            sat_p50 = s["p50"],
            sat_p75 = s["p75"],
            sat_p90 = s["p90"],
            sat_p95 = s["p95"],
            sat_p99 = s["p99"],
            frac_above_K = s["frac_above_K"],
            frac_above_2K = s["frac_above_2K"],
            max_round_peak = s["max_round_peak"],
            mean_round_peak = s["mean_round_peak"],
            max_frac_above_K = s["max_frac_above_K"],
            mean_frac_above_K_per_round = s["mean_frac_above_K_per_round"],
        ))
    end
    if !isempty(sat_rows)
        CSV.write(joinpath(output_dir, "per_run_saturation.csv"), DataFrame(sat_rows))
    end

    # 7. Decision/variance diagnostics
    if haskey(summary, "decision_diagnostics_summary")
        diag = summary["decision_diagnostics_summary"]
        diag_row = [(
            mean_action_probability_entropy = get(diag, "mean_mean_action_probability_entropy", 0.0),
            std_action_probability_entropy = get(diag, "std_mean_action_probability_entropy", 0.0),
            mean_visible_opportunities = get(diag, "mean_mean_visible_opportunities", 0.0),
            std_visible_opportunities = get(diag, "std_mean_visible_opportunities", 0.0),
            mean_utility_floor_rate = get(diag, "mean_utility_floor_rate", 0.0),
            mean_utility_ceiling_rate = get(diag, "mean_utility_ceiling_rate", 0.0),
            mean_roi_clamp_hit_rate = get(diag, "mean_roi_clamp_hit_rate", 0.0),
            mean_raw_mean_roi = get(diag, "mean_mean_raw_mean_roi", 0.0),
            mean_return_evidence = get(diag, "mean_mean_return_evidence", 0.0),
            mean_confidence_saturation_rate = get(diag, "mean_confidence_saturation_rate", 0.0),
            mean_visible_hhi_none = get(diag, "mean_visible_hhi_none", 0.0),
            mean_visible_hhi_basic = get(diag, "mean_visible_hhi_basic", 0.0),
            mean_visible_hhi_advanced = get(diag, "mean_visible_hhi_advanced", 0.0),
            mean_visible_hhi_premium = get(diag, "mean_visible_hhi_premium", 0.0),
            mean_visible_opportunities_none = get(diag, "mean_mean_visible_opportunities_none", 0.0),
            mean_visible_opportunities_basic = get(diag, "mean_mean_visible_opportunities_basic", 0.0),
            mean_visible_opportunities_advanced = get(diag, "mean_mean_visible_opportunities_advanced", 0.0),
            mean_visible_opportunities_premium = get(diag, "mean_mean_visible_opportunities_premium", 0.0),
            mean_info_quality_used_none = get(diag, "mean_mean_info_quality_used_none", 0.0),
            mean_info_quality_used_basic = get(diag, "mean_mean_info_quality_used_basic", 0.0),
            mean_info_quality_used_advanced = get(diag, "mean_mean_info_quality_used_advanced", 0.0),
            mean_info_quality_used_premium = get(diag, "mean_mean_info_quality_used_premium", 0.0),
            mean_total_ai_analysis_cost = get(diag, "mean_total_ai_analysis_cost", 0.0),
            mean_total_ai_decision_analysis_cost = get(diag, "mean_total_ai_decision_analysis_cost", 0.0),
            mean_total_ai_action_analysis_cost = get(diag, "mean_total_ai_action_analysis_cost", 0.0),
            mean_total_ai_analysis_cost_none = get(diag, "mean_total_ai_analysis_cost_none", 0.0),
            mean_total_ai_analysis_cost_basic = get(diag, "mean_total_ai_analysis_cost_basic", 0.0),
            mean_total_ai_analysis_cost_advanced = get(diag, "mean_total_ai_analysis_cost_advanced", 0.0),
            mean_total_ai_analysis_cost_premium = get(diag, "mean_total_ai_analysis_cost_premium", 0.0),
            mean_market_crowding_pressure = get(diag, "mean_market_crowding_pressure", 0.0),
            mean_market_opportunity_overlap = get(diag, "mean_market_opportunity_overlap", 0.0),
            mean_market_investment_concentration = get(diag, "mean_market_investment_concentration", 0.0),
            mean_market_opportunity_competition = get(diag, "mean_market_opportunity_competition", 0.0),
            mean_market_ai_herding_intensity = get(diag, "mean_market_ai_herding_intensity", 0.0),
            mean_market_ai_action_correlation = get(diag, "mean_market_ai_action_correlation", 0.0),
            mean_market_combo_reuse_pressure = get(diag, "mean_market_combo_reuse_pressure", 0.0),
        )]
        CSV.write(joinpath(output_dir, "decision_diagnostics.csv"), DataFrame(diag_row))

        component_rows = []
        component_keys = sort([
            replace(String(key), "mean_" => ""; count=1)
            for key in keys(diag)
            if startswith(String(key), "mean_knightian_")
        ])
        for key in component_keys
            push!(component_rows, (
                metric = key,
                mean = get(diag, "mean_" * key, 0.0),
                std = get(diag, "std_" * key, 0.0),
            ))
        end
        if !isempty(component_rows)
            CSV.write(joinpath(output_dir, "knightian_component_diagnostics.csv"), DataFrame(component_rows))
        end

        perception_component_rows = []
        perception_component_keys = sort([
            replace(String(key), "mean_" => ""; count=1)
            for key in keys(diag)
            if startswith(String(key), "mean_mean_perception_component_")
        ])
        for key in perception_component_keys
            push!(perception_component_rows, (
                metric = key,
                mean = get(diag, "mean_" * key, 0.0),
                std = get(diag, "std_" * key, 0.0),
            ))
        end
        if !isempty(perception_component_rows)
            CSV.write(
                joinpath(output_dir, "perception_component_diagnostics.csv"),
                DataFrame(perception_component_rows),
            )
        end
    end

    println("Results saved to: $output_dir")
end

"""
Print formatted results summary.
"""
function print_results(summary::Dict)
    stats = summary["summary_stats"]
    baseline = stats["none"]["mean_survival_rate"]
    paired_effects = get(summary, "paired_treatment_effects", Dict{String,Any}())

    println("\n" * "="^80)
    println("  MIXED-TIER EXPERIMENT RESULTS (Table 3 Equivalent)")
    println("="^80)

    # Panel A: Survival Rates
    println("\n  A. Final Survival Rates")
    println("-"^60)
    for tier in AI_TIERS
        s = stats[tier]
        @printf("    %-12s: %6.2f%% (±%.2f%%)\n",
                tier_display_label(tier), s["mean_survival_rate"]*100, s["std_survival_rate"]*100)
    end

    # Panel B: Treatment Effects
    println("\n  B. Treatment Effects vs No AI")
    println("-"^60)
    for tier in ["basic", "advanced", "premium"]
        effect = (stats[tier]["mean_survival_rate"] - baseline) * 100
        if haskey(paired_effects, tier)
            p = paired_effects[tier]
            @printf("    %-12s: %+.2f pp [%+.2f, %+.2f]\n",
                    tier_display_label(tier), Float64(p["mean_pp"]),
                    Float64(p["ci_lo_pp"]), Float64(p["ci_hi_pp"]))
        else
            @printf("    %-12s: %+.2f percentage points\n", tier_display_label(tier), effect)
        end
    end

    # Panel D & E: Activity Shares
    println("\n  D. Innovation Activity Share")
    println("-"^60)
    for tier in AI_TIERS
        @printf("    %-12s: %6.2f%%\n", tier_display_label(tier), stats[tier]["mean_innovate_share"]*100)
    end

    println("\n  E. Exploration Activity Share")
    println("-"^60)
    for tier in AI_TIERS
        @printf("    %-12s: %6.2f%%\n", tier_display_label(tier), stats[tier]["mean_explore_share"]*100)
    end

    # Panel F: Survivor Wealth
    println("\n  F. Survivor Wealth Percentiles")
    println("-"^60)
    @printf("    %-12s %12s %12s %12s\n", "Tier", "P50", "P90", "P95")
    for tier in AI_TIERS
        s = stats[tier]
        @printf("    %-12s %12.0f %12.0f %12.0f\n",
                tier_display_label(tier), s["mean_p50_capital"], s["mean_p90_capital"], s["mean_p95_capital"])
    end

    # Panel H: Innovation Volume
    println("\n  H. Innovations Per Agent")
    println("-"^60)
    for tier in AI_TIERS
        @printf("    %-12s: %.3f\n", tier_display_label(tier), stats[tier]["mean_innovations_per_agent"])
    end

    # Panel I: Capacity-saturation diagnostics
    if haskey(summary, "saturation_summary")
        sat = summary["saturation_summary"]
        println("\n  I. Capacity-Saturation Diagnostics (time-resolved across all rounds)")
        println("-"^60)
        @printf("    Obs per run (round×opp): %.0f\n",
                get(sat, "mean_n_obs", 0.0))
        @printf("    Saturation max overall:  %.2f ± %.2f\n",
                get(sat, "mean_max_overall", 0.0), get(sat, "std_max_overall", 0.0))
        @printf("    Saturation p95:          %.2f ± %.2f\n",
                get(sat, "mean_p95", 0.0), get(sat, "std_p95", 0.0))
        @printf("    Saturation p50/p75/p90:  %.2f / %.2f / %.2f\n",
                get(sat, "mean_p50", 0.0), get(sat, "mean_p75", 0.0),
                get(sat, "mean_p90", 0.0))
        @printf("    Frac above K_sat (all):  %.1f%% ± %.1f%%\n",
                100*get(sat, "mean_frac_above_K", 0.0),
                100*get(sat, "std_frac_above_K", 0.0))
        @printf("    Max round peak sat:      %.2f (mean %.2f)\n",
                get(sat, "mean_max_round_peak", 0.0),
                get(sat, "mean_mean_round_peak", 0.0))
        @printf("    Max-round frac above K:  %.1f%% (mean per-round %.1f%%)\n",
                100*get(sat, "mean_max_frac_above_K", 0.0),
                100*get(sat, "mean_mean_frac_above_K_per_round", 0.0))
    end

    if haskey(summary, "decision_diagnostics_summary")
        diag = summary["decision_diagnostics_summary"]
        println("\n  J. Decision/Variance Diagnostics")
        println("-"^60)
        @printf("    Action probability entropy: %.3f ± %.3f\n",
                get(diag, "mean_mean_action_probability_entropy", 0.0),
                get(diag, "std_mean_action_probability_entropy", 0.0))
        @printf("    Visible opportunities:      %.1f ± %.1f\n",
                get(diag, "mean_mean_visible_opportunities", 0.0),
                get(diag, "std_mean_visible_opportunities", 0.0))
        @printf("    Utility floor/ceiling hits: %.1f%% / %.1f%%\n",
                100*get(diag, "mean_utility_floor_rate", 0.0),
                100*get(diag, "mean_utility_ceiling_rate", 0.0))
        @printf("    ROI clamp hit rate:         %.1f%%\n",
                100*get(diag, "mean_roi_clamp_hit_rate", 0.0))
        @printf("    Raw ROI / return evidence: %.3f / %.3f\n",
                get(diag, "mean_mean_raw_mean_roi", 0.0),
                get(diag, "mean_mean_return_evidence", 0.0))
        @printf("    Confidence saturation:      %.1f%%\n",
                100*get(diag, "mean_confidence_saturation_rate", 0.0))
        @printf("    Visible HHI none/frontier:  %.3f / %.3f\n",
                get(diag, "mean_visible_hhi_none", 0.0),
                get(diag, "mean_visible_hhi_premium", 0.0))
        @printf("    Market recursion crowd/overlap/herd/corr: %.3f / %.3f / %.3f / %.3f\n",
                get(diag, "mean_market_crowding_pressure", 0.0),
                get(diag, "mean_market_opportunity_overlap", 0.0),
                get(diag, "mean_market_ai_herding_intensity", 0.0),
                get(diag, "mean_market_ai_action_correlation", 0.0))
        @printf("    AI analysis cost/round:     %.0f total (frontier %.0f)\n",
                get(diag, "mean_total_ai_analysis_cost", 0.0),
                get(diag, "mean_total_ai_analysis_cost_premium", 0.0))
    end

    println("\n  K. Confidence-Outcome Calibration")
    println("-"^60)
    @printf("    %-12s %10s %10s %10s %10s\n",
            "Tier", "raw gap", "abs gap", "pos%", "ROI p90")
    for tier in AI_TIERS
        s = stats[tier]
        @printf("    %-12s %10.3f %10.3f %9.1f%% %10.2f\n",
                tier_display_label(tier),
                get(s, "confidence_outcome_raw_gap_mean", 0.0),
                get(s, "confidence_outcome_abs_gap_mean", 0.0),
                100 * get(s, "confidence_outcome_positive_gap_rate", 0.0),
                get(s, "confidence_outcome_realized_multiple_p90", 0.0))
    end

    scorecard = get(summary, "paradox_mechanism_scorecard", build_paradox_mechanism_scorecard(summary))
    println("\n  L. Paradox of Future Knowledge Mechanism Scorecard")
    println("-"^60)
    @printf("    %-12s %8s %8s %9s %12s %8s\n",
            "Tier", "actor+", "info+", "horizon+", "channel", "surv")
    for tier in AI_TIERS
        s = scorecard[tier]
        @printf("    %-12s %+8.3f %+8.3f %+9.3f %12s %+8.2f\n",
                tier_display_label(tier),
                get(s, "first_order_knowledge_gain_vs_none", 0.0),
                get(s, "info_quality_gain_vs_none", 0.0),
                get(s, "epistemic_horizon_delta_vs_none", 0.0),
                get(s, "dominant_second_order_channel", "none"),
                get(s, "survival_effect_pp_vs_none", 0.0))
    end

    println("\n" * "="^80)
end

# ============================================================================
# MAIN
# ============================================================================

function main()
    println("="^80)
    println("  MIXED-POPULATION FIXED-TIER EXPERIMENT - FULL ANALYSIS")
    println("="^80)
    println("  Agents per run:     $N_AGENTS")
    println("  Agents per tier:    $AGENTS_PER_TIER")
    println("  Number of runs:     $N_RUNS")
    println("  Rounds per run:     $N_ROUNDS")
    println("  Threads available:  $(Threads.nthreads())")
    println("  Output directory:   $OUTPUT_DIR")
    println("="^80)

    mkpath(OUTPUT_DIR)
    write_run_provenance!(
        OUTPUT_DIR;
        script_name=basename(@__FILE__),
        parameters=Dict(
            "BASE_SEED" => BASE_SEED,
            "N_AGENTS" => N_AGENTS,
            "N_ROUNDS" => N_ROUNDS,
            "N_RUNS" => N_RUNS,
            "OUTPUT_DIR" => OUTPUT_DIR,
        ),
        notes=Dict(
            "design" => "fixed equal mixed-tier",
        ),
    )

    println("\nStarting $N_RUNS simulation runs...")
    start_time = time()

    all_results = Vector{Dict}(undef, N_RUNS)
    completed = Threads.Atomic{Int}(0)

    Threads.@threads for run_idx in 1:N_RUNS
        seed = BASE_SEED + run_idx
        try
            all_results[run_idx] = run_single_mixed_simulation(run_idx, seed)
        catch e
            all_results[run_idx] = Dict("run_idx" => run_idx, "status" => "error: $e")
            @warn "Run $run_idx failed: $e"
        end

        # Threads.atomic_add! returns the OLD value (Julia follows
        # C __atomic_fetch_add semantics). Add 1 to get the post-increment
        # count so the log prints 1..N (not 0..N-1) and the final N/N line
        # actually fires.
        c = Threads.atomic_add!(completed, 1) + 1
        if c % 10 == 0 || c == N_RUNS
            @printf("  Completed %d/%d runs (%.1fs elapsed)\n", c, N_RUNS, time() - start_time)
        end
    end

    @printf("\nAll runs complete! Total time: %.1f seconds\n", time() - start_time)

    println("\nAggregating results...")
    summary = aggregate_results(all_results)

    print_results(summary)
    save_results(summary, all_results, OUTPUT_DIR)

    return summary, all_results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
