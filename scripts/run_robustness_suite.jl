#!/usr/bin/env julia
#
# Robustness and sensitivity suite for the paradox of future knowledge.
#
# The curated set of placebo, mechanism-decomposition, alternative-specification,
# and boundary-condition checks for the model.
#
# Conditions are organized around four categories:
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

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GlimpseABM
using Statistics
using Random
using DataFrames
using CSV
using Dates
using Printf

include(joinpath(@__DIR__, "_safe_stats.jl"))
include(joinpath(@__DIR__, "_launch_metadata.jl"))

const AI_TIERS = ["none", "basic", "advanced", "premium"]
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
const SUITE_SEED_MODE = lowercase(get(ENV, "SUITE_SEED_MODE", "paired"))
const CONDITION_FILTER = strip(get(ENV, "CONDITIONS", ""))
const OUTPUT_DIR = get(ENV, "OUTPUT_DIR",
    joinpath(@__DIR__, "..", "results", "robustness_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"))

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
novelty_generalization_stress_on!(config::EmergentConfig) = (config.NOVELTY_NOISE_INVERSION_FACTOR = 0.4; nothing)
premium_exec_5x!(config::EmergentConfig) = (config.AI_EXECUTION_SUCCESS_MULTIPLIERS["premium"] = 5.0; nothing)
difficulty_scaled_ai_cost!(config::EmergentConfig) = (config.AI_DIFFICULTY_COST_SCALING = 1.0; nothing)
ai_complementarity_on!(config::EmergentConfig) = (config.AI_COMPLEMENTARITY_STRENGTH = 1.0; nothing)

function no_hallucination_overconfidence!(config::EmergentConfig)
    config.HALLUCINATION_INTENSITY = 0.0
    config.OVERCONFIDENCE_INTENSITY = 0.0
    return nothing
end

function cash_only_survival!(config::EmergentConfig)
    # Counterfactual: legacy cash-on-hand survival rule (deployed capital does
    # not count toward solvency). Measures the illiquidity contribution to the
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

function market_slack_high_capacity!(config::EmergentConfig)
    config.OPPORTUNITY_BASE_CAPACITY *= 2.0
    config.CROWDING_CAPACITY_RATIO_K = 2.25
    config.CROWDING_STRENGTH_LAMBDA = 0.75
    return nothing
end

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

function balanced_tier_assignments(n_agents::Int, rng::AbstractRNG)
    assignments = String[]
    per_tier = n_agents ÷ length(AI_TIERS)
    for tier in AI_TIERS
        append!(assignments, fill(tier, per_tier))
    end
    remainder = n_agents - length(assignments)
    for i in 1:remainder
        push!(assignments, AI_TIERS[mod1(i, length(AI_TIERS))])
    end
    shuffle!(rng, assignments)
    return assignments
end

function build_config(condition::RobustnessCondition, seed::Int)
    config = EmergentConfig(
        N_AGENTS = N_AGENTS,
        N_ROUNDS = N_ROUNDS,
        RANDOM_SEED = seed,
        AGENT_AI_MODE = condition.design == "emergent" ? "emergent" : "fixed",
    )
    condition.apply!(config)
    return config
end

function apply_balanced_fixed_tiers!(sim::EmergentSimulation, assignments::Vector{String})
    for (i, agent) in enumerate(sim.agents)
        new_tier = assignments[i]
        agent.fixed_ai_level = new_tier
        agent.current_ai_level = new_tier
        for tier in collect(keys(agent.subscription_accounts))
            GlimpseABM.cancel_subscription_schedule!(agent, tier)
        end
        if new_tier != "none"
            GlimpseABM.ensure_subscription_schedule!(agent, new_tier)
        end
    end
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
    return summarize_simulation(condition, sim, run_idx, seed)
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
        perception_telemetry = Dict(
            key => history_mean(sim, "mean_$(key)_$(tier)")
            for key in TIER_PERCEPTION_KEYS
        )

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
            :survival_rate => n_tier == 0 ? 0.0 : length(alive) / n_tier,
            :mean_final_capital => isempty(final_capitals) ? 0.0 : mean(final_capitals),
            :mean_in_flight_capital => isempty(in_flight_capitals) ? 0.0 : mean(in_flight_capitals),
            :mean_net_worth => isempty(net_worths) ? 0.0 : mean(net_worths),
            :liquidity_failure_rate => n_tier == 0 ? 0.0 : n_liquidity_fail / n_tier,
            :equity_failure_rate => n_tier == 0 ? 0.0 : n_equity_fail / n_tier,
            :tier_sector_invest_hhi => tier_sector_hhi,
            :tier_sector_overlap => tier_sector_overlap,
            :roi => total_invested > 0 ? (total_returned - total_invested) / total_invested : 0.0,
            :mean_competition => isempty(all_competition) ? 0.0 : mean(all_competition),
            :max_competition => isempty(all_competition) ? 0.0 : maximum(all_competition),
            :innovations_per_agent => n_tier == 0 ? 0.0 : sum(a.innovation_count for a in agents) / n_tier,
            :niches_per_agent => n_tier == 0 ? 0.0 : tier_total_niches / n_tier,
            :combinations_per_agent => n_tier == 0 ? 0.0 : tier_combinations / n_tier,
            :invest_share => total_actions > 0 ? action_counts["invest"] / total_actions : 0.0,
            :innovate_share => total_actions > 0 ? action_counts["innovate"] / total_actions : 0.0,
            :explore_share => total_actions > 0 ? action_counts["explore"] / total_actions : 0.0,
            :maintain_share => total_actions > 0 ? action_counts["maintain"] / total_actions : 0.0,
            :emergent_actor_ignorance => all_emergent_fields["all_agent_emergent_actor_ignorance"],
            :emergent_practical_indeterminism => all_emergent_fields["all_agent_emergent_practical_indeterminism"],
            :emergent_agentic_novelty => all_emergent_fields["all_agent_emergent_agentic_novelty"],
            :emergent_competitive_recursion => all_emergent_fields["all_agent_emergent_competitive_recursion"],
            :mean_visible_opportunities => history_mean(sim, "mean_visible_opportunities_$(tier)"),
            :mean_info_quality_used => history_mean(sim, "mean_info_quality_used_$(tier)"),
            :visible_hhi => history_mean(sim, "visible_hhi_$(tier)"),
        )
        for source in (all_emergent_fields, survivor_emergent_fields, perception_telemetry,
                       market_telemetry, confidence_outcome)
            for (key, value) in source
                row[Symbol(key)] = value
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
            "NOVELTY_GENERALIZATION_STRESS_ON", "mechanism_decomposition",
            "Enable an explicit AI generalization stress on novel, data-poor, stealth, and endogenous opportunities.",
            "Tests a stronger agentic-novelty interpretation as a stress cell, not a baseline assumption.",
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
            "Strong near-omniscience counterfactual for public information.",
            "fixed_mixed", "minimal", frontier_public_omniscience!),

        RobustnessCondition(
            "FRONTIER_EXEC_5X", "alternative_frontier_ai_specification",
            "Frontier AI receives 5x investment execution success multiplier.",
            "Tests whether execution dominance can overcome second-order future-knowledge costs.",
            "fixed_mixed", "minimal", premium_exec_5x!),

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
            "STRATEGIC_ANTICIPATION_ON", "alternative_frontier_ai_specification",
            "Enable AI-assisted anticipation of crowded shared signals.",
            "Operationalizes AGI as second-order strategic reasoning about other AI agents.",
            "fixed_mixed", "core", strategic_anticipation_on!),

        RobustnessCondition(
            "STRATEGIC_DIVERSIFICATION_ON", "alternative_frontier_ai_specification",
            "Enable anticipation plus equilibrium-aware diversification across near-top opportunities.",
            "Tests whether differentiated strategy attenuates herding and recursion.",
            "fixed_mixed", "core", differentiated_strategy_on!),

        RobustnessCondition(
            "AI_COMPLEMENTARITY_ON", "alternative_frontier_ai_specification",
            "AI accuracy depends on agent traits and sector knowledge.",
            "Operationalizes AGI as a complement to entrepreneurial judgment rather than a substitute.",
            "fixed_mixed", "core", ai_complementarity_on!),

        RobustnessCondition(
            "TOKEN_PRICING_ONLY", "alternative_frontier_ai_specification",
            "Use token-equivalent task costs only instead of subscription plus usage costs.",
            "Tests whether a pure compute-pricing interpretation changes the frontier result.",
            "fixed_mixed", "core", token_pricing_only!),

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
            "EMERGENT_ADOPTION_BASELINE", "boundary_conditions_generalizability",
            "Agents start without AI and choose tiers endogenously.",
            "Generalizability check against fixed equal-tier assignment.",
            "emergent", "minimal", noop!),
    ]
end

function selected_conditions()
    conditions = all_conditions()
    if !isempty(CONDITION_FILTER)
        requested = Set(strip.(split(CONDITION_FILTER, ",")))
        conditions = [c for c in conditions if c.name in requested]
        missing = setdiff(requested, Set(c.name for c in conditions))
        if !isempty(missing)
            error("Unknown CONDITIONS entries: $(join(collect(missing), ", "))")
        end
        return conditions
    end

    if SUITE_PRESET == "minimal"
        return [c for c in conditions if c.preset == "minimal"]
    elseif SUITE_PRESET == "core"
        return [c for c in conditions if c.preset in ("minimal", "core")]
    elseif SUITE_PRESET == "expanded"
        return conditions
    else
        error("Unknown SUITE_PRESET=$SUITE_PRESET. Use minimal, core, or expanded.")
    end
end

function print_manifest(conditions::Vector{RobustnessCondition})
    println("Robustness suite manifest")
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

function signed_effect_size(delta::Float64, band::Float64)::Float64
    return delta / max(band, eps(Float64))
end

function horizon_pressure(stats::AbstractDict)::Float64
    return mean([
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

function perceived_dimension_level(stats::AbstractDict, dim::String)::Float64
    return Float64(get(stats, "perceived_$(dim)_bounded_level", 0.0))
end

function perceived_dimension_gap_vs_prior(stats::AbstractDict, dim::String)::Float64
    return Float64(get(stats, "perceived_$(dim)_gap_vs_prior", 0.0))
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

function summarize_rows(per_run_df::DataFrame)
    rows = NamedTuple[]
    for group in groupby(per_run_df, [:condition, :category, :description, :theoretical_role, :design, :tier, :tier_rank])
        condition = first(group.condition)
        tier = first(group.tier)
        condition_rows = per_run_df[per_run_df.condition .== condition, :]
        none_rows = condition_rows[condition_rows.tier .== "none", :]
        none_survival = nrow(none_rows) == 0 ? NaN : mean(none_rows.survival_rate)
        survival_mean = mean(group.survival_rate)
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
            survival_sd = length(group.survival_rate) > 1 ? std(group.survival_rate) : 0.0,
            te_vs_none_pp = isnan(none_survival) ? NaN : (survival_mean - none_survival) * 100,
            mean_final_capital = mean(group.mean_final_capital),
            roi_mean = mean(group.roi),
            mean_competition = mean(group.mean_competition),
            max_competition = mean(group.max_competition),
            innovations_per_agent = mean(group.innovations_per_agent),
            invest_share = mean(group.invest_share),
            innovate_share = mean(group.innovate_share),
            explore_share = mean(group.explore_share),
            maintain_share = mean(group.maintain_share),
        ))
    end
    return DataFrame(rows)
end

function baseline_permutation_placebo(per_run_df::DataFrame; n_iter::Int=1000)
    baseline = per_run_df[per_run_df.condition .== "BASELINE", :]
    none = baseline[baseline.tier .== "none", :survival_rate]
    premium = baseline[baseline.tier .== "premium", :survival_rate]
    if length(none) < 2 || length(premium) < 2
        return nothing
    end
    actual = mean(premium) - mean(none)
    pool = vcat(collect(none), collect(premium))
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
            for tier in ("basic", "advanced", "premium")
                tier_rows = run_df[run_df.tier .== tier, :]
                nrow(tier_rows) == 0 && continue
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
        tier_stats["mean_survival_rate"] = Float64(get(tier_stats, "survival_rate", 0.0))
        tier_stats["mean_niches_per_agent"] = Float64(get(tier_stats, "niches_per_agent", 0.0))
        tier_stats["mean_combinations_per_agent"] = Float64(get(tier_stats, "combinations_per_agent", 0.0))
        stats[tier] = tier_stats
    end

    for tier in AI_TIERS
        if !haskey(stats, tier)
            stats[tier] = Dict{String,Any}(
                "n_runs" => 0,
                "mean_survival_rate" => 0.0,
                "mean_niches_per_agent" => 0.0,
                "mean_combinations_per_agent" => 0.0,
            )
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

function paradox_scorecard_rows(per_run_df::DataFrame)
    rows = Dict{Symbol,Any}[]
    for condition_df in groupby(per_run_df, [:condition, :category, :description, :theoretical_role, :design])
        summary = condition_summary_for_scorecard(DataFrame(condition_df))
        scorecard = build_paradox_mechanism_scorecard(summary)
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
                row[Symbol(key)] = get(s, key, 0.0)
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

function run_one_condition(condition::RobustnessCondition)
    cond_rows_per_run = Vector{Any}(undef, N_RUNS)
    Threads.@threads for run_idx in 1:N_RUNS
        cond_rows_per_run[run_idx] = run_one(condition, run_idx)
    end
    return reduce(vcat, cond_rows_per_run)
end

function main()
    conditions = selected_conditions()
    if "--help" in ARGS || "--list" in ARGS || get(ENV, "DRY_RUN", "0") == "1"
        print_manifest(conditions)
        return
    end

    print_manifest(conditions)
    mkpath(OUTPUT_DIR)
    write_run_provenance!(
        OUTPUT_DIR;
        script_name=basename(@__FILE__),
        parameters=Dict(
            "BASE_SEED" => BASE_SEED,
            "CONDITIONS" => CONDITION_FILTER == "" ? "selected_by_preset" : CONDITION_FILTER,
            "N_AGENTS" => N_AGENTS,
            "N_ROUNDS" => N_ROUNDS,
            "N_RUNS" => N_RUNS,
            "OUTPUT_DIR" => OUTPUT_DIR,
            "SUITE_PRESET" => SUITE_PRESET,
            "SUITE_SEED_MODE" => SUITE_SEED_MODE,
        ),
        notes=Dict(
            "suite" => "robustness and sensitivity suite",
        ),
    )
    per_cond_dir = joinpath(OUTPUT_DIR, "per_condition")
    mkpath(per_cond_dir)
    CSV.write(joinpath(OUTPUT_DIR, "robustness_suite_manifest.csv"),
        DataFrame([(name=c.name, category=c.category, description=c.description,
                    theoretical_role=c.theoretical_role, design=c.design,
                    preset=c.preset, seed_mode=SUITE_SEED_MODE) for c in conditions]))

    all_rows = Dict{Symbol,Any}[]
    n_conditions = length(conditions)
    t0 = time()

    for (ci, condition) in enumerate(conditions)
        cond_t0 = time()
        @printf("[%d/%d] Starting %s (%s, %s)\n",
            ci, n_conditions, condition.name, condition.category, condition.design)
        flush(stdout)

        cond_rows = run_one_condition(condition)
        append!(all_rows, cond_rows)

        CSV.write(joinpath(per_cond_dir, "$(condition.name).csv"), DataFrame(cond_rows))

        cond_elapsed = time() - cond_t0
        total_elapsed = time() - t0
        @printf("[%d/%d] Completed %s in %.1fs (cumulative %.1fs, %d/%d sim-rows)\n",
            ci, n_conditions, condition.name, cond_elapsed, total_elapsed,
            length(all_rows), n_conditions * N_RUNS * length(AI_TIERS))
        flush(stdout)

        # Reclaim memory between conditions to avoid GC death spiral on long runs.
        GC.gc()
    end

    per_run_df = DataFrame(all_rows)
    summary_df = summarize_rows(per_run_df)
    paired_df = paired_treatment_effects(per_run_df)
    delta_df = condition_delta_vs_baseline(per_run_df)
    scorecard_df = paradox_scorecard_rows(per_run_df)
    scorecard_delta_df = paradox_scorecard_delta_vs_baseline(scorecard_df)

    CSV.write(joinpath(OUTPUT_DIR, "robustness_suite_per_run.csv"), per_run_df)
    CSV.write(joinpath(OUTPUT_DIR, "robustness_suite_summary.csv"), summary_df)
    CSV.write(joinpath(OUTPUT_DIR, "robustness_suite_paired_treatment_effects.csv"), paired_df)
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

main()
