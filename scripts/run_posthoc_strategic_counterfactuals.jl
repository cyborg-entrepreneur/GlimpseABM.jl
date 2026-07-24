#!/usr/bin/env julia
"""
Run exploratory post hoc strategic counterfactual probes.

Outputs are written under `glimpse_abm/julia/results/posthoc_strategic_*` and
are not consumed by the primary exhibit pipeline.
"""

# Activate the project only for standalone execution. The package test harness
# include()s this script inside a module, and its isolated test environment does
# not expose Pkg unless the project declares it as a test dependency.
if abspath(PROGRAM_FILE) == @__FILE__
    import Pkg
    Pkg.activate(dirname(@__DIR__))
end

using GlimpseABM
using Random
using Statistics
using DataFrames
using CSV
using Dates
using Printf

include(joinpath(@__DIR__, "_fixed_tier_assignment.jl"))
include(joinpath(@__DIR__, "_launch_metadata.jl"))

const N_AGENTS = parse(Int, get(ENV, "N_AGENTS", "1000"))
const N_ROUNDS = parse(Int, get(ENV, "N_ROUNDS", "60"))
const N_RUNS = parse(Int, get(ENV, "N_RUNS", "25"))
const BASE_SEED = parse(Int, get(ENV, "BASE_SEED", "20260425"))
const NICHE_SIGMA = parse(Float64, get(ENV, "NICHE_SIGMA", "1.5"))
const NICHE_OPS = parse(Float64, get(ENV, "NICHE_OPS", "0.85"))
const DERIVED_GATE = parse(Float64, get(ENV, "DERIVED_GATE", "0.55"))
const OUT_DIR = get(ENV, "OUT_DIR",
    joinpath(dirname(@__DIR__), "results",
             "posthoc_strategic_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"))

struct PosthocCondition
    name::String
    family::String
    description::String
    apply!::Function
end

noop!(config::EmergentConfig) = nothing

function sequential_signal_baseline!(config::EmergentConfig)
    config.SEQUENTIAL_DECISIONS_ENABLED = true
    return nothing
end

function weak_targeting!(strength::Float64; max_multiplier::Float64 = 1.75,
                         weak_tiers::Vector{String} = ["none", "basic"])
    return function (config::EmergentConfig)
        config.POSTHOC_WEAK_TARGETING_ENABLED = true
        config.POSTHOC_WEAK_TARGETING_STRENGTH = strength
        config.POSTHOC_WEAK_TARGETING_MAX_MULTIPLIER = max_multiplier
        config.POSTHOC_WEAK_TARGETING_TIERS = ["premium"]
        config.POSTHOC_WEAK_TARGETING_WEAK_TIERS = weak_tiers
        return nothing
    end
end

function manipulation!(mode::String; signal_strength::Int = 2)
    return function (config::EmergentConfig)
        config.POSTHOC_MANIPULATION_ENABLED = true
        config.POSTHOC_MANIPULATION_MODE = mode
        config.POSTHOC_MANIPULATION_SIGNAL_STRENGTH = signal_strength
        config.POSTHOC_MANIPULATION_TIERS = ["premium"]
        config.POSTHOC_MANIPULATION_DECOY_TOP_K = 5
        config.SEQUENTIAL_DECISIONS_ENABLED = true
        return nothing
    end
end

function weak_agent_targeting!(; min_exposure::Float64 = 0.0,
                               weak_tiers::Vector{String} = ["none", "basic"])
    return function (config::EmergentConfig)
        config.POSTHOC_WEAK_AGENT_TARGETING_ENABLED = true
        config.POSTHOC_WEAK_AGENT_TARGETING_TIERS = ["premium"]
        config.POSTHOC_WEAK_AGENT_TARGETING_WEAK_TIERS = weak_tiers
        config.POSTHOC_WEAK_AGENT_TARGETING_MIN_EXPOSURE = min_exposure
        return nothing
    end
end

function combined_weak_decoy!(config::EmergentConfig)
    weak_targeting!(1.25; max_multiplier=2.0)(config)
    manipulation!("decoy"; signal_strength=2)(config)
    return nothing
end

function all_conditions()
    return PosthocCondition[
        PosthocCondition(
            "POSTHOC_BASELINE", "baseline",
            "Primary fixed-tier baseline under the post hoc runner.", noop!),
        PosthocCondition(
            "SEQUENTIAL_SIGNAL_BASELINE", "sequential_control",
            "Sequential early/late decision protocol with ordinary public signals, but no post hoc amplification or decoy signal.",
            sequential_signal_baseline!),
        PosthocCondition(
            "WEAK_TARGETING_LOW", "weak_targeting",
            "Frontier ranking gets a low bonus for opportunities with active no-AI/basic capital.",
            weak_targeting!(0.75; max_multiplier=1.50)),
        PosthocCondition(
            "WEAK_TARGETING_HIGH", "weak_targeting",
            "Frontier ranking gets a stronger bonus for opportunities with active no-AI/basic capital.",
            weak_targeting!(1.50; max_multiplier=2.25)),
        PosthocCondition(
            "STRONG_TARGETING_PLACEBO", "placebo",
            "Frontier ranking bonus targets advanced/frontier-occupied opportunities instead of weak-occupied ones.",
            weak_targeting!(1.50; max_multiplier=2.25, weak_tiers=["advanced", "premium"])),
        PosthocCondition(
            "WEAK_AGENT_TARGETING_ENDOGENOUS", "weak_agent_targeting",
            "Frontier agents restrict investment search to opportunities currently occupied by no-AI/basic capital.",
            weak_agent_targeting!()),
        PosthocCondition(
            "MANIPULATION_AMPLIFY", "manipulation",
            "Frontier early investors amplify the public signal attached to their actual investment.",
            manipulation!("amplify"; signal_strength=2)),
        PosthocCondition(
            "MANIPULATION_DECOY", "manipulation",
            "Frontier early investors add public signal to a visible decoy opportunity.",
            manipulation!("decoy"; signal_strength=2)),
        PosthocCondition(
            "WEAK_TARGETING_PLUS_DECOY", "combined",
            "Frontier weak targeting combined with decoy signal manipulation.",
            combined_weak_decoy!),
    ]
end

function reference_condition_name(condition::String, family::String)::String
    return family in ("manipulation", "combined") ?
        "SEQUENTIAL_SIGNAL_BASELINE" : "POSTHOC_BASELINE"
end

function selected_conditions()
    requested = strip(get(ENV, "CONDITIONS", ""))
    conditions = all_conditions()
    isempty(requested) && return conditions
    wanted = Set(strip.(split(requested, ",")))
    # A treatment-only filter must still carry the matched protocol control;
    # otherwise a completed run cannot produce a valid paired estimand.
    for condition in conditions
        if condition.name in wanted && condition.name != "POSTHOC_BASELINE"
            reference_name = reference_condition_name(condition.name, condition.family)
            push!(wanted, reference_name)
            reference_name == "SEQUENTIAL_SIGNAL_BASELINE" &&
                push!(wanted, "POSTHOC_BASELINE")
        end
    end
    picked = [c for c in conditions if c.name in wanted]
    missing = setdiff(wanted, Set(c.name for c in picked))
    isempty(missing) || error("Unknown CONDITIONS: $(join(sort!(collect(missing)), ", "))")
    return picked
end

function print_manifest(conditions::Vector{PosthocCondition})
    println("Post hoc strategic counterfactual manifest")
    println("N_AGENTS=$N_AGENTS  N_ROUNDS=$N_ROUNDS  N_RUNS=$N_RUNS  BASE_SEED=$BASE_SEED")
    println()
    for (idx, condition) in enumerate(conditions)
        reference = reference_condition_name(condition.name, condition.family)
        @printf("%2d. %-32s %-20s reference=%s\n",
                idx, condition.name, condition.family, reference)
        println("    $(condition.description)")
    end
end

"""Build every selected config and verify that every paired estimand has its control."""
function preflight_conditions(conditions::Vector{PosthocCondition})::Nothing
    N_AGENTS > 0 || error("N_AGENTS must be positive; got $N_AGENTS.")
    N_ROUNDS > 0 || error("N_ROUNDS must be positive; got $N_ROUNDS.")
    N_RUNS > 0 || error("N_RUNS must be positive; got $N_RUNS.")
    selected_names = Set(condition.name for condition in conditions)
    for condition in conditions
        build_config(condition, BASE_SEED)
        condition.name == "POSTHOC_BASELINE" && continue
        reference = reference_condition_name(condition.name, condition.family)
        reference in selected_names || error(
            "Selected condition $(condition.name) is missing matched reference $(reference).")
    end
    return nothing
end

function build_config(condition::PosthocCondition, seed::Int)
    config = EmergentConfig(
        N_AGENTS=N_AGENTS,
        N_ROUNDS=N_ROUNDS,
        RANDOM_SEED=seed,
        AGENT_AI_MODE="fixed",
    )
    config.NICHE_SIZE_LOG_SIGMA = NICHE_SIGMA
    config.OPS_COST_INTENSITY = NICHE_OPS
    config.DERIVED_KNOWLEDGE_QUALITY_GATE = DERIVED_GATE
    config.POSTHOC_WEAK_TARGETING_ENABLED = true
    config.POSTHOC_WEAK_TARGETING_STRENGTH = 0.0
    config.POSTHOC_WEAK_TARGETING_TIERS = ["premium"]
    condition.apply!(config)
    initialize!(config)
    return config
end

function build_simulation(condition::PosthocCondition, run_idx::Int)
    seed = BASE_SEED + run_idx
    rng = MersenneTwister(seed)
    config = build_config(condition, seed)
    sim = EmergentSimulation(
        config=config,
        initial_tier_distribution=Dict(t => 0.25 for t in AI_TIERS),
        seed=seed,
    )
    apply_balanced_fixed_tiers!(sim, balanced_tier_assignments(N_AGENTS, rng))
    return sim, seed
end

safe_mean(v) = isempty(v) ? NaN : mean(v)
sample_sd(v) = length(v) > 1 ? std(v) : NaN
ci95_halfwidth(v) = length(v) > 1 ? 1.96 * std(v) / sqrt(length(v)) : NaN

function finite_mean(values)::Float64
    vals = Float64[]
    for value in values
        if value isa Real && isfinite(Float64(value))
            push!(vals, Float64(value))
        end
    end
    return isempty(vals) ? NaN : mean(vals)
end

function history_mean(sim::EmergentSimulation, key::String)::Float64
    return finite_mean(get(h, key, NaN) for h in sim.history)
end

function investment_event_rows(condition::PosthocCondition, sim::EmergentSimulation,
                               run_idx::Int, seed::Int, round::Int)
    rows = NamedTuple[]
    for agent in sim.agents
        lo = agent.last_outcome
        lo isa Dict || continue
        get(lo, "round", -1) == round || continue
        get(lo, "action", "") == "invest" || continue
        push!(rows, (
            condition=condition.name,
            family=condition.family,
            run_idx=run_idx,
            seed=seed,
            round=round,
            agent_id=agent.id,
            tier=String(get(lo, "ai_behavior_level", get_ai_level(agent))),
            opportunity_id=String(get(lo, "opportunity_id", "")),
            amount=Float64(get(lo, "amount", 0.0)),
            posthoc_weak_exposure=Float64(get(lo, "posthoc_weak_exposure", 0.0)),
            posthoc_weak_targeting_multiplier=
                Float64(get(lo, "posthoc_weak_targeting_multiplier", 1.0)),
            posthoc_weak_agent_targeting_applied=
                Bool(get(lo, "posthoc_weak_agent_targeting_applied", false)),
            posthoc_weak_agent_target_pool_size=
                Int(get(lo, "posthoc_weak_agent_target_pool_size", 0)),
            posthoc_weak_agent_visible_pool_size=
                Int(get(lo, "posthoc_weak_agent_visible_pool_size", 0)),
            posthoc_weak_agent_best_visible_latent=
                Float64(get(lo, "posthoc_weak_agent_best_visible_latent", NaN)),
            posthoc_weak_agent_chosen_latent=
                Float64(get(lo, "posthoc_weak_agent_chosen_latent", NaN)),
            posthoc_weak_agent_foregone_latent_gap=
                Float64(get(lo, "posthoc_weak_agent_foregone_latent_gap", NaN)),
            posthoc_manipulation_signals_added=
                Float64(get(lo, "posthoc_manipulation_signals_added", 0.0)),
            posthoc_manipulation_signal_target=
                String(get(lo, "posthoc_manipulation_signal_target", "")),
            sequential_decision_role=
                String(get(lo, "sequential_decision_role", "simultaneous")),
            sequential_signals_available=
                Int(get(lo, "sequential_signals_available", 0)),
            sequential_signal_count_on_choice=
                Int(get(lo, "sequential_signal_count_on_choice", 0)),
            posthoc_manipulation_signals_available=
                Int(get(lo, "posthoc_manipulation_signals_available", 0)),
            posthoc_manipulation_signal_count_on_choice=
                Int(get(lo, "posthoc_manipulation_signal_count_on_choice", 0)),
        ))
    end
    return rows
end

function paradox_tier_rows(condition::PosthocCondition, sim::EmergentSimulation,
                           run_idx::Int, seed::Int)
    rows = NamedTuple[]
    all_emergent = aggregate_emergent_uncertainty_by_tier(
        sim.agents; include_dead=true)
    survivor_emergent = aggregate_emergent_uncertainty_by_tier(
        sim.agents; include_dead=false)

    for tier in AI_TIERS
        agents = [agent for agent in sim.agents
                  if String(isnothing(agent.fixed_ai_level) ?
                            get_ai_level(agent) : agent.fixed_ai_level) == tier]
        alive = [agent for agent in agents if agent.alive]
        confidence = GlimpseABM.confidence_outcome_stats(agents)
        tier_all = get(all_emergent, tier, Dict{String,Float64}())
        tier_survivor = get(survivor_emergent, tier, Dict{String,Float64}())

        push!(rows, (
            condition=condition.name,
            family=condition.family,
            run_idx=run_idx,
            seed=seed,
            tier=tier,
            n_agents=length(agents),
            survival_rate=isempty(agents) ? NaN : length(alive) / length(agents),
            mean_final_capital=isempty(agents) ? NaN :
                mean(get_capital(agent) for agent in agents),
            mean_visible_opportunities=
                history_mean(sim, "mean_visible_opportunities_$(tier)"),
            mean_info_quality_used=
                history_mean(sim, "mean_info_quality_used_$(tier)"),
            actor_ignorance_level=Float64(get(tier_all, "actor_ignorance", NaN)),
            practical_indeterminism_level=
                Float64(get(tier_all, "practical_indeterminism", NaN)),
            agentic_novelty_level=Float64(get(tier_all, "agentic_novelty", NaN)),
            competitive_recursion_level=
                Float64(get(tier_all, "competitive_recursion", NaN)),
            survivor_actor_ignorance_level=
                Float64(get(tier_survivor, "actor_ignorance", NaN)),
            survivor_practical_indeterminism_level=
                Float64(get(tier_survivor, "practical_indeterminism", NaN)),
            survivor_agentic_novelty_level=
                Float64(get(tier_survivor, "agentic_novelty", NaN)),
            survivor_competitive_recursion_level=
                Float64(get(tier_survivor, "competitive_recursion", NaN)),
            confidence_outcome_observations=
                Float64(get(confidence, "confidence_outcome_observations", 0.0)),
            confidence_outcome_agent_coverage=
                Float64(get(confidence, "confidence_outcome_agent_coverage", 0.0)),
            confidence_outcome_abs_gap_mean=
                Float64(get(confidence, "confidence_outcome_abs_gap_mean", NaN)),
            confidence_outcome_weighted_gap_mean=
                Float64(get(confidence, "confidence_outcome_weighted_gap_mean", NaN)),
            confidence_outcome_raw_gap_mean=
                Float64(get(confidence, "confidence_outcome_raw_gap_mean", NaN)),
            confidence_outcome_investment_gap_mean=
                Float64(get(confidence, "confidence_outcome_investment_gap_mean", NaN)),
            confidence_outcome_ai_gap_mean=
                Float64(get(confidence, "confidence_outcome_ai_gap_mean", NaN)),
            confidence_outcome_non_ai_gap_mean=
                Float64(get(confidence, "confidence_outcome_non_ai_gap_mean", NaN)),
            confidence_outcome_realized_multiple_mean=
                Float64(get(confidence, "confidence_outcome_realized_multiple_mean", NaN)),
            confidence_outcome_realized_multiple_std=
                Float64(get(confidence, "confidence_outcome_realized_multiple_std", NaN)),
            confidence_outcome_realized_multiple_p90=
                Float64(get(confidence, "confidence_outcome_realized_multiple_p90", NaN)),
            confidence_outcome_realized_multiple_p95=
                Float64(get(confidence, "confidence_outcome_realized_multiple_p95", NaN)),
        ))
    end
    return rows
end

function agent_rows(condition::PosthocCondition, sim::EmergentSimulation,
                    run_idx::Int, seed::Int)
    rows = NamedTuple[]
    for agent in sim.agents
        push!(rows, (
            condition=condition.name,
            family=condition.family,
            run_idx=run_idx,
            seed=seed,
            agent_id=agent.id,
            tier=String(isnothing(agent.fixed_ai_level) ? get_ai_level(agent) : agent.fixed_ai_level),
            survived=agent.alive,
            survival_rounds=agent.survival_rounds,
            final_capital=get_capital(agent),
            total_invested=agent.total_invested,
            total_returned=agent.total_returned,
            n_venture_ledger=length(agent.venture_ledger),
        ))
    end
    return rows
end

function run_one(condition::PosthocCondition, run_idx::Int)
    sim, seed = build_simulation(condition, run_idx)
    event_rows = NamedTuple[]
    for round in 1:N_ROUNDS
        step!(sim, round)
        append!(event_rows, investment_event_rows(condition, sim, run_idx, seed, round))
    end
    return agent_rows(condition, sim, run_idx, seed),
        event_rows,
        paradox_tier_rows(condition, sim, run_idx, seed)
end

function summarize(agent_df::DataFrame, event_df::DataFrame)
    rows = NamedTuple[]
    event_groups = isempty(event_df) ? Dict{Tuple{String,Int,String},DataFrame}() :
        Dict((String(first(g.condition)), Int(first(g.run_idx)), String(first(g.tier))) => DataFrame(g)
             for g in groupby(event_df, [:condition, :run_idx, :tier]))

    for g in groupby(agent_df, [:condition, :family, :run_idx, :tier])
        key = (String(first(g.condition)), Int(first(g.run_idx)), String(first(g.tier)))
        events = get(event_groups, key, DataFrame())
        targeted_events = isempty(events) ? DataFrame() :
            events[Bool.(events.posthoc_weak_agent_targeting_applied), :]
        late_events = isempty(events) ? DataFrame() :
            events[String.(events.sequential_decision_role) .== "late", :]
        push!(rows, (
            condition=String(first(g.condition)),
            family=String(first(g.family)),
            run_idx=Int(first(g.run_idx)),
            seed=Int(first(g.seed)),
            tier=String(first(g.tier)),
            n_agents=nrow(g),
            survival_rate=mean(Bool.(g.survived)),
            mean_final_capital=mean(Float64.(g.final_capital)),
            n_invest_events=nrow(events),
            mean_weak_exposure=isempty(events) ? NaN :
                mean(Float64.(events.posthoc_weak_exposure)),
            mean_targeting_multiplier=isempty(events) ? NaN :
                mean(Float64.(events.posthoc_weak_targeting_multiplier)),
            weak_agent_targeting_rate=isempty(events) ? NaN :
                mean(Bool.(events.posthoc_weak_agent_targeting_applied)),
            mean_weak_agent_target_pool_size=isempty(targeted_events) ? NaN :
                mean(Float64.(targeted_events.posthoc_weak_agent_target_pool_size)),
            mean_weak_agent_visible_pool_size=isempty(targeted_events) ? NaN :
                mean(Float64.(targeted_events.posthoc_weak_agent_visible_pool_size)),
            mean_weak_agent_foregone_latent_gap=isempty(targeted_events) ? NaN :
                mean(Float64.(targeted_events.posthoc_weak_agent_foregone_latent_gap)),
            manipulation_signals_added=isempty(events) ? 0.0 :
                sum(Float64.(events.posthoc_manipulation_signals_added)),
            late_invest_events=nrow(late_events),
            mean_sequential_signals_available=isempty(late_events) ? NaN :
                mean(Float64.(late_events.sequential_signals_available)),
            mean_manipulation_signals_available=isempty(late_events) ? NaN :
                mean(Float64.(late_events.posthoc_manipulation_signals_available)),
            mean_manipulation_signal_count_on_choice=isempty(late_events) ? NaN :
                mean(Float64.(late_events.posthoc_manipulation_signal_count_on_choice)),
            manipulation_signal_choice_exposure_rate=isempty(late_events) ? NaN :
                mean(Float64.(late_events.posthoc_manipulation_signal_count_on_choice) .> 0.0),
        ))
    end
    return DataFrame(rows)
end

function paired_effects(per_run::DataFrame)
    rows = NamedTuple[]
    for g in groupby(per_run[per_run.condition .!= "POSTHOC_BASELINE", :],
                     [:condition, :family, :tier])
        condition = String(first(g.condition))
        family = String(first(g.family))
        tier = String(first(g.tier))
        reference_name = reference_condition_name(condition, family)
        reference = per_run[
            (per_run.condition .== reference_name) .& (per_run.tier .== tier),
            [:run_idx, :seed, :survival_rate, :mean_final_capital],
        ]
        isempty(reference) && error(
            "Missing matched reference $(reference_name) for $(condition), tier=$(tier).")
        merged = innerjoin(
            g[:, [:run_idx, :seed, :survival_rate, :mean_final_capital,
                  :mean_weak_exposure, :mean_targeting_multiplier,
                  :weak_agent_targeting_rate, :mean_weak_agent_target_pool_size,
                  :mean_weak_agent_visible_pool_size,
                  :mean_weak_agent_foregone_latent_gap,
                  :manipulation_signals_added, :late_invest_events,
                  :mean_sequential_signals_available,
                  :mean_manipulation_signals_available,
                  :mean_manipulation_signal_count_on_choice,
                  :manipulation_signal_choice_exposure_rate]],
            rename(
                reference,
                :survival_rate => :reference_survival_rate,
                :mean_final_capital => :reference_mean_final_capital,
            ),
            on=[:run_idx, :seed],
        )
        nrow(merged) == nrow(g) || error(
            "Incomplete run pairing for $(condition) versus $(reference_name), " *
            "tier=$(tier): treatment rows=$(nrow(g)), paired rows=$(nrow(merged)).")
        survival_delta = merged.survival_rate .- merged.reference_survival_rate
        capital_delta = merged.mean_final_capital .-
            merged.reference_mean_final_capital
        survival_ci_halfwidth_pp = 100.0 * ci95_halfwidth(survival_delta)
        capital_ci_halfwidth = ci95_halfwidth(capital_delta)
        push!(rows, (
            condition=condition,
            family=family,
            reference_condition=reference_name,
            tier=tier,
            n_runs=nrow(merged),
            survival_delta_pp=100.0 * mean(survival_delta),
            survival_delta_sd_pp=100.0 * sample_sd(survival_delta),
            survival_delta_ci95_low_pp=
                100.0 * mean(survival_delta) - survival_ci_halfwidth_pp,
            survival_delta_ci95_high_pp=
                100.0 * mean(survival_delta) + survival_ci_halfwidth_pp,
            # Retain the treatment mean for backward-compatible readers, while
            # also exporting the correctly paired capital estimand.
            mean_final_capital=mean(merged.mean_final_capital),
            reference_mean_final_capital=
                mean(merged.reference_mean_final_capital),
            mean_final_capital_delta=mean(capital_delta),
            mean_final_capital_delta_sd=sample_sd(capital_delta),
            mean_final_capital_delta_ci95_low=
                mean(capital_delta) - capital_ci_halfwidth,
            mean_final_capital_delta_ci95_high=
                mean(capital_delta) + capital_ci_halfwidth,
            mean_weak_exposure=mean(skipmissing(merged.mean_weak_exposure)),
            mean_targeting_multiplier=mean(skipmissing(merged.mean_targeting_multiplier)),
            weak_agent_targeting_rate=mean(skipmissing(merged.weak_agent_targeting_rate)),
            mean_weak_agent_target_pool_size=
                mean(skipmissing(merged.mean_weak_agent_target_pool_size)),
            mean_weak_agent_visible_pool_size=
                mean(skipmissing(merged.mean_weak_agent_visible_pool_size)),
            mean_weak_agent_foregone_latent_gap=
                mean(skipmissing(merged.mean_weak_agent_foregone_latent_gap)),
            manipulation_signals_added=mean(merged.manipulation_signals_added),
            late_invest_events=mean(merged.late_invest_events),
            mean_sequential_signals_available=
                mean(skipmissing(merged.mean_sequential_signals_available)),
            mean_manipulation_signals_available=
                mean(skipmissing(merged.mean_manipulation_signals_available)),
            mean_manipulation_signal_count_on_choice=
                mean(skipmissing(merged.mean_manipulation_signal_count_on_choice)),
            manipulation_signal_choice_exposure_rate=
                mean(skipmissing(merged.manipulation_signal_choice_exposure_rate)),
        ))
    end
    return DataFrame(rows)
end

function paradox_premium_none_deltas(paradox_df::DataFrame)
    rows = NamedTuple[]
    for g in groupby(paradox_df, [:condition, :family, :run_idx])
        none = g[g.tier .== "none", :]
        premium = g[g.tier .== "premium", :]
        (isempty(none) || isempty(premium)) && continue
        n = none[1, :]
        p = premium[1, :]
        push!(rows, (
            condition=String(p.condition),
            family=String(p.family),
            run_idx=Int(p.run_idx),
            seed=Int(p.seed),
            premium_survival_rate=Float64(p.survival_rate),
            none_survival_rate=Float64(n.survival_rate),
            premium_minus_none_survival_pp=
                100.0 * (Float64(p.survival_rate) - Float64(n.survival_rate)),
            premium_minus_none_visible_opportunities=
                Float64(p.mean_visible_opportunities) - Float64(n.mean_visible_opportunities),
            premium_minus_none_info_quality=
                Float64(p.mean_info_quality_used) - Float64(n.mean_info_quality_used),
            premium_minus_none_actor_ignorance=
                Float64(p.actor_ignorance_level) - Float64(n.actor_ignorance_level),
            premium_minus_none_practical_indeterminism=
                Float64(p.practical_indeterminism_level) -
                Float64(n.practical_indeterminism_level),
            premium_minus_none_agentic_novelty=
                Float64(p.agentic_novelty_level) - Float64(n.agentic_novelty_level),
            premium_minus_none_competitive_recursion=
                Float64(p.competitive_recursion_level) -
                Float64(n.competitive_recursion_level),
            premium_minus_none_abs_confidence_gap=
                Float64(p.confidence_outcome_abs_gap_mean) -
                Float64(n.confidence_outcome_abs_gap_mean),
            premium_minus_none_realized_multiple_mean=
                Float64(p.confidence_outcome_realized_multiple_mean) -
                Float64(n.confidence_outcome_realized_multiple_mean),
            premium_minus_none_realized_multiple_std=
                Float64(p.confidence_outcome_realized_multiple_std) -
                Float64(n.confidence_outcome_realized_multiple_std),
            premium_minus_none_realized_multiple_p95=
                Float64(p.confidence_outcome_realized_multiple_p95) -
                Float64(n.confidence_outcome_realized_multiple_p95),
        ))
    end
    return DataFrame(rows)
end

function summarize_premium_none_deltas(delta_df::DataFrame)
    rows = NamedTuple[]
    for g in groupby(delta_df, [:condition, :family])
        push!(rows, (
            condition=String(first(g.condition)),
            family=String(first(g.family)),
            n_runs=nrow(g),
            survival_gap_pp_mean=mean(Float64.(g.premium_minus_none_survival_pp)),
            survival_gap_pp_sd=std(Float64.(g.premium_minus_none_survival_pp)),
            visible_opportunity_delta_mean=
                mean(Float64.(g.premium_minus_none_visible_opportunities)),
            info_quality_delta_mean=mean(Float64.(g.premium_minus_none_info_quality)),
            actor_ignorance_delta_mean=
                mean(Float64.(g.premium_minus_none_actor_ignorance)),
            practical_indeterminism_delta_mean=
                mean(Float64.(g.premium_minus_none_practical_indeterminism)),
            agentic_novelty_delta_mean=
                mean(Float64.(g.premium_minus_none_agentic_novelty)),
            competitive_recursion_delta_mean=
                mean(Float64.(g.premium_minus_none_competitive_recursion)),
            abs_confidence_gap_delta_mean=
                mean(Float64.(g.premium_minus_none_abs_confidence_gap)),
            realized_multiple_mean_delta_mean=
                mean(Float64.(g.premium_minus_none_realized_multiple_mean)),
            realized_multiple_std_delta_mean=
                mean(Float64.(g.premium_minus_none_realized_multiple_std)),
            realized_multiple_p95_delta_mean=
                mean(Float64.(g.premium_minus_none_realized_multiple_p95)),
        ))
    end
    return DataFrame(rows)
end

function condition_manifest_rows(conditions::Vector{PosthocCondition})::DataFrame
    rows = NamedTuple[]
    for condition in conditions
        cfg = build_config(condition, BASE_SEED)
        push!(rows, (
            condition=condition.name,
            family=condition.family,
            reference_condition=
                reference_condition_name(condition.name, condition.family),
            description=condition.description,
            sequential_decisions_enabled=cfg.SEQUENTIAL_DECISIONS_ENABLED,
            early_decision_fraction=cfg.EARLY_DECISION_FRACTION,
            signal_visibility_weight=cfg.SIGNAL_VISIBILITY_WEIGHT,
            weak_targeting_enabled=cfg.POSTHOC_WEAK_TARGETING_ENABLED,
            weak_targeting_strength=cfg.POSTHOC_WEAK_TARGETING_STRENGTH,
            weak_targeting_max_multiplier=
                cfg.POSTHOC_WEAK_TARGETING_MAX_MULTIPLIER,
            weak_targeting_tiers=join(cfg.POSTHOC_WEAK_TARGETING_TIERS, "|"),
            weak_targeting_weak_tiers=
                join(cfg.POSTHOC_WEAK_TARGETING_WEAK_TIERS, "|"),
            weak_agent_targeting_enabled=
                cfg.POSTHOC_WEAK_AGENT_TARGETING_ENABLED,
            weak_agent_targeting_min_exposure=
                cfg.POSTHOC_WEAK_AGENT_TARGETING_MIN_EXPOSURE,
            manipulation_enabled=cfg.POSTHOC_MANIPULATION_ENABLED,
            manipulation_mode=cfg.POSTHOC_MANIPULATION_MODE,
            manipulation_signal_strength=
                cfg.POSTHOC_MANIPULATION_SIGNAL_STRENGTH,
            manipulation_decoy_top_k=cfg.POSTHOC_MANIPULATION_DECOY_TOP_K,
            manipulation_tiers=join(cfg.POSTHOC_MANIPULATION_TIERS, "|"),
        ))
    end
    return DataFrame(rows)
end

function write_manifest(conditions::Vector{PosthocCondition})
    CSV.write(
        joinpath(OUT_DIR, "posthoc_condition_manifest.csv"),
        condition_manifest_rows(conditions),
    )
    open(joinpath(OUT_DIR, "POSTHOC_STRATEGIC_COUNTERFACTUALS_NOTE.md"), "w") do io
        println(io, "# Post Hoc Strategic Counterfactuals")
        println(io)
        println(io, "These simulations are exploratory post hoc probes. They are not read by the primary exhibit pipeline.")
        println(io)
        println(io, "- N_AGENTS=$(N_AGENTS)")
        println(io, "- N_ROUNDS=$(N_ROUNDS)")
        println(io, "- N_RUNS=$(N_RUNS)")
        println(io, "- BASE_SEED=$(BASE_SEED)")
        println(io, "- THREADS=$(Threads.nthreads())")
        println(io, "- NICHE_SIZE_LOG_SIGMA=$(NICHE_SIGMA)")
        println(io, "- OPS_COST_INTENSITY=$(NICHE_OPS)")
        println(io)
        for c in conditions
            println(io, "## $(c.name)")
            println(io)
            println(io, "Matched reference: `$(reference_condition_name(c.name, c.family))`")
            println(io)
            println(io, c.description)
            println(io)
        end
    end
end

function append_task_result!(
    agent_rows::Vector{NamedTuple},
    event_rows::Vector{NamedTuple},
    paradox_rows::Vector{NamedTuple},
    result,
)::Nothing
    agents, events, paradox = result
    append!(agent_rows, agents)
    append!(event_rows, events)
    append!(paradox_rows, paradox)
    return nothing
end

function collect_task_results(results, task_range)
    agent_rows = NamedTuple[]
    event_rows = NamedTuple[]
    paradox_rows = NamedTuple[]
    for idx in task_range
        append_task_result!(agent_rows, event_rows, paradox_rows, results[idx])
    end
    return agent_rows, event_rows, paradox_rows
end

function main()
    conditions = selected_conditions()
    if "--help" in ARGS || "--list" in ARGS || get(ENV, "DRY_RUN", "0") == "1"
        print_manifest(conditions)
        preflight_conditions(conditions)
        println("\nConfig preflight: all $(length(conditions)) post hoc condition configs build cleanly.")
        return nothing
    end

    mkpath(OUT_DIR)
    per_condition_dir = joinpath(OUT_DIR, "per_condition")
    mkpath(per_condition_dir)
    print_manifest(conditions)
    preflight_conditions(conditions)
    write_manifest(conditions)
    write_run_provenance!(
        OUT_DIR;
        script_name=basename(@__FILE__),
        parameters=Dict(
            "BASE_SEED" => BASE_SEED,
            "CONDITIONS" => strip(get(ENV, "CONDITIONS", "")) == "" ?
                "all" : strip(get(ENV, "CONDITIONS", "")),
            "DERIVED_GATE" => DERIVED_GATE,
            "N_AGENTS" => N_AGENTS,
            "NICHE_OPS" => NICHE_OPS,
            "NICHE_SIGMA" => NICHE_SIGMA,
            "N_ROUNDS" => N_ROUNDS,
            "N_RUNS" => N_RUNS,
            "OUT_DIR" => OUT_DIR,
            "RUN_TAG" => get(ENV, "RUN_TAG", ""),
            "SUBMISSION_GIT_COMMIT" =>
                get(ENV, "SUBMISSION_GIT_COMMIT", ""),
        ),
        notes=Dict(
            "design" => "seed-paired fixed-tier strategic post hoc",
            "reference_semantics" =>
                "manipulation/combined use SEQUENTIAL_SIGNAL_BASELINE; all other treatments use POSTHOC_BASELINE",
        ),
    )

    n_conditions = length(conditions)
    tasks = [(ci, condition, run_idx)
             for (ci, condition) in enumerate(conditions)
             for run_idx in 1:N_RUNS]
    n_tasks = length(tasks)
    results = Vector{Any}(undef, n_tasks)
    remaining = [Threads.Atomic{Int}(N_RUNS) for _ in conditions]
    completed_conditions = Threads.Atomic{Int}(0)
    write_lock = ReentrantLock()
    t0 = time()

    @printf("Running %d tasks (%d post hoc conditions x %d runs; N=%d, rounds=%d) on %d threads\n",
            n_tasks, n_conditions, N_RUNS, N_AGENTS, N_ROUNDS,
            Threads.nthreads())
    flush(stdout)
    Threads.@threads :greedy for task_idx in eachindex(tasks)
        ci, condition, run_idx = tasks[task_idx]
        results[task_idx] = run_one(condition, run_idx)
        if Threads.atomic_sub!(remaining[ci], 1) == 1
            first_idx = (ci - 1) * N_RUNS + 1
            last_idx = ci * N_RUNS
            agents, events, paradox =
                collect_task_results(results, first_idx:last_idx)
            lock(write_lock) do
                CSV.write(joinpath(per_condition_dir,
                                   "$(condition.name)_agent_outcomes.csv"),
                          DataFrame(agents))
                CSV.write(joinpath(per_condition_dir,
                                   "$(condition.name)_investment_events.csv"),
                          DataFrame(events))
                CSV.write(joinpath(per_condition_dir,
                                   "$(condition.name)_paradox_tier_summary.csv"),
                          DataFrame(paradox))
            end
            done = Threads.atomic_add!(completed_conditions, 1) + 1
            @printf("[%d/%d] completed %-32s in %.1fs since launch\n",
                    done, n_conditions, condition.name, time() - t0)
            flush(stdout)
        end
    end
    GC.gc()

    all_agent_rows, all_event_rows, all_paradox_rows =
        collect_task_results(results, eachindex(results))

    agent_df = DataFrame(all_agent_rows)
    event_df = DataFrame(all_event_rows)
    paradox_df = DataFrame(all_paradox_rows)

    # Persist the raw production outputs before any aggregation. A late
    # analysis failure must not erase a multi-hour ARC simulation run.
    CSV.write(joinpath(OUT_DIR, "posthoc_agent_outcomes.csv"), agent_df)
    CSV.write(joinpath(OUT_DIR, "posthoc_investment_events.csv"), event_df)
    CSV.write(joinpath(OUT_DIR, "posthoc_paradox_tier_summary.csv"), paradox_df)
    println("[checkpoint] raw post hoc outputs written before aggregation")
    flush(stdout)

    per_run = summarize(agent_df, event_df)
    effects = paired_effects(per_run)
    paradox_deltas = paradox_premium_none_deltas(paradox_df)
    paradox_delta_summary = summarize_premium_none_deltas(paradox_deltas)

    CSV.write(joinpath(OUT_DIR, "posthoc_per_run_summary.csv"), per_run)
    CSV.write(joinpath(OUT_DIR, "posthoc_paired_effects.csv"), effects)
    CSV.write(joinpath(OUT_DIR, "posthoc_premium_none_paradox_deltas.csv"),
              paradox_deltas)
    CSV.write(joinpath(OUT_DIR, "posthoc_premium_none_paradox_delta_summary.csv"),
              paradox_delta_summary)
    println("Wrote post hoc strategic outputs to $(OUT_DIR)")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
