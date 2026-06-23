#!/usr/bin/env julia
# Knightian uncertainty weight-family sensitivity sweep.
#
# This script varies coherent calibration families rather than individual
# constants. By default it scales driver weights while leaving baselines,
# thresholds, references, and final [0, 1] bounds unchanged.

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

function parse_float_list(raw::AbstractString)
    values = Float64[]
    for part in split(raw, ",")
        stripped = strip(part)
        isempty(stripped) && continue
        push!(values, parse(Float64, stripped))
    end
    return values
end

function parse_string_list(raw::AbstractString)
    values = String[]
    for part in split(raw, ",")
        stripped = strip(part)
        isempty(stripped) && continue
        push!(values, stripped)
    end
    return values
end

const N_AGENTS = parse(Int, get(ENV, "N_AGENTS", "1000"))
const N_ROUNDS = parse(Int, get(ENV, "N_ROUNDS", "60"))
const N_RUNS = parse(Int, get(ENV, "N_RUNS", "3"))
const BASE_SEED = parse(Int, get(ENV, "BASE_SEED", "20260425"))
const SWEEP_SEED_MODE = get(ENV, "SWEEP_SEED_MODE", "paired")
const SWEEP_MULTIPLIERS = parse_float_list(get(ENV, "SWEEP_MULTIPLIERS", "0.75,1.25"))
# Provides AI_TIERS, balanced_tier_assignments, apply_balanced_fixed_tiers!
include(joinpath(@__DIR__, "_fixed_tier_assignment.jl"))

const ALL_FAMILIES = [
    "producer_actor_ignorance",
    "producer_practical_indeterminism",
    "producer_agentic_novelty",
    "producer_competitive_recursion",
    "perception_evidence_reading",
    "perception_actor_ignorance",
    "perception_practical_indeterminism",
    "perception_agentic_novelty",
    "perception_competitive_recursion",
]

const SWEEP_FAMILIES = let raw = get(ENV, "SWEEP_FAMILIES", "")
    isempty(strip(raw)) ? ALL_FAMILIES : parse_string_list(raw)
end

const OUTPUT_DIR = get(
    ENV,
    "OUTPUT_DIR",
    joinpath(
        @__DIR__,
        "..",
        "results",
        "knightian_weight_family_sweep_$(Dates.format(now(), "yyyymmdd_HHMMSS"))",
    ),
)

starts_with_any(value::String, prefixes) = any(prefix -> startswith(value, prefix), prefixes)

function is_driver_weight_field(name::Symbol)
    text = String(name)
    has_driver_token = any(
        token -> occursin(token, text),
        ("weight", "reduction", "increase", "drag", "discount", "multiplier", "bump"),
    )
    is_baseline_token =
        text == "base_level" ||
        endswith(text, "_base") ||
        endswith(text, "_min") ||
        endswith(text, "_max") ||
        occursin("threshold", text) ||
        occursin("reference", text) ||
        occursin("baseline", text) ||
        occursin("decay", text)
    return has_driver_token && !is_baseline_token
end

function scale_struct_fields(weights, multiplier::Float64; predicate=is_driver_weight_field)
    pairs = Pair{Symbol,Any}[]
    for name in fieldnames(typeof(weights))
        value = getfield(weights, name)
        if value isa Real && predicate(name)
            push!(pairs, name => Float64(value) * multiplier)
        else
            push!(pairs, name => value)
        end
    end
    return typeof(weights)(; pairs...)
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

function scale_perception_family(
    weights::UncertaintyPerceptionWeights,
    multiplier::Float64,
    family::String,
)
    prefixes = if family == "perception_evidence_reading"
        (
            "opportunity_",
            "observable_",
            "evidence_",
            "ai_trust_",
            "avg_knowledge_",
            "discovery_",
            "truly_unknown_",
        )
    elseif family == "perception_actor_ignorance"
        ("actor_", "analytical_", "opportunity_signal_gap_")
    elseif family == "perception_practical_indeterminism"
        (
            "practical_",
            "system_",
            "timing_",
            "path_risk_",
            "agent_uncertainty_",
            "crowding_",
            "opportunity_overlap_",
            "opportunity_competition_",
            "ai_herding_pressure_",
        )
    elseif family == "perception_agentic_novelty"
        ("novelty_",)
    elseif family == "perception_competitive_recursion"
        ("recursion_", "strategic_opacity_")
    else
        error("Unknown perception family: $family")
    end

    return scale_struct_fields(weights, multiplier; predicate=name -> begin
        text = String(name)
        return starts_with_any(text, prefixes) && is_driver_weight_field(name)
    end)
end

function apply_family_multiplier!(config::EmergentConfig, family::String, multiplier::Float64)
    family == "baseline" && return

    producer = config.KNIGHTIAN_PRODUCER_WEIGHTS
    if family == "producer_actor_ignorance"
        config.KNIGHTIAN_PRODUCER_WEIGHTS = replace_producer_weights(
            producer;
            actor=scale_struct_fields(producer.actor_ignorance, multiplier),
        )
    elseif family == "producer_practical_indeterminism"
        config.KNIGHTIAN_PRODUCER_WEIGHTS = replace_producer_weights(
            producer;
            practical=scale_struct_fields(producer.practical_indeterminism, multiplier),
        )
    elseif family == "producer_agentic_novelty"
        config.KNIGHTIAN_PRODUCER_WEIGHTS = replace_producer_weights(
            producer;
            novelty=scale_struct_fields(producer.agentic_novelty, multiplier),
        )
    elseif family == "producer_competitive_recursion"
        config.KNIGHTIAN_PRODUCER_WEIGHTS = replace_producer_weights(
            producer;
            recursion=scale_struct_fields(producer.competitive_recursion, multiplier),
        )
    elseif startswith(family, "perception_")
        config.UNCERTAINTY_PERCEPTION_WEIGHTS = scale_perception_family(
            config.UNCERTAINTY_PERCEPTION_WEIGHTS,
            multiplier,
            family,
        )
    else
        error("Unknown sweep family: $family")
    end
end

function build_mixed_sim(seed::Int, family::String, multiplier::Float64)
    rng = Random.MersenneTwister(seed)
    config = EmergentConfig(
        N_AGENTS=N_AGENTS,
        N_ROUNDS=N_ROUNDS,
        RANDOM_SEED=seed,
        AGENT_AI_MODE="fixed",
        enable_round_logging=false,
    )
    apply_family_multiplier!(config, family, multiplier)

    initial_dist = Dict(tier => 0.25 for tier in AI_TIERS)
    sim = GlimpseABM.EmergentSimulation(
        config=config,
        initial_tier_distribution=initial_dist,
        seed=seed,
    )

    # rng is consumed only by the assignment shuffle, so drawing assignments
    # after construction is equivalent to drawing them before (same stream).
    apply_balanced_fixed_tiers!(sim, balanced_tier_assignments(N_AGENTS, rng))

    return sim
end

function numeric_history_value(row::Dict{String,Any}, key::String)
    value = get(row, key, 0.0)
    return value isa Real ? Float64(value) : 0.0
end

function tail_mean(history::Vector{Dict{String,Any}}, key::String; window::Int=10)
    isempty(history) && return NaN
    start_idx = max(1, length(history) - window + 1)
    # Per-tier round telemetry emits NaN for empty cells (no agents of that
    # tier in a round); keep only finite values so empty cells drop out of the
    # tail mean instead of deflating it. No finite values at all -> NaN.
    values = [
        v for v in (numeric_history_value(row, key) for row in history[start_idx:end])
        if isfinite(v)
    ]
    return isempty(values) ? NaN : mean(values)
end

function run_one(cell_idx::Int, run_idx::Int, family::String, multiplier::Float64, seed::Int)
    sim = build_mixed_sim(seed, family, multiplier)
    for round in 1:N_ROUNDS
        GlimpseABM.step!(sim, round)
    end

    result = Dict{String,Any}(
        "cell_idx" => cell_idx,
        "run_idx" => run_idx,
        "seed" => seed,
        "family" => family,
        "multiplier" => multiplier,
    )

    tier_survival = Dict{String,Float64}()
    for tier in AI_TIERS
        tier_agents = filter(agent -> agent.fixed_ai_level == tier, sim.agents)
        alive_count = count(agent -> agent.alive, tier_agents)
        n_tier = max(1, length(tier_agents))
        tier_survival[tier] = alive_count / n_tier
        result["$(tier)_survival"] = tier_survival[tier]
        result["$(tier)_innovations_per_agent"] =
            sum(agent.innovation_count for agent in tier_agents) / n_tier
        result["$(tier)_mean_capital"] =
            safe_mean([agent.resources.capital for agent in tier_agents])
    end

    none_survival = get(tier_survival, "none", 0.0)
    for tier in AI_TIERS
        result["$(tier)_te_pp_vs_none"] = (get(tier_survival, tier, 0.0) - none_survival) * 100
    end

    final_history = isempty(sim.history) ? Dict{String,Any}() : sim.history[end]
    for key in (
        "actor_ignorance",
        "practical_indeterminism",
        "agentic_novelty",
        "competitive_recursion",
        "market_crowding_pressure",
        "market_opportunity_overlap",
        "market_investment_concentration",
        "market_opportunity_competition",
        "market_ai_herding_intensity",
        "market_ai_action_correlation",
        "market_combo_reuse_pressure",
        "mean_visible_opportunities",
        "mean_info_quality_used_none",
        "mean_info_quality_used_basic",
        "mean_info_quality_used_advanced",
        "mean_info_quality_used_premium",
        "total_ai_analysis_cost",
    )
        result["final_$(key)"] = numeric_history_value(final_history, key)
        result["tail_mean_$(key)"] = tail_mean(sim.history, key)
    end

    for key in sort([
        k for k in keys(final_history)
        if startswith(k, "knightian_") || startswith(k, "mean_perception_component_")
    ])
        value = get(final_history, key, 0.0)
        if value isa Real
            result["final_$(key)"] = Float64(value)
        end
    end

    return result
end

function build_cells()
    cells = Vector{NamedTuple{(:family, :multiplier),Tuple{String,Float64}}}()
    push!(cells, (family="baseline", multiplier=1.0))
    for family in SWEEP_FAMILIES
        family in ALL_FAMILIES || error("Unknown family in SWEEP_FAMILIES: $family")
        for multiplier in SWEEP_MULTIPLIERS
            isapprox(multiplier, 1.0; atol=1e-12) && continue
            push!(cells, (family=family, multiplier=multiplier))
        end
    end
    return cells
end

function cell_seed(cell_idx::Int, run_idx::Int)
    if SWEEP_SEED_MODE == "paired"
        return BASE_SEED + run_idx
    elseif SWEEP_SEED_MODE == "independent"
        return BASE_SEED + cell_idx * 10_000 + run_idx
    end
    error("Unknown SWEEP_SEED_MODE=$(SWEEP_SEED_MODE); use paired or independent")
end

function summarize_results(results::Vector{Dict{String,Any}}, cells)
    numeric_keys = sort(unique(
        key for row in results for (key, value) in row
        if value isa Real && !(key in ("cell_idx", "run_idx", "seed", "multiplier"))
    ))

    summaries = Dict{String,Any}[]
    for (idx, cell) in enumerate(cells)
        rows = filter(row -> row["cell_idx"] == idx, results)
        summary = Dict{String,Any}(
            "cell_idx" => idx,
            "family" => cell.family,
            "multiplier" => cell.multiplier,
            "n_runs" => length(rows),
        )
        for key in numeric_keys
            values = [Float64(row[key]) for row in rows if haskey(row, key) && row[key] isa Real]
            summary["mean_$(key)"] = safe_mean(values)
            summary["std_$(key)"] = safe_std(values)
        end
        push!(summaries, summary)
    end
    return summaries
end

function print_summary(summaries)
    println("\n" * "="^80)
    println("  KNIGHTIAN WEIGHT-FAMILY SWEEP RESULTS")
    println("="^80)
    @printf(
        "  %-38s %6s %7s %7s %7s %7s %7s %7s %7s %7s\n",
        "family",
        "mult",
        "none",
        "basic",
        "adv",
        "prem",
        "TE_b",
        "TE_a",
        "TE_p",
        "horiz",
    )
    println("  " * "-"^96)
    for row in summaries
        horizon = safe_mean([
            Float64(get(row, "mean_tail_mean_practical_indeterminism", 0.0)),
            Float64(get(row, "mean_tail_mean_agentic_novelty", 0.0)),
            Float64(get(row, "mean_tail_mean_competitive_recursion", 0.0)),
        ])
        @printf(
            "  %-38s %6.2f %7.3f %7.3f %7.3f %7.3f %+7.2f %+7.2f %+7.2f %7.3f\n",
            row["family"],
            row["multiplier"],
            get(row, "mean_none_survival", 0.0),
            get(row, "mean_basic_survival", 0.0),
            get(row, "mean_advanced_survival", 0.0),
            get(row, "mean_premium_survival", 0.0),
            get(row, "mean_basic_te_pp_vs_none", 0.0),
            get(row, "mean_advanced_te_pp_vs_none", 0.0),
            get(row, "mean_premium_te_pp_vs_none", 0.0),
            horizon,
        )
    end
end

function main()
    cells = build_cells()
    mkpath(OUTPUT_DIR)

    println("="^80)
    println("  KNIGHTIAN WEIGHT-FAMILY SENSITIVITY SWEEP")
    println("="^80)
    println("  Agents per run:     $N_AGENTS")
    println("  Runs per cell:      $N_RUNS")
    println("  Rounds per run:     $N_ROUNDS")
    println("  Families:           $(join(SWEEP_FAMILIES, ", "))")
    println("  Multipliers:        $(join(SWEEP_MULTIPLIERS, ", "))")
    println("  Seed mode:          $SWEEP_SEED_MODE")
    println("  Cells:              $(length(cells))")
    println("  Threads available:  $(Threads.nthreads())")
    println("  Output directory:   $OUTPUT_DIR")
    println("="^80)

    total_runs = length(cells) * N_RUNS
    all_results = Vector{Dict{String,Any}}(undef, total_runs)
    completed = Threads.Atomic{Int}(0)
    start_time = time()

    Threads.@threads for idx in 1:total_runs
        cell_idx = ((idx - 1) ÷ N_RUNS) + 1
        run_idx = ((idx - 1) % N_RUNS) + 1
        cell = cells[cell_idx]
        seed = cell_seed(cell_idx, run_idx)
        all_results[idx] = run_one(cell_idx, run_idx, cell.family, cell.multiplier, seed)

        count_done = Threads.atomic_add!(completed, 1) + 1
        if count_done % 10 == 0 || count_done == total_runs
            @printf("  Completed %d/%d runs (%.1fs elapsed)\n", count_done, total_runs, time() - start_time)
        end
    end

    summaries = summarize_results(all_results, cells)
    print_summary(summaries)

    CSV.write(joinpath(OUTPUT_DIR, "weight_family_sweep_per_run.csv"), DataFrame(all_results))
    CSV.write(joinpath(OUTPUT_DIR, "weight_family_sweep_summary.csv"), DataFrame(summaries))

    println("\nCSVs saved to: $OUTPUT_DIR")
    println("="^80)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
