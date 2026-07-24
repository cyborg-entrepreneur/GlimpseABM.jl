#!/usr/bin/env julia
#
# Focused identification sweep for the canonical outstanding-capital/capacity
# payoff constraint. This is deliberately separate from the manuscript suite:
# it crosses the literature-bounded (lambda, gamma, kappa) grid, then varies the
# mean and dispersion of opportunity capacity one factor at a time.
#
# Usage:
#   DRY_RUN=1 julia --project=. scripts/run_capacity_identification_sweep.jl
#   N_RUNS=50 julia --threads=auto --project=. scripts/run_capacity_identification_sweep.jl
#   CELLS=CANONICAL_IK,CAPACITY_MEAN_HALF,CAPACITY_MEAN_DOUBLE \
#     N_AGENTS=64 N_ROUNDS=8 N_RUNS=2 julia --project=. scripts/run_capacity_identification_sweep.jl

if abspath(PROGRAM_FILE) == @__FILE__
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
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
const N_RUNS = parse(Int, get(ENV, "N_RUNS", "50"))
const BASE_SEED = parse(Int, get(ENV, "BASE_SEED", "20260425"))
const OUTPUT_DIR = get(ENV, "OUTPUT_DIR", joinpath(
    @__DIR__, "..", "results",
    "capacity_identification_$(Dates.format(now(), "yyyymmdd_HHMMSS"))",
))
const CELL_FILTER = strip(get(ENV, "CELLS", ""))

struct CapacityCell
    name::String
    lambda::Float64
    gamma::Float64
    kappa::Float64
    capacity_scale::Float64
    capacity_sigma::Float64
    action_hhi_blend::Float64
    decision_crowding_aversion::Float64
    role::String
end

function all_cells()::Vector{CapacityCell}
    cells = CapacityCell[]
    for lambda in (1.0, 1.5, 2.0), gamma in (1.0, 1.5, 2.0), kappa in (1.0, 1.5, 2.0)
        is_default = lambda == 1.5 && gamma == 1.5 && kappa == 1.5
        name = is_default ? "CANONICAL_IK" :
            @sprintf("GRID_L%.1f_G%.1f_K%.1f", lambda, gamma, kappa)
        push!(cells, CapacityCell(
            name, lambda, gamma, kappa, 1.0, 1.5, 0.0, 0.0,
            is_default ? "canonical" : "literature_bounded_structural_grid",
        ))
    end

    append!(cells, [
        CapacityCell("CAPACITY_MEAN_HALF", 1.5, 1.5, 1.5, 0.5, 1.5, 0.0, 0.0,
                     "one_factor_capacity_mean"),
        CapacityCell("CAPACITY_MEAN_DOUBLE", 1.5, 1.5, 1.5, 2.0, 1.5, 0.0, 0.0,
                     "one_factor_capacity_mean"),
        CapacityCell("CAPACITY_TAIL_TRUNCATED", 1.5, 1.5, 1.5, 1.0, 0.0, 0.0, 0.0,
                     "one_factor_capacity_dispersion"),
        CapacityCell("CAPACITY_TAIL_MODERATE", 1.5, 1.5, 1.5, 1.0, 0.75, 0.0, 0.0,
                     "one_factor_capacity_dispersion"),
        CapacityCell("CROWDING_ACTION_HHI_BLEND", 1.5, 1.5, 1.5, 1.0, 1.5, 0.30, 0.0,
                     "previous_payoff_specification"),
        CapacityCell("CAPACITY_AWARE_SELECTION", 1.5, 1.5, 1.5, 1.0, 1.5, 0.0, 1.0,
                     "strategic_capacity_boundary"),
    ])
    return cells
end

function selected_cells()::Vector{CapacityCell}
    cells = all_cells()
    isempty(CELL_FILTER) && return cells
    requested = Set(strip.(split(CELL_FILTER, ",")))
    selected = [cell for cell in cells if cell.name in requested]
    missing = setdiff(requested, Set(cell.name for cell in selected))
    isempty(missing) || error("Unknown CELLS entries: $(join(sort!(collect(missing)), ", "))")
    return selected
end

function build_simulation(cell::CapacityCell, seed::Int)::EmergentSimulation
    cfg = EmergentConfig(
        N_AGENTS=N_AGENTS,
        N_ROUNDS=N_ROUNDS,
        RANDOM_SEED=seed,
        AGENT_AI_MODE="fixed",
    )
    # Match the manuscript suite's canonical non-capacity settings.
    cfg.NICHE_SIZE_LOG_SIGMA = cell.capacity_sigma
    cfg.OPS_COST_INTENSITY = 0.85
    cfg.DERIVED_KNOWLEDGE_QUALITY_GATE = 0.55
    cfg.CROWDING_STRENGTH_LAMBDA = cell.lambda
    cfg.CROWDING_CONVEXITY_GAMMA = cell.gamma
    cfg.CROWDING_CAPACITY_RATIO_K = cell.kappa
    cfg.CROWDING_INDEX_BLEND = cell.action_hhi_blend
    cfg.DECISION_CROWDING_AVERSION_WEIGHT = cell.decision_crowding_aversion
    cfg.OPPORTUNITY_BASE_CAPACITY *= cell.capacity_scale
    cfg.enable_round_logging = false

    rng = MersenneTwister(seed)
    sim = EmergentSimulation(
        config=cfg,
        initial_tier_distribution=Dict(tier => 0.25 for tier in AI_TIERS),
        seed=seed,
        run_id="capacity_$(cell.name)_$(seed)",
    )
    apply_balanced_fixed_tiers!(sim, balanced_tier_assignments(N_AGENTS, rng))
    return sim
end

safe_mean(values) = isempty(values) ? NaN : mean(values)

function top_fraction_share(values, fraction::Float64)::Float64
    ordered = sort(Float64[v for v in values if isfinite(v) && v >= 0.0]; rev=true)
    isempty(ordered) && return NaN
    total = sum(ordered)
    total > 0.0 || return NaN
    k = clamp(ceil(Int, fraction * length(ordered)), 1, length(ordered))
    return sum(@view ordered[1:k]) / total
end

"""Diagnostic Pareto-tail estimate over the largest `fraction` of positive values."""
function hill_alpha(values, fraction::Float64=0.10)::Float64
    ordered = sort(Float64[v for v in values if isfinite(v) && v > 0.0]; rev=true)
    n = length(ordered)
    k = floor(Int, fraction * n)
    (k >= 5 && k < n) || return NaN
    threshold = ordered[k + 1]
    threshold > 0.0 || return NaN
    denominator = sum(log(ordered[idx] / threshold) for idx in 1:k)
    return denominator > 0.0 ? k / denominator : NaN
end

finite_mean(values) = begin
    kept = Float64[v for v in values if isfinite(v)]
    isempty(kept) ? NaN : mean(kept)
end

function run_one(cell::CapacityCell, run_idx::Int)
    seed = BASE_SEED + run_idx
    sim = build_simulation(cell, seed)
    for round in 1:N_ROUNDS
        GlimpseABM.step!(sim, round)
    end

    survival = Dict{String,Float64}()
    for tier in AI_TIERS
        agents = [agent for agent in sim.agents if GlimpseABM.get_ai_level(agent) == tier]
        survival[tier] = safe_mean(Float64[agent.alive for agent in agents])
    end

    sat_entry = Float64[]
    sat_maturity = Float64[]
    action_hhi = Float64[]
    applied_multiplier = Float64[]
    capacities = Float64[]
    deal_return_multiples = Float64[]
    for agent in sim.agents, event in agent.venture_ledger
        push!(deal_return_multiples, event[1])
        push!(capacities, event[8])
        push!(sat_entry, event[10])
        push!(sat_maturity, event[11])
        push!(action_hhi, event[12])
        effective_load = event[11] + cell.action_hhi_blend * event[12]
        excess = max(0.0, effective_load / cell.kappa - 1.0)
        push!(applied_multiplier, exp(-cell.lambda * excess^cell.gamma))
    end

    penalty_active = isempty(applied_multiplier) ? NaN :
        mean(Float64[mult < 1.0 - 1e-12 for mult in applied_multiplier])
    wealth_multiples = Float64[
        max(0.0, agent.resources.capital) /
        max(agent.resources.performance.initial_equity, eps(Float64))
        for agent in sim.agents
    ]
    return (
        cell=cell.name,
        role=cell.role,
        run_idx=run_idx,
        seed=seed,
        lambda=cell.lambda,
        gamma=cell.gamma,
        kappa=cell.kappa,
        capacity_scale=cell.capacity_scale,
        capacity_sigma=cell.capacity_sigma,
        action_hhi_blend=cell.action_hhi_blend,
        decision_crowding_aversion=cell.decision_crowding_aversion,
        survival_none=survival["none"],
        survival_basic=survival["basic"],
        survival_advanced=survival["advanced"],
        survival_premium=survival["premium"],
        frontier_effect_pp=100.0 * (survival["premium"] - survival["none"]),
        mean_capacity_exposure=safe_mean(capacities),
        mean_saturation_entry=safe_mean(sat_entry),
        mean_saturation_maturity=safe_mean(sat_maturity),
        mean_action_hhi=safe_mean(action_hhi),
        fraction_penalty_active=penalty_active,
        mean_crowding_multiplier=safe_mean(applied_multiplier),
        n_matured_investments=length(applied_multiplier),
        wealth_top_1pct_share=top_fraction_share(wealth_multiples, 0.01),
        wealth_top_10pct_share=top_fraction_share(wealth_multiples, 0.10),
        wealth_hill_alpha_top10pct=hill_alpha(wealth_multiples),
        wealth_max_multiple=isempty(wealth_multiples) ? NaN : maximum(wealth_multiples),
        deal_return_top_1pct_share=top_fraction_share(deal_return_multiples, 0.01),
        deal_return_top_10pct_share=top_fraction_share(deal_return_multiples, 0.10),
        deal_return_hill_alpha_top10pct=hill_alpha(deal_return_multiples),
    )
end

function aggregate_results(raw::DataFrame)::DataFrame
    rows = NamedTuple[]
    for group in groupby(raw, :cell)
        effects = Float64.(group.frontier_effect_pp)
        n = length(effects)
        push!(rows, (
            cell=String(group.cell[1]),
            role=String(group.role[1]),
            lambda=Float64(group.lambda[1]),
            gamma=Float64(group.gamma[1]),
            kappa=Float64(group.kappa[1]),
            capacity_scale=Float64(group.capacity_scale[1]),
            capacity_sigma=Float64(group.capacity_sigma[1]),
            action_hhi_blend=Float64(group.action_hhi_blend[1]),
            decision_crowding_aversion=Float64(group.decision_crowding_aversion[1]),
            n_runs=n,
            frontier_effect_pp=mean(effects),
            frontier_effect_se_pp=n > 1 ? std(effects) / sqrt(n) : NaN,
            survival_none=mean(group.survival_none),
            survival_premium=mean(group.survival_premium),
            mean_capacity_exposure=mean(group.mean_capacity_exposure),
            mean_saturation_entry=mean(group.mean_saturation_entry),
            mean_saturation_maturity=mean(group.mean_saturation_maturity),
            mean_action_hhi=mean(group.mean_action_hhi),
            fraction_penalty_active=mean(group.fraction_penalty_active),
            mean_crowding_multiplier=mean(group.mean_crowding_multiplier),
            n_matured_investments=sum(group.n_matured_investments),
            wealth_top_1pct_share=finite_mean(group.wealth_top_1pct_share),
            wealth_top_10pct_share=finite_mean(group.wealth_top_10pct_share),
            wealth_hill_alpha_top10pct=finite_mean(group.wealth_hill_alpha_top10pct),
            wealth_max_multiple=finite_mean(group.wealth_max_multiple),
            deal_return_top_1pct_share=finite_mean(group.deal_return_top_1pct_share),
            deal_return_top_10pct_share=finite_mean(group.deal_return_top_10pct_share),
            deal_return_hill_alpha_top10pct=finite_mean(group.deal_return_hill_alpha_top10pct),
        ))
    end
    return sort!(DataFrame(rows), [:role, :cell])
end

function main()
    cells = selected_cells()
    println("Capacity-identification sweep (canonical pure I/K)")
    println("N_AGENTS=$N_AGENTS N_ROUNDS=$N_ROUNDS N_RUNS=$N_RUNS cells=$(length(cells))")
    for cell in cells
        @printf("%-34s lambda=%.2f gamma=%.2f kappa=%.2f cap=%.2fx sigma=%.2f HHI=%.2f decision=%.2f\n",
            cell.name, cell.lambda, cell.gamma, cell.kappa, cell.capacity_scale,
            cell.capacity_sigma, cell.action_hhi_blend, cell.decision_crowding_aversion)
    end
    if get(ENV, "DRY_RUN", "0") == "1" || "--list" in ARGS
        return nothing
    end

    mkpath(OUTPUT_DIR)
    write_run_provenance!(
        OUTPUT_DIR;
        script_name=basename(@__FILE__),
        parameters=Dict(
            "BASE_SEED" => BASE_SEED,
            "CELLS" => isempty(CELL_FILTER) ? "all" : CELL_FILTER,
            "N_AGENTS" => N_AGENTS,
            "N_ROUNDS" => N_ROUNDS,
            "N_RUNS" => N_RUNS,
            "OUTPUT_DIR" => OUTPUT_DIR,
            "RUN_TAG" => get(ENV, "RUN_TAG", ""),
            "SUBMISSION_GIT_COMMIT" =>
                get(ENV, "SUBMISSION_GIT_COMMIT", ""),
        ),
        notes=Dict(
            "design" => "seed-paired capacity-identification sweep",
        ),
    )
    CSV.write(joinpath(OUTPUT_DIR, "capacity_identification_manifest.csv"), DataFrame([(
        cell=cell.name,
        role=cell.role,
        lambda=cell.lambda,
        gamma=cell.gamma,
        kappa=cell.kappa,
        capacity_scale=cell.capacity_scale,
        capacity_sigma=cell.capacity_sigma,
        action_hhi_blend=cell.action_hhi_blend,
        decision_crowding_aversion=cell.decision_crowding_aversion,
    ) for cell in cells]))
    tasks = [(cell, run_idx) for cell in cells for run_idx in 1:N_RUNS]
    results = Vector{Any}(undef, length(tasks))
    done = Threads.Atomic{Int}(0)
    Threads.@threads for idx in eachindex(tasks)
        cell, run_idx = tasks[idx]
        results[idx] = run_one(cell, run_idx)
        completed = Threads.atomic_add!(done, 1) + 1
        completed % max(1, min(25, length(tasks))) == 0 &&
            println("  completed $completed / $(length(tasks))")
    end

    raw = DataFrame(results)
    CSV.write(joinpath(OUTPUT_DIR, "capacity_identification_raw.csv"), raw)
    println("[checkpoint] raw capacity-identification results written before aggregation")
    flush(stdout)
    summary = aggregate_results(raw)
    CSV.write(joinpath(OUTPUT_DIR, "capacity_identification_summary.csv"), summary)
    println("Output: $OUTPUT_DIR")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
