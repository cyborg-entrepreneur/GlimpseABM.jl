# Open-action extension tests (the design notes
# §Open-action extension): A1 ENABLE_PIVOT + A2 ENABLE_DIRECTED_CREATION.
#
# Battery (mirrors test_strategy_ladder.jl's engineering pattern):
# 1. Config validation — malformed haircut bounds / negative strength error
# loudly at initialize! and at first use.
# 2. Default neutrality — both flags off is bit-identical to the implicit
# default model; the open-action debug counter stays 0 over a full
# flags-off simulation; an enabled simulation does enter the code.
# 3. Accounting — a constructed pivot recovers committed × haircut exactly,
# reduces opp.total_invested exactly once, removes the record (maturity
# can never double-pay), and a later death-release of a stale reference
# cannot double-release (idempotent with A5's marker).
# 4. Observability — pivot decision and creation-bias density invariant to
# hidden-field mutations (latent_*, hidden_factors, contains_hallucination,
# actual_accuracy); responsive to observable mutations.
# 5. Behavioral A1 — under perceived-crowding deterioration an ENABLE_PIVOT
# agent exits a holding a default agent keeps; in-sim pivot telemetry > 0.
# 6. Behavioral A2 — with a crowded-vs-sparse perceived density, directed
# creation shifts the innovation-sector distribution toward sparse
# sectors vs the deterministic default (fixed seed, many draws).
# 7. Interaction — the OPEN_ACTION_AGI_NATIVE_MARKET cell (both flags +
# agi_native on all tiers) builds and a tiny sim runs end-to-end.

using Test
using Random
using Statistics

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using GlimpseABM
using GlimpseABM: KnowledgeSignal, ActorIgnorance, ExecutionRisk,
    PracticalIndeterminism, InnovationSignal, AgenticNovelty,
    CompetitionSignal, CompetitiveRecursion, Perception

include("test_helpers.jl")

# ── helpers (oa_ prefix: runtests.jl includes all test files in one scope) ──

"Perception whose only strategy/pivot-relevant dial is perceived crowding."
function oa_test_perception(; crowding::Float64 = 0.0,
                              ai_share::Float64 = 0.0,
                              recursion::Float64 = 0.0)::Perception
    _empty = Dict{String,Float64}()
    Perception(
        KnowledgeSignal(0.5, 0.5, _empty, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ActorIgnorance(0.5, 0.5, _empty, 0.0, 0.0, 0.0, 0.0),
        ExecutionRisk(0.5, _empty, 0.0, 0.0, crowding, 0.0, 0.0),
        PracticalIndeterminism(0.5, 0.5, _empty, _empty, 0.0, 0.0, 0.0, crowding, 0.0, 0.0),
        InnovationSignal(0.5, 0.5, 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0),
        AgenticNovelty(0.5, 0.5, 0.5, _empty, 0.0, 0.0, 0, 0, 0.0,
                       Dict{String,Any}(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        CompetitionSignal(0.5, 0.0, _empty, ai_share, 0.0, 0.0),
        CompetitiveRecursion(recursion, recursion, 0.0, _empty, _empty, 0.0, 0.0, 0.0, ai_share),
        Dict{String,Float64}(),
        0.0,
        Dict{String,Float64}(),
        0.5,
        Dict{String,Float64}(
            "decision_confidence" => 0.5,
            "return_evidence" => 0.0,
            "confidence_saturation_hit" => 0.0,
       ),
   )
end

function oa_test_config(; pivot::Bool = false, directed::Bool = false,
                          strategy_mode::Union{String,Nothing} = nothing,
                          n_agents::Int = 4, kwargs...)
    config = if isnothing(strategy_mode)
        EmergentConfig(; N_AGENTS = n_agents, N_ROUNDS = 2, RANDOM_SEED = 42,
                       ENABLE_PIVOT = pivot, ENABLE_DIRECTED_CREATION = directed,
                       enable_round_logging = false, kwargs...)
    else
        EmergentConfig(; N_AGENTS = n_agents, N_ROUNDS = 2, RANDOM_SEED = 42,
                       ENABLE_PIVOT = pivot, ENABLE_DIRECTED_CREATION = directed,
                       STRATEGY_MODE = strategy_mode, enable_round_logging = false,
                       kwargs...)
    end
    GlimpseABM.initialize!(config)
    return config
end

function oa_test_opportunity(id::String; sector::String = "tech",
                             competition::Float64 = 0.0)
    return Opportunity(
        id = id,
        latent_return_potential = 1.5,
        latent_failure_potential = 0.4,
        complexity = 0.4,
        sector = sector,
        competition = competition,
        discovered = true,
   )
end

"""
Agent holding one constructed in-flight investment (the keys _execute_invest!
stores). Returns (agent, opp, investment).
"""
function oa_agent_with_investment(config;
        amount::Float64 = 50_000.0,
        round_invested::Int = 2,
        maturity_round::Int = 14,
        estimated_return::Float64 = 1.2,
        decision_confidence::Float64 = 0.2,
        crowding_at_commitment::Float64 = 0.0,
        opp_base_invested::Float64 = 100_000.0,
        seed::Int = 5)
    agent = EmergentAgent(1, config; primary_sector = "tech",
                          fixed_ai_level = "premium", rng = MersenneTwister(seed))
    opp = oa_test_opportunity("opp_pivot_$(seed)")
    opp.total_invested = opp_base_invested + amount
    investment = Dict{String,Any}(
        "opportunity_id" => opp.id,
        "opportunity" => opp,
        "amount" => amount,
        "round_invested" => round_invested,
        "maturity_round" => maturity_round,
        "ai_level" => "premium",
        "ai_label" => "premium",
        "estimated_return" => estimated_return,
        "sector" => opp.sector,
        "decision_confidence" => decision_confidence,
        "perceived_crowding_at_commitment" => crowding_at_commitment,
   )
    push!(agent.active_investments, investment)
    return agent, opp, investment
end

function oa_fingerprint_sim(config; seed::Int = 4242, rounds::Int = 6)
    sim = EmergentSimulation(config = config, seed = seed)
    for r in 1:rounds
        GlimpseABM.step!(sim, r)
    end
    return (
        capitals = [GlimpseABM.get_capital(a) for a in sim.agents],
        alive = [a.alive for a in sim.agents],
        actions = [copy(a.action_history) for a in sim.agents],
        sim = sim,
   )
end

# ── 1. config validation ─────────────────────────────────────────────────────

@testset "Open action: malformed config errors loudly" begin
    @test_throws ErrorException GlimpseABM.initialize!(
        EmergentConfig(PIVOT_HAIRCUT_FLOOR = 0.8, PIVOT_HAIRCUT_CEILING = 0.4))
    @test_throws ErrorException GlimpseABM.initialize!(
        EmergentConfig(PIVOT_HAIRCUT_CEILING = 1.4))
    @test_throws ErrorException GlimpseABM.initialize!(
        EmergentConfig(PIVOT_HAIRCUT_FLOOR = -0.1))
    @test_throws ErrorException GlimpseABM.initialize!(
        EmergentConfig(DIRECTED_CREATION_STRENGTH = -1.0))
 # F3a: the promoted pivot-trigger fields are validated too.
    @test_throws ErrorException GlimpseABM.initialize!(
        EmergentConfig(PIVOT_REDEPLOY_MARGIN = -0.05))
    @test_throws ErrorException GlimpseABM.initialize!(
        EmergentConfig(PIVOT_CONVICTION_BASE = 0.5))
    @test_throws ErrorException GlimpseABM.initialize!(
        EmergentConfig(PIVOT_DETERIORATION_GAIN = -1.0))

 # Defaults are valid (validation lives at initialize!; the per-round
 # re-validation in the pivot review was removed as hot-path hygiene — F6).
    @test GlimpseABM.initialize!(EmergentConfig()) isa EmergentConfig
end

# ── 2. default neutrality ────────────────────────────────────────────────────

@testset "Open action: default neutrality (flags off bit-identical)" begin
 # F6: the debug counter is increment-gated behind a flag (production
 # default false); the structural-neutrality assertions enable it so the
 # ==0 guarantee stays a guarantee about reachable code paths.
    GlimpseABM.set_open_action_count_enabled!(true)
    GlimpseABM.reset_open_action_eval_count!()
    cfg_default = EmergentConfig(N_AGENTS = 24, N_ROUNDS = 6, RANDOM_SEED = 7,
                                 enable_round_logging = false)
    GlimpseABM.initialize!(cfg_default)
    fp_default = oa_fingerprint_sim(cfg_default; seed = 9001)
 # Structural proof: no open-action code path is reachable when disabled.
    @test GlimpseABM.open_action_eval_count() == 0

    cfg_off = EmergentConfig(N_AGENTS = 24, N_ROUNDS = 6, RANDOM_SEED = 7,
                             ENABLE_PIVOT = false, ENABLE_DIRECTED_CREATION = false,
                             enable_round_logging = false)
    GlimpseABM.initialize!(cfg_off)
    fp_off = oa_fingerprint_sim(cfg_off; seed = 9001)
    @test fp_default.capitals == fp_off.capitals
    @test fp_default.alive == fp_off.alive
    @test fp_default.actions == fp_off.actions
    @test GlimpseABM.open_action_eval_count() == 0

 # An enabled simulation DOES enter open-action code (the counter guard is
 # alive, not vacuous).
    cfg_on = EmergentConfig(N_AGENTS = 24, N_ROUNDS = 6, RANDOM_SEED = 7,
                            ENABLE_PIVOT = true, ENABLE_DIRECTED_CREATION = true,
                            enable_round_logging = false)
    GlimpseABM.initialize!(cfg_on)
    oa_fingerprint_sim(cfg_on; seed = 9001)
    @test GlimpseABM.open_action_eval_count() > 0
    GlimpseABM.reset_open_action_eval_count!()

 # With the debug flag OFF (production default), the counter never moves
 # even in an enabled simulation — the increments are debug-only overhead.
    GlimpseABM.set_open_action_count_enabled!(false)
    cfg_on2 = EmergentConfig(N_AGENTS = 24, N_ROUNDS = 6, RANDOM_SEED = 7,
                             ENABLE_PIVOT = true, ENABLE_DIRECTED_CREATION = true,
                             enable_round_logging = false)
    GlimpseABM.initialize!(cfg_on2)
    oa_fingerprint_sim(cfg_on2; seed = 9001)
    @test GlimpseABM.open_action_eval_count() == 0
    GlimpseABM.set_open_action_count_enabled!(true)

 # Disabled review short-circuits before ANY computation, even with a
 # pivot-ripe portfolio in hand.
    cfg_hold = oa_test_config(pivot = false)
    agent, opp, _ = oa_agent_with_investment(cfg_hold; estimated_return = 0.0)
    invested_before = opp.total_invested
    GlimpseABM.reset_open_action_eval_count!()
    @test GlimpseABM.review_and_pivot_investments!(
        agent, 8, oa_test_perception(crowding = 0.9)) == (0, 0.0, 0.0, Set{String}())
    @test GlimpseABM.open_action_eval_count() == 0
    @test length(agent.active_investments) == 1
    @test opp.total_invested == invested_before
    GlimpseABM.set_open_action_count_enabled!(false)
end

# ── 3. accounting ────────────────────────────────────────────────────────────

@testset "Open action A1: pivot accounting (conservation, single release, no double payout)" begin
    config = oa_test_config(pivot = true)
 # Constructed trigger: committed at crowding 0.0 / confidence 0.2 with a
 # 1.2× expected multiple; current perceived crowding 0.9 ⇒ deterioration
 # 0.9 × (1.5 − 0.2) clamps to d_eff = 1.0 ⇒ expected residual 0 < any
 # recoverable value ⇒ pivot. At round 8 (invested round 2, ttm 12):
 # progress 0.5 ⇒ haircut 0.40 + 0.35·0.5 = 0.575 ⇒ recovered = 28,750.
    agent, opp, investment = oa_agent_with_investment(config)
    capital_before = GlimpseABM.get_capital(agent)
    returned_before = agent.total_returned
    base_invested = opp.total_invested - 50_000.0
    perception = oa_test_perception(crowding = 0.9)
    tier_roi_before = length(get(agent.ai_learning.tier_roi_history, "premium", Float64[]))
    tier_belief_before = GlimpseABM.get_tier_belief_mean(agent.ai_learning, "premium")
    conf_obs_before = agent.confidence_outcome_diagnostics.obs_count

    n, recovered, committed, pivoted_ids = GlimpseABM.review_and_pivot_investments!(
        agent, 8, perception)
    @test n == 1
    @test committed == 50_000.0
    @test recovered == 50_000.0 * 0.575                      # exact haircut
    @test pivoted_ids == Set([opp.id])                       # F3c: ids reported
    @test GlimpseABM.get_capital(agent) == capital_before + recovered
    @test agent.total_returned == returned_before + recovered
    @test opp.total_invested == base_invested                # released exactly once
    @test isempty(agent.active_investments)                  # record removed

 # Honest learning record, flagged as censored salvage (F2d).
    last = agent.recent_outcomes[end]
    @test last["action"] == "pivot"
    @test last["success"] == false
    @test last["amount"] == 50_000.0
    @test last["capital_returned"] == recovered
    @test last["cash_multiple"] ≈ 0.575
    @test last["censored_salvage"] === true
    @test investment["censored_salvage"] === true            # flag on the source record too

 # F2d tier evidence: the recovery multiple entered the invested tier's
 # ROI history and shifted the tier belief downward (0.575 < 1).
    roi_hist = agent.ai_learning.tier_roi_history["premium"]
    @test length(roi_hist) == tier_roi_before + 1
    @test roi_hist[end] ≈ 0.575 - 1.0
    @test GlimpseABM.get_tier_belief_mean(agent.ai_learning, "premium") < tier_belief_before

 # F2c calibration: one confidence-outcome observation, on the investment
 # channel (a pivot resolves an investment), mirroring maturity.
    diag = agent.confidence_outcome_diagnostics
    @test diag.obs_count == conf_obs_before + 1
    @test diag.investment_obs_count >= 1

 # F2e Boolean channels: success/failure counters stay maturity-only.
    @test agent.success_count == 0
    @test agent.failure_count == 0

 # Maturity processing cannot double-pay: at the original maturity round
 # there is nothing left to mature and capital is unchanged.
    sim = EmergentSimulation(config = config, seed = 77)
    capital_at_pivot = GlimpseABM.get_capital(agent)
    matured = GlimpseABM.process_matured_investments!(agent, sim.market, 14)
    @test isempty(matured)
    @test GlimpseABM.get_capital(agent) == capital_at_pivot

 # Second review is a no-op (idempotent).
    @test GlimpseABM.review_and_pivot_investments!(agent, 9, perception) ==
          (0, 0.0, 0.0, Set{String}())
    @test opp.total_invested == base_invested

 # Death after pivot cannot double-release — even a STALE reference to the
 # pivoted record is skipped via the shared A5 release marker.
    push!(agent.active_investments, investment)
    GlimpseABM._release_inflight_capital_at_death!(agent)
    @test opp.total_invested == base_invested

 # And the review never touches already-due investments (they belong to
 # maturity processing).
    agent2, opp2, _ = oa_agent_with_investment(config; maturity_round = 8,
                                               estimated_return = 0.0, seed = 6)
    invested2 = opp2.total_invested
    @test GlimpseABM.review_and_pivot_investments!(agent2, 8, perception) ==
          (0, 0.0, 0.0, Set{String}())
    @test length(agent2.active_investments) == 1
    @test opp2.total_invested == invested2
end

# ── 3b. pivot mirrors maturity: exposure + ledger (the design notes F2) ──────────

@testset "Open action A1: pivot records a forecast-disconfirmation exposure (raw estimate)" begin
    config = oa_test_config(pivot = true)
    perception = oa_test_perception(crowding = 0.9)

 # The exposure must be scored against the commitment-time INSTRUMENT
 # estimate (F1 raw), not the decision-basis estimated_return. Store
 # diverging values: decision 1.0 (shaded), instrument 2.0 (raw).
    agent, opp, investment = oa_agent_with_investment(config; estimated_return = 1.0)
    investment["instrument_estimated_return"] = 2.0
    env = KnightianUncertaintyEnvironment(config; rng = MersenneTwister(404))

    n, recovered, _, _ = GlimpseABM.review_and_pivot_investments!(
        agent, 8, perception; uncertainty_env = env)
    @test n == 1
    events = env.ai_uncertainty_signals["observable_disconfirmation_events"]
    @test length(events) == 1
    ev = events[1]
    @test ev["ai_level"] == "premium"
    @test ev["ai_assisted"] === true            # from the invested tier label
    @test ev["success"] === false
 # return_error = |actual − pred| / max(1, |pred|) with actual =
 # recovered/amount = 0.575 and pred = the RAW 2.0 (not the shaded 1.0).
    @test ev["return_error"] ≈ abs(recovered / 50_000.0 - 2.0) / 2.0

 # A legacy record with no estimate at all is EXCLUDED from exposure
 # recording (F1 no-estimate rule) but still pivots on the neutral 1.0.
    agent2, _, inv2 = oa_agent_with_investment(config; seed = 6)
    delete!(inv2, "estimated_return")
    env2 = KnightianUncertaintyEnvironment(config; rng = MersenneTwister(405))
    n2, _, _, _ = GlimpseABM.review_and_pivot_investments!(
        agent2, 8, perception; uncertainty_env = env2)
    @test n2 == 1
    @test isempty(env2.ai_uncertainty_signals["observable_disconfirmation_events"])
end

@testset "Open action A1: round ledger closes over pivot recoveries (F2a)" begin
 # Inject a write-off-grade off-market holding so the production round
 # pivots it (same construction as the in-sim telemetry test below), then
 # check the round PnL: capital_returned[invest] must include the pivot
 # recovery, so net_capital_flow_invest - pivot_capital_recovered equals
 # the flow that maturities + new deploys alone would give.
    cfg = EmergentConfig(N_AGENTS = 16, N_ROUNDS = 3, RANDOM_SEED = 11,
                         ENABLE_PIVOT = true, enable_round_logging = false)
    GlimpseABM.initialize!(cfg)
    sim = EmergentSimulation(config = cfg, seed = 501)
    opp = oa_test_opportunity("opp_ledger_writeoff")
    amount = 10_000.0
    opp.total_invested += amount
    push!(sim.agents[1].active_investments, Dict{String,Any}(
        "opportunity_id" => opp.id,
        "opportunity" => opp,
        "amount" => amount,
        "round_invested" => 0,
        "maturity_round" => 8,
        "ai_level" => "none",
        "ai_label" => "none",
        "estimated_return" => 0.0,
        "decision_confidence" => 0.0,
        "perceived_crowding_at_commitment" => 0.0,
        "sector" => opp.sector,
   ))
    stats = GlimpseABM.step!(sim, 1)
    @test stats["pivot_count"] >= 1
    recovered = stats["pivot_capital_recovered"]
    @test recovered > 0.0
 # Round 1 of a fresh sim has no maturities (n_matured == 0), so the
 # invest-side returns this round are EXACTLY the pivot recoveries — the
 # strict closure the old ledger leaked (pivot cash credited to agents but
 # absent from capital_returned["invest"]).
    @test stats["n_matured"] == 0
    returned = stats["total_capital_returned_invest"]
    deployed = stats["total_capital_deployed_invest"]
    @test returned ≈ recovered
    @test stats["net_capital_flow_invest"] ≈ returned - deployed
end

# ── 4. observability audit ───────────────────────────────────────────────────

@testset "Open action: observability (hidden fields inert, observables live)" begin
    config = oa_test_config(pivot = true, directed = true)

 # A1: the pivot decision pivots on the agent's STORED commitment-time
 # instrument estimate (F1/F3c: decision-time-stored beliefs — a visible
 # field the agent itself persisted) and must be invariant to every hidden
 # field. Borderline construction: no crowding deterioration, so the
 # stored estimate alone decides — 0.3 (pivot) vs 1.5 (hold).
    function run_review(; stored_return::Float64, mutate_hidden::Bool, seed::Int)
        agent, opp, investment = oa_agent_with_investment(config;
            estimated_return = stored_return, decision_confidence = 0.9,
            crowding_at_commitment = 0.5, seed = seed)
        investment["instrument_estimated_return"] = stored_return
        if mutate_hidden
            opp.latent_return_potential = 99.0
            opp.latent_failure_potential = 0.99
        end
        result = GlimpseABM.review_and_pivot_investments!(
            agent, 8, oa_test_perception(crowding = 0.5))
        return result
    end

    base = run_review(stored_return = 0.3, mutate_hidden = false, seed = 21)
    @test base[1] == 1                       # stored low estimate ⇒ pivot
    hidden = run_review(stored_return = 0.3, mutate_hidden = true, seed = 21)
    @test hidden == base                     # hidden mutations: EXACTLY unchanged
    held = run_review(stored_return = 1.5, mutate_hidden = false, seed = 21)
    @test held == (0, 0.0, 0.0, Set{String}())  # observable change flips the decision

 # A2: the perceived density estimate reads only sector + competition
 # traces of the visible set.
    opps = [oa_test_opportunity("d1"; sector = "tech", competition = 0.8),
            oa_test_opportunity("d2"; sector = "tech", competition = 0.6),
            oa_test_opportunity("d3"; sector = "healthcare", competition = 0.0)]
    density_base = GlimpseABM.perceived_sector_density(opps)
    for o in opps
        o.latent_return_potential = 99.0
        o.latent_failure_potential = 0.99
    end
    @test GlimpseABM.perceived_sector_density(opps) == density_base   # latents inert
    opps[3].competition = 0.9                                         # observable
    @test GlimpseABM.perceived_sector_density(opps) != density_base
    @test density_base["tech"] > density_base["healthcare"]           # crowded from this seat
    @test sum(values(density_base)) ≈ 1.0
end

# ── 5. behavioral A1 ─────────────────────────────────────────────────────────

@testset "Open action A1: pivot agent exits a deteriorated holding a default agent keeps" begin
    perception_crowded = oa_test_perception(crowding = 0.9)

 # Same holding, same perceived deterioration; only the flag differs.
    cfg_pivot = oa_test_config(pivot = true)
    cfg_hold = oa_test_config(pivot = false)
    agent_p, opp_p, _ = oa_agent_with_investment(cfg_pivot; seed = 31)
    agent_h, opp_h, _ = oa_agent_with_investment(cfg_hold; seed = 31)

    n_p, _, _ = GlimpseABM.review_and_pivot_investments!(agent_p, 8, perception_crowded)
    n_h, _, _ = GlimpseABM.review_and_pivot_investments!(agent_h, 8, perception_crowded)
    @test n_p == 1 && isempty(agent_p.active_investments)   # pivot agent exits
    @test n_h == 0 && length(agent_h.active_investments) == 1  # default holds

 # The rule is selective, not a blanket exit: with no deterioration, high
 # original conviction, and a healthy expected multiple, the pivot agent
 # holds too.
    agent_s, _, _ = oa_agent_with_investment(cfg_pivot;
        estimated_return = 1.5, decision_confidence = 0.9,
        crowding_at_commitment = 0.9, seed = 32)
    n_s, _, _ = GlimpseABM.review_and_pivot_investments!(agent_s, 8, perception_crowded)
    @test n_s == 0 && length(agent_s.active_investments) == 1

 # In-sim telemetry: inject a write-off-grade holding (stored estimate 0 ⇒
 # expected residual 0 ⇒ pivot regardless of perceived crowding) and step
 # one production round; the per-round pivot count column registers it.
    cfg_sim = EmergentConfig(N_AGENTS = 16, N_ROUNDS = 3, RANDOM_SEED = 11,
                             ENABLE_PIVOT = true, enable_round_logging = false)
    GlimpseABM.initialize!(cfg_sim)
    sim = EmergentSimulation(config = cfg_sim, seed = 501)
 # Off-market holding with a stored write-off-grade estimate. Since F3c the
 # trigger always reads the decision-time-STORED commitment estimate (the
 # F1 instrument value), never a same-round cache re-query.
    opp = oa_test_opportunity("opp_injected_writeoff")
    amount = 10_000.0
    opp.total_invested += amount
    push!(sim.agents[1].active_investments, Dict{String,Any}(
        "opportunity_id" => opp.id,
        "opportunity" => opp,
        "amount" => amount,
        "round_invested" => 0,
        "maturity_round" => 8,
        "ai_level" => "none",
        "ai_label" => "none",
        "estimated_return" => 0.0,
        "decision_confidence" => 0.0,
        "perceived_crowding_at_commitment" => 0.0,
        "sector" => opp.sector,
   ))
    stats = GlimpseABM.step!(sim, 1)
    @test stats["pivot_count"] >= 1
    @test stats["pivot_capital_recovered"] > 0.0
    @test 0.40 <= stats["pivot_recovery_rate"] <= 0.75
    @test !any(inv -> get(inv, "opportunity_id", "") == opp.id,
               sim.agents[1].active_investments)

 # Flags-off runs emit the columns with empty-cell conventions intact:
 # zero count (no events is data), NaN rate (undefined over zero pivots).
    cfg_none = EmergentConfig(N_AGENTS = 8, N_ROUNDS = 2, RANDOM_SEED = 11,
                              enable_round_logging = false)
    GlimpseABM.initialize!(cfg_none)
    sim_none = EmergentSimulation(config = cfg_none, seed = 502)
    stats_none = GlimpseABM.step!(sim_none, 1)
    @test stats_none["pivot_count"] == 0
    @test isnan(stats_none["pivot_recovery_rate"])
end

# ── 6. behavioral A2 ─────────────────────────────────────────────────────────

@testset "Open action A2: directed creation shifts sector choice toward sparse sectors" begin
    config = oa_test_config(directed = true, n_agents = 8)
    sim = EmergentSimulation(config = config, seed = 909)
    engine = sim.innovation_engine
    knowledge = collect(values(engine.knowledge_base.knowledge_pieces))[1:2]
    agent = sim.agents[1]
    sectors = collect(config.SECTORS)
    @test length(sectors) >= 2

 # Reactive default: deterministic knowledge-domain mapping, zero RNG.
    anchor = GlimpseABM.determine_innovation_sector(engine, agent, knowledge)
    @test anchor in sectors || anchor == "tech"
    @test GlimpseABM.determine_innovation_sector(engine, agent, knowledge) == anchor

 # Perceived density: ALL visible mass on the anchor sector (maximally
 # crowded from this seat); every other sector is sparse.
    density = Dict{String,Float64}(anchor => 1.0)
    rng = MersenneTwister(99)
    n_draws = 400
    counts = Dict{String,Int}()
    for _ in 1:n_draws
        s = GlimpseABM.determine_innovation_sector(
            engine, agent, knowledge; sector_density = density, rng = rng)
        counts[s] = get(counts, s, 0) + 1
    end
    anchor_share = get(counts, anchor, 0) / n_draws
    sparse_share = 1.0 - anchor_share
    @test anchor_share < 0.5          # vs 1.0 under the deterministic default
    @test sparse_share > 0.5          # creation redirected toward sparse sectors
    @test length(keys(counts)) >= 2   # genuinely distributed, not a swap

 # Uniform perceived density ⇒ weights reduce to the anchor prior (the
 # knowledge linkage is redirected, never severed).
    uniform = Dict{String,Float64}(s => 1.0 / length(sectors) for s in sectors)
    w = GlimpseABM.directed_sector_weights(sectors, uniform, anchor, 1.0)
    @test length(w) == length(sectors)
    @test w[findfirst(==(anchor), sectors)] ≈ GlimpseABM.DIRECTED_CREATION_ANCHOR_PRIOR
    @test all(w[i] ≈ 1.0 for i in eachindex(sectors) if sectors[i] != anchor)

 # Strength 0 ⇒ density-blind (prior only); higher strength ⇒ stronger
 # tilt away from the crowded sector.
    w0 = GlimpseABM.directed_sector_weights(sectors, density, anchor, 0.0)
    w2 = GlimpseABM.directed_sector_weights(sectors, density, anchor, 2.0)
    i_anchor = findfirst(==(anchor), sectors)
    @test w0[i_anchor] ≈ GlimpseABM.DIRECTED_CREATION_ANCHOR_PRIOR
    @test w2[i_anchor] / sum(w2) <
          GlimpseABM.directed_sector_weights(sectors, density, anchor, 1.0)[i_anchor] /
          sum(GlimpseABM.directed_sector_weights(sectors, density, anchor, 1.0))

 # Disabled config ignores any provided density (no redirection, no RNG).
    cfg_off = oa_test_config(directed = false, n_agents = 8)
    sim_off = EmergentSimulation(config = cfg_off, seed = 909)
    @test GlimpseABM.determine_innovation_sector(
        sim_off.innovation_engine, sim_off.agents[1], knowledge;
        sector_density = density, rng = MersenneTwister(99)) ==
        GlimpseABM.determine_innovation_sector(
            sim_off.innovation_engine, sim_off.agents[1], knowledge)
end

# ── 7. interaction: the maximal AGI-robustness cell ──────────────────────────

@testset "Open action: OPEN_ACTION_AGI_NATIVE_MARKET cell runs end-to-end" begin
 # Both open-action channels + composite AGI-native strategy on all tiers
 # (pre-registered P8 cell; suite row OPEN_ACTION_AGI_NATIVE_MARKET).
    config = EmergentConfig(N_AGENTS = 24, N_ROUNDS = 8, RANDOM_SEED = 7,
                            ENABLE_PIVOT = true, ENABLE_DIRECTED_CREATION = true,
                            STRATEGY_MODE = "agi_native",
                            STRATEGY_TIERS = ["none", "basic", "advanced", "premium"],
                            enable_round_logging = false)
    GlimpseABM.initialize!(config)
    GlimpseABM.set_open_action_count_enabled!(true)
    GlimpseABM.reset_open_action_eval_count!()
    GlimpseABM.reset_strategy_eval_count!()
    fp = oa_fingerprint_sim(config; seed = 1234, rounds = 8)
    @test length(fp.capitals) == 24
    @test all(isfinite, fp.capitals)
    @test GlimpseABM.open_action_eval_count() > 0     # open-action channels live
    @test GlimpseABM.strategy_eval_count() > 0        # strategy ladder live
    @test haskey(fp.sim.history[end], "pivot_count")  # telemetry columns present
    GlimpseABM.reset_open_action_eval_count!()
    GlimpseABM.reset_strategy_eval_count!()
    GlimpseABM.set_open_action_count_enabled!(false)
end

# ── pure-function contracts ──────────────────────────────────────────────────

@testset "Open action: pure function contracts" begin
    config = oa_test_config(pivot = true)
 # Haircut: linear floor → ceiling on maturity progress, clamped.
    @test GlimpseABM.pivot_haircut(config, 0.0) == 0.40
    @test GlimpseABM.pivot_haircut(config, 1.0) == 0.75
    @test GlimpseABM.pivot_haircut(config, 0.5) ≈ 0.575
    @test GlimpseABM.pivot_haircut(config, -1.0) == 0.40
    @test GlimpseABM.pivot_haircut(config, 2.0) == 0.75

 # Trigger: pivot ⇔ expected residual < recoverable × (1 + margin).
    hold, residual, recoverable = GlimpseABM.pivot_trigger(config;
        progress = 0.5, crowding_now = 0.0, crowding_at_commit = 0.0,
        decision_confidence = 0.5, est_multiple = 1.2)
    @test !hold && residual == 1.2 && recoverable ≈ 0.575
 # A residual just under the margin band pivots; just over holds. The
 # margin is a config field since F3a (default 0.05, formerly a module
 # constant).
    margin = config.PIVOT_REDEPLOY_MARGIN
    @test margin == 0.05
    below, _, _ = GlimpseABM.pivot_trigger(config;
        progress = 0.5, crowding_now = 0.0, crowding_at_commit = 0.0,
        decision_confidence = 0.5, est_multiple = 0.575 * (1 + margin) - 1e-9)
    above, _, _ = GlimpseABM.pivot_trigger(config;
        progress = 0.5, crowding_now = 0.0, crowding_at_commit = 0.0,
        decision_confidence = 0.5, est_multiple = 0.575 * (1 + margin) + 1e-9)
    @test below && !above
 # Conviction modulates the deterioration bite: same signal, low-confidence
 # committer folds, high-conviction committer holds.
    low_conf, _, _ = GlimpseABM.pivot_trigger(config;
        progress = 0.2, crowding_now = 0.7, crowding_at_commit = 0.0,
        decision_confidence = 0.1, est_multiple = 1.3)
    high_conf, _, _ = GlimpseABM.pivot_trigger(config;
        progress = 0.2, crowding_now = 0.7, crowding_at_commit = 0.0,
        decision_confidence = 0.95, est_multiple = 1.3)
    @test low_conf && !high_conf
 # Crowding improvement (now < commit) never counts as deterioration.
    improved, residual_i, _ = GlimpseABM.pivot_trigger(config;
        progress = 0.2, crowding_now = 0.1, crowding_at_commit = 0.9,
        decision_confidence = 0.5, est_multiple = 1.3)
    @test !improved && residual_i == 1.3
end

# ── F3a: deterioration gain — bit-identity at 1.0 and monotone liveness ──────

@testset "Open action A1: PIVOT_DETERIORATION_GAIN (bit-identity at 1.0, monotone)" begin
    trigger_at(gain; kwargs...) = begin
        cfg = oa_test_config(pivot = true, PIVOT_DETERIORATION_GAIN = gain)
        GlimpseABM.pivot_trigger(cfg; kwargs...)
    end
 # Bit-identity at gain 1.0: the explicit formula with the original
 # hardcoded factors reproduces the trigger exactly (IEEE x·1.0 ≡ x).
    for (prog, cn, cc, conf, est) in (
            (0.5, 0.4, 0.1, 0.3, 1.2), (0.2, 0.7, 0.0, 0.1, 1.3),
            (0.9, 0.55, 0.31, 0.62, 0.97), (0.5, 0.0, 0.0, 0.5, 1.2))
        cfg = oa_test_config(pivot = true)   # default gain = 1.0
        got = GlimpseABM.pivot_trigger(cfg; progress = prog, crowding_now = cn,
            crowding_at_commit = cc, decision_confidence = conf, est_multiple = est)
        d_eff = clamp(clamp(clamp(cn, 0, 1) - clamp(cc, 0, 1), 0, 1) * (1.5 - conf), 0.0, 1.0)
        expected_residual = max(est, 0.0) * (1.0 - d_eff)
        recoverable = GlimpseABM.pivot_haircut(cfg, prog)
        @test got[2] === expected_residual
        @test got[1] === (expected_residual <
                          recoverable * (1.0 + cfg.PIVOT_REDEPLOY_MARGIN))
    end
 # Monotone: with a deterioration the original gain leaves sub-trigger, a
 # higher gain converts holds into pivots (never the reverse).
    kw = (progress = 0.3, crowding_now = 0.28, crowding_at_commit = 0.0,
          decision_confidence = 0.5, est_multiple = 1.1)
    p1, r1, _ = trigger_at(1.0; kw...)
    p2, r2, _ = trigger_at(2.0; kw...)
    p3, r3, _ = trigger_at(3.0; kw...)
    @test r1 >= r2 >= r3                 # residual shrinks monotonically in gain
    @test !p1 && p3                      # observed-range deterioration only fires at higher gain
 # And zero deterioration never triggers regardless of gain.
    calm = (progress = 0.3, crowding_now = 0.0, crowding_at_commit = 0.0,
            decision_confidence = 0.5, est_multiple = 1.1)
    @test !trigger_at(5.0; calm...)[1]
end

# ── F2e: pivot records excluded from Boolean success channels ────────────────

@testset "Open action A1: success-rate channels exclude pivots, return evidence keeps them" begin
    invest_win = Dict{String,Any}("action" => "invest", "success" => true,
                                  "ai_used" => true, "cash_multiple" => 1.5)
    invest_loss = Dict{String,Any}("action" => "invest", "success" => false,
                                   "ai_used" => true, "cash_multiple" => 0.6)
    pivots = [Dict{String,Any}("action" => "pivot", "success" => false,
                               "ai_used" => true, "cash_multiple" => 0.5,
                               "censored_salvage" => true) for _ in 1:3]
    base = GlimpseABM._recent_outcome_experience_stats(
        Dict{String,Any}[invest_win, invest_loss])
    with_pivots = GlimpseABM._recent_outcome_experience_stats(
        Dict{String,Any}[invest_win, invest_loss, pivots...])
 # Boolean channels unchanged by the pivot records...
    @test with_pivots["recent_success_rate"] == base["recent_success_rate"] == 0.5
    @test with_pivots["ai_success_rate"] == base["ai_success_rate"] == 0.5
 #...but the cash multiples DO carry the loss into return evidence.
    @test with_pivots["n_roi_outcomes"] == 5
    @test with_pivots["return_evidence"] < base["return_evidence"]
 # Pivot-only history: Boolean channels stay at the neutral prior.
    only_pivots = GlimpseABM._recent_outcome_experience_stats(
        Dict{String,Any}[pivots...])
    @test only_pivots["recent_success_rate"] == 0.5
    @test only_pivots["ai_success_rate"] == 0.5
    @test only_pivots["n_roi_outcomes"] == 3
end

# ── F3c: same-round hygiene (no re-entry; utilities see post-pivot state) ────

@testset "Open action A1: just-pivoted opportunity cannot be re-entered this round" begin
    config = oa_test_config(pivot = true)
    sim = EmergentSimulation(config = config, seed = 808)
    mc = GlimpseABM.get_market_conditions(sim.market)

    function decide_with_holding(; with_holding::Bool)
        agent = EmergentAgent(1, config; primary_sector = "tech",
                              fixed_ai_level = "none", rng = MersenneTwister(99))
 # Force invest selection: overwhelming persistent invest taste (kept
 # below the exp(u/T) overflow point so the softmax stays finite).
        agent.action_biases["invest"] = 100.0
        opp = oa_test_opportunity("opp_reentry"; competition = 0.0)
        if with_holding
            opp.total_invested += 10_000.0
            push!(agent.active_investments, Dict{String,Any}(
                "opportunity_id" => opp.id,
                "opportunity" => opp,
                "amount" => 10_000.0,
                "round_invested" => 2,
                "maturity_round" => 14,
                "ai_level" => "none",
                "ai_label" => "none",
                "estimated_return" => 0.0,    # write-off grade ⇒ pivots
                "decision_confidence" => 0.0,
                "perceived_crowding_at_commitment" => 0.0,
                "sector" => opp.sector,
           ))
        end
        outcome = GlimpseABM.make_decision!(
            agent, [opp], mc, sim.market, 8)
        return agent, opp, outcome
    end

 # Control: without a holding the forced-invest agent invests in the opp.
    _, _, outcome_free = decide_with_holding(with_holding = false)
    @test outcome_free["action"] == "invest"
    @test outcome_free["opportunity_id"] == "opp_reentry"

 # With the write-off holding, the review pivots it FIRST; the only visible
 # opportunity is the just-pivoted one, so the forced invest falls through
 # to maintain instead of re-entering at full price (haircut churn).
    agent, opp, outcome = decide_with_holding(with_holding = true)
    @test outcome["pivot_count"] == 1
    @test outcome["action"] == "maintain"
    @test isempty(agent.active_investments)
end

@testset "Open action A1: ALL action utilities see post-pivot capital/portfolio (F3c ordering)" begin
    config = oa_test_config(pivot = true)
    sim = EmergentSimulation(config = config, seed = 909)
    mc = GlimpseABM.get_market_conditions(sim.market)
    amount = 10_000.0

 # Two byte-identical agents with the same write-off holding. For the TWIN
 # the review is run EXTERNALLY before make_decision! (its internal review
 # is then a no-op); for the PIVOT agent the internal review must do the
 # work. If — and only if — the internal review runs before every utility
 # computation, both agents enter the utility pass in exactly the same
 # state with exactly the same RNG stream (the review consumes none), so
 # utilities and the chosen action must match exactly.
    function build_agent(tag::String)
        agent = EmergentAgent(1, config; primary_sector = "tech",
                              fixed_ai_level = "none", rng = MersenneTwister(4242))
        opp_held = oa_test_opportunity("opp_ordering_held_$(tag)")
        opp_held.total_invested += amount
        push!(agent.active_investments, Dict{String,Any}(
            "opportunity_id" => opp_held.id,
            "opportunity" => opp_held,
            "amount" => amount,
            "round_invested" => 2,
            "maturity_round" => 14,
            "ai_level" => "none",
            "ai_label" => "none",
            "estimated_return" => 0.0,
            "decision_confidence" => 0.0,
            "perceived_crowding_at_commitment" => 0.0,
            "sector" => opp_held.sector,
       ))
        return agent
    end

    agent_pivot = build_agent("a")
    agent_twin = build_agent("b")
    n_ext, recovered_ext, _, _ = GlimpseABM.review_and_pivot_investments!(
        agent_twin, 8, GlimpseABM.empty_perception())
    @test n_ext == 1 && recovered_ext ≈ amount * GlimpseABM.pivot_haircut(config, 0.5)

    opp_a = oa_test_opportunity("opp_ordering_visible_a")
    opp_b = oa_test_opportunity("opp_ordering_visible_b")
    outcome_pivot = GlimpseABM.make_decision!(agent_pivot, [opp_a], mc, sim.market, 8)
    outcome_twin = GlimpseABM.make_decision!(agent_twin, [opp_b], mc, sim.market, 8)
    @test outcome_pivot["pivot_count"] == 1
    @test outcome_twin["pivot_count"] == 0
    @test outcome_pivot["utilities"] == outcome_twin["utilities"]
    @test outcome_pivot["action"] == outcome_twin["action"]
    @test GlimpseABM.get_capital(agent_pivot) == GlimpseABM.get_capital(agent_twin)
end
