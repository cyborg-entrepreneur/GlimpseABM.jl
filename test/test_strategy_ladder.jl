# AGI strategy ladder tests (the strategy-ladder design notes)
#
# Battery:
#   1. Config validation — unknown STRATEGY_MODE / STRATEGY_TIERS error loudly
#      at initialize! and at first strategy_active use (never fall through to
#      reactive).
#   2. Default neutrality — reactive mode short-circuits before any strategy
#      computation: dispatcher inactive, debug counter stays 0 over a full
#      simulation, and a sim with mode unset is exactly equal to a sim with
#      explicit "reactive".
#   3. Tier gating — mode active with STRATEGY_TIERS=["premium"] leaves a
#      basic-tier agent's scores/utilities bit-identical to reactive.
#   4. Observability audit — mutating hidden fields (latent_*, hidden_factors,
#      contains_hallucination, actual_accuracy) leaves every strategy output
#      EXACTLY unchanged; mutating an observable input changes it.
#   5. Behavioral, one per rung — S1 flips the chosen opportunity away from
#      the consensus-legible option; S2 flips it toward the high-private-edge
#      option; S3 raises explore/innovate utility and shifts relative invest
#      scores toward the low-own-confidence opportunity; composite runs all
#      gates simultaneously without error.
#   6. G1 invariants (2026-06-09 fix) — the S2 multiplier is centered WITHIN
#      the choice set: mean multiplier over a choice set ≡ 1 (no flat
#      invest-propensity suppression), higher-familiarity opportunities get
#      strictly higher multipliers (re-ranking direction), single-candidate
#      sets are exactly neutral.
#   7. G2 — co-activating the legacy STRATEGIC_ANTICIPATION channel with the
#      canonical S1 modes errors loudly; diversification (a re-draw channel)
#      and non-S1 modes stay allowed.
#   8. G4 — the debug eval counter only increments behind the test flag
#      (default off; reset_strategy_eval_count!() enables it), so production
#      ladder cells pay no contended atomics while the ==0 neutrality pins
#      stay meaningful.

using Test
using Random
using Statistics

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using GlimpseABM
using GlimpseABM: KnowledgeSignal, ActorIgnorance, ExecutionRisk,
    PracticalIndeterminism, InnovationSignal, AgenticNovelty,
    CompetitionSignal, CompetitiveRecursion, Perception

include("test_helpers.jl")

# ── helpers ──────────────────────────────────────────────────────────────────

"""
Perception with controllable strategy-relevant observables (mirrors
empty_perception, models.jl): `ai_share` → CompetitionSignal/
CompetitiveRecursion ai_usage_share, `recursion` → recursion level,
`crowding` → practical-indeterminism / execution-risk crowding_pressure.
"""
function strategy_test_perception(;
    ai_share::Float64 = 0.0,
    recursion::Float64 = 0.0,
    crowding::Float64 = 0.0,
)::Perception
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

function strategy_test_config(; mode::String = "reactive",
                                tiers::Vector{String} = ["none", "basic", "advanced", "premium"])
    config = EmergentConfig(
        N_AGENTS = 4, N_ROUNDS = 2, RANDOM_SEED = 42,
        STRATEGY_MODE = mode, STRATEGY_TIERS = tiers,
        enable_round_logging = false,
    )
    GlimpseABM.initialize!(config)
    return config
end

function strategy_test_opportunity(id::String; sector::String = "tech",
                                   competition::Float64 = 0.0,
                                   complexity::Float64 = 0.4)
    return Opportunity(
        id = id,
        latent_return_potential = 1.5,
        latent_failure_potential = 0.4,
        complexity = complexity,
        sector = sector,
        competition = competition,
        discovered = true,
    )
end

"Pre-populate an InformationSystem cache so the production ranking path runs
with controlled Information objects (cache hits: no charge, no RNG draws)."
function seeded_info_system(config, entries::Vector{<:Tuple})
    sys = InformationSystem(config)
    for (opp_id, tier, agent_id, info) in entries
        sys.information_cache[(opp_id, tier, agent_id)] = info
    end
    return sys
end

# Run a short fixed-seed simulation and return an exact behavioral fingerprint.
function run_fingerprint_sim(config; seed::Int = 4242, rounds::Int = 6)
    sim = EmergentSimulation(config = config, seed = seed)
    for r in 1:rounds
        GlimpseABM.step!(sim, r)
    end
    return (
        capitals = [GlimpseABM.get_capital(a) for a in sim.agents],
        alive = [a.alive for a in sim.agents],
        actions = [copy(a.action_history) for a in sim.agents],
    )
end

# ── 1. config validation ─────────────────────────────────────────────────────

@testset "Strategy ladder: unknown mode errors loudly" begin
    bad = EmergentConfig(STRATEGY_MODE = "equilibrium_solver")
    @test_throws ErrorException GlimpseABM.initialize!(bad)
    # First-use path (config built without initialize!) must also fail loudly,
    # never fall through to reactive.
    @test_throws ErrorException strategy_active(bad, "premium")

    bad_tiers = EmergentConfig(STRATEGY_TIERS = ["premium", "ultrapremium"])
    @test_throws ErrorException GlimpseABM.initialize!(bad_tiers)

    # All valid modes pass validation.
    for mode in STRATEGY_MODES
        ok = EmergentConfig(STRATEGY_MODE = String(mode))
        @test GlimpseABM.initialize!(ok) isa EmergentConfig
    end
end

# ── 2. default neutrality ────────────────────────────────────────────────────

@testset "Strategy ladder: default neutrality (reactive bit-identical)" begin
    # Dispatcher inactive for reactive mode on every tier.
    config = strategy_test_config()
    for tier in ["none", "basic", "advanced", "premium"]
        @test !strategy_active(config, tier)
    end

    # Structural proof that no strategy code path is reachable when inactive:
    # the debug counter (incremented by every strategy component computation)
    # stays at 0 across an entire reactive simulation.
    GlimpseABM.reset_strategy_eval_count!()
    cfg_default = EmergentConfig(N_AGENTS = 24, N_ROUNDS = 6, RANDOM_SEED = 7,
                                 enable_round_logging = false)
    GlimpseABM.initialize!(cfg_default)
    fp_default = run_fingerprint_sim(cfg_default; seed = 9001)
    @test GlimpseABM.strategy_eval_count() == 0

    # Mode unset (struct default) == mode explicitly "reactive": exact
    # equality of final capitals, survival flags, and action histories.
    cfg_reactive = EmergentConfig(N_AGENTS = 24, N_ROUNDS = 6, RANDOM_SEED = 7,
                                  STRATEGY_MODE = "reactive",
                                  enable_round_logging = false)
    GlimpseABM.initialize!(cfg_reactive)
    fp_reactive = run_fingerprint_sim(cfg_reactive; seed = 9001)
    @test fp_default.capitals == fp_reactive.capitals
    @test fp_default.alive == fp_reactive.alive
    @test fp_default.actions == fp_reactive.actions
    @test GlimpseABM.strategy_eval_count() == 0

    # And an active-mode simulation DOES enter strategy code (counter guard is
    # alive, not vacuous).
    cfg_active = EmergentConfig(N_AGENTS = 24, N_ROUNDS = 6, RANDOM_SEED = 7,
                                STRATEGY_MODE = "agi_native",
                                enable_round_logging = false)
    GlimpseABM.initialize!(cfg_active)
    run_fingerprint_sim(cfg_active; seed = 9001)
    @test GlimpseABM.strategy_eval_count() > 0
    GlimpseABM.reset_strategy_eval_count!()
end

# ── 3. tier gating ───────────────────────────────────────────────────────────

@testset "Strategy ladder: tier gating leaves out-of-scope tiers bit-identical" begin
    config_reactive = strategy_test_config()
    config_gated = strategy_test_config(mode = "agi_native", tiers = ["premium"])

    @test strategy_active(config_gated, "premium")
    @test !strategy_active(config_gated, "basic")
    @test !strategy_active(config_gated, "none")

    opps = [strategy_test_opportunity("opp_g1"; competition = 0.6),
            strategy_test_opportunity("opp_g2"; sector = "retail")]
    mc = test_market_conditions()
    perception = strategy_test_perception(ai_share = 0.9, recursion = 0.8, crowding = 0.9)

    # Same trait seed; configs differ only in the strategy fields, which do not
    # touch trait sampling.
    agent_r = EmergentAgent(1, config_reactive; primary_sector = "tech",
                            fixed_ai_level = "basic", rng = MersenneTwister(31))
    agent_g = EmergentAgent(1, config_gated; primary_sector = "tech",
                            fixed_ai_level = "basic", rng = MersenneTwister(31))

    info1 = Information(estimated_return = 1.5, estimated_uncertainty = 0.3, confidence = 0.8)
    info2 = Information(estimated_return = 1.4, estimated_uncertainty = 0.6, confidence = 0.3)
    sys_r = seeded_info_system(config_reactive,
        [("opp_g1", "basic", 1, deepcopy(info1)), ("opp_g2", "basic", 1, deepcopy(info2))])
    sys_g = seeded_info_system(config_gated,
        [("opp_g1", "basic", 1, deepcopy(info1)), ("opp_g2", "basic", 1, deepcopy(info2))])

    GlimpseABM.reset_strategy_eval_count!()
    evals_r = GlimpseABM.evaluate_portfolio_opportunities(
        agent_r, opps, mc, perception; ai_level = "basic", info_system = sys_r)
    evals_g = GlimpseABM.evaluate_portfolio_opportunities(
        agent_g, opps, mc, perception; ai_level = "basic", info_system = sys_g)
    @test [e.final_score for e in evals_r] == [e.final_score for e in evals_g]
    @test [e.estimated_return for e in evals_r] == [e.estimated_return for e in evals_g]
    @test [e.opportunity.id for e in evals_r] == [e.opportunity.id for e in evals_g]

    u_r = GlimpseABM.calculate_investment_utility(
        agent_r, opps, mc, perception; ai_level = "basic", info_system = sys_r)
    u_g = GlimpseABM.calculate_investment_utility(
        agent_g, opps, mc, perception; ai_level = "basic", info_system = sys_g)
    @test u_r === u_g
    # No strategy computation ran for the gated-out tier.
    @test GlimpseABM.strategy_eval_count() == 0
end

# ── 4. observability audit ───────────────────────────────────────────────────

@testset "Strategy ladder: observability audit (hidden fields inert, observables live)" begin
    config = strategy_test_config(mode = "agi_native")
    perception = strategy_test_perception(ai_share = 0.8, recursion = 0.7, crowding = 0.6)
    knowledge = Dict{String,Float64}("tech" => 0.8)
    opp = strategy_test_opportunity("opp_audit"; competition = 0.4)
    info = Information(estimated_return = 1.4, estimated_uncertainty = 0.3, confidence = 0.7)
    # Two-sector choice set so the audited opp has a nonzero within-set edge
    # deviation (G1 form: the multiplier centers on the choice-set mean).
    opps = [opp, strategy_test_opportunity("opp_blind"; sector = "retail")]

    s1_base = GlimpseABM.strategy_shaded_return(config, 1.4, info, perception)
    ctx_base = GlimpseABM.strategy_edge_context(config, knowledge, opps)
    mult_base = GlimpseABM.strategy_score_multiplier(
        config, ctx_base, opp, info, perception)
    sys = seeded_info_system(config, [
        ("opp_audit", "premium", 1, info),
        ("opp_blind", "premium", 1,
         Information(estimated_return = 1.2, estimated_uncertainty = 0.7, confidence = 0.3)),
    ])
    shift_base = GlimpseABM.strategy_complement_action_shift(
        config, opps, sys, "premium", 1, perception)
    @test s1_base < 1.4            # S1 shade engaged (sanity: active inputs)
    @test mult_base != 1.0         # S2 edge engaged (above-mean familiarity)
    @test shift_base > 0.0         # S3a sees the low-confidence target

    # Mutate every forbidden hidden field (design doc observability audit).
    for o in opps
        o.latent_return_potential = 99.0
        o.latent_failure_potential = 0.99
    end
    for i in (info, sys.information_cache[("opp_blind", "premium", 1)])
        i.hidden_factors["bias"] = 0.9
        i.contains_hallucination = true
        i.actual_accuracy = 0.01
    end

    @test GlimpseABM.strategy_shaded_return(config, 1.4, info, perception) === s1_base
    # Rebuild the edge context AFTER the hidden-field mutation: the whole
    # two-pass S2 path (edges, choice-set mean, multiplier) must be inert.
    ctx_mut = GlimpseABM.strategy_edge_context(config, knowledge, opps)
    @test ctx_mut.edges == ctx_base.edges
    @test ctx_mut.edge_mean === ctx_base.edge_mean
    @test GlimpseABM.strategy_score_multiplier(
        config, ctx_mut, opp, info, perception) === mult_base
    @test GlimpseABM.strategy_complement_action_shift(
        config, opps, sys, "premium", 1, perception) === shift_base

    # Mutating OBSERVABLE inputs must change outputs.
    low_conf = Information(estimated_return = 1.4, estimated_uncertainty = 0.3, confidence = 0.2)
    @test GlimpseABM.strategy_shaded_return(config, 1.4, low_conf, perception) != s1_base
    poor_knowledge = Dict{String,Float64}("tech" => 0.05)
    ctx_poor = GlimpseABM.strategy_edge_context(config, poor_knowledge, opps)
    @test GlimpseABM.strategy_score_multiplier(
        config, ctx_poor, opp, info, perception) != mult_base
    sys.information_cache[("opp_audit", "premium", 1)] =
        Information(estimated_return = 1.4, estimated_uncertainty = 0.3, confidence = 0.2)
    @test GlimpseABM.strategy_complement_action_shift(
        config, opps, sys, "premium", 1, perception) != shift_base
end

# ── 5. behavioral: one test per rung ─────────────────────────────────────────

@testset "Strategy ladder S1: chosen opportunity shifts away from consensus-legible option" begin
    config_r = strategy_test_config()
    config_s1 = strategy_test_config(mode = "consensus_discounting")
    mc = test_market_conditions()
    # High AI adoption + high perceived crowding: the regime where anticipated
    # congestion bites.
    perception = strategy_test_perception(ai_share = 1.0, recursion = 0.8, crowding = 1.0)

    # Two opportunities, same sector/complexity. "crowded" is consensus-legible
    # (the agent's OWN instrument is highly confident ⇒ rivals' instruments are
    # too) with a slightly better estimate; "quiet" is data-poor.
    opp_crowded = strategy_test_opportunity("crowded")
    opp_quiet = strategy_test_opportunity("quiet")
    opps = [opp_crowded, opp_quiet]
    info_crowded = Information(estimated_return = 1.6, estimated_uncertainty = 0.3, confidence = 0.95)
    info_quiet = Information(estimated_return = 1.5, estimated_uncertainty = 0.3, confidence = 0.35)

    agent_r = EmergentAgent(1, config_r; primary_sector = "tech",
                            fixed_ai_level = "premium", rng = MersenneTwister(11))
    agent_s = EmergentAgent(1, config_s1; primary_sector = "tech",
                            fixed_ai_level = "premium", rng = MersenneTwister(11))
    sys_r = seeded_info_system(config_r,
        [("crowded", "premium", 1, deepcopy(info_crowded)), ("quiet", "premium", 1, deepcopy(info_quiet))])
    sys_s = seeded_info_system(config_s1,
        [("crowded", "premium", 1, deepcopy(info_crowded)), ("quiet", "premium", 1, deepcopy(info_quiet))])

    evals_r = GlimpseABM.evaluate_portfolio_opportunities(
        agent_r, opps, mc, perception; ai_level = "premium", info_system = sys_r)
    evals_s = GlimpseABM.evaluate_portfolio_opportunities(
        agent_s, opps, mc, perception; ai_level = "premium", info_system = sys_s)

    @test evals_r[1].opportunity.id == "crowded"   # reactive: oracle's #1
    @test evals_s[1].opportunity.id == "quiet"     # S1: anticipates the pile-in
    # The shade acts on the stored decision-time expected return too (sizing
    # uses the strategy-adjusted belief).
    crowded_s = only(e for e in evals_s if e.opportunity.id == "crowded")
    @test crowded_s.estimated_return < 1.6
end

@testset "Strategy ladder S2: chosen opportunity shifts toward private edge" begin
    config_r = strategy_test_config()
    config_s2 = strategy_test_config(mode = "comparative_advantage")
    mc = test_market_conditions()
    perception = strategy_test_perception()   # S2 needs no market commonality

    # "home" sector tech (agent deeply familiar), "away" sector retail (no
    # familiarity). The away option has the larger raw estimate, so reactive
    # ranking prefers it; private-edge re-ranking flips to founder-market fit.
    opp_home = strategy_test_opportunity("home"; sector = "tech", complexity = 0.5)
    opp_away = strategy_test_opportunity("away"; sector = "retail", complexity = 0.5)
    opps = [opp_home, opp_away]
    info_home = Information(estimated_return = 1.4, estimated_uncertainty = 0.3, confidence = 0.6)
    info_away = Information(estimated_return = 1.8, estimated_uncertainty = 0.3, confidence = 0.6)

    results = Dict{String,String}()
    for (label, config) in (("reactive", config_r), ("s2", config_s2))
        agent = EmergentAgent(1, config; primary_sector = "tech",
                              fixed_ai_level = "premium", rng = MersenneTwister(13))
        empty!(agent.resources.knowledge)
        agent.resources.knowledge["tech"] = 0.9
        agent.resources.knowledge["retail"] = 0.0
        agent.traits["competence"] = 0.5
        agent.traits["analytical_ability"] = 0.5
        sys = seeded_info_system(config,
            [("home", "premium", 1, deepcopy(info_home)), ("away", "premium", 1, deepcopy(info_away))])
        evals = GlimpseABM.evaluate_portfolio_opportunities(
            agent, opps, mc, perception; ai_level = "premium", info_system = sys)
        results[label] = evals[1].opportunity.id
    end

    @test results["reactive"] == "away"   # raw estimate wins reactively
    @test results["s2"] == "home"         # private edge flips the choice
end

@testset "Strategy ladder S3: complement seeking under high signal-commonality" begin
    config_r = strategy_test_config()
    config_s3 = strategy_test_config(mode = "complement_seeking")
    mc = test_market_conditions()
    perception_hi = strategy_test_perception(ai_share = 1.0, recursion = 0.8, crowding = 0.5)

    opp_legible = strategy_test_opportunity("legible")
    opp_blind = strategy_test_opportunity("blind")
    opps = [opp_legible, opp_blind]
    info_legible = Information(estimated_return = 1.5, estimated_uncertainty = 0.3, confidence = 0.9)
    info_blind = Information(estimated_return = 1.5, estimated_uncertainty = 0.7, confidence = 0.3)

    agent = EmergentAgent(1, config_s3; primary_sector = "tech",
                          fixed_ai_level = "premium", rng = MersenneTwister(17))
    sys = seeded_info_system(config_s3,
        [("legible", "premium", 1, info_legible), ("blind", "premium", 1, info_blind)])

    # (a) The action shift is positive when low-own-confidence targets are
    # visible and shared signals dominate, and it raises explore + innovate
    # utility relative to reactive.
    shift = GlimpseABM.strategy_complement_action_shift(
        config_s3, opps, sys, "premium", 1, perception_hi)
    @test shift > 0.0
    explore_reactive = GlimpseABM.calculate_exploration_utility(
        agent, perception_hi; ai_level = "premium")
    explore_s3 = GlimpseABM.calculate_exploration_utility(
        agent, perception_hi; ai_level = "premium", strategy_shift = shift)
    @test explore_s3 > explore_reactive
    innovate_reactive = GlimpseABM.calculate_innovation_utility(
        agent, mc, perception_hi; ai_level = "premium")
    innovate_s3 = GlimpseABM.calculate_innovation_utility(
        agent, mc, perception_hi; ai_level = "premium", strategy_shift = shift)
    @test innovate_s3 > innovate_reactive
    # With no commonality there is no shift (the strategy is conditional, not
    # a blanket explore bonus).
    @test GlimpseABM.strategy_complement_action_shift(
        config_s3, opps, sys, "premium", 1, strategy_test_perception()) == 0.0

    # (b) Within invest ranking, the oracle-blind opportunity becomes
    # RELATIVELY more attractive (low-confidence penalty softened).
    score_ratio = Dict{String,Float64}()
    for (label, config) in (("reactive", config_r), ("s3", config_s3))
        a = EmergentAgent(1, config; primary_sector = "tech",
                          fixed_ai_level = "premium", rng = MersenneTwister(17))
        s = seeded_info_system(config,
            [("legible", "premium", 1, deepcopy(info_legible)), ("blind", "premium", 1, deepcopy(info_blind))])
        evals = GlimpseABM.evaluate_portfolio_opportunities(
            a, opps, mc, perception_hi; ai_level = "premium", info_system = s)
        by_id = Dict(e.opportunity.id => e.final_score for e in evals)
        score_ratio[label] = by_id["blind"] / by_id["legible"]
    end
    @test score_ratio["s3"] > score_ratio["reactive"]
end

@testset "Strategy ladder composite (agi_native): all gates active simultaneously" begin
    config = strategy_test_config(mode = "agi_native")
    perception = strategy_test_perception(ai_share = 0.9, recursion = 0.8, crowding = 0.8)
    knowledge = Dict{String,Float64}("tech" => 0.9)
    opp = strategy_test_opportunity("composite")
    opp_away = strategy_test_opportunity("composite_away"; sector = "retail")
    info = Information(estimated_return = 1.5, estimated_uncertainty = 0.6, confidence = 0.3)

    # S1, S2, and S3b all fire on the same evaluation (two-sector choice set
    # so the S2 within-set deviation is nonzero).
    @test GlimpseABM.strategy_shaded_return(config, 1.5, info, perception) < 1.5
    ctx = GlimpseABM.strategy_edge_context(config, knowledge, [opp, opp_away])
    mult = GlimpseABM.strategy_score_multiplier(
        config, ctx, opp, info, perception)
    @test mult != 1.0
    sys = seeded_info_system(config, [("composite", "premium", 1, info)])
    @test GlimpseABM.strategy_complement_action_shift(
        config, [opp], sys, "premium", 1, perception) > 0.0

    # End-to-end: a short full simulation with every gate active runs clean
    # and actually enters strategy code.
    GlimpseABM.reset_strategy_eval_count!()
    cfg_sim = EmergentConfig(N_AGENTS = 24, N_ROUNDS = 6, RANDOM_SEED = 7,
                             STRATEGY_MODE = "agi_native",
                             enable_round_logging = false)
    GlimpseABM.initialize!(cfg_sim)
    fp = run_fingerprint_sim(cfg_sim; seed = 1234)
    @test length(fp.capitals) == 24
    @test GlimpseABM.strategy_eval_count() > 0
    GlimpseABM.reset_strategy_eval_count!()
end

# ── pure-function unit checks (formula contracts) ───────────────────────────

@testset "Strategy ladder: pure function contracts" begin
    # S1 forecast: product of three [0,1] observables, each necessary.
    @test consensus_congestion_forecast(1.0, 1.0, 1.0) == 1.0
    @test consensus_congestion_forecast(0.0, 1.0, 1.0) == 0.0
    @test consensus_congestion_forecast(1.0, 0.0, 1.0) == 0.0
    @test consensus_congestion_forecast(1.0, 1.0, 0.0) == 0.0
    @test consensus_congestion_forecast(0.5, 0.8, 0.5) ≈ 0.2

    # S2 edge (G1 form): per-opportunity, falls back to sector familiarity on
    # the base Opportunity (no knowledge_components field); bounded; no
    # agent-constant trait term (constants cannot re-rank a choice set).
    knowledge = Dict{String,Float64}("tech" => 0.9, "retail" => 0.2)
    opp_t = strategy_test_opportunity("pf_tech"; sector = "tech")
    @test private_edge(opp_t, knowledge, 0.7) == 0.7   # explicit fallback
    @test private_edge(opp_t, knowledge, 1.4) == 1.0   # clamped
    @test private_edge(opp_t, knowledge, -0.2) == 0.0  # clamped

    # S3 shift: zero whenever either commonality or availability is zero;
    # capped at 0.25 × strength.
    @test complement_shift(0.0, 1.0, 1.0) == 0.0
    @test complement_shift(1.0, 0.0, 1.0) == 0.0
    @test complement_shift(1.0, 1.0, 1.0) == 0.25
    @test complement_shift(1.0, 1.0, 0.0) == 0.0
end

# ── 6. G1 invariants: within-choice-set centering ────────────────────────────

@testset "Strategy ladder G1: S2 multiplier centered within the choice set" begin
    config_s2 = strategy_test_config(mode = "comparative_advantage")
    perception = strategy_test_perception()
    knowledge = Dict{String,Float64}("tech" => 0.9, "retail" => 0.2)
    opps = [strategy_test_opportunity("g1_home"; sector = "tech"),
            strategy_test_opportunity("g1_near"; sector = "retail"),
            strategy_test_opportunity("g1_far"; sector = "service")]

    ctx = GlimpseABM.strategy_edge_context(config_s2, knowledge, opps)
    @test !isnothing(ctx)
    # Edges are the per-opportunity familiarity gradient (fallback path).
    @test ctx.edges["g1_home"] > ctx.edges["g1_near"] > ctx.edges["g1_far"]
    @test ctx.edge_mean ≈ sum(values(ctx.edges)) / length(opps)

    # NEW INVARIANT (replaces the old absolute-0.5 centering pin): the mean
    # multiplier over the opportunities scored in one call is EXACTLY 1 — the
    # level confound is removed by construction; only ranking signal remains.
    mults = [GlimpseABM.strategy_score_multiplier(config_s2, ctx, o, nothing, perception)
             for o in opps]
    @test mean(mults) ≈ 1.0 atol = 1e-12
    # Re-ranking direction: the higher-familiarity opportunity gets the
    # strictly higher multiplier.
    @test mults[1] > mults[2] > mults[3]
    @test mults[1] > 1.0 > mults[3]

    # A single-candidate choice set is exactly neutral (deviation ≡ 0): S2
    # re-ranks, it never shifts the invest-vs-not level.
    ctx_one = GlimpseABM.strategy_edge_context(config_s2, knowledge, opps[1:1])
    @test GlimpseABM.strategy_score_multiplier(
        config_s2, ctx_one, opps[1], nothing, perception) == 1.0

    # Context is nothing when S2 is inactive for the mode or the set is empty.
    @test isnothing(GlimpseABM.strategy_edge_context(
        strategy_test_config(), knowledge, opps))                 # reactive
    @test isnothing(GlimpseABM.strategy_edge_context(
        strategy_test_config(mode = "consensus_discounting"), knowledge, opps))
    @test isnothing(GlimpseABM.strategy_edge_context(config_s2, knowledge, Opportunity[]))

    # End-to-end through the production ranking hook: with IDENTICAL
    # information on equal-estimate opportunities, the S2 cell preserves the
    # choice-set's mean-1 property at the multiplier level while flipping the
    # ranking toward familiarity (behavioral S2 test covers the flip; here we
    # pin that BOTH hook sites consume the same two-pass context without
    # erroring on a mixed-sector pool).
    agent = EmergentAgent(1, config_s2; primary_sector = "tech",
                          fixed_ai_level = "premium", rng = MersenneTwister(23))
    empty!(agent.resources.knowledge)
    agent.resources.knowledge["tech"] = 0.9
    agent.resources.knowledge["retail"] = 0.2
    mc = test_market_conditions()
    info = Information(estimated_return = 1.4, estimated_uncertainty = 0.3, confidence = 0.6)
    sys = seeded_info_system(config_s2,
        [(o.id, "premium", 1, deepcopy(info)) for o in opps])
    evals = GlimpseABM.evaluate_portfolio_opportunities(
        agent, opps, mc, perception; ai_level = "premium", info_system = sys)
    @test evals[1].opportunity.id == "g1_home"
    u = GlimpseABM.calculate_investment_utility(
        agent, opps, mc, perception; ai_level = "premium", info_system = sys)
    @test isfinite(u) && u > 0.0
end

# ── 7. G2: legacy operationalization cannot co-activate with canonical S1 ────

@testset "Strategy ladder G2: legacy anticipation co-activation forbidden" begin
    # Two operationalizations of anticipatory congestion discounting must
    # never compose multiplicatively: S1 is canonical (design doc,
    # Canonicality); the legacy flag is an independent-implementation
    # robustness check run in its own cell.
    for mode in ("consensus_discounting", "agi_native")
        bad = EmergentConfig(STRATEGY_MODE = mode,
                             STRATEGIC_ANTICIPATION_ENABLED = true)
        @test_throws ErrorException GlimpseABM.initialize!(bad)
        # First-use path must also fail loudly (config built w/o initialize!).
        @test_throws ErrorException strategy_active(bad, "premium")
    end
    # Modes that do not implement the S1 channel may co-run the legacy
    # multiplier (different channels, no composition on the same path).
    for mode in ("reactive", "comparative_advantage", "complement_seeking")
        ok = EmergentConfig(STRATEGY_MODE = mode,
                            STRATEGIC_ANTICIPATION_ENABLED = true)
        @test GlimpseABM.initialize!(ok) isa EmergentConfig
    end
    # Diversification is a post-ranking stochastic RE-DRAW
    # (select_portfolio_evaluation), not a second discount on the score path:
    # deliberately allowed alongside every ladder mode.
    ok_div = EmergentConfig(STRATEGY_MODE = "agi_native",
                            STRATEGIC_DIVERSIFICATION_ENABLED = true)
    @test GlimpseABM.initialize!(ok_div) isa EmergentConfig
end

# ── 8. G4: debug counter gated behind the test flag ──────────────────────────

@testset "Strategy ladder G4: eval counter increments only behind the test flag" begin
    config = strategy_test_config(mode = "agi_native")
    perception = strategy_test_perception(ai_share = 0.9, recursion = 0.8, crowding = 0.8)
    info = Information(estimated_return = 1.5, estimated_uncertainty = 0.6, confidence = 0.3)

    # Flag off (the production default): strategy computations run but never
    # touch the atomic — no contended atomics in production ladder cells.
    GlimpseABM.reset_strategy_eval_count!(enable = false)
    @test !GlimpseABM.strategy_eval_count_enabled()
    @test GlimpseABM.strategy_shaded_return(config, 1.5, info, perception) < 1.5
    @test GlimpseABM.strategy_eval_count() == 0

    # reset_strategy_eval_count!() enables counting (test convention): the
    # ==0 neutrality pins and >0 liveness pins keep their meaning without
    # callers managing the flag explicitly.
    GlimpseABM.reset_strategy_eval_count!()
    @test GlimpseABM.strategy_eval_count_enabled()
    @test GlimpseABM.strategy_shaded_return(config, 1.5, info, perception) < 1.5
    @test GlimpseABM.strategy_eval_count() > 0

    # Neutrality pin stays meaningful with the flag enabled: a full reactive
    # simulation still increments nothing (no strategy code is reachable).
    GlimpseABM.reset_strategy_eval_count!()
    cfg = EmergentConfig(N_AGENTS = 12, N_ROUNDS = 3, RANDOM_SEED = 7,
                         enable_round_logging = false)
    GlimpseABM.initialize!(cfg)
    run_fingerprint_sim(cfg; seed = 77, rounds = 3)
    @test GlimpseABM.strategy_eval_count() == 0

    # Leave the flag in its production default.
    GlimpseABM.set_strategy_eval_count_enabled!(false)
end
