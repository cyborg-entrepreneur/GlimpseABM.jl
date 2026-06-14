# Phase 2 regression: Perception typed schema + nested-struct shim + snapshot isolation.
# Mirrors test_market_conditions.jl design. Guards against the v3.3.2 / v3.3.3
# "live mutable dict reference" bug class that bit MarketConditions twice
# before snapshot isolation was added.

using Test
using Random

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using GlimpseABM

@testset "Perception schema + nested struct types" begin
    cfg = EmergentConfig(N_AGENTS=20, N_ROUNDS=1, RANDOM_SEED=42)
    GlimpseABM.initialize!(cfg)
    market = MarketEnvironment(cfg; rng=MersenneTwister(42))
    env = KnightianUncertaintyEnvironment(cfg; rng=MersenneTwister(42))
    mc = GlimpseABM.get_market_conditions(market)

    agent_traits = Dict{String,Float64}(
        "competence" => 0.6, "ai_trust" => 0.5, "exploration_tendency" => 0.4,
        "innovativeness" => 0.5, "uncertainty_tolerance" => 0.5,
        "analytical_ability" => 0.5, "risk_tolerance" => 0.5,
   )

    perception = GlimpseABM.perceive_uncertainty(
        env, agent_traits, market.opportunities, mc;
        ai_level="basic",
        agent_knowledge=Set(["tech"]),
        sector_knowledge=Dict("tech" => 0.5),
        action_history=String[],
   )

    @test perception isa Perception
    @test perception.knowledge_signal isa KnowledgeSignal
    @test perception.actor_ignorance isa ActorIgnorance
    @test perception.execution_risk isa ExecutionRisk
    @test perception.practical_indeterminism isa PracticalIndeterminism
    @test perception.innovation_signal isa InnovationSignal
    @test perception.agentic_novelty isa AgenticNovelty
    @test perception.competition_signal isa CompetitionSignal
    @test perception.competitive_recursion isa CompetitiveRecursion
    @test perception.crowding_metrics isa Dict{String,Float64}
    @test perception.volatility_metric isa Float64
    @test perception.action_profile isa Dict{String,Float64}
    @test perception.decision_confidence isa Float64

 # Aliased "level" fields mirror their canonical fields
    @test perception.actor_ignorance.level == perception.actor_ignorance.ignorance_level
    @test perception.practical_indeterminism.level == perception.practical_indeterminism.indeterminism_level
    @test perception.agentic_novelty.level == perception.agentic_novelty.novelty_potential
    @test perception.competitive_recursion.level == perception.competitive_recursion.recursion_level

 # All "level" reads stay bounded
    @test 0.0 <= perception.actor_ignorance.level <= 1.0
    @test 0.0 <= perception.practical_indeterminism.level <= 1.0
    @test 0.0 <= perception.agentic_novelty.level <= 1.0
    @test 0.0 <= perception.competitive_recursion.level <= 1.0
    @test 0.02 <= perception.decision_confidence <= 0.98

 # Immutability: top-level + nested
    @test_throws ErrorException perception.decision_confidence = 0.5
    @test_throws ErrorException perception.actor_ignorance.level = 0.5

 # Dict-shim: get/getindex/haskey across nested structs
    @test get(perception, "decision_confidence", 999.0) == perception.decision_confidence
    @test get(perception, "nonexistent_key", 42) == 42
    @test perception["actor_ignorance"] === perception.actor_ignorance
    @test get(perception["actor_ignorance"], "level", 0.0) == perception.actor_ignorance.level
    @test haskey(perception, "agentic_novelty")
    @test haskey(perception.agentic_novelty, "level")
    @test !haskey(perception, "missing")
    @test_throws KeyError perception["missing"]
end

@testset "Observable opportunity signals are cached in information context" begin
    cfg = EmergentConfig(N_AGENTS=20, N_ROUNDS=1, RANDOM_SEED=441)
    GlimpseABM.initialize!(cfg)
    market = MarketEnvironment(cfg; rng=MersenneTwister(441))
    visible = market.opportunities[1:min(5, length(market.opportunities))]
    weights = GlimpseABM._perception_weights(cfg)

    info_context = GlimpseABM._observable_information_context(
        visible,
        Set{String}(["tech"]),
        Dict("tech" => 0.5),
        length(cfg.SECTORS),
        weights,
   )
    expected_signals = [
        GlimpseABM._observable_opportunity_signal(opp, weights)
        for opp in visible
    ]

    @test info_context.opportunity_signals == expected_signals
    @test length(info_context.opportunity_signals) == length(visible)

    empty_context = GlimpseABM._observable_information_context(
        typeof(visible)(),
        Set{String}(),
        Dict{String,Float64}(),
        length(cfg.SECTORS),
        weights,
   )
    @test isempty(empty_context.opportunity_signals)
end

@testset "Perception snapshot isolation (v3.3.2 regression class)" begin
 # If perception.crowding_metrics holds a reference to the live
 # market_conditions.crowding_metrics dict, mutating the market later
 # in the round would silently change all already-cached Perception
 # objects. This was the v3.3.2 bug class for MarketConditions; we
 # explicitly snapshot-isolate crowding_metrics in uncertainty.jl when
 # constructing Perception.
    cfg = EmergentConfig(N_AGENTS=10, N_ROUNDS=1, RANDOM_SEED=42)
    GlimpseABM.initialize!(cfg)
    market = MarketEnvironment(cfg; rng=MersenneTwister(42))
    env = KnightianUncertaintyEnvironment(cfg; rng=MersenneTwister(42))
    mc = GlimpseABM.get_market_conditions(market)

 # Seed crowding_metrics so we have a baseline to mutate after the fact
    mc.crowding_metrics["sentinel_pre_perception"] = 0.123

    agent_traits = Dict{String,Float64}(
        "competence" => 0.5, "ai_trust" => 0.5, "exploration_tendency" => 0.5,
        "innovativeness" => 0.5, "uncertainty_tolerance" => 0.5,
        "analytical_ability" => 0.5, "risk_tolerance" => 0.5,
   )

    perception = GlimpseABM.perceive_uncertainty(
        env, agent_traits, market.opportunities, mc;
        ai_level="none",
        agent_knowledge=Set{String}(),
        sector_knowledge=Dict{String,Float64}(),
        action_history=String[],
   )

    @test haskey(perception.crowding_metrics, "sentinel_pre_perception")
    @test perception.crowding_metrics["sentinel_pre_perception"] == 0.123
    @test perception.crowding_metrics !== mc.crowding_metrics  # different objects

 # Mutate the source dict AFTER perception is constructed
    mc.crowding_metrics["sentinel_post_perception"] = 0.999
    mc.crowding_metrics["sentinel_pre_perception"] = 99.0  # mutate in place

    @test !haskey(perception.crowding_metrics, "sentinel_post_perception")
    @test perception.crowding_metrics["sentinel_pre_perception"] == 0.123
end

@testset "Perception uses measured Knightian state as prior" begin
    cfg = EmergentConfig(N_AGENTS=10, N_ROUNDS=1, RANDOM_SEED=930)
    GlimpseABM.initialize!(cfg)
    market = MarketEnvironment(cfg; rng=MersenneTwister(930))
    mc = GlimpseABM.get_market_conditions(market)
    traits = Dict{String,Float64}(
        "competence" => 0.5,
        "ai_trust" => 0.5,
        "exploration_tendency" => 0.5,
        "innovativeness" => 0.5,
        "uncertainty_tolerance" => 0.5,
        "analytical_ability" => 0.5,
        "risk_tolerance" => 0.5,
   )

    function perceived_with_prior(actor_prior, recursion_prior)
        env = KnightianUncertaintyEnvironment(cfg; rng=MersenneTwister(931))
        env.actor_ignorance_state["level"] = actor_prior
        env.competitive_recursion_state["level"] = recursion_prior
        env._last_environment_state = Dict{String,Any}(
            "actor_ignorance" => Dict{String,Any}("level" => actor_prior),
            "practical_indeterminism" => Dict{String,Any}("level" => 0.4),
            "agentic_novelty" => Dict{String,Any}("level" => 0.5),
            "competitive_recursion" => Dict{String,Any}(
                "level" => recursion_prior,
                "ai_action_correlation" => 0.6,
                "combo_reuse_pressure" => 0.5,
           ),
       )
        return GlimpseABM.perceive_uncertainty(
            env,
            traits,
            market.opportunities,
            mc;
            ai_level="advanced",
            agent_knowledge=Set(["tech"]),
            sector_knowledge=Dict("tech" => 0.5),
            action_history=String["invest", "explore"],
            agent_id=1,
       )
    end

    low = perceived_with_prior(0.15, 0.10)
    high = perceived_with_prior(0.85, 0.80)

    @test high.actor_ignorance.level > low.actor_ignorance.level
    @test high.competitive_recursion.level > low.competitive_recursion.level
    @test high.confidence_diagnostics["perception_actor_ignorance_prior"] ≈ 0.85
    @test high.confidence_diagnostics["perception_competitive_recursion_prior"] ≈ 0.80
    @test haskey(high.confidence_diagnostics, "perception_actor_ignorance_raw_score")
    @test haskey(high.confidence_diagnostics, "perception_actor_ignorance_normalized_score")
    @test haskey(high.confidence_diagnostics, "perception_actor_ignorance_bounded_level")
    @test haskey(high.confidence_diagnostics, "perception_competitive_recursion_gap_vs_prior")
end

@testset "Decision confidence does not directly inflate with raw info quality" begin
    low_info_confidence = GlimpseABM._decision_confidence_from_components(
        total_uncertainty=0.42,
        competence_trait=0.65,
        recent_success_rate=0.55,
        ai_success_rate=0.55,
        avg_ai_trust=0.5,
        mean_roi=0.08,
        risk_tolerance=0.5,
        hallucination_rate=0.0,
        info_quality=0.25,
   )
    high_info_confidence = GlimpseABM._decision_confidence_from_components(
        total_uncertainty=0.42,
        competence_trait=0.65,
        recent_success_rate=0.55,
        ai_success_rate=0.55,
        avg_ai_trust=0.5,
        mean_roi=0.08,
        risk_tolerance=0.5,
        hallucination_rate=0.0,
        info_quality=0.97,
   )

    @test isapprox(high_info_confidence, low_info_confidence; atol=1e-12)
end

@testset "Raw AI tier does not mechanically reshape fixed-state perception" begin
    cfg = EmergentConfig(N_AGENTS=20, N_ROUNDS=1, RANDOM_SEED=925)
    GlimpseABM.initialize!(cfg)
    market = MarketEnvironment(cfg; rng=MersenneTwister(925))
    mc = GlimpseABM.get_market_conditions(market)

    visible = Opportunity[
        Opportunity(id="obs_tech", discovered=true, sector="tech", complexity=0.35,
                    novelty_score=0.20, combination_uncertainty=0.15,
                    competition=0.20, market_share=0.30,
                    capital_requirements=100_000.0, capacity=1_000_000.0),
        Opportunity(id="obs_service", discovered=true, sector="service", complexity=0.55,
                    novelty_score=0.30, combination_uncertainty=0.25,
                    competition=0.15, market_share=0.25,
                    capital_requirements=80_000.0, capacity=900_000.0),
        Opportunity(id="obs_manufacturing", discovered=true, sector="manufacturing", complexity=0.70,
                    novelty_score=0.45, combination_uncertainty=0.35,
                    competition=0.10, market_share=0.20,
                    capital_requirements=120_000.0, capacity=1_200_000.0),
    ]
    traits = Dict{String,Float64}(
        "competence" => 0.62,
        "ai_trust" => 0.58,
        "exploration_tendency" => 0.47,
        "innovativeness" => 0.51,
        "uncertainty_tolerance" => 0.53,
        "analytical_ability" => 0.61,
        "risk_tolerance" => 0.49,
   )
    sector_knowledge = Dict("tech" => 0.55, "service" => 0.45, "manufacturing" => 0.35)
    action_history = ["invest", "explore", "maintain", "innovate", "invest"]

    perceive_for(tier) = GlimpseABM.perceive_uncertainty(
        KnightianUncertaintyEnvironment(cfg; rng=MersenneTwister(926)),
        traits,
        visible,
        mc;
        ai_level=tier,
        agent_knowledge=Set(["tech", "service"]),
        sector_knowledge=sector_knowledge,
        action_history=action_history,
        agent_id=1,
        recent_outcomes=Dict{String,Any}[],
   )

    basic = perceive_for("basic")
    premium = perceive_for("premium")

    basic_levels = (
        basic.actor_ignorance.level,
        basic.practical_indeterminism.level,
        basic.agentic_novelty.level,
        basic.competitive_recursion.level,
        basic.decision_confidence,
        basic.knowledge_signal.info_quality,
        basic.knowledge_signal.info_breadth,
   )
    premium_levels = (
        premium.actor_ignorance.level,
        premium.practical_indeterminism.level,
        premium.agentic_novelty.level,
        premium.competitive_recursion.level,
        premium.decision_confidence,
        premium.knowledge_signal.info_quality,
        premium.knowledge_signal.info_breadth,
   )

    for (basic_value, premium_value) in zip(basic_levels, premium_levels)
        @test isapprox(basic_value, premium_value; atol=1e-12)
    end

    sparse = GlimpseABM.perceive_uncertainty(
        KnightianUncertaintyEnvironment(cfg; rng=MersenneTwister(927)),
        traits,
        visible[1:1],
        mc;
        ai_level="premium",
        agent_knowledge=Set(["tech", "service"]),
        sector_knowledge=sector_knowledge,
        action_history=action_history,
        agent_id=1,
        recent_outcomes=Dict{String,Any}[],
   )
    rich = GlimpseABM.perceive_uncertainty(
        KnightianUncertaintyEnvironment(cfg; rng=MersenneTwister(928)),
        traits,
        visible,
        mc;
        ai_level="premium",
        agent_knowledge=Set(["tech", "service"]),
        sector_knowledge=sector_knowledge,
        action_history=action_history,
        agent_id=1,
        recent_outcomes=Dict{String,Any}[],
   )

    @test rich.knowledge_signal.info_breadth > sparse.knowledge_signal.info_breadth
    @test rich.actor_ignorance.level < sparse.actor_ignorance.level
end

@testset "Production decisions use agent-local perception buffers" begin
    cfg = EmergentConfig(N_AGENTS=4, N_ROUNDS=1, RANDOM_SEED=919)
    GlimpseABM.initialize!(cfg)
    sim = EmergentSimulation(config=cfg, seed=919)
    agent = sim.agents[1]
    mc = GlimpseABM.get_market_conditions(
        sim.market;
        uncertainty_state=GlimpseABM.get_uncertainty_state(sim.uncertainty_env),
   )
    opps = GlimpseABM.get_opportunities_for_agent(sim.market, agent)
    isempty(opps) && (opps = GlimpseABM.get_available_opportunities(sim.market))

    @test !haskey(sim.uncertainty_env._agent_short_term_buffers, agent.id)

    GlimpseABM.make_decision!(
        agent,
        opps,
        mc,
        sim.market,
        1;
        uncertainty_env=sim.uncertainty_env,
        info_system=sim.info_system,
        innovation_engine=sim.innovation_engine,
   )

    @test haskey(sim.uncertainty_env._agent_short_term_buffers, agent.id)
    agent_buffers = sim.uncertainty_env._agent_short_term_buffers[agent.id]
    @test all(!isempty(get(agent_buffers, metric, Float64[])) for metric in
              ("actor_ignorance", "practical_indeterminism", "agentic_novelty", "competitive_recursion"))
end

@testset "Recent outcomes enter production decision confidence" begin
    cfg = EmergentConfig(N_AGENTS=4, N_ROUNDS=1, RANDOM_SEED=920)
    GlimpseABM.initialize!(cfg)
    market = MarketEnvironment(cfg; rng=MersenneTwister(920))
    env = KnightianUncertaintyEnvironment(cfg; rng=MersenneTwister(921))
    info_system = InformationSystem(cfg)
    mc = GlimpseABM.get_market_conditions(
        market;
        uncertainty_state=GlimpseABM.get_uncertainty_state(env),
   )

    neutral = EmergentAgent(1, cfg; primary_sector="tech", initial_capital=5_000_000.0,
                            fixed_ai_level="none", rng=MersenneTwister(922))
    burned = EmergentAgent(2, cfg; primary_sector="tech", initial_capital=5_000_000.0,
                           fixed_ai_level="none", rng=MersenneTwister(922))
    burned.traits = copy(neutral.traits)
    burned.resources.knowledge = copy(neutral.resources.knowledge)

    @test hasfield(typeof(burned), :recent_outcomes)
    push!(burned.recent_outcomes, Dict{String,Any}(
        "action" => "invest",
        "success" => false,
        "investment_amount" => 100_000.0,
        "capital_returned" => 20_000.0,
        "ai_used" => false,
   ))

    opps = GlimpseABM.get_available_opportunities(market)
    neutral_outcome = GlimpseABM.make_decision!(
        neutral,
        opps,
        mc,
        market,
        1;
        uncertainty_env=env,
        info_system=info_system,
   )
    burned_outcome = GlimpseABM.make_decision!(
        burned,
        opps,
        mc,
        market,
        1;
        uncertainty_env=env,
        info_system=info_system,
   )

    @test burned_outcome["perception"].decision_confidence <
          neutral_outcome["perception"].decision_confidence
end

@testset "Recent outcome buffer records observable production outcomes" begin
    cfg = EmergentConfig(N_AGENTS=3, N_ROUNDS=2, RANDOM_SEED=921, RECENT_OUTCOME_MEMORY=2)
    GlimpseABM.initialize!(cfg)
    market = MarketEnvironment(cfg; rng=MersenneTwister(921))
    mc = GlimpseABM.get_market_conditions(market)
    agent = EmergentAgent(1, cfg; primary_sector="tech", initial_capital=5_000_000.0,
                          fixed_ai_level="none", rng=MersenneTwister(922))
    opp = Opportunity(
        id="recent-buffer-matured-opp",
        latent_return_potential=1.2,
        latent_failure_potential=0.1,
        discovered=true,
        sector="tech",
        time_to_maturity=1,
        capacity=1_000_000.0,
        config=cfg,
        rng=MersenneTwister(923),
   )

    push!(agent.active_investments, Dict{String,Any}(
        "opportunity" => opp,
        "amount" => 100_000.0,
        "maturity_round" => 1,
        "ai_level" => "none",
        "ai_label" => "none",
        "estimated_return" => 1.1,
        "decision_confidence" => 0.66,
   ))

    matured = GlimpseABM.process_matured_investments!(agent, market, 1; market_conditions=mc)
    @test length(matured) == 1
    @test length(agent.recent_outcomes) == 1
    @test agent.recent_outcomes[end]["action"] == "invest"
    @test haskey(agent.recent_outcomes[end], "return_multiple")

    GlimpseABM.execute_action!(agent, "explore", market, 1)
    @test length(agent.recent_outcomes) == 2
    @test agent.recent_outcomes[end]["action"] == "explore"

    GlimpseABM.execute_action!(agent, "innovate", market, 2; market_conditions=mc)
    @test length(agent.recent_outcomes) == 2
    @test [o["action"] for o in agent.recent_outcomes] == ["explore", "innovate"]
end

@testset "Typed Perception decision confidence is read for telemetry" begin
    cfg = EmergentConfig(N_AGENTS=4, N_ROUNDS=1, RANDOM_SEED=921)
    GlimpseABM.initialize!(cfg)
    market = MarketEnvironment(cfg; rng=MersenneTwister(923))
    env = KnightianUncertaintyEnvironment(cfg; rng=MersenneTwister(924))
    mc = GlimpseABM.get_market_conditions(market)
    agent = EmergentAgent(1, cfg; primary_sector="tech", initial_capital=5_000_000.0,
                          fixed_ai_level="premium", rng=MersenneTwister(925))
    perception = GlimpseABM.perceive_uncertainty(
        env,
        agent.traits,
        market.opportunities,
        mc;
        ai_level="premium",
        agent_knowledge=Set(keys(agent.resources.knowledge)),
        sector_knowledge=agent.resources.knowledge,
        action_history=agent.action_history,
        ai_learning_profile=agent.ai_learning,
        agent_id=agent.id,
   )

    action = Dict{String,Any}(
        "perception" => perception,
        "ai_confidence" => 0.01,
   )

    @test GlimpseABM._action_decision_confidence(action) == perception.decision_confidence
end

println("Perception regression tests passed.")
