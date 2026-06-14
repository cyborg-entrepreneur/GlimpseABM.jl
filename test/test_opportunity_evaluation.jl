using Test
using Random

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using GlimpseABM
include("test_helpers.jl")

@testset "OpportunityEvaluation struct schema + shim" begin
    opp = Opportunity(
        id="opp_test",
        latent_return_potential=1.8,
        latent_failure_potential=0.25,
        complexity=0.5,
   )

 # Default keyword constructor produces typed-zero AI fields when ai_info
 # is nothing (matches the dict producer that previously omitted those keys
 # entirely when no Information was available).
    ev = OpportunityEvaluation(;
        opportunity=opp,
        final_score=1.42,
        ai_level_used="basic",
        estimated_return=1.7,
   )

 # Type invariants
    @test ev isa OpportunityEvaluation
    @test ev.opportunity === opp
    @test ev.final_score == 1.42
    @test ev.ai_level_used == "basic"
    @test ev.estimated_return == 1.7
    @test ev.competition_at_evaluation == 0.0
    @test ev.ai_info === nothing
    @test ev.ai_contains_hallucination === false
    @test ev.ai_confidence == 0.0
    @test ev.ai_actual_accuracy == 0.0

 # Immutability — the struct must not be reassignable
    @test_throws ErrorException ev.final_score = 2.0

 # Dict-shim reads (Phase 1 of migration; some diagnostic tests still use
 # dict-style access)
    @test ev["opportunity"] === opp
    @test ev["final_score"] == 1.42
    @test get(ev, "final_score", 999.0) == 1.42
    @test get(ev, "ai_actual_accuracy", 0.5) == 0.0  # field exists → field wins
    @test get(ev, "nonexistent_key", 42) == 42
    @test haskey(ev, "final_score")
    @test haskey(ev, "ai_info")
    @test !haskey(ev, "nonexistent_key")
    @test_throws KeyError ev["nonexistent_key"]

 # With AI info populated, the AI fields propagate
    info = Information(
        estimated_return=1.65,
        estimated_uncertainty=0.18,
        confidence=0.72,
        actual_accuracy=0.81,
        contains_hallucination=true,
   )
    ev_ai = OpportunityEvaluation(;
        opportunity=opp,
        final_score=2.1,
        ai_level_used="premium",
        estimated_return=1.65,
        competition_at_evaluation=0.35,
        ai_info=info,
        ai_contains_hallucination=info.contains_hallucination,
        ai_confidence=Float64(info.confidence),
        ai_actual_accuracy=Float64(info.actual_accuracy),
   )
    @test ev_ai.ai_info === info
    @test ev_ai.ai_contains_hallucination === true
    @test ev_ai.ai_confidence == 0.72
    @test ev_ai.ai_actual_accuracy == 0.81
    @test ev_ai.competition_at_evaluation == 0.35
end

@testset "OpportunityEvaluation sort order matches pre-struct behavior" begin
 # Regression test for the sort! call inside evaluate_portfolio_opportunities:
 # the dict producer sorted descending by "final_score"; the struct producer
 # must sort descending by.final_score, producing the identical permutation.
    opps = [
        Opportunity(id="a", latent_return_potential=1.0, latent_failure_potential=0.1, complexity=0.3),
        Opportunity(id="b", latent_return_potential=1.0, latent_failure_potential=0.1, complexity=0.3),
        Opportunity(id="c", latent_return_potential=1.0, latent_failure_potential=0.1, complexity=0.3),
    ]
    scores = [0.5, 2.0, 1.0]
    evals = [
        OpportunityEvaluation(;
            opportunity=opps[i],
            final_score=scores[i],
            ai_level_used="none",
            estimated_return=1.0,
       )
        for i in 1:3
    ]

    sort!(evals, by=e -> e.final_score, rev=true)
    @test [e.opportunity.id for e in evals] == ["b", "c", "a"]
    @test [e.final_score for e in evals] == [2.0, 1.0, 0.5]
end

@testset "AI mechanism hooks fire on runtime paths" begin
    config = EmergentConfig()
    GlimpseABM.initialize!(config)
    config.AI_EXECUTION_SUCCESS_MULTIPLIERS["premium"] = 4.0
    config.AI_INFORMATION_QUALITY_BOOSTS["premium"] = 0.40
    config.AI_COMPLEMENTARITY_STRENGTH = 1.0
    config.AI_DIFFICULTY_COST_SCALING = 1.0
    config.AI_COST_MODEL = "fixed"
    config.STRATEGIC_ANTICIPATION_ENABLED = true
    config.STRATEGIC_ANTICIPATION_STRENGTH = 1.0

    opp = Opportunity(
        id="reviewer_probe",
        latent_return_potential=2.0,
        latent_failure_potential=0.55,
        complexity=0.8,
        sector="tech",
        competition=10.0,
        config=config,
   )
    market_conditions = test_market_conditions(volatility=0.1)

    baseline = GlimpseABM.realized_return(opp, market_conditions, "none"; rng=MersenneTwister(123))
    premium = GlimpseABM.realized_return(opp, market_conditions, "premium"; rng=MersenneTwister(123))
    @test premium > baseline

    info_system = InformationSystem(config)
    traits = Dict(
        "analytical_ability" => 0.9,
        "competence" => 0.9,
        "market_awareness" => 0.9,
        "ai_trust" => 0.9,
   )
    knowledge = Dict("tech" => 0.9)
    info = GlimpseABM.get_information(
        info_system,
        opp,
        "premium";
        agent_id=1,
        agent_traits=traits,
        sector_knowledge=knowledge,
        rng=MersenneTwister(321),
   )
    @test isapprox(info.hidden_factors["info_quality_used"], 0.99; atol=1e-12)
    @test info.hidden_factors["ai_complementarity_delta"] > 0.3

    agent = EmergentAgent(1, config; initial_capital=1_000_000.0, fixed_ai_level="none", rng=MersenneTwister(2))
    agent.traits["analytical_ability"] = 0.9
    agent.traits["uncertainty_tolerance"] = 0.5
    @test isapprox(
        GlimpseABM.ai_analysis_call_cost(agent, "basic"; opportunity=opp),
        config.AI_LEVELS["basic"].per_use_cost * (1.0 + opp.complexity),
   )
    @test isapprox(
        GlimpseABM.ai_analysis_call_cost(agent, "advanced"; opportunity=opp),
        config.AI_LEVELS["advanced"].per_use_cost * opp.complexity,
   )
    @test GlimpseABM.strategic_anticipation_multiplier(
        agent,
        opp,
        "premium",
        config.AI_LEVELS["premium"].info_quality,
        empty_perception(),
        5,
   ) < 0.8
    @test GlimpseABM.strategic_anticipation_multiplier(
        agent,
        opp,
        "none",
        0.25,
        empty_perception(),
        5,
   ) == 1.0

    @test GlimpseABM.behavior_ai_level(config, "premium") == "premium"
    @test GlimpseABM.counts_as_ai_use(config, "premium") === true
    config.SHAM_AI_LABELS_USE_NONE = true
    @test GlimpseABM.behavior_ai_level(config, "premium") == "none"
    @test GlimpseABM.counts_as_ai_use(config, "premium") === false
    config.SHAM_AI_LABELS_USE_NONE = false

    config.STRATEGIC_DIVERSIFICATION_ENABLED = true
    config.STRATEGIC_DIVERSIFICATION_STRENGTH = 10.0
    config.STRATEGIC_DIVERSIFICATION_TOP_K = 3
    config.STRATEGIC_DIVERSIFICATION_SCORE_BAND = 0.20
    crowded = Opportunity(id="crowded", latent_return_potential=2.0, latent_failure_potential=0.2,
        complexity=0.4, competition=10.0, config=config)
    uncrowded = Opportunity(id="uncrowded", latent_return_potential=1.9, latent_failure_potential=0.2,
        complexity=0.4, competition=0.0, config=config)
    evals = OpportunityEvaluation[
        OpportunityEvaluation(; opportunity=crowded, final_score=1.00, ai_level_used="premium",
            estimated_return=2.0, competition_at_evaluation=10.0),
        OpportunityEvaluation(; opportunity=uncrowded, final_score=0.98, ai_level_used="premium",
            estimated_return=1.9, competition_at_evaluation=0.0),
    ]
    sort!(evals, by=e -> e.final_score, rev=true)
    nonbest = 0
    for seed in 1:20
        agent.rng = MersenneTwister(seed)
        selected = GlimpseABM.select_portfolio_evaluation(agent, evals, "premium", empty_perception())
        nonbest += selected.opportunity.id != "crowded" ? 1 : 0
    end
    @test nonbest > 0
end

@testset "Matured investments use tier stored at commitment" begin
    config = EmergentConfig()
    GlimpseABM.initialize!(config)
    config.AI_EXECUTION_SUCCESS_MULTIPLIERS["premium"] = 4.0

    opp = Opportunity(
        id="maturity_probe",
        latent_return_potential=2.0,
        latent_failure_potential=0.55,
        complexity=0.6,
        sector="tech",
        config=config,
   )
    market_conditions = test_market_conditions(volatility=0.1)
    expected_multiple = GlimpseABM.realized_return(opp, market_conditions, "premium"; rng=MersenneTwister(42))

    agent = EmergentAgent(2, config; initial_capital=1_000_000.0, fixed_ai_level="none", rng=MersenneTwister(7))
    agent.rng = MersenneTwister(42)
    amount = 10_000.0
    before = GlimpseABM.get_capital(agent)
    push!(agent.active_investments, Dict{String,Any}(
        "amount" => amount,
        "maturity_round" => 1,
        "opportunity" => opp,
        "ai_level" => "premium",
        "estimated_return" => 1.5,
   ))
    market = MarketEnvironment(config; rng=MersenneTwister(10))

    outcomes = GlimpseABM.process_matured_investments!(
        agent,
        market,
        1;
        market_conditions=market_conditions,
   )

    @test length(outcomes) == 1
    @test outcomes[1]["ai_level"] == "premium"
    @test isapprox(GlimpseABM.get_capital(agent) - before, amount * expected_multiple)
end

println("OpportunityEvaluation regression tests passed.")
