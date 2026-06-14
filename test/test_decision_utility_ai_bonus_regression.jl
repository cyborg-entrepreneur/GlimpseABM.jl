using Test
using Random

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using GlimpseABM
include("test_helpers.jl")

function decision_bonus_fixture()
    config = EmergentConfig()
    GlimpseABM.initialize!(config)
    agent = EmergentAgent(
        1,
        config;
        initial_capital=1_000_000.0,
        primary_sector="tech",
        fixed_ai_level="none",
        rng=MersenneTwister(11),
   )
    for key in keys(agent.traits)
        agent.traits[key] = 0.5
    end
    agent.competence = 0.5
    agent.innovativeness = 0.5
    return config, agent
end

@testset "AI information quality does not directly boost decision utility" begin
    config, agent = decision_bonus_fixture()
    market_conditions = test_market_conditions(volatility=0.2, round=1)
    perception = empty_perception()
    opportunities = Opportunity[
        Opportunity(
            id="fixed_input_probe",
            latent_return_potential=1.8,
            latent_failure_potential=0.2,
            complexity=0.4,
            sector="tech",
            config=config,
       ),
    ]

    none_invest = GlimpseABM.calculate_investment_utility(
        agent,
        opportunities,
        market_conditions,
        perception;
        ai_level="none",
        info_system=nothing,
   )
    premium_invest = GlimpseABM.calculate_investment_utility(
        agent,
        opportunities,
        market_conditions,
        perception;
        ai_level="premium",
        info_system=nothing,
   )
    @test isapprox(premium_invest, none_invest; atol=1e-12)

    none_innovate = GlimpseABM.calculate_innovation_utility(
        agent,
        market_conditions,
        perception;
        ai_level="none",
   )
    premium_innovate = GlimpseABM.calculate_innovation_utility(
        agent,
        market_conditions,
        perception;
        ai_level="premium",
   )
    @test isapprox(premium_innovate, none_innovate; atol=1e-12)
end

@testset "Tier labels do not directly discount fixed opportunity evidence" begin
    config, agent = decision_bonus_fixture()
    market_conditions = test_market_conditions(volatility=0.2, round=1)
    opportunity = Opportunity(
        id="fixed_evidence_probe",
        latent_return_potential=1.8,
        latent_failure_potential=0.2,
        complexity=0.4,
        sector="tech",
        config=config,
   )

    none_score = GlimpseABM.evaluate_opportunity_basic(
        agent,
        opportunity,
        market_conditions;
        estimated_return=1.8,
        estimated_uncertainty=0.25,
        confidence=0.8,
        ai_level="none",
   )
    premium_score = GlimpseABM.evaluate_opportunity_basic(
        agent,
        opportunity,
        market_conditions;
        estimated_return=1.8,
        estimated_uncertainty=0.25,
        confidence=0.8,
        ai_level="premium",
   )

    @test isapprox(premium_score, none_score; atol=1e-12)
end

@testset "Legacy no-engine innovation fallback has no direct AI novelty boost" begin
    config = EmergentConfig()
    GlimpseABM.initialize!(config)
    config.INNOVATION_PROBABILITY = 1.0
    market = MarketEnvironment(config; rng=MersenneTwister(10))

    function fallback_outcome(tier::String)
        agent = EmergentAgent(
            1,
            config;
            initial_capital=1_000_000.0,
            primary_sector="tech",
            fixed_ai_level=tier,
            rng=MersenneTwister(48),
       )
        agent.competence = 1.0
        agent.innovativeness = 1.0
        return GlimpseABM.execute_action!(
            agent,
            "innovate",
            market,
            1;
            innovation_engine=nothing,
       )
    end

    none_outcome = fallback_outcome("none")
    premium_outcome = fallback_outcome("premium")

    @test none_outcome["success"] == premium_outcome["success"] == true
    @test none_outcome["is_new_combination"] == premium_outcome["is_new_combination"]
end

println("Decision-utility AI bonus regression tests passed.")
