using Test
using Random

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using GlimpseABM
include("test_helpers.jl")

function derivative_investment_outcome(tier::String, seed::Int)::Bool
    config = EmergentConfig()
    GlimpseABM.initialize!(config)
    market = MarketEnvironment(config; rng=MersenneTwister(700 + seed))
    agent = EmergentAgent(
        1,
        config;
        initial_capital=1_000_000.0,
        primary_sector="tech",
        fixed_ai_level=tier,
        rng=MersenneTwister(seed),
    )
    opp = Opportunity(
        id="same_market_opportunity",
        latent_return_potential=1.4,
        latent_failure_potential=0.25,
        complexity=0.4,
        discovered=true,
        competition=1.0,
        path_dependency=0.2,
        novelty_score=0.1,
        sector="tech",
        capital_requirements=10_000.0,
        config=config,
        rng=MersenneTwister(900 + seed),
    )
    outcome = GlimpseABM.execute_action!(
        agent,
        "invest",
        market,
        1;
        opportunity=opp,
        estimated_return=1.2,
    )
    @test outcome["success"] === true
    return Bool(outcome["invested_derivative"])
end

function perceived_novelty_with_derivative_rate(derivative_rate::Float64)::Float64
    config = EmergentConfig()
    GlimpseABM.initialize!(config)
    env = KnightianUncertaintyEnvironment(config; rng=MersenneTwister(41))

    base_drivers = Dict{String,Any}(
        "combo_rate" => 0.25,
        "reuse_pressure" => 0.20,
        "new_possibility_rate" => 0.10,
        "adoption_rate" => 0.20,
        "derivative_adoption_rate" => 0.20,
    )
    env.agentic_novelty_state["novelty_level"] = 0.55
    env.agentic_novelty_state["new_possibility_rate"] = 0.10
    env.agentic_novelty_state["new_possibilities"] = 3
    env.agentic_novelty_state["innovation_births"] = 2
    env.agentic_novelty_state["scarcity_signal"] = 0.50
    env.agentic_novelty_state["recent_scarcity_signal"] = 0.50
    env.agentic_novelty_state["drivers"] = base_drivers
    env.agentic_novelty_state["tier_drivers"] = Dict{String,Any}(
        "combo_rate" => Dict("none" => 0.25),
        "reuse_pressure" => Dict("none" => 0.20),
        "new_possibility_rate" => Dict("none" => 0.10),
        "adoption_rate" => Dict("none" => derivative_rate),
        "derivative_adoption_rate" => Dict("none" => derivative_rate),
    )

    traits = Dict(
        "competence" => 0.5,
        "ai_trust" => 0.5,
        "exploration_tendency" => 0.5,
        "innovativeness" => 0.5,
        "uncertainty_tolerance" => 0.5,
        "analytical_ability" => 0.5,
        "risk_tolerance" => 0.5,
    )

    perception = GlimpseABM.perceive_uncertainty(
        env,
        traits,
        Opportunity[],
        test_market_conditions(volatility=0.1, round=1);
        ai_level="none",
        agent_id=1,
    )
    return perception.agentic_novelty.level
end

@testset "Derivative investment labels are not tier-only AI effects" begin
    mismatches = count(
        seed -> derivative_investment_outcome("none", seed) != derivative_investment_outcome("premium", seed),
        1:100,
    )
    @test mismatches == 0
end

@testset "Derivative adoption lowers tier-specific perceived novelty" begin
    low_derivative = perceived_novelty_with_derivative_rate(0.0)
    high_derivative = perceived_novelty_with_derivative_rate(1.0)

    @test high_derivative < low_derivative
end

println("Derivative-novelty tests passed.")
