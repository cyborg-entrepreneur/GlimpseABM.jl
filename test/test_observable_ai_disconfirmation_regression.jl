using Test
using Random

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using GlimpseABM

function disconfirmation_fixture(;
    hidden_hallucination::Bool,
    estimated_return::Float64,
    capital_returned::Float64,
    ai_confidence::Float64,
    recent_accuracy::Vector{Float64} = Float64[],
    analytical_ability::Float64 = 0.5,
)
    config = EmergentConfig(N_AGENTS=1, N_ROUNDS=1, RANDOM_SEED=919)
    config.enable_round_logging = false
    GlimpseABM.initialize!(config)

    sim = EmergentSimulation(config=config, seed=919)
    agent = sim.agents[1]
    agent.fixed_ai_level = "premium"
    agent.current_ai_level = "premium"
    agent.traits["analytical_ability"] = analytical_ability
    agent.resources.knowledge["retail"] = 0.8
    agent.resources.knowledge["tech"] = 0.8
    agent.resources.capabilities["uncertainty_management"] = 0.8
    agent.resources.capabilities["opportunity_evaluation"] = 0.8
    agent.ai_learning.accuracy_estimates["market_analysis"] = copy(recent_accuracy)

    investment_amount = 100.0
    matured = Dict{String,Any}(
        "ai_level" => "premium",
        "ai_used" => true,
        "investment_amount" => investment_amount,
        "capital_returned" => capital_returned,
        "return_multiple" => capital_returned / investment_amount,
        "estimated_return" => estimated_return,
        "ai_confidence" => ai_confidence,
        "ai_analysis_domain" => "market_analysis",
        "success" => capital_returned > investment_amount,
        "ai_contains_hallucination" => hidden_hallucination,
    )

    result = GlimpseABM._apply_matured_ai_learning!(sim, agent, matured)
    return result, agent, sim
end

@testset "Hidden hallucination flags do not drive agent learning" begin
    false_result, false_agent, false_sim = disconfirmation_fixture(
        hidden_hallucination=false,
        estimated_return=2.0,
        capital_returned=20.0,
        ai_confidence=0.95,
    )
    true_result, true_agent, true_sim = disconfirmation_fixture(
        hidden_hallucination=true,
        estimated_return=2.0,
        capital_returned=20.0,
        ai_confidence=0.95,
    )

    @test false_result.hidden_hallucination === false
    @test true_result.hidden_hallucination === true
    @test false_result.suspected_ai_failure == true_result.suspected_ai_failure
    @test isapprox(false_result.disconfirmation_score, true_result.disconfirmation_score; atol=1e-12)
    @test false_agent.ai_learning.hallucination_experiences ==
          true_agent.ai_learning.hallucination_experiences
    @test false_agent.ai_learning.domain_trust == true_agent.ai_learning.domain_trust
    @test false_agent.resources.knowledge == true_agent.resources.knowledge
    @test false_agent.resources.capabilities == true_agent.resources.capabilities

    false_events = false_sim.uncertainty_env.ai_uncertainty_signals["observable_disconfirmation_events"]
    true_events = true_sim.uncertainty_env.ai_uncertainty_signals["observable_disconfirmation_events"]
    @test length(false_events) == 1
    @test length(true_events) == 1
    @test false_events[1]["suspected_ai_failure"] == true_events[1]["suspected_ai_failure"]
    @test false_events[1]["severity"] == true_events[1]["severity"]
end

@testset "Observable high-confidence misses trigger AI disconfirmation" begin
    result, agent, sim = disconfirmation_fixture(
        hidden_hallucination=false,
        estimated_return=2.0,
        capital_returned=20.0,
        ai_confidence=0.95,
    )

    @test result.suspected_ai_failure
    @test agent.ai_learning.hallucination_experiences["market_analysis"] == 1
    @test agent.ai_learning.domain_trust["market_analysis"] < 0.5
    @test agent.resources.knowledge["retail"] < 0.8

    events = sim.uncertainty_env.ai_uncertainty_signals["observable_disconfirmation_events"]
    @test length(events) == 1
    @test events[1]["suspected_ai_failure"] === true
    @test events[1]["severity"] > 0.0
end

@testset "Hidden hallucination flags without observable miss stay diagnostic" begin
    false_result, false_agent, false_sim = disconfirmation_fixture(
        hidden_hallucination=false,
        estimated_return=1.0,
        capital_returned=98.0,
        ai_confidence=0.95,
    )
    true_result, true_agent, true_sim = disconfirmation_fixture(
        hidden_hallucination=true,
        estimated_return=1.0,
        capital_returned=98.0,
        ai_confidence=0.95,
    )

    @test !false_result.suspected_ai_failure
    @test !true_result.suspected_ai_failure
    @test false_agent.ai_learning.hallucination_experiences["market_analysis"] == 0
    @test true_agent.ai_learning.hallucination_experiences["market_analysis"] == 0
    @test false_agent.ai_learning.domain_trust == true_agent.ai_learning.domain_trust
    @test false_agent.resources.knowledge == true_agent.resources.knowledge

    false_events = false_sim.uncertainty_env.ai_uncertainty_signals["observable_disconfirmation_events"]
    true_events = true_sim.uncertainty_env.ai_uncertainty_signals["observable_disconfirmation_events"]
    @test false_events[1]["suspected_ai_failure"] === false
    @test true_events[1]["suspected_ai_failure"] === false
end

@testset "Recent domain misses make moderate disconfirmation stronger" begin
    clean_result, _, _ = disconfirmation_fixture(
        hidden_hallucination=false,
        estimated_return=1.6,
        capital_returned=80.0,
        ai_confidence=0.95,
        recent_accuracy=Float64[],
    )
    repeated_miss_result, repeated_agent, _ = disconfirmation_fixture(
        hidden_hallucination=false,
        estimated_return=1.6,
        capital_returned=80.0,
        ai_confidence=0.95,
        recent_accuracy=[0.0, 0.0, 0.1, 0.0, 0.1],
    )

    @test repeated_miss_result.disconfirmation_score > clean_result.disconfirmation_score
    @test !clean_result.suspected_ai_failure
    @test repeated_miss_result.suspected_ai_failure
    @test repeated_agent.ai_learning.hallucination_experiences["market_analysis"] == 1
end

println("Observable AI disconfirmation regression tests passed.")
