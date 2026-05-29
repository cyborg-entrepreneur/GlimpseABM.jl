using Test
using Random

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using GlimpseABM

function hidden_ai_signal_actions(n_exposures::Int, n_hallucinations::Int)
    return [
        Dict{String,Any}(
            "action" => "maintain",
            "agent_id" => i,
            "round" => 1,
            "ai_level_used" => "premium",
            "ai_behavior_level" => "premium",
            "ai_used" => true,
            "ai_contains_hallucination" => i <= n_hallucinations,
            "ai_confidence" => 0.8,
            "ai_actual_accuracy" => i <= n_hallucinations ? 0.2 : 0.8,
            "ai_analysis_domain" => "market_analysis",
        )
        for i in 1:n_exposures
    ]
end

function signal_test_context(; seed::Int = 515)
    config = EmergentConfig()
    config.N_AGENTS = 100
    config.N_ROUNDS = 1
    config.enable_round_logging = false
    GlimpseABM.initialize!(config)

    market = MarketEnvironment(config; rng=MersenneTwister(seed))
    env = KnightianUncertaintyEnvironment(config; rng=MersenneTwister(seed + 1))
    return config, market, env
end

function measure_signal_state(
    env::KnightianUncertaintyEnvironment,
    market::MarketEnvironment,
    actions::Vector{Dict{String,Any}},
)
    GlimpseABM.record_ai_signals!(env, 1, actions)
    state = GlimpseABM.measure_uncertainty_state!(env, market, actions, Innovation[], 1)
    return state
end

function hidden_ai_signal_state(n_exposures::Int, n_hallucinations::Int; seed::Int = 515)
    _, market, env = signal_test_context(seed=seed)
    actions = hidden_ai_signal_actions(n_exposures, n_hallucinations)
    state = measure_signal_state(env, market, actions)
    return env, market, state
end

function observable_ai_signal_state(n_exposures::Int, n_failures::Int; seed::Int = 515)
    _, market, env = signal_test_context(seed=seed)
    actions = hidden_ai_signal_actions(n_exposures, 0)

    for i in 1:n_exposures
        failed = i <= n_failures
        GlimpseABM.record_observable_ai_disconfirmation!(
            env,
            1,
            i,
            "premium",
            "market_analysis";
            suspected_ai_failure=failed,
            disconfirmation_score=failed ? 0.90 : 0.05,
            severity=failed ? 0.80 : 0.0,
            return_error=failed ? 1.2 : 0.05,
            ai_confidence=0.8,
            accuracy_score=failed ? 0.2 : 0.8,
            success=!failed,
        )
    end

    state = measure_signal_state(env, market, actions)
    return env, market, state
end

function signal_perception(
    env::KnightianUncertaintyEnvironment,
    market::MarketEnvironment,
    state::Dict{String,Dict{String,Any}},
)
    uncertainty_snapshot = Dict{String,Any}(k => v for (k, v) in state)
    conditions = GlimpseABM.get_market_conditions(market; uncertainty_state=uncertainty_snapshot)
    traits = Dict{String,Float64}(
        "competence" => 0.5,
        "ai_trust" => 0.5,
        "exploration_tendency" => 0.5,
        "innovativeness" => 0.5,
        "uncertainty_tolerance" => 0.5,
        "analytical_ability" => 0.5,
        "risk_tolerance" => 0.5,
    )
    return GlimpseABM.perceive_uncertainty(
        env,
        traits,
        GlimpseABM.get_available_opportunities(market),
        conditions;
        ai_level="premium",
    )
end

@testset "Hidden AI error telemetry is diagnostic only" begin
    clean_env, clean_market, clean_state = hidden_ai_signal_state(20, 0)
    hidden_env, hidden_market, hidden_state = hidden_ai_signal_state(20, 20)

    @test clean_state["actor_ignorance"]["hidden_hallucination_rate"] == 0.0
    @test hidden_state["actor_ignorance"]["hidden_hallucination_rate"] == 1.0
    @test clean_state["actor_ignorance"]["ai_disconfirmation_rate"] == 0.0
    @test hidden_state["actor_ignorance"]["ai_disconfirmation_rate"] == 0.0
    @test isapprox(
        clean_state["actor_ignorance"]["level"],
        hidden_state["actor_ignorance"]["level"];
        atol=1e-12,
    )

    clean_perception = signal_perception(clean_env, clean_market, clean_state)
    hidden_perception = signal_perception(hidden_env, hidden_market, hidden_state)
    @test isapprox(
        clean_perception.competitive_recursion.strategic_opacity,
        hidden_perception.competitive_recursion.strategic_opacity;
        atol=1e-12,
    )
end

@testset "Observable AI disconfirmation is exposure-normalized" begin
    _, _, small_state = observable_ai_signal_state(10, 1)
    _, _, large_state = observable_ai_signal_state(100, 10)

    @test small_state["actor_ignorance"]["ai_signal_exposures"] == 10
    @test large_state["actor_ignorance"]["ai_signal_exposures"] == 100
    @test isapprox(small_state["actor_ignorance"]["ai_disconfirmation_rate"], 0.1; atol=1e-12)
    @test isapprox(large_state["actor_ignorance"]["ai_disconfirmation_rate"], 0.1; atol=1e-12)
    @test isapprox(
        small_state["actor_ignorance"]["level"],
        large_state["actor_ignorance"]["level"];
        atol=1e-12,
    )

    low_env, low_market, low_state = observable_ai_signal_state(100, 10)
    high_env, high_market, high_state = observable_ai_signal_state(10, 10)

    @test high_state["actor_ignorance"]["ai_disconfirmation_rate"] >
          low_state["actor_ignorance"]["ai_disconfirmation_rate"]
    @test high_state["actor_ignorance"]["level"] >
          low_state["actor_ignorance"]["level"]

    small_perception = signal_perception(low_env, low_market, low_state)
    large_perception = signal_perception(high_env, high_market, high_state)
    @test large_perception.competitive_recursion.strategic_opacity >
          small_perception.competitive_recursion.strategic_opacity
end

println("AI signal uncertainty regression tests passed.")
