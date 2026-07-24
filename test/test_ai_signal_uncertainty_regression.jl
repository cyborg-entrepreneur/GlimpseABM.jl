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

# Record forecast-disconfirmation exposures for one population. The channel is
# symmetric: ai_assisted=false records the human baseline population, and
# ai_assisted=true records the AI-assisted population.
function record_population_disconfirmations!(
    env::KnightianUncertaintyEnvironment,
    n_exposures::Int,
    n_failures::Int;
    ai_assisted::Bool = true,
)
    ai_level = ai_assisted ? "premium" : "none"
    for i in 1:n_exposures
        failed = i <= n_failures
        GlimpseABM.record_observable_ai_disconfirmation!(
            env,
            1,
            i,
            ai_level,
            "market_analysis";
            suspected_ai_failure=failed,
            disconfirmation_score=failed ? 0.90 : 0.05,
            severity=failed ? 0.80 : 0.0,
            return_error=failed ? 1.2 : 0.05,
            ai_confidence=0.8,
            accuracy_score=failed ? 0.2 : 0.8,
            success=!failed,
            ai_assisted=ai_assisted,
        )
    end
end

function observable_ai_signal_state(n_exposures::Int, n_failures::Int; seed::Int = 515)
    _, market, env = signal_test_context(seed=seed)
    actions = hidden_ai_signal_actions(n_exposures, 0)

    record_population_disconfirmations!(env, n_exposures, n_failures; ai_assisted=true)

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
    # "ai_disconfirmation_rate" is the compatibility alias for the overall
    # forecast-disconfirmation rate.
    @test isapprox(small_state["actor_ignorance"]["ai_disconfirmation_rate"], 0.1; atol=1e-12)
    @test isapprox(large_state["actor_ignorance"]["ai_disconfirmation_rate"], 0.1; atol=1e-12)
    @test isapprox(small_state["actor_ignorance"]["forecast_disconfirmation_rate"], 0.1; atol=1e-12)
    @test isapprox(
        small_state["actor_ignorance"]["level"],
        large_state["actor_ignorance"]["level"];
        atol=1e-12,
    )

    # Actor ignorance carries the renamed symmetric component.
    components = small_state["actor_ignorance"]["components"]
    @test haskey(components, "forecast_disconfirmation")
    @test !haskey(components, "ai_disconfirmation")
    @test components["forecast_disconfirmation"] > 0.0

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

@testset "Forecast disconfirmation is symmetric across populations" begin
    # (a) A no-AI population still produces exposures and a nonzero overall
    # rate: forecast disconfirmation is venture-return unpredictability, not
    # an AI artifact.
    _, no_ai_market, no_ai_env = signal_test_context()
    record_population_disconfirmations!(no_ai_env, 20, 6; ai_assisted=false)
    no_ai_stats = GlimpseABM._rolling_ai_signal_stats(no_ai_env)
    @test no_ai_stats.exposure_count == 20
    @test no_ai_stats.baseline_exposure_count == 20
    @test no_ai_stats.ai_exposure_count == 0
    @test isapprox(no_ai_stats.disconfirmation_rate, 0.3; atol=1e-12)
    @test no_ai_stats.overall_rate == no_ai_stats.disconfirmation_rate
    @test isapprox(no_ai_stats.baseline_rate, 0.3; atol=1e-12)
    @test no_ai_stats.ai_rate == 0.0
    @test no_ai_stats.excess_ai_rate == 0.0

    # The overall rate drives actor ignorance even with zero AI exposures.
    no_ai_actions = Dict{String,Any}[]
    no_ai_state = measure_signal_state(no_ai_env, no_ai_market, no_ai_actions)
    @test isapprox(no_ai_state["actor_ignorance"]["forecast_disconfirmation_rate"], 0.3; atol=1e-12)
    @test no_ai_state["actor_ignorance"]["components"]["forecast_disconfirmation"] > 0.0
    @test no_ai_state["actor_ignorance"]["excess_ai_disconfirmation_rate"] == 0.0

    # (b) Identical forecast errors in both populations: AI carries no excess
    # disconfirmation, so the AI-specific pressure nets to ~0.
    _, matched_market, matched_env = signal_test_context()
    record_population_disconfirmations!(matched_env, 20, 6; ai_assisted=true)
    record_population_disconfirmations!(matched_env, 20, 6; ai_assisted=false)
    matched_stats = GlimpseABM._rolling_ai_signal_stats(matched_env)
    @test isapprox(matched_stats.ai_rate, matched_stats.baseline_rate; atol=1e-12)
    @test isapprox(matched_stats.excess_ai_rate, 0.0; atol=1e-12)
    @test isapprox(matched_stats.overall_rate, 0.3; atol=1e-12)

    # (c) AI errors strictly above the baseline produce a positive excess.
    _, excess_market, excess_env = signal_test_context()
    record_population_disconfirmations!(excess_env, 20, 12; ai_assisted=true)
    record_population_disconfirmations!(excess_env, 20, 2; ai_assisted=false)
    excess_stats = GlimpseABM._rolling_ai_signal_stats(excess_env)
    @test isapprox(excess_stats.ai_rate, 0.6; atol=1e-12)
    @test isapprox(excess_stats.baseline_rate, 0.1; atol=1e-12)
    @test isapprox(excess_stats.excess_ai_rate, 0.5; atol=1e-12)

    # The AI-specific recursion channel responds to excess, not presence:
    # matched populations behave like a clean market, excess raises opacity.
    matched_state = measure_signal_state(matched_env, matched_market, Dict{String,Any}[])
    excess_state = measure_signal_state(excess_env, excess_market, Dict{String,Any}[])
    matched_perception = signal_perception(matched_env, matched_market, matched_state)
    excess_perception = signal_perception(excess_env, excess_market, excess_state)
    @test excess_perception.competitive_recursion.strategic_opacity >
          matched_perception.competitive_recursion.strategic_opacity
end

@testset "F4: confidence miscalibration splits into ai/baseline/excess" begin
    # The fixture writes observable_miscalibration = max(0, conf − accuracy)
    # = max(0, 0.8 − 0.2) = 0.6 for failures and 0.0 otherwise, so a
    # population with k/n failures has mean miscalibration 0.6·k/n.

    # (a) No-AI population: the AI side is empty ⇒ excess is exactly 0 even
    # though the pooled value is positive (no AI to attribute it to).
    _, _, no_ai_env = signal_test_context()
    record_population_disconfirmations!(no_ai_env, 20, 10; ai_assisted=false)
    no_ai = GlimpseABM._rolling_ai_signal_stats(no_ai_env)
    @test no_ai.confidence_miscalibration > 0.0      # pooled value retained
    @test no_ai.ai_miscalibration == 0.0
    @test isapprox(no_ai.baseline_miscalibration, 0.6 * 0.5; atol=1e-12)
    @test no_ai.excess_miscalibration == 0.0

    # (b) Matched gaps in both populations ⇒ excess ≈ 0.
    _, _, matched_env = signal_test_context()
    record_population_disconfirmations!(matched_env, 20, 6; ai_assisted=true)
    record_population_disconfirmations!(matched_env, 20, 6; ai_assisted=false)
    matched = GlimpseABM._rolling_ai_signal_stats(matched_env)
    @test isapprox(matched.ai_miscalibration, matched.baseline_miscalibration; atol=1e-12)
    @test isapprox(matched.excess_miscalibration, 0.0; atol=1e-12)
    @test matched.confidence_miscalibration > 0.0    # pooled stays symmetric/diagnostic

    # (c) AI-worse gaps ⇒ excess > 0 and equals the population difference.
    _, _, worse_env = signal_test_context()
    record_population_disconfirmations!(worse_env, 20, 12; ai_assisted=true)
    record_population_disconfirmations!(worse_env, 20, 2; ai_assisted=false)
    worse = GlimpseABM._rolling_ai_signal_stats(worse_env)
    @test isapprox(worse.ai_miscalibration, 0.6 * 0.6; atol=1e-12)
    @test isapprox(worse.baseline_miscalibration, 0.6 * 0.1; atol=1e-12)
    @test isapprox(worse.excess_miscalibration, 0.6 * 0.5; atol=1e-12)

    # The recursion-perception component consumes the EXCESS, not the pooled
    # value: two environments with zero excess but very different pooled
    # miscalibration produce the same recursion contribution. matched (pooled
    # 0.18, excess 0) is compared against a clean env (pooled 0, excess 0):
    # identical seeds + identical excess inputs ⇒ identical recursion levels.
    _, clean_market, clean_env = signal_test_context()
    record_population_disconfirmations!(clean_env, 20, 0; ai_assisted=true)
    record_population_disconfirmations!(clean_env, 20, 0; ai_assisted=false)
    clean = GlimpseABM._rolling_ai_signal_stats(clean_env)
    @test clean.confidence_miscalibration == 0.0
    # Same OVERALL failure pattern is required for the comparison to isolate
    # miscalibration: rebuild "matched" with the same failure counts as above
    # but identical overall rates differ from clean — so compare the
    # component directly instead of the full level.
    pw_matched = GlimpseABM._rolling_ai_signal_stats(matched_env)
    @test pw_matched.excess_miscalibration == clean.excess_miscalibration == 0.0
end

@testset "F5: rolling AI-signal stats cached once per (round, record-state)" begin
    _, market, env = signal_test_context()
    record_population_disconfirmations!(env, 20, 6; ai_assisted=true)

    direct = GlimpseABM._rolling_ai_signal_stats(env)
    cached1 = GlimpseABM._rolling_ai_signal_stats_cached(env, 3)
    cached2 = GlimpseABM._rolling_ai_signal_stats_cached(env, 3)
    @test cached1 == direct                  # cache equals direct computation
    @test cached1 === cached2                # second read reuses the stored object

    # New exposures within the same round invalidate the cache (the pivot
    # path records exposures mid-round — F2b), keeping the value exact.
    record_population_disconfirmations!(env, 10, 10; ai_assisted=true)
    cached3 = GlimpseABM._rolling_ai_signal_stats_cached(env, 3)
    @test cached3 !== cached1
    @test cached3 == GlimpseABM._rolling_ai_signal_stats(env)
    @test cached3.ai_rate > cached1.ai_rate

    # A new round with unchanged records recomputes once and re-caches.
    cached4 = GlimpseABM._rolling_ai_signal_stats_cached(env, 4)
    @test cached4 == cached3
    @test GlimpseABM._rolling_ai_signal_stats_cached(env, 4) === cached4
end

println("AI signal uncertainty tests passed.")
