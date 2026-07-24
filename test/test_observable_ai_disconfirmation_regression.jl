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
    ai_level::String = "premium",
)
    config = EmergentConfig(N_AGENTS=1, N_ROUNDS=1, RANDOM_SEED=919)
    config.enable_round_logging = false
    GlimpseABM.initialize!(config)

    sim = EmergentSimulation(config=config, seed=919)
    agent = sim.agents[1]
    agent.fixed_ai_level = ai_level
    agent.current_ai_level = ai_level
    agent.traits["analytical_ability"] = analytical_ability
    agent.resources.knowledge["retail"] = 0.8
    agent.resources.knowledge["tech"] = 0.8
    agent.resources.capabilities["uncertainty_management"] = 0.8
    agent.resources.capabilities["opportunity_evaluation"] = 0.8
    agent.ai_learning.accuracy_estimates["market_analysis"] = copy(recent_accuracy)

    investment_amount = 100.0
    matured = Dict{String,Any}(
        "ai_level" => ai_level,
        "ai_used" => ai_level != "none",
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

@testset "AI-assisted matured outcomes are flagged in the recorded event" begin
    _, _, sim = disconfirmation_fixture(
        hidden_hallucination=false,
        estimated_return=2.0,
        capital_returned=20.0,
        ai_confidence=0.95,
    )
    events = sim.uncertainty_env.ai_uncertainty_signals["observable_disconfirmation_events"]
    @test length(events) == 1
    @test events[1]["ai_assisted"] === true
end

# ── F1: raw instrument estimate drives scoring ─────────────────────────────

function f1_fixture(matured_extra::Dict{String,Any}; ai_level::String = "premium")
    config = EmergentConfig(N_AGENTS=1, N_ROUNDS=1, RANDOM_SEED=919)
    config.enable_round_logging = false
    GlimpseABM.initialize!(config)
    sim = EmergentSimulation(config=config, seed=919)
    agent = sim.agents[1]
    agent.fixed_ai_level = ai_level
    agent.current_ai_level = ai_level
    matured = merge(Dict{String,Any}(
        "ai_level" => ai_level,
        "ai_used" => ai_level != "none",
        "investment_amount" => 100.0,
        "capital_returned" => 20.0,
        "return_multiple" => 0.2,
        "ai_confidence" => 0.95,
        "ai_analysis_domain" => "market_analysis",
        "success" => false,
        "ai_contains_hallucination" => false,
    ), matured_extra)
    result = GlimpseABM._apply_matured_ai_learning!(sim, agent, matured)
    return result, agent, sim
end

@testset "F1: disconfirmation is scored against the RAW instrument estimate" begin
    # Same raw instrument estimate, different (shaded) decision estimates ⇒
    # identical scoring: the decision-basis value is ignored by the scorer.
    shaded_a, _, _ = f1_fixture(Dict{String,Any}(
        "estimated_return" => 1.2, "instrument_estimated_return" => 2.0,
        "has_return_estimate" => true))
    shaded_b, _, _ = f1_fixture(Dict{String,Any}(
        "estimated_return" => 0.7, "instrument_estimated_return" => 2.0,
        "has_return_estimate" => true))
    @test shaded_a.disconfirmation_score == shaded_b.disconfirmation_score
    @test shaded_a.suspected_ai_failure == shaded_b.suspected_ai_failure

    # Different raw estimates DO change the score (the raw key is live, not
    # decorative).
    raw_low, _, _ = f1_fixture(Dict{String,Any}(
        "estimated_return" => 1.2, "instrument_estimated_return" => 0.2,
        "has_return_estimate" => true))
    @test raw_low.disconfirmation_score != shaded_a.disconfirmation_score

    # Records without the raw key fall back to estimated_return.
    fallback, _, fallback_sim = f1_fixture(Dict{String,Any}("estimated_return" => 2.0))
    @test fallback.disconfirmation_score == shaded_a.disconfirmation_score
    @test length(fallback_sim.uncertainty_env.ai_uncertainty_signals["observable_disconfirmation_events"]) == 1
end

@testset "F1: outcomes with NO estimate are excluded (no perfect-forecast dilution)" begin
    # has_return_estimate=false (production marker for an investment that
    # never stored an estimate): no exposure recorded and no AI-trust learning.
    result, agent, sim = f1_fixture(Dict{String,Any}(
        "estimated_return" => 0.2,
        "has_return_estimate" => false))
    @test result.applied === false
    @test result.was_accurate === nothing
    @test result.suspected_ai_failure === false
    @test isempty(sim.uncertainty_env.ai_uncertainty_signals["observable_disconfirmation_events"])
    @test isempty(get(agent.ai_learning.accuracy_estimates, "market_analysis", Float64[]))

    # The maturity path itself produces the marker: an investment without any
    # estimate keys matures into has_return_estimate=false and contributes NO
    # estimation-error observation to the agent's uncertainty metrics.
    config = EmergentConfig(N_AGENTS=1, N_ROUNDS=1, RANDOM_SEED=920)
    config.enable_round_logging = false
    GlimpseABM.initialize!(config)
    sim2 = EmergentSimulation(config=config, seed=920)
    agent2 = sim2.agents[1]
    opp = Opportunity(id="f1_no_estimate", latent_return_potential=1.2,
                      latent_failure_potential=0.4, complexity=0.4,
                      sector="tech", competition=0.0, discovered=true)
    push!(agent2.active_investments, Dict{String,Any}(
        "opportunity_id" => opp.id,
        "opportunity" => opp,
        "amount" => 1000.0,
        "round_invested" => 0,
        "maturity_round" => 1,
        "ai_level" => "premium",
        "ai_label" => "premium",
        "sector" => opp.sector,
    ))
    errs_before = length(agent2.uncertainty_metrics.estimation_errors)
    matured = GlimpseABM.process_matured_investments!(agent2, sim2.market, 1)
    @test length(matured) == 1
    @test matured[1]["has_return_estimate"] === false
    @test length(agent2.uncertainty_metrics.estimation_errors) == errs_before
    # Returns/competition tracking (estimate-independent) still recorded.
    @test !isempty(agent2.uncertainty_metrics.return_history)

    # With an estimate the maturity path emits the raw key + marker and the
    # estimation error IS recorded against the raw value.
    sim3 = EmergentSimulation(config=config, seed=921)
    agent3 = sim3.agents[1]
    opp3 = Opportunity(id="f1_with_estimate", latent_return_potential=1.2,
                       latent_failure_potential=0.4, complexity=0.4,
                       sector="tech", competition=0.0, discovered=true)
    push!(agent3.active_investments, Dict{String,Any}(
        "opportunity_id" => opp3.id,
        "opportunity" => opp3,
        "amount" => 1000.0,
        "round_invested" => 0,
        "maturity_round" => 1,
        "ai_level" => "premium",
        "ai_label" => "premium",
        "sector" => opp3.sector,
        "estimated_return" => 0.9,                 # decision basis (shaded)
        "instrument_estimated_return" => 1.6,      # raw instrument value
    ))
    matured3 = GlimpseABM.process_matured_investments!(agent3, sim3.market, 1)
    @test matured3[1]["has_return_estimate"] === true
    @test matured3[1]["instrument_estimated_return"] == 1.6
    @test matured3[1]["estimated_return"] == 0.9   # decision basis preserved
    err = agent3.uncertainty_metrics.estimation_errors[end]
    actual = matured3[1]["return_multiple"]
    @test err ≈ abs(1.6 - actual) / max(0.01, abs(actual))  # raw, not shaded
end

@testset "F1: _execute_invest! persists both the decision and instrument estimates" begin
    config = EmergentConfig(N_AGENTS=1, N_ROUNDS=1, RANDOM_SEED=922)
    config.enable_round_logging = false
    GlimpseABM.initialize!(config)
    sim = EmergentSimulation(config=config, seed=922)
    agent = sim.agents[1]
    opp = Opportunity(id="f1_invest", latent_return_potential=1.2,
                      latent_failure_potential=0.4, complexity=0.4,
                      sector="tech", competition=0.0, discovered=true)
    outcome = GlimpseABM.execute_action!(agent, "invest", sim.market, 1;
        opportunity=opp, estimated_return=0.8, instrument_estimated_return=1.7)
    @test outcome["success"] === true
    inv = agent.active_investments[end]
    @test inv["estimated_return"] == 0.8                  # decision basis (sizing/scoring)
    @test inv["instrument_estimated_return"] == 1.7       # raw instrument value
    @test outcome["instrument_estimated_return"] == 1.7
    # Without the kwarg the two coincide.
    opp2 = Opportunity(id="f1_invest2", latent_return_potential=1.2,
                       latent_failure_potential=0.4, complexity=0.4,
                       sector="tech", competition=0.0, discovered=true)
    GlimpseABM.execute_action!(agent, "invest", sim.market, 1;
        opportunity=opp2, estimated_return=0.8)
    inv2 = agent.active_investments[end]
    @test inv2["instrument_estimated_return"] == inv2["estimated_return"] == 0.8
end

@testset "No-AI matured outcomes feed the symmetric forecast-disconfirmation channel" begin
    # A matured outcome with a large forecast miss produces an exposure even
    # without AI. The event is flagged as the human baseline population and
    # AI-trust learning stays untouched.
    result, agent, sim = disconfirmation_fixture(
        hidden_hallucination=false,
        estimated_return=2.0,
        capital_returned=20.0,
        ai_confidence=0.5,
        ai_level="none",
    )

    @test result.applied === false
    @test result.was_accurate === nothing

    events = sim.uncertainty_env.ai_uncertainty_signals["observable_disconfirmation_events"]
    @test length(events) == 1
    @test events[1]["ai_assisted"] === false
    @test events[1]["ai_level"] == "none"
    @test events[1]["suspected_ai_failure"] === true
    @test events[1]["severity"] > 0.0

    # The exposure flows into the rolling stats as baseline, never as AI:
    # a no-AI population yields a nonzero OVERALL rate with zero excess.
    stats = GlimpseABM._rolling_ai_signal_stats(sim.uncertainty_env)
    @test stats.exposure_count == 1
    @test stats.baseline_exposure_count == 1
    @test stats.ai_exposure_count == 0
    @test stats.disconfirmation_rate > 0.0
    @test stats.overall_rate == stats.disconfirmation_rate
    @test stats.baseline_rate > 0.0
    @test stats.ai_rate == 0.0
    @test stats.excess_ai_rate == 0.0

    # No AI-trust learning or hallucination-penalty mechanics for no-AI agents.
    @test agent.ai_learning.hallucination_experiences["market_analysis"] == 0
    @test agent.ai_learning.domain_trust["market_analysis"] == 0.5
    @test agent.resources.knowledge["retail"] == 0.8
end

println("Observable AI disconfirmation tests passed.")
