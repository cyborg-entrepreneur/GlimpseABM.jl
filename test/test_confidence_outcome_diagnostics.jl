using Test
using Random

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using GlimpseABM

@testset "Confidence-outcome diagnostics" begin
    cfg = EmergentConfig(N_AGENTS=1, N_ROUNDS=1, RANDOM_SEED=11)
    GlimpseABM.initialize!(cfg)
    agent = EmergentAgent(1, cfg; rng=MersenneTwister(11))

    GlimpseABM.record_confidence_outcome_observation!(
        agent, 0.80, 0.50; ai_used=true, channel="investment"
   )
    GlimpseABM.record_confidence_outcome_observation!(
        agent, 0.20, 2.00; ai_used=false, channel="innovate"
   )

    diag = agent.confidence_outcome_diagnostics
    @test diag.obs_count == 2
    @test diag.investment_obs_count == 1
    @test diag.innovate_explore_obs_count == 1
    @test diag.ai_obs_count == 1
    @test diag.non_ai_obs_count == 1
    @test diag.positive_gap_count == 1
    @test diag.negative_gap_count == 1
    @test diag.realized_multiples == [0.5, 2.0]

    stats = GlimpseABM.confidence_outcome_stats([agent])
    @test stats["confidence_outcome_observed_agents"] == 1
    @test stats["confidence_outcome_agent_coverage"] == 1.0
    @test stats["confidence_outcome_observations"] == 2
    @test stats["confidence_outcome_positive_gap_rate"] == 0.5
    @test stats["confidence_outcome_negative_gap_rate"] == 0.5
    @test stats["confidence_outcome_investment_observations"] == 1
    @test stats["confidence_outcome_innovate_explore_observations"] == 1
    @test stats["confidence_outcome_ai_observations"] == 1
    @test stats["confidence_outcome_non_ai_observations"] == 1
    @test stats["confidence_outcome_realized_multiple_mean"] == 1.25
    @test stats["confidence_outcome_realized_multiple_p50"] == 1.25
    @test stats["confidence_outcome_abs_gap_mean"] > 0.0

    snap = GlimpseABM.snapshot(agent, 1)
    @test haskey(snap, "confidence_outcome_weighted_gap")
    @test haskey(snap, "confidence_outcome_obs_count")
    @test !haskey(snap, "paradox_signal")
    @test !haskey(snap, "paradox_obs_count")
end

@testset "Simulation summary uses confidence-outcome naming" begin
    cfg = EmergentConfig(N_AGENTS=4, N_ROUNDS=2, RANDOM_SEED=12)
    cfg.enable_round_logging = false
    GlimpseABM.initialize!(cfg)
    sim = EmergentSimulation(config=cfg, seed=12, output_dir="/tmp/glimpse_confidence_outcome_test")
    GlimpseABM.run!(sim)

    stats = GlimpseABM.summary_stats(sim)
    @test haskey(stats, "confidence_outcome_raw_gap_mean")
    @test haskey(stats, "confidence_outcome_abs_gap_mean")
    @test haskey(stats, "confidence_outcome_positive_gap_rate")
    @test haskey(stats, "confidence_outcome_negative_gap_rate")
    @test haskey(stats, "confidence_outcome_realized_multiple_mean")
    @test !haskey(stats, "mean_paradox_signal")
    @test !haskey(stats, "mean_abs_paradox_signal")
    @test !haskey(stats, "paradox_coverage")
end
