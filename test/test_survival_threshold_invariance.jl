using Test
using Random

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using GlimpseABM

function _matched_agent_pair(config::EmergentConfig)
    none = EmergentAgent(1, config; primary_sector="tech", initial_capital=5_000_000.0,
                         fixed_ai_level="none", rng=MersenneTwister(1))
    premium = EmergentAgent(2, config; primary_sector="tech", initial_capital=5_000_000.0,
                            fixed_ai_level="premium", rng=MersenneTwister(2))
    none.traits["ai_trust"] = 1.0
    premium.traits["ai_trust"] = 1.0
    return none, premium
end

@testset "AI tier and trust do not change objective survival thresholds" begin
    config = EmergentConfig(N_AGENTS=2, N_ROUNDS=1, RANDOM_SEED=1)
    GlimpseABM.initialize!(config)
    config.INSOLVENCY_GRACE_ROUNDS = 1

    @testset "liquidity threshold is tier-invariant" begin
        none, premium = _matched_agent_pair(config)
        GlimpseABM.set_capital!(none, 1_800_000.0)
        GlimpseABM.set_capital!(premium, 1_800_000.0)

        @test GlimpseABM.check_survival!(none, 1) === false
        @test GlimpseABM.check_survival!(premium, 1) === false
        @test none.failure_reason == "liquidity_failure"
        @test premium.failure_reason == "liquidity_failure"
    end

    @testset "equity threshold is tier-invariant" begin
        none, premium = _matched_agent_pair(config)
        # Above tech liquidity threshold ($1.95M) but below the global equity
        # ratio floor: 0.40 * $5M = $2.0M.
        GlimpseABM.set_capital!(none, 1_975_000.0)
        GlimpseABM.set_capital!(premium, 1_975_000.0)

        @test GlimpseABM.check_survival!(none, 1) === false
        @test GlimpseABM.check_survival!(premium, 1) === false
        @test none.failure_reason == "equity_failure"
        @test premium.failure_reason == "equity_failure"
    end
end

println("Survival threshold invariance tests passed.")
