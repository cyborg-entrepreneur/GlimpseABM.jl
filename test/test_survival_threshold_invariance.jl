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

@testset "SURVIVAL_COUNTS_INFLIGHT: in-flight capital counts toward net worth" begin
    # initial_capital = $5M  ->  equity floor = 0.40 * $5M = $2M; tech liquidity
    # floor ~ $1.95M. Deployed-but-maturing (in-flight) capital should lift
    # effective net worth under the toggle, flipping the survival outcome.
    function _inflight_agent(; cash::Float64, inflight::Float64, counts_inflight::Bool)
        cfg = EmergentConfig(N_AGENTS=2, N_ROUNDS=1, RANDOM_SEED=1)
        GlimpseABM.initialize!(cfg)
        cfg.INSOLVENCY_GRACE_ROUNDS = 1
        cfg.SURVIVAL_COUNTS_INFLIGHT = counts_inflight
        a = EmergentAgent(1, cfg; primary_sector="tech", initial_capital=5_000_000.0,
                          fixed_ai_level="premium", rng=MersenneTwister(1))
        GlimpseABM.set_capital!(a, cash)
        if inflight > 0.0
            push!(a.active_investments, Dict{String,Any}("amount" => inflight))
        end
        return a
    end

    @testset "liquidity leg" begin
        # $1M cash is below the liquidity floor; $4M in-flight lifts net worth to $5M.
        @test GlimpseABM.check_survival!(
            _inflight_agent(cash=1_000_000.0, inflight=4_000_000.0, counts_inflight=true), 1) === true
        a_off = _inflight_agent(cash=1_000_000.0, inflight=4_000_000.0, counts_inflight=false)
        @test GlimpseABM.check_survival!(a_off, 1) === false
        @test a_off.failure_reason == "liquidity_failure"
    end

    @testset "equity leg" begin
        # $1.975M cash clears the liquidity floor but is below the $2M equity floor
        # (ratio 0.395 < 0.40); $1M in-flight lifts the ratio above the floor.
        @test GlimpseABM.check_survival!(
            _inflight_agent(cash=1_975_000.0, inflight=1_000_000.0, counts_inflight=true), 1) === true
        a_off = _inflight_agent(cash=1_975_000.0, inflight=1_000_000.0, counts_inflight=false)
        @test GlimpseABM.check_survival!(a_off, 1) === false
        @test a_off.failure_reason == "equity_failure"
    end
end

println("Survival threshold invariance tests passed.")
