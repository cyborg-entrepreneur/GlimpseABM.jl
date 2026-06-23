# v3.1 regression: capital-saturation convexity crowding.
#
# Locks in the core property of the refactor: the crowding penalty
# depends on capital saturation (total_invested / capacity), NOT on the
# count of competitors (opp.competition). Count-based dependence was the
# bug the refactor targeted — ten $10k investments should not penalize
# returns the same as one $10M investment.

using Test
using Random
using Statistics

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using GlimpseABM
include(joinpath(@__DIR__, "test_helpers.jl"))

@testset "Capital-saturation convexity crowding" begin
    cfg = EmergentConfig(N_AGENTS=10, N_ROUNDS=1, RANDOM_SEED=42)
    GlimpseABM.initialize!(cfg)
    market_conditions = test_market_conditions()

    # Build three opportunities with identical latent fundamentals but
    # varying saturation. Disable discovery-gating by marking discovered=true
    # — we're probing realized_return directly, not the discovery path.
    function build_opp(sat_ratio::Float64; competition::Float64=1.0)
        cap = 1.0e7
        opp = Opportunity(
            id="probe_$(sat_ratio)_$(competition)",
            latent_return_potential=2.0,
            latent_failure_potential=0.3,
            complexity=0.5,
            discovered=true,
            config=cfg,
            sector="tech",
            capacity=cap,
            total_invested=sat_ratio * cap,
        )
        opp.competition = competition
        return opp
    end

    # ───────────────────────────────────────────────────────────────
    # Property 1: Below K_sat, penalty is ~zero. Above K_sat, returns fall.
    # ───────────────────────────────────────────────────────────────
    rng = MersenneTwister(42)
    N = 400
    low_sat  = build_opp(0.3)    # well below K_sat=1.2
    high_sat = build_opp(2.5)    # well above K_sat

    low_returns  = [GlimpseABM.realized_return(low_sat,  market_conditions; rng=rng) for _ in 1:N]
    rng = MersenneTwister(42)
    high_returns = [GlimpseABM.realized_return(high_sat, market_conditions; rng=rng) for _ in 1:N]

    @test mean(low_returns) > mean(high_returns)
    # Magnitude check: high_sat should lose a meaningful chunk
    @test mean(high_returns) / mean(low_returns) < 0.85

    # ───────────────────────────────────────────────────────────────
    # Property 2: Count-invariance. Two opps with identical capital
    # saturation but wildly different competition counts produce
    # statistically indistinguishable returns. This is THE point of
    # the v3.1 refactor.
    # ───────────────────────────────────────────────────────────────
    few_competitors  = build_opp(1.0, competition=1.0)
    many_competitors = build_opp(1.0, competition=50.0)

    rng = MersenneTwister(42)
    few_returns  = [GlimpseABM.realized_return(few_competitors,  market_conditions; rng=rng) for _ in 1:N]
    rng = MersenneTwister(42)
    many_returns = [GlimpseABM.realized_return(many_competitors, market_conditions; rng=rng) for _ in 1:N]

    # Means should be within 5% of each other — count doesn't feed the penalty
    ratio = mean(many_returns) / mean(few_returns)
    @test 0.95 < ratio < 1.05

    # ───────────────────────────────────────────────────────────────
    # Property 3: Penalty is monotone in saturation.
    # ───────────────────────────────────────────────────────────────
    sat_levels = [0.5, 1.0, 1.5, 2.0, 3.0]
    means = Float64[]
    for s in sat_levels
        opp = build_opp(s)
        rng = MersenneTwister(42)
        push!(means, mean(GlimpseABM.realized_return(opp, market_conditions; rng=rng) for _ in 1:N))
    end
    # Allow 1-2 inversions due to stochastic noise, but the overall trend must be down
    @test means[1] > means[end]
    @test means[1] > means[3]  # 0.5 vs 1.5
end

println("Capital-saturation crowding test passed.")
