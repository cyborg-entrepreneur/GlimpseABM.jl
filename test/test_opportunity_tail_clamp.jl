# test/test_opportunity_tail_clamp.jl
#
# Guards the opportunity-value tail controls added 2026-06-14:
# RETURN_RANGE_MAX_MULT — scales the per-sector return_range[2] draw ceiling (market.jl:159).
# This is THE binding cap on opportunity returns.
# RETURN_CLAMP_MAX — global latent_return ceiling (market.jl:201,1491,1541,1616).
# LOG_SIGMA_CAP — ceiling on the lognormal log_sigma (market.jl:155).
#
# Key fact this test pins: at the defaults, the binding cap is the per-SECTOR ceiling
# (max return_range[2] = 4.0x, tech), NOT the 25x global clamp — so the 25x clamp never
# fires at baseline. Raising RETURN_RANGE_MAX_MULT is what releases the venture-realistic
# right tail (the unicorn-tail baseline). Defaults are byte-identical to the prior
# hardcoded clamps.

using Test
using Random

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using GlimpseABM

@testset "Opportunity tail controls" begin
 # Defaults reproduce the prior hardcoded clamps exactly (byte-identity guard).
    cfg0 = EmergentConfig()
    @test cfg0.RETURN_RANGE_MAX_MULT == 1.0
    @test cfg0.RETURN_CLAMP_MAX == 25.0
    @test cfg0.LOG_SIGMA_MULT == 1.0
    @test cfg0.LOG_SIGMA_CAP == 1.0

    function initial_returns(; mult=1.0, clampmax=25.0, sigmamult=1.0, sigmacap=1.0, seed=7, n=2000)
        config = EmergentConfig()
        config.N_AGENTS = n
        config.RETURN_RANGE_MAX_MULT = mult
        config.RETURN_CLAMP_MAX = clampmax
        config.LOG_SIGMA_MULT = sigmamult
        config.LOG_SIGMA_CAP = sigmacap
        GlimpseABM.initialize!(config)
        market = MarketEnvironment(config; rng=MersenneTwister(seed))
        return [o.latent_return_potential for o in market.opportunities]
    end

 # At defaults the per-sector ceiling binds (max return_range[2] = 4.0, tech), well below
 # the 25x global clamp — confirming the 25x clamp is not the binding cap at baseline.
    rets_default = initial_returns()
    @test maximum(rets_default) <= 4.0 + 1e-6
    @test minimum(rets_default) >= 0.5 - 1e-6

 # Raising RETURN_RANGE_MAX_MULT releases the tail PAST the old sector ceiling — proves the
 # multiplier is wired at the binding site (market.jl:159), not the dead 25x global clamp.
    rets_heavy = initial_returns(mult=25.0, clampmax=100.0, sigmacap=2.0)
    @test maximum(rets_heavy) > 4.0
    @test maximum(rets_heavy) <= 100.0 + 1e-6

 # LOG_SIGMA_MULT is the tail-heaviness driver: at high ceilings, scaling sigma
 # produces a materially heavier right tail than raising the ceiling lever alone
 # (raw sector sigma ~0.32-0.45 is already below the cap, so the cap is inert).
    rets_sigma = initial_returns(mult=250.0, clampmax=2000.0, sigmamult=4.0, sigmacap=3.0)
    @test maximum(rets_sigma) > maximum(rets_heavy)

 # Determinism: identical config + seed -> identical draws.
    @test initial_returns(seed=11) == initial_returns(seed=11)
end
