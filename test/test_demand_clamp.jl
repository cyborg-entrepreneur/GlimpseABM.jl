# Regression: demand adjustment clamp.
#
# Earlier, get_demand_adjustments could produce return_penalty=15.67 and
# failure_pressure=-9.36 under crowded sectors due to unchecked multiplicative
# compounding of crowd_excess, crowd_relief, and clearing_ratio terms. The
# negative failure then got clamp-saved to 0.05 at models.jl:224, making
# oversubscribed sectors functionally immortal — an obvious physics violation.
#
# This test installs a crowded-sector state into the market, calls
# get_demand_adjustments directly, and asserts both outputs fall in a sane
# positive range.

using Test
using Random

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using GlimpseABM

@testset "Demand adjustment sane bounds" begin
    cfg = EmergentConfig(N_AGENTS=50, N_ROUNDS=5, RANDOM_SEED=42)
    GlimpseABM.initialize!(cfg)
    market = MarketEnvironment(cfg; rng=MersenneTwister(42))

    # Force an extreme crowding + extreme clearing ratio state. This exactly
    # reproduces the audit probe conditions.
    market.crowding_metrics["share_invest"] = 0.95   # far above threshold
    market.sector_clearing_index["tech"] = 10.0      # extreme hot market

    # First clear the cache so get_demand_adjustments recomputes
    empty!(market.sector_demand_adjustments)

    adj = GlimpseABM.get_demand_adjustments(market, "tech")
    ret = adj["return"]
    fail = adj["failure"]

    # Both must be finite, positive, and within sane economic bounds. Earlier,
    # the unclamped compound formulas produced return=15.67 and failure=-9.36
    # in this exact scenario.
    @test isfinite(ret)
    @test isfinite(fail)
    @test ret > 0.0     # returns can't be negative
    @test fail > 0.0    # failure pressure can't be negative
    @test ret <= 3.0    # clamp ceiling
    @test fail <= 3.0   # clamp ceiling

    # Cold-market counterpart: low crowding, low clearing ratio.
    empty!(market.sector_demand_adjustments)
    market.crowding_metrics["share_invest"] = 0.10
    market.sector_clearing_index["tech"] = 0.10

    adj2 = GlimpseABM.get_demand_adjustments(market, "tech")
    @test isfinite(adj2["return"])
    @test isfinite(adj2["failure"])
    @test adj2["return"] > 0.0
    @test adj2["failure"] > 0.0
    @test adj2["return"] <= 3.0
    @test adj2["failure"] <= 3.0
end

println("Demand clamp test passed.")
