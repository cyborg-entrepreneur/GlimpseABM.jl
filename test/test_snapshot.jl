# v3.3.2 regression: MarketConditions is a true snapshot.
#
# Pre-v3.3.2 get_market_conditions passed market.tier_invest_share (and the
# other dict-valued fields) by reference. Mutating the market after the
# snapshot was taken mutated the snapshot too. Reviewer probe confirmed
# `same_dict_ref=true`. Now dicts are copy/deepcopy'd at construction.

using Test
using Random

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using GlimpseABM

@testset "MarketConditions snapshot isolation" begin
    cfg = EmergentConfig(N_AGENTS=20, N_ROUNDS=1, RANDOM_SEED=42)
    GlimpseABM.initialize!(cfg)
    market = MarketEnvironment(cfg; rng=MersenneTwister(42))

    market.tier_invest_share["premium"] = 0.25
    market.sector_clearing_index["tech"] = 1.5
    market.crowding_metrics["crowding_index"] = 0.3
    market.sector_demand_adjustments["tech"] = Dict{String,Float64}("return"=>1.1, "failure"=>0.95)

    mc = GlimpseABM.get_market_conditions(market)

    # Capture pre-mutation values
    tier_before  = mc.tier_invest_share["premium"]
    clear_before = mc.sector_clearing_index["tech"]
    crowd_before = mc.crowding_metrics["crowding_index"]
    sector_before = mc.sector_demand_adjustments["tech"]["return"]

    # Now mutate the LIVE market after the snapshot.
    market.tier_invest_share["premium"] = 0.77
    market.sector_clearing_index["tech"] = 99.0
    market.crowding_metrics["crowding_index"] = 0.99
    market.sector_demand_adjustments["tech"]["return"] = 2.0

    # Snapshot must be unchanged.
    @test mc.tier_invest_share["premium"] == tier_before
    @test mc.sector_clearing_index["tech"] == clear_before
    @test mc.crowding_metrics["crowding_index"] == crowd_before
    @test mc.sector_demand_adjustments["tech"]["return"] == sector_before

    # Direct identity check — separate objects, not same reference.
    @test mc.tier_invest_share !== market.tier_invest_share
    @test mc.sector_clearing_index !== market.sector_clearing_index
    @test mc.crowding_metrics !== market.crowding_metrics
    @test mc.sector_demand_adjustments !== market.sector_demand_adjustments
    # Nested inner dict also separate (deepcopy)
    @test mc.sector_demand_adjustments["tech"] !== market.sector_demand_adjustments["tech"]
end

println("MarketConditions snapshot test passed.")
