# v3.0 regression: MarketConditions typed schema.
# Every production consumer of market_conditions accesses a typed field.
# Guards against re-introducing the silent-zero dataflow bugs fixed in v2.7,
# v2.9, and v2.12.

using Test
using Random

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using GlimpseABM

@testset "MarketConditions schema" begin
    cfg = EmergentConfig(N_AGENTS=20, N_ROUNDS=1, RANDOM_SEED=42)
    market = MarketEnvironment(cfg; rng=MersenneTwister(42))
    mc = GlimpseABM.get_market_conditions(market)

    # Type invariants
    @test mc isa MarketConditions
    @test mc.regime isa String
    @test mc.volatility isa Float64
    @test mc.regime_return_multiplier isa Float64
    @test mc.regime_failure_multiplier isa Float64
    @test mc.round isa Int
    @test mc.tier_invest_share isa Dict{String,Float64}
    @test mc.sector_clearing_index isa Dict{String,Float64}
    @test mc.aggregate_clearing_ratio isa Float64
    @test mc.crowding_metrics isa Dict{String,Float64}
    @test mc.sector_demand_adjustments isa Dict{String,Dict{String,Float64}}
    @test mc.avg_competition isa Float64
    @test mc.uncertainty_state isa Dict{String,Any}
    @test mc.extras isa Dict{String,Any}

    # Non-degenerate values in production path
    @test isfinite(mc.avg_competition)
    @test isfinite(mc.volatility)

    # Immutability — catches accidental post-construction mutation
    @test_throws ErrorException mc.regime = "crisis"

    # Dict-shim backward compat (for any straggling get(mc, "X", …) sites)
    @test get(mc, "regime", "missing") == mc.regime
    @test get(mc, "nonexistent_key", 42) == 42
    @test haskey(mc, "regime")
    @test !haskey(mc, "nonexistent_key")
    @test mc["regime"] == mc.regime

    # uncertainty_state can be injected at construction. v3.3.3: snapshot is
    # deepcopied, so the struct holds a separate object with equal contents —
    # not the same reference. Mutating `us` after construction must NOT
    # mutate the snapshot.
    us = Dict{String,Any}("actor_ignorance" => Dict{String,Any}("level"=>0.5))
    mc2 = GlimpseABM.get_market_conditions(market; uncertainty_state=us)
    @test mc2.uncertainty_state !== us          # separate object
    @test mc2.uncertainty_state == us           # equal contents at snapshot time
    us["actor_ignorance"]["level"] = 0.99
    @test mc2.uncertainty_state["actor_ignorance"]["level"] == 0.5  # isolated
    @test mc2.regime == mc.regime
end

println("MarketConditions schema test passed.")
