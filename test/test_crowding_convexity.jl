# Capital-saturation convexity crowding tests.
#
# The crowding penalty depends on capital saturation (total_invested /
# capacity), not on the count of competitors (opp.competition). Ten $10k
# investments should not penalize returns the same as one $10M investment.

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
    low_sat  = build_opp(0.3)    # well below canonical kappa=1.5
    high_sat = build_opp(2.5)    # well above K_sat

    # The shared helper is the single formula used by both payoff realization
    # and maturity telemetry. Lock its exact pure-I/K equation, then verify the
    # named robustness blend changes only effective load.
    market_conditions.crowding_metrics["crowding_index"] = 0.60
    pure_terms = GlimpseABM.capacity_crowding_terms(
        high_sat, market_conditions, cfg)
    pure_excess = max(0.0,
        2.5 / cfg.CROWDING_CAPACITY_RATIO_K - 1.0)
    pure_expected = exp(-cfg.CROWDING_STRENGTH_LAMBDA *
        pure_excess^cfg.CROWDING_CONVEXITY_GAMMA)
    @test pure_terms.effective_load == 2.5
    @test pure_terms.multiplier ≈ pure_expected

    blend_cfg = deepcopy(cfg)
    blend_cfg.CROWDING_INDEX_BLEND = 0.30
    blend_terms = GlimpseABM.capacity_crowding_terms(
        high_sat, market_conditions, blend_cfg)
    @test blend_terms.effective_load ≈ 2.5 + 0.30 * 0.60
    @test blend_terms.multiplier < pure_terms.multiplier

    low_returns  = [GlimpseABM.realized_return(low_sat,  market_conditions; rng=rng) for _ in 1:N]
    rng = MersenneTwister(42)
    high_returns = [GlimpseABM.realized_return(high_sat, market_conditions; rng=rng) for _ in 1:N]

    @test mean(low_returns) > mean(high_returns)
    # Magnitude check: high_sat should lose a meaningful chunk
    @test mean(high_returns) / mean(low_returns) < 0.85

    # ───────────────────────────────────────────────────────────────
    # Property 2: Count-invariance. Two opps with identical capital
    # saturation but wildly different competition counts produce
    # statistically indistinguishable returns.
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
    # Pure I/K canonical form: paired draws are exactly unchanged through the
    # threshold, then fall monotonically once the local capital load exceeds it.
    @test means[1] == means[2] == means[3]
    @test means[3] > means[4] > means[5]
    @test means[1] > means[end]
end

println("Capital-saturation crowding test passed.")
