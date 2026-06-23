# v3.2 regression: confidence × signal_score sizing.
#
# Before v3.2 Julia sized every bet as min(capital · max_fraction,
# opp.capital_requirements), regardless of how confident the agent was or
# how strongly the evaluation scored the opportunity. Python already
# multiplied by (confidence · signal_score). This test locks in the port:
# sizing scales with both signals.

using Test
using Random

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using GlimpseABM

@testset "Confidence/signal-scaled invest sizing" begin
    cfg = EmergentConfig(N_AGENTS=5, N_ROUNDS=1, RANDOM_SEED=42)
    GlimpseABM.initialize!(cfg)
    market = MarketEnvironment(cfg; rng=MersenneTwister(42))

    # Build an agent with known capital
    agent = EmergentAgent(1, cfg; rng=MersenneTwister(42))
    GlimpseABM.set_capital!(agent, 1_000_000.0)  # pin for predictable math

    opp = market.opportunities[1]
    opp.capital_requirements = 1.0  # so min_required doesn't override

    # ─────────────────────────────────────────────────────────────
    # Property 1: high confidence + high score invests more than low.
    # ─────────────────────────────────────────────────────────────
    # Reset capital between runs since the function deducts.
    function run_invest(confidence::Float64, signal_score::Float64)
        GlimpseABM.set_capital!(agent, 1_000_000.0)
        cap_before = GlimpseABM.get_capital(agent)
        outcome = GlimpseABM.execute_action!(
            agent, "invest", market, 1;
            opportunity=opp,
            confidence=confidence,
            signal_score=signal_score,
        )
        cap_after = GlimpseABM.get_capital(agent)
        return cap_before - cap_after
    end

    low_bet  = run_invest(0.2, 0.3)
    high_bet = run_invest(0.9, 1.5)

    @test high_bet > low_bet
    # With target_fraction 0.033, capital 1M:
    #   low  ≈ 1M · 0.033 · 0.2 · 0.3  = $1,980
    #   high ≈ 1M · 0.033 · 0.9 · 1.5  = $44,550 (but capped at max_fraction · capital = $37,000)
    # Expect roughly ≥10× spread between low and high.
    @test high_bet / low_bet >= 10.0

    # ─────────────────────────────────────────────────────────────
    # Property 2: legacy call (no confidence/signal) falls back to flat
    # sizing — preserves backward compat for tests and replay paths.
    # ─────────────────────────────────────────────────────────────
    GlimpseABM.set_capital!(agent, 1_000_000.0)
    cap_before = GlimpseABM.get_capital(agent)
    opp.capital_requirements = 10_000.0
    GlimpseABM.execute_action!(agent, "invest", market, 1; opportunity=opp)
    cap_after = GlimpseABM.get_capital(agent)
    flat_bet = cap_before - cap_after
    # Legacy path: min(capital · max_fraction, capital_requirements)
    # = min(37_000, 10_000) = 10_000
    @test isapprox(flat_bet, 10_000.0; atol=1.0)

    # ─────────────────────────────────────────────────────────────
    # Property 3: minimum funding floor. When desired < min_fraction *
    # requirements and the agent CAN afford min_required, bump up.
    # ─────────────────────────────────────────────────────────────
    opp.capital_requirements = 100_000.0  # min_required = 0.25 * 100k = 25k
    # confidence=0.15 · signal=0.1 → desired = 1M · 0.033 · 0.15 · 0.1 = $495
    # Well below min_required $25k, agent has $1M → floor should kick in.
    GlimpseABM.set_capital!(agent, 1_000_000.0)
    cap_before = GlimpseABM.get_capital(agent)
    GlimpseABM.execute_action!(agent, "invest", market, 1; opportunity=opp,
                                confidence=0.15, signal_score=0.1)
    floor_bet = cap_before - GlimpseABM.get_capital(agent)
    # Floor bumps to $25k (min_fraction * requirements), capped at max_invest $37k.
    @test floor_bet >= 25_000.0
    @test floor_bet <= 37_000.0 + 1.0
end

println("Confidence-sizing test passed.")
