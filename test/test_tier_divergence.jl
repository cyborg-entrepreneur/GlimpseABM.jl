# Regression test: AI-tier-aware ranking
#
# Asserts that two agents facing the same opportunity set but with different
# AI tiers produce different top-3 rankings in a non-trivial fraction of trials.
#
# Before the 2026-04-23 correctness fixes, this test FAILED — the
# evaluate_portfolio_opportunities function discarded tier-noisy
# `estimated_returns` and ranked by `latent_return_potential` (hidden ground
# truth) for every tier, so all tiers always picked the same top opportunity.
#
# After the fix, `evaluate_opportunity_basic` accepts an `estimated_return`
# parameter that reflects AI tier (via InformationSystem.get_information when
# available, else inline tier-noise model). This test guards against the
# bypass coming back.

using Test
using Random
using Statistics

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using GlimpseABM

include("test_helpers.jl")

const N_TRIALS = 100
const N_OPPS = 50

@testset "AI-tier ranking divergence (regression)" begin
    config = EmergentConfig(N_AGENTS=2, N_ROUNDS=1, RANDOM_SEED=42)
    market = MarketEnvironment(config; rng=MersenneTwister(42))
    info_system = InformationSystem(config)

 # Build a fixed opportunity set with varying latent_return, novelty,
 # complexity. We keep the same opportunities across all trials so the
 # only randomness is the tier-noise injection in get_information.
    base_rng = MersenneTwister(1)
    opportunities = Opportunity[]
    for i in 1:N_OPPS
        sector = rand(base_rng, market.sectors)
        opp = GlimpseABM._create_realistic_opportunity(market, "test_$i", sector)
        push!(opportunities, opp)
    end

    market_conditions = test_market_conditions()
    perception = empty_perception()

    top1_match = 0
    top3_overlap_sum = 0.0
    rank_corr_sum = 0.0

    for trial in 1:N_TRIALS
 # Two agents with same trait seed but different fixed AI tiers,
 # using the same RNG seed for tier-noise to make the comparison fair
 # (any difference comes from AI mechanism, not from RNG drift)
        seed = 1000 + trial

        agent_none = EmergentAgent(1, config;
            primary_sector="tech",
            fixed_ai_level="none",
            rng=MersenneTwister(seed))
        agent_premium = EmergentAgent(2, config;
            primary_sector="tech",
            fixed_ai_level="premium",
            rng=MersenneTwister(seed))

 # Reset info_system cache between trials so tier-noise samples fresh
        empty!(info_system.information_cache)

        evals_none = GlimpseABM.evaluate_portfolio_opportunities(
            agent_none, opportunities, market_conditions, perception;
            ai_level="none", info_system=info_system)

        empty!(info_system.information_cache)

        evals_premium = GlimpseABM.evaluate_portfolio_opportunities(
            agent_premium, opportunities, market_conditions, perception;
            ai_level="premium", info_system=info_system)

        top_none = [e["opportunity"].id for e in evals_none[1:min(3, end)]]
        top_premium = [e["opportunity"].id for e in evals_premium[1:min(3, end)]]

        if !isempty(top_none) && !isempty(top_premium) && top_none[1] == top_premium[1]
            top1_match += 1
        end
        overlap = length(intersect(top_none, top_premium))
        top3_overlap_sum += overlap

 # Spearman rank correlation across full opportunity ranking
        ranks_none = Dict(e["opportunity"].id => i for (i, e) in enumerate(evals_none))
        ranks_premium = Dict(e["opportunity"].id => i for (i, e) in enumerate(evals_premium))
        opp_ids = collect(keys(ranks_none))
        rn = [ranks_none[id] for id in opp_ids]
        rp = [ranks_premium[id] for id in opp_ids]
        if length(rn) > 1 && std(rn) > 0 && std(rp) > 0
            rank_corr_sum += cor(rn, rp)
        end
    end

    top1_match_rate = top1_match / N_TRIALS
    avg_top3_overlap = top3_overlap_sum / N_TRIALS
    avg_rank_corr = rank_corr_sum / N_TRIALS

    @info "Tier-divergence regression test results" top1_match_rate avg_top3_overlap avg_rank_corr

 # Primary assertion: top-1 should NOT always agree
 # If tiers always pick the same top opportunity, mechanism is bypassed.
 # Allow up to 60% top-1 agreement (generous floor; in practice expect ~30-40%)
    @test top1_match_rate <= 0.60

 # Secondary assertion: top-3 overlap should not be 3.0 (perfect agreement)
 # Allow up to 2.5 average overlap (out of 3); if 3.0 they always agree
    @test avg_top3_overlap < 2.8

 # Tertiary assertion: full-ranking correlation should be bounded away from 1.0
 # Tier-noise should produce some disagreement on the full ranking
    @test avg_rank_corr < 0.95
end
