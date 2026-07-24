"""
Empirical tier-balance test.

The paper's causal claim rests on random assignment of AI tier being
orthogonal to agent endowments (capital, traits, sector). Orthogonality is
structural: `balanced_tier_assignments` shuffles a fixed tier vector using
only agent index, never agent state. This test pins that property empirically.

Design: construct real N=1000 simulations (primary-design scale) for several
seeds, apply the balanced fixed-tier assignment, and check that per-tier means
of initial capital and core traits match the pooled means within a small
multiple of the standard error. Seeds are fixed, so the test is deterministic.

Wrapped in a module so the shared helper's `const AI_TIERS` stays isolated.
"""
module TestTierBalance

using Test
using Random
using Statistics
using GlimpseABM

include(joinpath(@__DIR__, "..", "scripts", "_fixed_tier_assignment.jl"))

@testset "Tier assignment is empirically balanced on endowments" begin
    seeds = [20260426, 20260427, 20260428]
    n_agents = 1000

    # Pooled per-tier samples across seeds: measure => tier => values
    measures = ["capital", "uncertainty_tolerance", "innovativeness", "competence"]
    samples = Dict(m => Dict(t => Float64[] for t in AI_TIERS) for m in measures)

    for seed in seeds
        config = EmergentConfig(
            N_AGENTS=n_agents,
            N_ROUNDS=5,
            RANDOM_SEED=seed,
            AGENT_AI_MODE="fixed",
        )
        initial_dist = Dict(t => 0.25 for t in AI_TIERS)
        sim = GlimpseABM.EmergentSimulation(
            config=config,
            initial_tier_distribution=initial_dist,
            seed=seed,
            output_dir=mktempdir(),
        )
        assignments = balanced_tier_assignments(n_agents, MersenneTwister(seed))
        apply_balanced_fixed_tiers!(sim, assignments)

        for agent in sim.agents
            tier = agent.fixed_ai_level
            # Pre-run, capital IS initial capital (no dynamics have run).
            push!(samples["capital"][tier], GlimpseABM.get_capital(agent))
            push!(samples["uncertainty_tolerance"][tier], agent.uncertainty_tolerance)
            push!(samples["innovativeness"][tier], agent.innovativeness)
            push!(samples["competence"][tier], agent.competence)
        end
    end

    n_per_tier = length(seeds) * (n_agents ÷ length(AI_TIERS))  # 750

    for m in measures
        pooled = vcat((samples[m][t] for t in AI_TIERS)...)
        pooled_mean = mean(pooled)
        pooled_sd = std(pooled)
        se = pooled_sd / sqrt(n_per_tier)
        for t in AI_TIERS
            @test length(samples[m][t]) == n_per_tier
            # |tier mean - pooled mean| within 4 standard errors. Deterministic
            # given the fixed seeds; 4 SE leaves comfortable headroom while
            # catching any systematic tier-endowment coupling.
            @test abs(mean(samples[m][t]) - pooled_mean) <= 4.0 * se
        end
    end
end

end # module
