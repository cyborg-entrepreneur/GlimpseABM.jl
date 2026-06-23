"""
Regression tests for the shared fixed-tier assignment helpers
(scripts/_fixed_tier_assignment.jl).

These pin the PRIMARY paper design: mixed fixed-tier, N=1000 -> exactly
250 agents per tier, locked for the run. The helpers were consolidated
from byte-similar private copies in seven mixed-tier drivers on
2026-06-09; this file is the regression pin for the single-sourced
implementation.

Wrapped in a module so the shared file's `const AI_TIERS` and helper
function definitions stay isolated from other test files in Main.
"""
module TestFixedTierAssignment

using Test
using Random
using GlimpseABM

include(joinpath(@__DIR__, "..", "scripts", "_fixed_tier_assignment.jl"))

tier_counts(assignments) = Dict(t => count(==(t), assignments) for t in AI_TIERS)

@testset "Fixed-tier assignment (primary design)" begin

    @testset "N=1000 -> exactly 250 per tier" begin
        assignments = balanced_tier_assignments(1000, MersenneTwister(42))
        @test length(assignments) == 1000
        counts = tier_counts(assignments)
        for tier in AI_TIERS
            @test counts[tier] == 250
        end
    end

    @testset "N=1002 remainder round-robin in AI_TIERS order" begin
        assignments = balanced_tier_assignments(1002, MersenneTwister(42))
        @test length(assignments) == 1002
        counts = tier_counts(assignments)
        # Remainder of 2 goes to the first two tiers in AI_TIERS order.
        @test counts["none"] == 251
        @test counts["basic"] == 251
        @test counts["advanced"] == 250
        @test counts["premium"] == 250
    end

    @testset "Seeded shuffle: pairing invariant" begin
        a1 = balanced_tier_assignments(1000, MersenneTwister(20260425))
        a2 = balanced_tier_assignments(1000, MersenneTwister(20260425))
        # Same seed -> identical assignment vector, so conditions sharing a
        # seed sequence get identical agent->tier maps (within-seed pairing).
        @test a1 == a2
        a3 = balanced_tier_assignments(1000, MersenneTwister(20260426))
        @test a1 != a3
        # Different seed reshuffles but preserves exact counts.
        @test tier_counts(a1) == tier_counts(a3)
    end

    @testset "apply_balanced_fixed_tiers! on a real simulation" begin
        n_agents = 24
        seed = 4242
        # AI_COST_MODEL pinned to "hybrid": the subscription-hygiene guard
        # below is ABOUT seat-fee schedules, which only exist under the
        # hybrid/fixed backends (token-core migration 2026-06-11 made "token"
        # the struct default). The token-default case is asserted separately
        # at the end of this testset.
        config = EmergentConfig(
            N_AGENTS=n_agents,
            N_ROUNDS=5,
            RANDOM_SEED=seed,
            AGENT_AI_MODE="fixed",
            AI_COST_MODEL="hybrid",
        )
        initial_dist = Dict(t => 0.25 for t in AI_TIERS)
        sim = GlimpseABM.EmergentSimulation(
            config=config,
            initial_tier_distribution=initial_dist,
            seed=seed,
            output_dir=mktempdir(),
        )
        rng = MersenneTwister(seed)
        assignments = balanced_tier_assignments(n_agents, rng)
        apply_balanced_fixed_tiers!(sim, assignments)

        @test length(sim.agents) == n_agents
        counts = tier_counts(assignments)
        for tier in AI_TIERS
            @test counts[tier] == 6
        end

        for (i, agent) in enumerate(sim.agents)
            assigned = assignments[i]
            @test agent.fixed_ai_level == assigned
            @test agent.current_ai_level == assigned

            # Subscription hygiene (v2.9 billing bug guard): no agent may
            # hold a schedule for any tier other than its assignment.
            scheduled = collect(keys(agent.subscription_accounts))
            @test all(t -> t == assigned, scheduled)

            ai_config = sim.config.AI_LEVELS[assigned]
            if assigned == "none"
                @test isempty(agent.subscription_accounts)
            elseif ai_config.cost_type == "subscription" && ai_config.cost > 0
                # Subscription-priced tiers (advanced/premium under the
                # default hybrid cost model) carry exactly one schedule:
                # the assigned tier's.
                @test scheduled == [assigned]
                @test get(agent.subscription_accounts, assigned, 0) > 0
            else
                # Per-use tiers (basic) carry no subscription schedule;
                # ensure_subscription_schedule! is a deliberate no-op.
                @test isempty(agent.subscription_accounts)
            end
        end

        # Token-core default: under AI_COST_MODEL="token" (the struct
        # default), NO tier carries a subscription schedule — assignment
        # must not create seat-fee accounts that the backend will never bill.
        token_config = EmergentConfig(
            N_AGENTS=n_agents,
            N_ROUNDS=5,
            RANDOM_SEED=seed,
            AGENT_AI_MODE="fixed",
        )
        @test token_config.AI_COST_MODEL == "token"
        token_sim = GlimpseABM.EmergentSimulation(
            config=token_config,
            initial_tier_distribution=initial_dist,
            seed=seed,
            output_dir=mktempdir(),
        )
        apply_balanced_fixed_tiers!(token_sim, assignments)
        for agent in token_sim.agents
            @test isempty(agent.subscription_accounts)
        end
    end

end

end # module
