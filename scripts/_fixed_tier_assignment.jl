"""
Shared fixed-tier assignment helpers for mixed fixed-tier analysis scripts.

PRIMARY PAPER DESIGN: mixed fixed-tier, N=1000 -> exactly 250 agents per
tier (none/basic/advanced/premium), locked for the whole run
(AGENT_AI_MODE="fixed" + agent.fixed_ai_level). Emergent adoption is a
robustness design only.

v3.5.21: extracted from run_reviewer_robustness_suite.jl (following the
_safe_stats.jl shared-include pattern) so the suite and the six sibling
mixed-tier drivers all share one regression-pinned implementation.
Previously each driver carried a byte-similar private copy named
`create_tier_assignments` plus an inline tier-override loop; those copies
were verified algorithmically identical before consolidation
.

Regression coverage: test/test_fixed_tier_assignment.jl.
"""

using Random
using GlimpseABM

const AI_TIERS = ["none", "basic", "advanced", "premium"]

"""
    balanced_tier_assignments(n_agents::Int, rng::AbstractRNG) -> Vector{String}

PRIMARY-design tier assignment: exact integer counts, not multinomial
sampling. Each tier in `AI_TIERS` receives exactly `n_agents ÷ 4` agents;
any remainder is distributed round-robin in `AI_TIERS` order (so N=1002
yields none=251, basic=251, advanced=250, premium=250). The vector is then
shuffled with the caller-supplied seeded `rng`, so the same seed produces
the same assignment vector — the pairing invariant that lets conditions
sharing a seed sequence be compared within-seed.

For the headline N=1000 design this is exactly 250 agents per tier.
"""
function balanced_tier_assignments(n_agents::Int, rng::AbstractRNG)
    assignments = String[]
    per_tier = n_agents ÷ length(AI_TIERS)
    for tier in AI_TIERS
        append!(assignments, fill(tier, per_tier))
    end
    remainder = n_agents - length(assignments)
    for i in 1:remainder
        push!(assignments, AI_TIERS[mod1(i, length(AI_TIERS))])
    end
    shuffle!(rng, assignments)
    return assignments
end

"""
    apply_balanced_fixed_tiers!(sim::EmergentSimulation, assignments::Vector{String})

Override each agent's tier after simulation construction, with full
subscription hygiene.

The constructor samples tiers from `initial_tier_distribution` and starts a
subscription schedule for each agent's originally-sampled tier. Setting
`fixed_ai_level` without resetting subscriptions corrupts billing — the
documented v2.9 bug where an agent pays for tier X while using tier Y. So
for every agent we (1) set both `fixed_ai_level` and `current_ai_level` to
the assigned tier, (2) cancel ALL existing subscription schedules, and
(3) ensure a schedule for the new tier (a no-op for tiers without
subscription pricing, e.g. "none" and per-use "basic").
"""
function apply_balanced_fixed_tiers!(sim::EmergentSimulation, assignments::Vector{String})
    for (i, agent) in enumerate(sim.agents)
        new_tier = assignments[i]
        agent.fixed_ai_level = new_tier
        agent.current_ai_level = new_tier
        for tier in collect(keys(agent.subscription_accounts))
            GlimpseABM.cancel_subscription_schedule!(agent, tier)
        end
        if new_tier != "none"
            GlimpseABM.ensure_subscription_schedule!(agent, new_tier)
        end
    end
    return sim
end
