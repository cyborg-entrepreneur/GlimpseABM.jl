# Audit-addendum knob tests (the design notes):
# AI_BIAS_INTENSITY (NO_AI_BIAS cell) + WEALTH_COMPUTE_SCALING
# (WEALTH_SCALED_COMPUTE cell).
#
# Battery (mirrors test_emergence_audit.jl):
# 1. Defaults — both knobs default to their neutral values.
# 2. Bias behavioral — intensity 0 zeroes bias_applied where the domain
# capability carries nonzero bias; intensity scales it linearly.
# 3. Bias default neutrality — explicit 1.0 is bit-identical to implicit
# (multiplication by exactly 1.0 is an IEEE identity; fingerprint sim).
# 4. Wealth behavioral — under scaling=1.0 a capital-rich agent's perceived
# menu is larger than a capital-poor agent's from the same pool; under
# the 0.0 default, capital does not move the menu size.
# 5. Wealth default neutrality — explicit 0.0 bit-identical to implicit
# (the guard skips the block entirely).

using Test
using Random
using Statistics

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using GlimpseABM

# ── helpers (ra_ prefix: runtests.jl includes all test files in one scope) ──

function ra_config(; n_agents::Int = 24, kwargs...)
    config = EmergentConfig(; N_AGENTS = n_agents, N_ROUNDS = 8,
                            RANDOM_SEED = 42, enable_round_logging = false,
                            kwargs...)
    GlimpseABM.initialize!(config)
    return config
end

const RA_TIER_MIX = Dict("none" => 0.25, "basic" => 0.25,
                         "advanced" => 0.25, "premium" => 0.25)

function ra_fingerprint_sim(config; seed::Int = 4242, rounds::Int = 6)
    sim = EmergentSimulation(config = config, seed = seed,
                             initial_tier_distribution = RA_TIER_MIX)
    for r in 1:rounds
        GlimpseABM.step!(sim, r)
    end
    return (
        capitals = [GlimpseABM.get_capital(a) for a in sim.agents],
        alive = [a.alive for a in sim.agents],
        actions = [copy(a.action_history) for a in sim.agents],
   )
end

function ra_test_opportunity(id::String)
    return Opportunity(
        id = id,
        latent_return_potential = 3.0,
        latent_failure_potential = 0.3,
        complexity = 0.4,
        sector = "tech",
        discovered = true,
   )
end

# Force a known nonzero bias into EVERY premium domain capability so the
# bias_applied assertions do not depend on which domain determine_domain picks.
function ra_set_premium_bias!(config, bias::Float64)
    caps = config.AI_DOMAIN_CAPABILITIES["premium"]
    for domain in collect(keys(caps))
        old = caps[domain]
        caps[domain] = GlimpseABM.AIDomainCapability(
            old.accuracy, old.hallucination_rate, bias)
    end
    return config
end

# ── 1. defaults ──────────────────────────────────────────────────────────────

@testset "Robustness addendum: knob defaults are neutral" begin
    config = EmergentConfig()
    @test config.AI_BIAS_INTENSITY == 1.0
    @test config.WEALTH_COMPUTE_SCALING == 0.0
end

# ── 2. bias behavioral ──────────────────────────────────────────────────────

@testset "AI_BIAS_INTENSITY scales bias_applied" begin
    opp = ra_test_opportunity("ra_bias_opp")
    for (intensity, expected) in ((0.0, 0.0), (0.5, 0.1), (1.0, 0.2))
        config = ra_config(AI_BIAS_INTENSITY = intensity,
                           HALLUCINATION_INTENSITY = 0.0)
        ra_set_premium_bias!(config, 0.2)
        sys = GlimpseABM.InformationSystem(config)
        info = get_information(sys, opp, "premium";
                               agent_id = 1, rng = MersenneTwister(7))
        @test isapprox(info.bias_applied, expected; atol = 1e-12)
    end
end

# ── 3. bias default neutrality ──────────────────────────────────────────────

@testset "AI_BIAS_INTENSITY=1.0 explicit is bit-identical to implicit" begin
    base = ra_fingerprint_sim(ra_config())
    explicit = ra_fingerprint_sim(ra_config(AI_BIAS_INTENSITY = 1.0))
    @test base.capitals == explicit.capitals
    @test base.alive == explicit.alive
    @test base.actions == explicit.actions
end

# ── 4. wealth behavioral ────────────────────────────────────────────────────

@testset "WEALTH_COMPUTE_SCALING widens rich menus, narrows poor ones" begin
 # Pool large enough that the visibility budget binds for premium.
    pool = [ra_test_opportunity("ra_w_opp_$(i)") for i in 1:120]

    function menu_sizes(scaling::Float64)
        config = ra_config(WEALTH_COMPUTE_SCALING = scaling)
        sim = EmergentSimulation(config = config, seed = 99,
                                 initial_tier_distribution = RA_TIER_MIX)
        rich, poor = sim.agents[1], sim.agents[2]
        for (agent, ratio) in ((rich, 4.0), (poor, 0.25))
            eq = agent.resources.performance.initial_equity
            agent.resources.capital = ratio * eq
        end
        n_rich = mean(length(GlimpseABM.get_perceived_opportunities(
            sim.market, pool, "premium", rich)) for _ in 1:20)
        n_poor = mean(length(GlimpseABM.get_perceived_opportunities(
            sim.market, pool, "premium", poor)) for _ in 1:20)
        return n_rich, n_poor
    end

    n_rich_on, n_poor_on = menu_sizes(1.0)
    @test n_rich_on > n_poor_on + 10  # 16x budget gap is decisive, not marginal

 # Default off: capital does not move the menu.
    n_rich_off, n_poor_off = menu_sizes(0.0)
    @test abs(n_rich_off - n_poor_off) < 10
end

# ── 5. wealth default neutrality ────────────────────────────────────────────

@testset "WEALTH_COMPUTE_SCALING=0.0 explicit is bit-identical to implicit" begin
    base = ra_fingerprint_sim(ra_config())
    explicit = ra_fingerprint_sim(ra_config(WEALTH_COMPUTE_SCALING = 0.0))
    @test base.capitals == explicit.capitals
    @test base.alive == explicit.alive
    @test base.actions == explicit.actions
end
