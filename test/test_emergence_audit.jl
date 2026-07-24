# Emergence mechanism tests: AI_ERROR_CORRELATION (rho) and
# DECISION_TEMPERATURE (T).
#
# Battery (mirrors test_strategy_ladder.jl / test_open_action_space.jl):
#   1. Config validation — rho outside [0,1] / T <= 0 error loudly at
#      initialize! AND at first use (get_information / make_decision!).
#   2. Default neutrality — rho=0 + T=1 explicit is bit-identical to the
#      implicit default (fingerprint sim), and the rho=0 path consumes NO
#      common draws (common_error_cache stays empty over a full sim).
#   3. P9 variance preservation — the per-agent estimate-error variance is
#      ~1 at rho=0 AND rho=1 (the variance-preserving blend), while the
#      cross-agent error correlation flips ~0 -> ~1; constructed directly on
#      get_information outputs for two agents, same opp/round.
#   4. P9 accuracy invariance — mean |estimated_return - latent| unchanged
#      in rho (the entire point of the dial).
#   5. P9 cache semantics — one common draw per (opp, tier) per round,
#      shared across same-tier agents, distinct across tiers, cleared with
#      the info cache each round (clear_cache! + simulation step!).
#   6. P10 behavioral — at small fixed-seed scale, T=2.0 raises pooled
#      action-mix entropy vs T=1.0 and T=0.5 lowers it.

using Test
using Random
using Statistics

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using GlimpseABM

# ── helpers (ea_ prefix: runtests.jl includes all test files in one scope) ──

function ea_config(; n_agents::Int = 24, kwargs...)
    config = EmergentConfig(; N_AGENTS = n_agents, N_ROUNDS = 8,
                            RANDOM_SEED = 42, enable_round_logging = false,
                            kwargs...)
    GlimpseABM.initialize!(config)
    return config
end

const EA_TIER_MIX = Dict("none" => 0.25, "basic" => 0.25,
                         "advanced" => 0.25, "premium" => 0.25)

function ea_fingerprint_sim(config; seed::Int = 4242, rounds::Int = 6,
                            tiers = EA_TIER_MIX)
    sim = EmergentSimulation(config = config, seed = seed,
                             initial_tier_distribution = tiers)
    for r in 1:rounds
        GlimpseABM.step!(sim, r)
    end
    return (
        capitals = [GlimpseABM.get_capital(a) for a in sim.agents],
        alive = [a.alive for a in sim.agents],
        actions = [copy(a.action_history) for a in sim.agents],
        sim = sim,
    )
end

function ea_test_opportunity(id::String)
    return Opportunity(
        id = id,
        # Mid-range latent return: the [1.1, 25.0] clamp is > 5 sigma away at
        # the maximum error scale (1 - 0.35) * 0.5, so the eps reconstruction
        # below is clamp-free.
        latent_return_potential = 3.0,
        latent_failure_potential = 0.3,
        complexity = 0.4,
        sector = "tech",
        discovered = true,
    )
end

"""
Reconstruct the standard-normal estimate-error draw eps from a get_information
output. Requires HALLUCINATION_INTENSITY = 0 (no estimate overwrite) and the
default NOVELTY_NOISE_INVERSION_FACTOR = 0 (generalization multiplier = 1):
estimated_return = latent + eps * (1 - actual_accuracy) * 0.5 + bias * 0.3.
"""
function ea_recover_eps(info, opp)::Float64
    return (info.estimated_return - opp.latent_return_potential -
            info.bias_applied * 0.3) / ((1.0 - info.actual_accuracy) * 0.5)
end

"""
Per-draw estimate errors for two same-tier agents over `n` simulated rounds
(clear_cache! between draws = the round boundary). Both agent rngs are
freshly seeded, so paired calls across rho settings see IDENTICAL accuracy/
hallucination-chain draws (the agent stream consumes the same draw count at
every rho — pinned by the bit-identity test).
"""
function ea_paired_errors(rho::Float64; n::Int = 2000, tier::String = "premium")
    config = ea_config(AI_ERROR_CORRELATION = rho, HALLUCINATION_INTENSITY = 0.0)
    sys = GlimpseABM.InformationSystem(config; common_error_seed = 777)
    opp = ea_test_opportunity("ea_var_opp")
    rng1 = MersenneTwister(11)
    rng2 = MersenneTwister(22)
    eps1 = Float64[]; eps2 = Float64[]
    err1 = Float64[]
    for _ in 1:n
        GlimpseABM.clear_cache!(sys)  # round boundary: fresh common draw
        info1 = get_information(sys, opp, tier; agent_id = 1, rng = rng1)
        info2 = get_information(sys, opp, tier; agent_id = 2, rng = rng2)
        push!(eps1, ea_recover_eps(info1, opp))
        push!(eps2, ea_recover_eps(info2, opp))
        push!(err1, abs(info1.estimated_return - opp.latent_return_potential))
    end
    return eps1, eps2, err1
end

# ── 1. config validation ─────────────────────────────────────────────────────

@testset "Emergence audit: malformed config errors loudly" begin
    @test_throws ErrorException GlimpseABM.initialize!(
        EmergentConfig(AI_ERROR_CORRELATION = -0.1))
    @test_throws ErrorException GlimpseABM.initialize!(
        EmergentConfig(AI_ERROR_CORRELATION = 1.5))
    @test_throws ErrorException GlimpseABM.initialize!(
        EmergentConfig(DECISION_TEMPERATURE = 0.0))
    @test_throws ErrorException GlimpseABM.initialize!(
        EmergentConfig(DECISION_TEMPERATURE = -1.0))

    # Valid endpoints pass.
    @test GlimpseABM.initialize!(
        EmergentConfig(AI_ERROR_CORRELATION = 1.0,
                       DECISION_TEMPERATURE = 0.5)) isa EmergentConfig

    # First-use guard, rho (get_information): a config mutated AFTER
    # initialize! still fails loudly at the blend site.
    cfg = ea_config()
    cfg.AI_ERROR_CORRELATION = 1.5
    sys = GlimpseABM.InformationSystem(cfg)
    @test_throws ErrorException get_information(
        sys, ea_test_opportunity("ea_bad_rho"), "premium";
        agent_id = 1, rng = MersenneTwister(1))

    # First-use guard, T (make_decision! softmax): mutate after the sim's
    # initialize! ran; the very first decision round errors.
    cfg_t = ea_config(n_agents = 8)
    sim = EmergentSimulation(config = cfg_t, seed = 99,
                             initial_tier_distribution = EA_TIER_MIX)
    sim.config.DECISION_TEMPERATURE = -1.0
    @test_throws ErrorException GlimpseABM.step!(sim, 1)
end

# ── 2. default neutrality (bit-identity at rho = 0, T = 1) ───────────────────

@testset "Emergence audit: default neutrality (rho=0, T=1 bit-identical)" begin
    cfg_default = ea_config()
    fp_default = ea_fingerprint_sim(cfg_default; seed = 9001)

    cfg_explicit = ea_config(AI_ERROR_CORRELATION = 0.0,
                             DECISION_TEMPERATURE = 1.0)
    fp_explicit = ea_fingerprint_sim(cfg_explicit; seed = 9001)

    @test fp_default.capitals == fp_explicit.capitals
    @test fp_default.alive == fp_explicit.alive
    @test fp_default.actions == fp_explicit.actions

    # Structural proof that rho = 0 consumes NO common draws: get! always
    # caches, so an empty cache after a full mixed-tier simulation means the
    # common-error stream was never touched (the short-circuit fired before
    # any new RNG use).
    @test isempty(fp_default.sim.info_system.common_error_cache)
    @test isempty(fp_explicit.sim.info_system.common_error_cache)

    # And an rho-enabled simulation DOES touch the common path (the pin above
    # is alive, not vacuous): entries appear during a round. Checked at the
    # system level to be independent of round-1 visibility timing.
    cfg_on = ea_config(AI_ERROR_CORRELATION = 1.0)
    sys_on = GlimpseABM.InformationSystem(cfg_on; common_error_seed = 3)
    get_information(sys_on, ea_test_opportunity("ea_alive"), "premium";
                    agent_id = 1, rng = MersenneTwister(5))
    @test length(sys_on.common_error_cache) == 1
end

# ── 3 + 4. P9: variance preservation, correlation flip, accuracy invariance ──

@testset "Emergence audit P9: variance preserved, correlation 0 -> 1, accuracy invariant" begin
    eps1_r0, eps2_r0, err_r0 = ea_paired_errors(0.0)
    eps1_r1, eps2_r1, err_r1 = ea_paired_errors(1.0)

    # Variance preservation: per-agent standardized error variance ~ 1 at
    # BOTH endpoints (Var = rho + (1 - rho) = 1 by the blend algebra).
    @test isapprox(var(eps1_r0), 1.0; atol = 0.15)
    @test isapprox(var(eps2_r0), 1.0; atol = 0.15)
    @test isapprox(var(eps1_r1), 1.0; atol = 0.15)
    @test isapprox(var(eps2_r1), 1.0; atol = 0.15)

    # Cross-agent correlation: ~0 at rho = 0 (fully idiosyncratic), exactly
    # the SAME draw at rho = 1 (sqrt(1)*eps_common + sqrt(0)*eps_idio).
    @test abs(cor(eps1_r0, eps2_r0)) < 0.1
    @test all(isapprox.(eps1_r1, eps2_r1; atol = 1e-8))
    @test cor(eps1_r1, eps2_r1) > 0.999

    # Accuracy invariance: mean absolute estimate error unchanged in rho.
    # The agent rngs are seeded identically across the two settings and the
    # agent stream consumes the same number of draws at every rho, so the
    # error SCALES are draw-for-draw identical — only the standard-normal
    # eps source differs.
    @test isapprox(mean(err_r0), mean(err_r1); rtol = 0.08)

    # Intermediate rho keeps the variance budget too.
    eps1_r5, eps2_r5, _ = ea_paired_errors(0.5; n = 2000)
    @test isapprox(var(eps1_r5), 1.0; atol = 0.15)
    c5 = cor(eps1_r5, eps2_r5)
    @test 0.25 < c5 < 0.75  # commonality rises with rho, strictly between endpoints
end

# ── 5. P9: common-draw cache semantics ───────────────────────────────────────

@testset "Emergence audit P9: common draw shared per (opp, tier), cleared per round" begin
    cfg = ea_config(AI_ERROR_CORRELATION = 1.0, HALLUCINATION_INTENSITY = 0.0)
    sys = GlimpseABM.InformationSystem(cfg; common_error_seed = 55)
    opp_a = ea_test_opportunity("ea_cache_a")
    opp_b = ea_test_opportunity("ea_cache_b")

    # Two same-tier agents, same opp, same round: ONE common draw.
    info1 = get_information(sys, opp_a, "premium"; agent_id = 1, rng = MersenneTwister(1))
    info2 = get_information(sys, opp_a, "premium"; agent_id = 2, rng = MersenneTwister(2))
    @test length(sys.common_error_cache) == 1
    @test isapprox(ea_recover_eps(info1, opp_a), ea_recover_eps(info2, opp_a); atol = 1e-8)

    # A different TIER on the same opp gets its OWN common draw.
    info3 = get_information(sys, opp_a, "advanced"; agent_id = 3, rng = MersenneTwister(3))
    @test length(sys.common_error_cache) == 2
    @test !isapprox(ea_recover_eps(info3, opp_a), ea_recover_eps(info1, opp_a); atol = 1e-8)

    # A different OPP gets its own draw; the "none" tier is exempt entirely.
    get_information(sys, opp_b, "premium"; agent_id = 1, rng = MersenneTwister(4))
    @test length(sys.common_error_cache) == 3
    get_information(sys, opp_b, "none"; agent_id = 5, rng = MersenneTwister(6))
    @test length(sys.common_error_cache) == 3

    # clear_cache! clears BOTH caches (the round boundary).
    GlimpseABM.clear_cache!(sys)
    @test isempty(sys.common_error_cache)
    @test isempty(sys.information_cache)

    # Simulation-level: step! clears the common cache at the START of every
    # round (a sentinel inserted between rounds is gone after the next step).
    cfg_sim = ea_config(n_agents = 12, AI_ERROR_CORRELATION = 1.0)
    sim = EmergentSimulation(config = cfg_sim, seed = 321,
                             initial_tier_distribution = EA_TIER_MIX)
    GlimpseABM.step!(sim, 1)
    sim.info_system.common_error_cache[("ea_sentinel", "premium")] = 1.23
    GlimpseABM.step!(sim, 2)
    @test !haskey(sim.info_system.common_error_cache, ("ea_sentinel", "premium"))
end

# ── 6. P10: behavioral — temperature moves action-mix entropy ────────────────

"Shannon entropy (nats) of the pooled executed-action mix across agents/rounds."
function ea_action_entropy(temperature::Float64; seeds = (101, 202, 303))
    counts = Dict{String,Int}()
    for seed in seeds
        cfg = ea_config(n_agents = 32, DECISION_TEMPERATURE = temperature)
        fp = ea_fingerprint_sim(cfg; seed = seed, rounds = 10)
        for hist in fp.actions, a in hist
            counts[a] = get(counts, a, 0) + 1
        end
    end
    total = sum(values(counts))
    return -sum(p * log(p) for p in (c / total for c in values(counts)) if p > 0)
end

@testset "Emergence audit P10: T=2 flattens, T=0.5 sharpens the action mix" begin
    ent_low = ea_action_entropy(0.5)
    ent_mid = ea_action_entropy(1.0)
    ent_high = ea_action_entropy(2.0)
    @test ent_high > ent_mid
    @test ent_mid > ent_low
end
