# Regression tests for engine-invariant fixes.
# Each testset pins one fix; see the design notes for the
# underlying findings.

using Test
using GlimpseABM
using Random
using Statistics

include("test_helpers.jl")

@testset "engine-invariant fixes" begin

    # ------------------------------------------------------------------
    # T1-1: ai_herding_intensity must measure concentration, not volume.
    # ------------------------------------------------------------------
    @testset "Herding intensity: concentration not saturation" begin
        function run_herding_rounds(concentrated::Bool; n_rounds::Int=30)
            # N=1000 -> ~40 market opportunities, enough to distinguish
            # dispersed (20 distinct targets) from concentrated (one target).
            cfg = EmergentConfig(N_AGENTS=1000, N_ROUNDS=n_rounds, RANDOM_SEED=77)
            GlimpseABM.initialize!(cfg)
            market = MarketEnvironment(cfg; rng=MersenneTwister(77))
            env = KnightianUncertaintyEnvironment(cfg; rng=MersenneTwister(78))
            opp_ids = [opp.id for opp in market.opportunities]
            @assert length(opp_ids) >= 20
            intensity = 0.0
            for r in 1:n_rounds
                actions = Dict{String,Any}[]
                # 20 AI invests: all on one opportunity (concentrated) or
                # spread across 20 distinct opportunities (dispersed).
                for k in 1:20
                    target = concentrated ? opp_ids[1] : opp_ids[k]
                    push!(actions, Dict{String,Any}(
                        "action" => "invest",
                        "agent_id" => k,
                        "ai_level_used" => "premium",
                        "ai_behavior_level" => "premium",
                        "ai_used" => true,
                        "chosen_opportunity_details" => Dict{String,Any}("id" => target),
                    ))
                end
                # 20 non-AI maintains so AI participation is interior.
                for k in 21:40
                    push!(actions, Dict{String,Any}(
                        "action" => "maintain",
                        "agent_id" => k,
                        "ai_level_used" => "none",
                        "ai_used" => false,
                    ))
                end
                GlimpseABM.record_ai_signals!(env, r, actions)
                measure_uncertainty_state!(env, market, actions, Innovation[], r)
                intensity = Float64(get(env.practical_indeterminism_state,
                                        "ai_herding_intensity", -1.0))
            end
            return intensity
        end

        dispersed = run_herding_rounds(false)
        concentrated = run_herding_rounds(true)

        # Dispersed AI investing is the null case: no herding signal even
        # after 30 sustained rounds (the old stock/flow metric pinned at 1.0
        # here within ~10 rounds).
        @test 0.0 <= dispersed < 0.10
        # All-on-one-opportunity is herding: clearly positive, clearly above
        # the dispersed null, and not saturated at the clamp.
        @test concentrated > 0.25
        @test concentrated > dispersed + 0.2
        @test concentrated < 1.0
    end

    @testset "Herding intensity: zero without AI activity" begin
        cfg = EmergentConfig(N_AGENTS=50, N_ROUNDS=5, RANDOM_SEED=79)
        GlimpseABM.initialize!(cfg)
        market = MarketEnvironment(cfg; rng=MersenneTwister(79))
        env = KnightianUncertaintyEnvironment(cfg; rng=MersenneTwister(80))
        for r in 1:5
            actions = [Dict{String,Any}(
                "action" => "invest",
                "agent_id" => k,
                "ai_level_used" => "none",
                "ai_used" => false,
                "chosen_opportunity_details" =>
                    Dict{String,Any}("id" => market.opportunities[1].id),
            ) for k in 1:10]
            GlimpseABM.record_ai_signals!(env, r, actions)
            measure_uncertainty_state!(env, market, actions, Innovation[], r)
        end
        @test Float64(get(env.practical_indeterminism_state,
                          "ai_herding_intensity", -1.0)) == 0.0
    end

    # ------------------------------------------------------------------
    # T2-5: volatility EWMA keeps state on the unscaled axis.
    # ------------------------------------------------------------------
    @testset "Volatility EWMA steady state matches decay/scaling semantics" begin
        cfg = EmergentConfig(N_AGENTS=10, N_ROUNDS=10, RANDOM_SEED=81)
        GlimpseABM.initialize!(cfg)
        env = KnightianUncertaintyEnvironment(cfg; rng=MersenneTwister(81))
        shares_a = [1.0, 0.0, 0.0, 0.0]
        shares_b = [0.0, 1.0, 0.0, 0.0]
        ai_shares = [1.0, 0.0, 0.0, 0.0]
        vol = 0.0
        for i in 1:300
            shares = isodd(i) ? shares_a : shares_b
            vol = GlimpseABM._update_volatility_state!(env, shares, ai_shares, nothing)
        end
        # Constant raw signal: action_delta = 0.5 each call, ai/market deltas 0.
        raw = cfg.UNCERTAINTY_ACTION_VARIANCE_WEIGHT * 0.5
        expected = raw * cfg.UNCERTAINTY_VOLATILITY_SCALING
        # Pre-fix the feedback compounded the scaling: steady state was
        # ~(1-d)/(1-d*s) * s * raw ≈ 0.21*expected — far outside this band.
        @test isapprox(vol, expected; rtol=0.05)
    end

    # ------------------------------------------------------------------
    # T2-6: knowledge_gap driver of actor ignorance is alive.
    # ------------------------------------------------------------------
    @testset "Actor-ignorance knowledge gap responds to coverage" begin
        cfg = EmergentConfig(N_AGENTS=10, N_ROUNDS=5, RANDOM_SEED=82)
        GlimpseABM.initialize!(cfg)
        market = MarketEnvironment(cfg; rng=MersenneTwister(82))
        kb = KnowledgeBase(cfg)
        GlimpseABM.initialize_base_knowledge!(kb)
        piece_ids = collect(keys(kb.knowledge_pieces))
        @test length(piece_ids) > 4
        # Sparse coverage: each agent holds 1 piece of a much larger base.
        for agent_id in 1:10
            kb.agent_knowledge[agent_id] = Set([piece_ids[1]])
        end
        env = KnightianUncertaintyEnvironment(cfg; knowledge_base=kb,
                                              rng=MersenneTwister(83))
        actions = [Dict{String,Any}("action" => "maintain", "agent_id" => k,
                                    "ai_level_used" => "none", "ai_used" => false)
                   for k in 1:10]
        measure_uncertainty_state!(env, market, actions, Innovation[], 1)
        components = env.actor_ignorance_state["components"]
        sparse_gap = Float64(get(components, "knowledge_gap", -1.0))
        # The old opportunities-per-agent normalizer froze this at exactly 0.
        @test sparse_gap > 0.05

        # Full coverage: every agent holds every piece -> gap collapses.
        for agent_id in 1:10
            kb.agent_knowledge[agent_id] = Set(piece_ids)
        end
        env2 = KnightianUncertaintyEnvironment(cfg; knowledge_base=kb,
                                               rng=MersenneTwister(84))
        measure_uncertainty_state!(env2, market, actions, Innovation[], 1)
        full_gap = Float64(get(env2.actor_ignorance_state["components"],
                               "knowledge_gap", -1.0))
        @test full_gap < sparse_gap
        @test full_gap <= 0.01
    end

    # ------------------------------------------------------------------
    # T1-3: latent failure floor respects calibrated sector ranges.
    # ------------------------------------------------------------------
    @testset "Service-sector failure heterogeneity survives the global clamp" begin
        cfg = EmergentConfig(N_AGENTS=100, N_ROUNDS=5, RANDOM_SEED=85)
        GlimpseABM.initialize!(cfg)
        market = MarketEnvironment(cfg; rng=MersenneTwister(85))
        failures = Float64[]
        for i in 1:300
            opp = GlimpseABM._create_realistic_opportunity(market, "svc_test_$i", "service")
            push!(failures, opp.latent_failure_potential)
        end
        profile = cfg.SECTOR_PROFILES["service"]
        # Pre-fix every draw was pinned to exactly 0.1 (above the sector max
        # of ~0.093). Now the calibrated range must be populated with real
        # within-sector variation.
        @test minimum(failures) < 0.09
        @test std(failures) > 0.005
        @test maximum(failures) <= profile.failure_range[2] + 1e-9
        @test count(==(0.1), failures) < length(failures) ÷ 4
    end

    # ------------------------------------------------------------------
    # T2-3: dead agents release in-flight capital from opportunities.
    # ------------------------------------------------------------------
    @testset "Dead agent releases in-flight capital exactly once" begin
        cfg = EmergentConfig(N_AGENTS=2, N_ROUNDS=10, RANDOM_SEED=86,
                             SURVIVAL_COUNTS_INFLIGHT=false,
                             INSOLVENCY_GRACE_ROUNDS=1)
        GlimpseABM.initialize!(cfg)
        market = MarketEnvironment(cfg; rng=MersenneTwister(86))
        agent = EmergentAgent(1, cfg; primary_sector="tech",
                              initial_capital=5_000_000.0,
                              fixed_ai_level="none", rng=MersenneTwister(87))
        opp = Opportunity(
            id="death-release", latent_return_potential=1.5,
            latent_failure_potential=0.1, discovered=true, sector="tech",
            time_to_maturity=8, total_invested=250_000.0, capacity=1_000_000.0,
            config=cfg, rng=MersenneTwister(88))
        push!(agent.active_investments, Dict{String,Any}(
            "opportunity" => opp, "amount" => 100_000.0, "maturity_round" => 9,
            "ai_level" => "none", "ai_label" => "none",
            "estimated_return" => 1.2,
            "competition_at_entry" => 0.0,
            "capacity_saturation_at_entry" => 0.25))

        GlimpseABM.set_capital!(agent, 0.0)
        agent.insolvency_rounds = cfg.INSOLVENCY_GRACE_ROUNDS  # at the limit
        alive = check_survival!(agent, 3)
        @test !alive
        @test !agent.alive
        # The dead agent's outstanding 100k stake is released...
        @test isapprox(opp.total_invested, 150_000.0; atol=1e-6)
        # ...exactly once: neither a second death-release nor the (skipped)
        # maturity path may decrement again.
        GlimpseABM._release_inflight_capital_at_death!(agent)
        @test isapprox(opp.total_invested, 150_000.0; atol=1e-6)
        matured = GlimpseABM.process_matured_investments!(agent, market, 9)
        @test isempty(matured)
        @test isapprox(opp.total_invested, 150_000.0; atol=1e-6)
    end

    # ------------------------------------------------------------------
    # T2-1: innovation returns are recorded -> ROIC is a real signal.
    # ------------------------------------------------------------------
    @testset "Innovate ROIC reflects realized cash, not -1.0 forever" begin
        cfg = EmergentConfig(N_AGENTS=2, N_ROUNDS=5, RANDOM_SEED=89)
        GlimpseABM.initialize!(cfg)
        market = MarketEnvironment(cfg; rng=MersenneTwister(89))
        agent = EmergentAgent(1, cfg; primary_sector="tech",
                              initial_capital=2_000_000.0,
                              fixed_ai_level="none", rng=MersenneTwister(90))
        # Legacy fallback path (no engine) is deterministic enough here: both
        # success (return = spend * multiplier) and failure (12% recovery)
        # must move ROIC off the -1.0 floor.
        outcome = GlimpseABM._execute_innovate!(agent, market, 1, Dict{String,Any}())
        roic = GlimpseABM.compute_roic(agent.resources.performance, "innovate")
        @test roic > -0.95
        deployed = agent.resources.performance.deployed_by_action["innovate"]
        returned = get(agent.resources.performance.returned_by_action, "innovate", 0.0)
        @test deployed > 0.0
        @test returned > 0.0
    end

    # ------------------------------------------------------------------
    # T2-4: regime chain steps at most once per round (legacy path removed).
    # ------------------------------------------------------------------
    @testset "No second regime-transition path exists" begin
        @test !isdefined(GlimpseABM, :_transition_regime!)
    end

    # ------------------------------------------------------------------
    # Test-suite gap (a)(i): default config primitives are tier-neutral.
    # ------------------------------------------------------------------
    @testset "Default config tier primitives are neutral" begin
        cfg = EmergentConfig()
        for (tier, v) in cfg.AI_EXECUTION_SUCCESS_MULTIPLIERS
            @test v == 1.0
        end
        for (tier, v) in cfg.AI_QUALITY_BOOST
            @test v == 0.0
        end
        for (tier, v) in cfg.AI_INFORMATION_QUALITY_BOOSTS
            @test v == 0.0
        end
        @test cfg.AI_NOVELTY_UPLIFT == 0.0
        @test cfg.STRATEGIC_ANTICIPATION_ENABLED == false
        @test cfg.STRATEGIC_ANTICIPATION_STRENGTH == 0.0
    end

    # ------------------------------------------------------------------
    # Test-suite gap (e): v3.5 power-law right tail.
    # ------------------------------------------------------------------
    @testset "Power-law tail: alpha governs tail mass; ceiling holds" begin
        function sample_returns(alpha::Float64; n::Int=20_000)
            cfg = EmergentConfig(N_AGENTS=10, N_ROUNDS=5, RANDOM_SEED=91)
            cfg.POWER_LAW_SHAPE_A = alpha
            GlimpseABM.initialize!(cfg)
            mc = test_market_conditions()
            opp = Opportunity(
                id="tail-test", latent_return_potential=2.0,
                latent_failure_potential=0.1, discovered=true, sector="tech",
                time_to_maturity=6, config=cfg, rng=MersenneTwister(92))
            rng = MersenneTwister(93)
            return [GlimpseABM.realized_return(opp, mc, "none"; rng=rng) for _ in 1:n]
        end

        heavy = sample_returns(2.2)
        light = sample_returns(3.5)
        # Heavier tail (lower alpha) must put strictly more mass above 5x.
        @test count(>(5.0), heavy) > count(>(5.0), light)
        @test count(>(3.0), heavy) > 0          # the tail exists at all
        @test maximum(heavy) <= 200.0 + 1e-9    # scarcity-gated hard ceiling
        # Typical outcomes stay in a sane venture band.
        @test 0.3 < median(heavy) < 2.5
    end

    # ------------------------------------------------------------------
    # Test-suite gap (c): the crowding penalty is actually convex.
    # ------------------------------------------------------------------
    @testset "Crowding penalty convexity (second difference)" begin
        cfg = EmergentConfig(N_AGENTS=10, N_ROUNDS=5, RANDOM_SEED=94)
        GlimpseABM.initialize!(cfg)
        mc = test_market_conditions()
        capacity = 1_000_000.0
        # Saturation levels in the over-capacity (penalty-active) region,
        # equally spaced so second differences are meaningful.
        sat_levels = [2.0, 2.5, 3.0, 3.5, 4.0]
        log_means = Float64[]
        for s in sat_levels
            opp = Opportunity(
                id="convexity-$(s)", latent_return_potential=2.0,
                latent_failure_potential=0.1, discovered=true, sector="tech",
                time_to_maturity=6, capacity=capacity,
                total_invested=s * capacity, config=cfg,
                rng=MersenneTwister(95))
            rng = MersenneTwister(96)
            vals = [GlimpseABM.realized_return(opp, mc, "none"; rng=rng) for _ in 1:6_000]
            push!(log_means, log(mean(vals)))
        end
        # Monotone decreasing in saturation...
        @test issorted(log_means; rev=true)
        # ...and convex penalty: log-mean decline accelerates (second
        # differences negative). A linear-in-saturation penalty would have
        # ~zero second differences and fail this.
        d2 = diff(diff(log_means))
        @test all(d2 .< 0.0)
    end

    # ------------------------------------------------------------------
    # Phantom invests: with no visible opportunities, "invest" cannot be
    # sampled or recorded.
    # ------------------------------------------------------------------
    @testset "No phantom invest actions when no opportunities visible" begin
        cfg = EmergentConfig(N_AGENTS=2, N_ROUNDS=5, RANDOM_SEED=97)
        GlimpseABM.initialize!(cfg)
        market = MarketEnvironment(cfg; rng=MersenneTwister(97))
        mc = GlimpseABM.get_market_conditions(market)
        agent = EmergentAgent(1, cfg; primary_sector="tech",
                              initial_capital=2_000_000.0,
                              fixed_ai_level="none", rng=MersenneTwister(98))
        for trial in 1:60
            outcome = GlimpseABM.make_decision!(agent, Opportunity[], mc, market, trial)
            @test get(outcome, "action", "") != "invest"
            probs = get(outcome, "action_probabilities", Dict{String,Float64}())
            @test get(probs, "invest", 0.0) == 0.0
        end
    end
end
