# Diagnostic invariant tests — designed to catch the data-flow / semantic
# bugs that static inspection couldn't see. Each test sets up a small,
# targeted scenario and asserts a numeric invariant that would have failed
# under one of the v2/v2.1/v2.2/v2.3-era bugs.
#
# These are NOT smoke tests ("does it crash?"). Each assertion encodes
# "the mechanism produced the value it should produce."

using Test
using Random
using Statistics

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using GlimpseABM

include("test_helpers.jl")

const N_OPPS_DIAG = 20

"""Build a 1-agent fixed-tier setup with a known opportunity pool."""
function _diag_setup(tier::String; seed::Int = 42, n_opps::Int = N_OPPS_DIAG)
    cfg = EmergentConfig(N_AGENTS=1, N_ROUNDS=1, RANDOM_SEED=seed,
                         AGENT_AI_MODE="fixed",
                         AI_COST_MODEL="fixed")
    market = MarketEnvironment(cfg; rng=MersenneTwister(seed))
    info_system = InformationSystem(cfg)
    agent = EmergentAgent(1, cfg;
        primary_sector="tech",
        fixed_ai_level=tier,
        rng=MersenneTwister(seed))

    base_rng = MersenneTwister(seed + 1)
    opps = Opportunity[]
    for i in 1:n_opps
        sector = rand(base_rng, market.sectors)
        push!(opps, GlimpseABM._create_realistic_opportunity(market, "diag_$i", sector))
    end
    return cfg, market, info_system, agent, opps
end

@testset "Diagnostics" begin

    # ------------------------------------------------------------------
    # 1. Per-use cost accumulates for Basic tier
    # ------------------------------------------------------------------
    @testset "basic per-use cost is actually deducted" begin
        cfg, market, info_system, agent, opps = _diag_setup("basic")
        capital_before = get_capital(agent)
        per_use = Float64(cfg.AI_LEVELS["basic"].per_use_cost) * cfg.AI_COST_INTENSITY
        @test per_use > 0  # Sanity

        market_conditions = test_market_conditions()
        perception = empty_perception()
        evals = GlimpseABM.evaluate_portfolio_opportunities(
            agent, opps, market_conditions, perception;
            ai_level="basic", info_system=info_system,
        )
        capital_after = get_capital(agent)
        deducted = capital_before - capital_after
        # Should be ~length(opps) × per_use (one charge per fresh get_information)
        expected = length(opps) * per_use
        @test isapprox(deducted, expected, rtol=0.01)
    end

    # ------------------------------------------------------------------
    # 2. Per-use cost dedupe across utility + ranking calls
    # ------------------------------------------------------------------
    @testset "per-use cost dedupe on cache hit" begin
        cfg, market, info_system, agent, opps = _diag_setup("basic")
        per_use = Float64(cfg.AI_LEVELS["basic"].per_use_cost) * cfg.AI_COST_INTENSITY

        market_conditions = test_market_conditions()
        perception = empty_perception()

        # First call: ALL fresh — full charge
        c0 = get_capital(agent)
        GlimpseABM.evaluate_portfolio_opportunities(
            agent, opps, market_conditions, perception;
            ai_level="basic", info_system=info_system,
        )
        c1 = get_capital(agent)
        first_deduct = c0 - c1

        # Second call SAME ROUND, same opps: all cache hits → zero charge
        GlimpseABM.evaluate_portfolio_opportunities(
            agent, opps, market_conditions, perception;
            ai_level="basic", info_system=info_system,
        )
        c2 = get_capital(agent)
        second_deduct = c1 - c2

        @test first_deduct > 0
        @test isapprox(second_deduct, 0.0, atol=1.0)  # Cache hits → no charge
    end

    @testset "production novelty generalization stress is wired" begin
        function production_info_with_stress(factor::Float64)
            cfg, _market, _info_system, agent, opps = _diag_setup("premium"; seed=303)
            cfg.NOVELTY_NOISE_INVERSION_FACTOR = factor
            opp = opps[1]
            opp.novelty_score = 0.95
            opp.combination_uncertainty = 0.90
            opp.truly_unknown = true
            opp.created_by = 99
            info_system = InformationSystem(cfg)
            return GlimpseABM.get_information(info_system, opp, "premium";
                agent_id=agent.id,
                rng=MersenneTwister(909))
        end

        baseline = production_info_with_stress(0.0)
        stressed = production_info_with_stress(0.4)

        @test baseline.hidden_factors["novelty_generalization_pressure"] == 0.0
        @test stressed.hidden_factors["novelty_generalization_pressure"] > 0.0
        @test stressed.actual_accuracy < baseline.actual_accuracy
        @test stressed.confidence < baseline.confidence
    end

    @testset "hybrid token costs vary by tier and task difficulty" begin
        cfg, _market, _info_system, agent, opps = _diag_setup("basic"; seed=404)
        cfg.AI_COST_MODEL = "hybrid"

        simple = opps[1]
        simple.complexity = 0.1
        simple.novelty_score = 0.05
        simple.combination_uncertainty = 0.05

        complex = opps[2]
        complex.complexity = 1.8
        complex.novelty_score = 0.9
        complex.combination_uncertainty = 0.9

        basic_cost = GlimpseABM.ai_analysis_call_cost(agent, "basic";
            opportunity=simple,
            action="market_analysis",
            n_candidates=Float64(length(opps)),
        )
        advanced_cost = GlimpseABM.ai_analysis_call_cost(agent, "advanced";
            opportunity=simple,
            action="market_analysis",
            n_candidates=Float64(length(opps)),
        )
        premium_simple_cost = GlimpseABM.ai_analysis_call_cost(agent, "premium";
            opportunity=simple,
            action="market_analysis",
            n_candidates=Float64(length(opps)),
        )
        premium_complex_cost = GlimpseABM.ai_analysis_call_cost(agent, "premium";
            opportunity=complex,
            action="market_analysis",
            n_candidates=Float64(length(opps)),
        )

        @test 0.0 < basic_cost < advanced_cost < premium_simple_cost
        @test premium_complex_cost > premium_simple_cost

        basic_usage = GlimpseABM.estimate_ai_token_usage(agent, "basic"; action="exploration")
        premium_usage = GlimpseABM.estimate_ai_token_usage(agent, "premium"; action="exploration")
        @test premium_usage["effective_tokens"] > basic_usage["effective_tokens"]
    end

    @testset "hybrid difficulty-scaling robustness knob affects token costs" begin
        cfg, _market, _info_system, agent, opps = _diag_setup("premium"; seed=405)
        cfg.AI_COST_MODEL = "hybrid"

        opp = opps[1]
        opp.complexity = 1.8
        opp.novelty_score = 0.5
        opp.combination_uncertainty = 0.5

        cfg.AI_DIFFICULTY_COST_SCALING = 0.0
        base_cost = GlimpseABM.ai_analysis_call_cost(agent, "premium";
            opportunity=opp,
            action="market_analysis",
            n_candidates=Float64(length(opps)),
        )
        cfg.AI_DIFFICULTY_COST_SCALING = 1.0
        scaled_cost = GlimpseABM.ai_analysis_call_cost(agent, "premium";
            opportunity=opp,
            action="market_analysis",
            n_candidates=Float64(length(opps)),
        )

        @test scaled_cost > base_cost
    end

    @testset "frontier AI aliases map to canonical internal tier" begin
        @test GlimpseABM.normalize_ai_label("frontier_ai") == "premium"
        @test GlimpseABM.normalize_ai_label("frontier") == "premium"
        # Backward compatibility for older scripts/configs.
        @test GlimpseABM.normalize_ai_label("premium_ai") == "premium"
    end

    @testset "failed AI exploration is still charged" begin
        cfg = EmergentConfig(N_AGENTS=1, N_ROUNDS=1, RANDOM_SEED=505,
                             AGENT_AI_MODE="fixed",
                             AI_COST_MODEL="hybrid",
                             DISCOVERY_PROBABILITY=0.0)
        market = MarketEnvironment(cfg; rng=MersenneTwister(505))
        agent = EmergentAgent(1, cfg;
            primary_sector="tech",
            initial_capital=1_000_000.0,
            fixed_ai_level="basic",
            rng=MersenneTwister(505))

        c0 = get_capital(agent)
        outcome = GlimpseABM.execute_action!(agent, "explore", market, 1)
        deducted = c0 - get_capital(agent)

        @test outcome["success"] == false
        @test outcome["ai_analysis_cost"] > 0.0
        @test outcome["ai_token_cost"] > 0.0
        @test deducted > outcome["explore_cost"]
        @test isapprox(deducted, outcome["explore_cost"] + outcome["ai_analysis_cost"]; rtol=1e-8)
    end

    @testset "decision AI analysis cost is surfaced separately" begin
        cfg, market, info_system, agent, opps = _diag_setup("basic"; seed=5051)
        cfg.AI_COST_MODEL = "hybrid"
        outcome = GlimpseABM.make_decision!(
            agent,
            opps,
            test_market_conditions(),
            market,
            1;
            info_system=info_system,
            ai_level_override="basic",
        )

        @test outcome["ai_decision_analysis_cost"] > 0.0
        @test outcome["ai_total_analysis_cost"] >= outcome["ai_decision_analysis_cost"]
        @test outcome["ai_decision_token_cost"] > 0.0
        @test outcome["ai_total_token_cost"] >= outcome["ai_decision_token_cost"]
    end

    @testset "round stats report AI analysis cost outside action deployment" begin
        cfg = EmergentConfig(N_AGENTS=1, N_ROUNDS=1, RANDOM_SEED=5052)
        sim = EmergentSimulation(config=cfg, seed=5052, run_id="ai_cost_diagnostic")
        action = Dict{String,Any}(
            "action" => "explore",
            "ai_behavior_level" => "premium",
            "ai_decision_analysis_cost" => 5.0,
            "ai_action_analysis_cost" => 7.0,
            "ai_total_analysis_cost" => 12.0,
            "explore_cost" => 100.0,
        )

        stats = GlimpseABM.compile_round_stats(
            sim,
            1,
            Dict{String,Any}[action],
            Dict{String,Any}[],
            Dict{String,Dict{String,Any}}(),
        )

        @test stats["total_ai_analysis_cost"] == 12.0
        @test stats["total_ai_decision_analysis_cost"] == 5.0
        @test stats["total_ai_action_analysis_cost"] == 7.0
        @test stats["total_ai_analysis_cost_premium"] == 12.0
        @test stats["total_capital_deployed_explore"] == 100.0
    end

    @testset "exploration token cost responds to visible search context" begin
        function run_with_visible_context(all_visible::Bool)
            cfg = EmergentConfig(N_AGENTS=1, N_ROUNDS=1, RANDOM_SEED=506,
                                 AGENT_AI_MODE="fixed",
                                 AI_COST_MODEL="hybrid",
                                 DISCOVERY_PROBABILITY=0.0)
            market = MarketEnvironment(cfg; rng=MersenneTwister(506))
            for (i, opp) in enumerate(market.opportunities)
                opp.discovered = all_visible
                opp.complexity = all_visible ? 1.8 : 0.1
                opp.novelty_score = all_visible ? 0.95 : 0.05
                opp.combination_uncertainty = all_visible ? 0.95 : 0.05
                opp.sector = cfg.SECTORS[mod1(i, length(cfg.SECTORS))]
            end
            agent = EmergentAgent(1, cfg;
                primary_sector="tech",
                initial_capital=1_000_000.0,
                fixed_ai_level="premium",
                rng=MersenneTwister(506))
            return GlimpseABM.execute_action!(agent, "explore", market, 1)
        end

        hidden = run_with_visible_context(false)
        visible = run_with_visible_context(true)

        @test visible["ai_search_visible_opportunities"] > hidden["ai_search_visible_opportunities"]
        @test visible["ai_search_sector_diversity"] > hidden["ai_search_sector_diversity"]
        @test visible["ai_search_context_complexity"] > hidden["ai_search_context_complexity"]
        @test visible["ai_effective_search_tokens"] > hidden["ai_effective_search_tokens"]
        @test visible["ai_analysis_cost"] > hidden["ai_analysis_cost"]
    end

    @testset "exploration learning does not use raw info_breadth shortcut" begin
        function run_with_breadth(info_breadth::Float64)
            cfg = EmergentConfig(N_AGENTS=1, N_ROUNDS=1, RANDOM_SEED=606,
                                 AGENT_AI_MODE="fixed",
                                 AI_COST_MODEL="hybrid",
                                 DISCOVERY_PROBABILITY=1.0)
            base = cfg.AI_LEVELS["basic"]
            cfg.AI_LEVELS["basic"] = GlimpseABM.AILevelConfig(
                base.cost,
                base.cost_type,
                base.info_quality,
                info_breadth,
                base.per_use_cost,
            )
            market = MarketEnvironment(cfg; rng=MersenneTwister(606))
            agent = EmergentAgent(1, cfg;
                primary_sector="tech",
                initial_capital=1_000_000.0,
                fixed_ai_level="basic",
                rng=MersenneTwister(606))
            return GlimpseABM.execute_action!(agent, "explore", market, 1)
        end

        low = run_with_breadth(0.10)
        high = run_with_breadth(0.95)

        @test isapprox(low["ai_analysis_cost"], high["ai_analysis_cost"]; atol=1e-12)
        @test isapprox(low["ai_effective_search_tokens"], high["ai_effective_search_tokens"]; atol=1e-12)
        @test isapprox(low["knowledge_gain"], high["knowledge_gain"]; atol=1e-12)
        @test low["created_niche"] == high["created_niche"]
        @test low["discovered_niche"] == high["discovered_niche"]
    end

    # ------------------------------------------------------------------
    # 3. Per-agent cache key produces within-tier heterogeneity
    # ------------------------------------------------------------------
    @testset "two premium agents get different estimates per opp" begin
        cfg, market, info_system, agent_a, opps = _diag_setup("premium"; seed=1001)
        # Same setup but different agent id and rng state
        agent_b = EmergentAgent(2, cfg;
            primary_sector="tech",
            fixed_ai_level="premium",
            rng=MersenneTwister(2001))

        differing = 0
        for opp in opps
            ia = GlimpseABM.get_information(info_system, opp, "premium";
                agent_id=agent_a.id, rng=agent_a.rng)
            ib = GlimpseABM.get_information(info_system, opp, "premium";
                agent_id=agent_b.id, rng=agent_b.rng)
            if !isapprox(ia.estimated_return, ib.estimated_return)
                differing += 1
            end
        end
        # If cache were keyed only by (opp, tier), every estimate would match
        # and `differing` would be 0. With per-agent keying, almost all should differ.
        @test differing >= length(opps) - 2
    end

    # ------------------------------------------------------------------
    # 4. Frontier AI subscription billing matches monthly cost per round
    # ------------------------------------------------------------------
    @testset "frontier AI subscription bills full monthly cost per round" begin
        # Pins the subscription backend explicitly: this test is ABOUT seat-fee
        # billing, which only exists under hybrid/fixed (token-core migration
        # 2026-06-11 made "token" the struct default, with no subscriptions).
        cfg = EmergentConfig(N_AGENTS=1, N_ROUNDS=1, RANDOM_SEED=42,
                             AGENT_AI_MODE="fixed", AI_COST_MODEL="hybrid")
        agent = EmergentAgent(1, cfg;
            primary_sector="tech",
            fixed_ai_level="premium",
            rng=MersenneTwister(42))
        GlimpseABM.ensure_subscription_schedule!(agent, "premium")
        c0 = get_capital(agent)
        charged = GlimpseABM.charge_subscription_installment!(agent, "premium")
        c1 = get_capital(agent)
        # Documented: $3500/month × AI_COST_INTENSITY
        expected = 3500.0 * cfg.AI_COST_INTENSITY
        @test isapprox(charged, expected, rtol=0.01)
        @test isapprox(c0 - c1, expected, rtol=0.01)
    end

    # ------------------------------------------------------------------
    # 5. Investment-amount aggregation matches sum of per-action amounts
    # ------------------------------------------------------------------
    @testset "opp.total_invested matches summed action amounts" begin
        cfg = EmergentConfig(N_AGENTS=20, N_ROUNDS=2, RANDOM_SEED=42,
                             AGENT_AI_MODE="fixed")
        sim = EmergentSimulation(config=cfg, seed=42,
            initial_tier_distribution=Dict("none"=>1.0))
        GlimpseABM.run!(sim)

        # Each opp.total_invested is decremented as investments mature, so
        # this only matches in the absence of maturity events. We assert a
        # weaker invariant: total_invested across opps is non-negative and
        # bounded by total agent capital deployed.
        agg_opp = sum(o.total_invested for o in sim.market.opportunities; init=0.0)
        agg_agent = sum(a.total_invested for a in sim.agents; init=0.0)
        @test agg_opp >= 0
        @test agg_agent >= 0
        @test agg_opp <= agg_agent + 1.0  # Opp total ≤ what agents reported deploying
    end

    # ------------------------------------------------------------------
    # 6. Competition accumulates when many agents invest in the same opp
    # ------------------------------------------------------------------
    @testset "competition rises with concentrated investment" begin
        cfg = EmergentConfig(N_AGENTS=2, N_ROUNDS=1, RANDOM_SEED=42)
        market = MarketEnvironment(cfg; rng=MersenneTwister(42))
        opp = first(market.opportunities)
        # Hammer one opp with 100 invest updates
        for _ in 1:100
            GlimpseABM.update_opportunity_competition!(market, opp, 0.05)
        end
        # With per-investment decay removed, competition should be far above
        # the ~0.7 cap that v2.2 hit. Pre-fix value was capped at 0.7;
        # post-fix should be ≥ 1.0.
        @test opp.competition >= 1.0
    end

    # ------------------------------------------------------------------
    # 7. Population-scaling idempotency — no double-multiplication
    # ------------------------------------------------------------------
    @testset "initialize! is idempotent for population scaling" begin
        cfg = EmergentConfig(N_AGENTS=4000, RANDOM_SEED=42)
        GlimpseABM.initialize!(cfg)
        cap_after_first = cfg.OPPORTUNITY_BASE_CAPACITY
        thresh_after_first = cfg.DISRUPTION_COMPETITION_THRESHOLD
        # Second call should be a no-op
        GlimpseABM.initialize!(cfg)
        @test cfg.OPPORTUNITY_BASE_CAPACITY == cap_after_first
        @test cfg.DISRUPTION_COMPETITION_THRESHOLD == thresh_after_first
        # And third
        GlimpseABM.initialize!(cfg)
        @test cfg.OPPORTUNITY_BASE_CAPACITY == cap_after_first
    end

    # ------------------------------------------------------------------
    # 8. RNG reproducibility — two markets with same seed produce same opp.capacity
    # ------------------------------------------------------------------
    @testset "opportunity capacity is reproducible across seeded markets" begin
        cfg = EmergentConfig(N_AGENTS=10, RANDOM_SEED=42)
        m1 = MarketEnvironment(cfg; rng=MersenneTwister(42))
        m2 = MarketEnvironment(cfg; rng=MersenneTwister(42))
        caps1 = [o.capacity for o in m1.opportunities]
        caps2 = [o.capacity for o in m2.opportunities]
        @test length(caps1) == length(caps2)
        @test caps1 == caps2
    end

    # ------------------------------------------------------------------
    # 9. Fixed mode → get_ai_level always returns the fixed tier
    # ------------------------------------------------------------------
    @testset "fixed mode locks tier" begin
        cfg = EmergentConfig(N_AGENTS=5, N_ROUNDS=2, RANDOM_SEED=42,
                             AGENT_AI_MODE="fixed")
        sim = EmergentSimulation(config=cfg, seed=42,
            initial_tier_distribution=Dict("premium"=>1.0))
        for agent in sim.agents
            @test get_ai_level(agent) == "premium"
        end
        GlimpseABM.run!(sim)
        for agent in sim.agents
            @test get_ai_level(agent) == "premium"
        end
    end

    # ------------------------------------------------------------------
    # 10. Emergent mode → fixed_ai_level is nothing, current can change
    # ------------------------------------------------------------------
    @testset "emergent mode allows tier switching" begin
        cfg = EmergentConfig(N_AGENTS=5, N_ROUNDS=1, RANDOM_SEED=42,
                             AGENT_AI_MODE="emergent")
        sim = EmergentSimulation(config=cfg, seed=42,
            initial_tier_distribution=Dict("premium"=>1.0))
        # Emergent agents must have nothing in fixed_ai_level so choose_ai_level fires
        for agent in sim.agents
            @test isnothing(agent.fixed_ai_level) || agent.fixed_ai_level == "none"
        end
    end

    # (Former testset 11 covered update_lifecycle! stage transitions; the
    # staged opportunity lifecycle was excised as unreachable.)

    # ------------------------------------------------------------------
    # 12. Niche / spawn opportunity is visible to its creator (via override)
    # ------------------------------------------------------------------
    @testset "niche opportunity is visible to creator via created_by override" begin
        cfg = EmergentConfig(N_AGENTS=2, RANDOM_SEED=42)
        market = MarketEnvironment(cfg; rng=MersenneTwister(42))
        niche_opp = GlimpseABM.create_niche_opportunity(
            market,
            "tech_specialized",
            1,
            5;
            sector_novelty_context=0.80,
            sector_uncertainty_context=0.70,
            search_effort=1.2,
            niche_creation_evidence=0.30,
        )
        # v2.7: discovered starts false; creator sees via created_by override.
        @test niche_opp.discovered == false
        @test niche_opp.created_by == 1
        @test niche_opp.novelty_score > 0.50
        @test niche_opp.combination_uncertainty > 0.50
        @test niche_opp.required_discovery_threshold > 0.0
        @test !isnothing(niche_opp.combination_signature)
        @test niche_opp.component_scarcity > 0.50
        # Creator-visibility check: agent with id=1 should see this opp in
        # get_opportunities_for_agent even though discovered=false.
        traits = Dict("exploration_tendency"=>0.5, "market_awareness"=>0.5,
                      "ai_trust"=>0.5, "uncertainty_tolerance"=>0.5)
        creator = EmergentAgent(1, cfg; primary_sector="tech", fixed_ai_level="none",
                                rng=MersenneTwister(42))
        visible = GlimpseABM.get_opportunities_for_agent(market, creator)
        @test niche_opp.id in [o.id for o in visible]
    end

    @testset "base-sector knowledge transfers to niche sectors with opacity attenuation" begin
        cfg = EmergentConfig(N_AGENTS=1, RANDOM_SEED=46)
        @test GlimpseABM.base_sector_name("tech_specialized", cfg.SECTORS) == "tech"

        legible = (
            novelty_score = 0.10,
            combination_uncertainty = 0.10,
            required_discovery_threshold = 0.00,
            truly_unknown = false,
        )
        opaque = (
            novelty_score = 0.90,
            combination_uncertainty = 0.90,
            required_discovery_threshold = 0.80,
            truly_unknown = true,
        )

        base_legible = GlimpseABM.sector_familiarity(
            Dict("tech" => 1.0),
            "tech_specialized";
            opportunity=legible,
            config=cfg,
            default=0.0,
        )
        base_opaque = GlimpseABM.sector_familiarity(
            Dict("tech" => 1.0),
            "tech_specialized";
            opportunity=opaque,
            config=cfg,
            default=0.0,
        )
        exact = GlimpseABM.sector_familiarity(
            Dict("tech_specialized" => 1.0),
            "tech_specialized";
            opportunity=opaque,
            config=cfg,
            default=0.0,
        )

        @test 0.0 < base_opaque < base_legible < 1.0
        @test exact == 1.0

        market = MarketEnvironment(cfg; rng=MersenneTwister(46))
        niche_opp = GlimpseABM.create_niche_opportunity(
            market,
            "tech_specialized",
            2,
            5;
            sector_novelty_context=0.80,
            sector_uncertainty_context=0.70,
            search_effort=1.2,
            niche_creation_evidence=0.30,
        )
        agent = EmergentAgent(1, cfg;
            primary_sector="tech",
            fixed_ai_level="premium",
            rng=MersenneTwister(46))
        agent.resources.knowledge["tech"] = 1.0
        delete!(agent.resources.knowledge, "tech_specialized")

        p_base = GlimpseABM.opportunity_discovery_probability(market, agent, niche_opp, "premium")
        agent.resources.knowledge["tech_specialized"] = 1.0
        p_exact = GlimpseABM.opportunity_discovery_probability(market, agent, niche_opp, "premium")

        @test 0.0 < p_base < p_exact
    end

    @testset "innovation-spawned opportunities carry source ontology" begin
        cfg = EmergentConfig(N_AGENTS=2, RANDOM_SEED=43)
        market = MarketEnvironment(cfg; rng=MersenneTwister(43))
        innovation = Innovation(
            id="innov_probe",
            type="radical",
            knowledge_components=["k1", "k2"],
            novelty=0.80,
            quality=0.70,
            round_created=4,
            creator_id=1,
            success=true,
            ai_assisted=true,
            sector="tech",
            combination_signature="k1||k2",
            cash_multiple=3.0,
            scarcity=0.90,
            is_new_combination=true,
            ai_level_used="premium",
        )

        spawned = GlimpseABM.spawn_opportunity_from_innovation!(market, innovation, 3.0)

        @test spawned.discovered == false
        @test spawned.created_by == 1
        @test spawned.origin_innovation_id == innovation.id
        @test spawned.combination_signature == "k1||k2"
        @test isapprox(spawned.component_scarcity, 0.90; atol=1e-12)
        @test isapprox(spawned.novelty_score, 0.80; atol=1e-12)
        @test spawned.combination_uncertainty > 0.50
        @test spawned.required_discovery_threshold > 0.0
    end

    @testset "simulation niche creation action carries ontology context" begin
        cfg = EmergentConfig(N_AGENTS=2, RANDOM_SEED=44)
        sim = EmergentSimulation(config=cfg, seed=44, run_id="niche_context_probe")
        action = Dict{String,Any}(
            "action" => "explore",
            "exploration_type" => "niche_discovery",
            "agent_id" => 1,
            "discovered_sector" => "tech_specialized",
            "sector_novelty_context" => 0.90,
            "sector_uncertainty_context" => 0.85,
            "ai_search_effort" => 1.4,
            "niche_creation_evidence" => 0.32,
        )

        ids = GlimpseABM._create_niche_opportunities_from_action!(sim, action, 7)
        created = [sim.market.opportunity_map[id] for id in ids]

        @test !isempty(ids)
        @test action["new_opportunity_id"] == first(ids)
        @test action["new_opportunity_ids"] == ids
        @test all(opp -> opp.discovered == false, created)
        @test all(opp -> opp.created_by == 1, created)
        @test all(opp -> opp.discovery_round == 7, created)
        @test all(opp -> opp.sector == "tech_specialized", created)
        @test all(opp -> opp.novelty_score > 0.55, created)
        @test all(opp -> opp.combination_uncertainty > 0.55, created)
        @test all(opp -> opp.required_discovery_threshold > 0.0, created)
        @test all(opp -> opp.combination_signature == "niche:tech:specialized", created)
        @test all(opp -> opp.component_scarcity > 0.55, created)
    end

    @testset "niche taxonomy emerges and reaches market creation" begin
        cfg = EmergentConfig(N_AGENTS=2, RANDOM_SEED=45)
        market = MarketEnvironment(cfg; rng=MersenneTwister(45))
        agent = EmergentAgent(1, cfg;
            primary_sector="tech",
            fixed_ai_level="premium",
            rng=MersenneTwister(45))
        agent.traits["exploration_tendency"] = 0.95
        agent.traits["market_awareness"] = 0.90
        agent.resources.knowledge["tech"] = 0.80

        taxonomy = GlimpseABM._select_niche_taxonomy(
            agent,
            market,
            "tech",
            ["tech", "service"],
            10;
            sector_novelty=0.90,
            sector_uncertainty=0.85,
            search_effort=1.5,
            niche_creation_evidence=0.30,
            current_knowledge=0.80,
            knowledge_gap=0.20,
            exploration_tendency=0.95,
            market_awareness=0.90,
        )

        @test startswith(taxonomy.niche_sector, "tech_")
        @test haskey(taxonomy.scores, taxonomy.modifier)
        @test taxonomy.scores["specialized"] > taxonomy.scores["standard"]

        sim = EmergentSimulation(config=cfg, seed=45, run_id="niche_taxonomy_probe")
        action = Dict{String,Any}(
            "action" => "explore",
            "exploration_type" => "niche_discovery",
            "agent_id" => 1,
            "discovered_sector" => "tech",
            "niche_modifier" => "digital",
            "niche_sector" => "tech_digital",
            "sector_novelty_context" => 0.75,
            "sector_uncertainty_context" => 0.65,
            "ai_search_effort" => 1.1,
            "niche_creation_evidence" => 0.28,
        )

        ids = GlimpseABM._create_niche_opportunities_from_action!(sim, action, 8)
        created = [sim.market.opportunity_map[id] for id in ids]

        @test !isempty(created)
        @test all(opp -> opp.sector == "tech_digital", created)
        @test all(opp -> opp.combination_signature == "niche:tech:digital", created)
    end

    # ------------------------------------------------------------------
    # 13. Fixed-mode runs are tier-homogeneous (ATE design invariant)
    # ------------------------------------------------------------------
    @testset "fixed-mode tier population is homogeneous" begin
        # Critical for ATE estimation: when a fixed-mode experiment runs the
        # "premium" cell, every agent must be premium. There must be no
        # mixed-tier competition contaminating the treatment assignment.
        for tier in ["none", "basic", "advanced", "premium"]
            cfg = EmergentConfig(N_AGENTS=20, N_ROUNDS=1, RANDOM_SEED=42,
                                 AGENT_AI_MODE="fixed")
            sim = EmergentSimulation(config=cfg, seed=42,
                initial_tier_distribution=Dict(tier => 1.0))
            tiers = unique(get_ai_level(a) for a in sim.agents)
            @test tiers == [tier]
            # And no agent should drift to another tier across rounds
            GlimpseABM.run!(sim)
            tiers_after = unique(get_ai_level(a) for a in sim.agents)
            @test tiers_after == [tier]
        end
    end

end

println("All diagnostic tests passed!")
