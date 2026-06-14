"""
Test suite for GlimpseABM.jl
"""

using Test
using GlimpseABM
using Random
using DataFrames

@testset "GlimpseABM.jl" begin

    @testset "Configuration" begin
 # Test default config creation
        config = EmergentConfig()
        @test config.N_AGENTS == 1000
        @test config.N_ROUNDS == 120
        @test config.RANDOM_SEED == 42

 # Test sectors populated
        GlimpseABM.initialize!(config)
        @test !isempty(config.SECTORS)
        @test "tech" in config.SECTORS

 # Test scaled opportunities
        n_opps = GlimpseABM.get_scaled_opportunities(config, 1000)
        @test n_opps >= config.MIN_OPPORTUNITIES
    end

    @testset "Models" begin
 # Test Opportunity creation
        opp = Opportunity(
            id="test_opp_1",
            latent_return_potential=1.5,
            latent_failure_potential=0.3,
            complexity=0.5
       )
        @test opp.id == "test_opp_1"
        @test opp.latent_return_potential == 1.5
        @test opp.discovered == false

 # Test Information creation
        info = Information(
            estimated_return=1.2,
            estimated_uncertainty=0.3,
            confidence=0.8
       )
        @test info.estimated_return == 1.2
        @test GlimpseABM.quality_score(info) > 0

 # Test Innovation creation
        innov = Innovation(
            id="innov_1",
            type="incremental",
            knowledge_components=["comp1", "comp2"],
            novelty=0.7,
            quality=0.6,
            round_created=1,
            creator_id=1
       )
        @test innov.novelty == 0.7

 # Test AILearningProfile
        profile = AILearningProfile()
        @test haskey(profile.domain_trust, "market_analysis")
        @test profile.domain_trust["market_analysis"] == 0.5
    end

    @testset "Agents" begin
        config = EmergentConfig()
        GlimpseABM.initialize!(config)

        rng = MersenneTwister(42)

 # Test agent creation
        agent = EmergentAgent(1, config; rng=rng)
        @test agent.id == 1
        @test agent.alive == true
        @test GlimpseABM.get_capital(agent) > 0

 # Test capital operations
        initial_capital = GlimpseABM.get_capital(agent)
        GlimpseABM.set_capital!(agent, initial_capital + 1000)
        @test GlimpseABM.get_capital(agent) == initial_capital + 1000

 # Test AI level
        @test GlimpseABM.get_ai_level(agent) == "none"
        agent.fixed_ai_level = "premium"
        @test GlimpseABM.get_ai_level(agent) == "premium"
    end

    @testset "Market" begin
        config = EmergentConfig()
        GlimpseABM.initialize!(config)

        rng = MersenneTwister(42)

 # Test market creation
        market = MarketEnvironment(config; rng=rng)
        @test market.market_regime == "normal"
        @test !isempty(market.opportunities)

 # Test get available opportunities
        opps = GlimpseABM.get_available_opportunities(market)
        @test !isempty(opps)

 # Test market conditions
        conditions = GlimpseABM.get_market_conditions(market)
        @test haskey(conditions, "regime")
        @test conditions["regime"] == "normal"
    end

    @testset "Uncertainty" begin
        config = EmergentConfig()
        rng = MersenneTwister(42)

 # Test uncertainty environment creation
        unc_env = KnightianUncertaintyEnvironment(config; rng=rng)

 # Test initial state
        @test haskey(unc_env.actor_ignorance_state, "level")
        @test haskey(unc_env.practical_indeterminism_state, "level")
        @test haskey(unc_env.agentic_novelty_state, "level")
        @test haskey(unc_env.competitive_recursion_state, "level")

 # Test composite uncertainty
        composite = GlimpseABM.get_composite_uncertainty(unc_env)
        @test composite >= 0.0
        @test composite <= 1.0
    end

    @testset "Simulation Mini Run" begin
 # Create minimal config for fast test
        config = EmergentConfig()
        config.N_AGENTS = 10
        config.N_ROUNDS = 5
        config.enable_round_logging = false
        GlimpseABM.initialize!(config)

 # Run simulation
        sim = EmergentSimulation(
            config=config,
            output_dir="/tmp/glimpse_test",
            run_id="test_run",
            seed=42
       )

        GlimpseABM.run!(sim)

 # Verify simulation ran
        @test sim.current_round == 5
        @test !isempty(sim.history)

 # Check some agents survived
        survivors = count(a -> a.alive, sim.agents)
        @test survivors >= 0

 # Test history to DataFrame
        df = GlimpseABM.history_to_dataframe(sim)
        @test nrow(df) == 5

 # Test summary stats
        stats = GlimpseABM.summary_stats(sim)
        @test haskey(stats, "final_survival_rate")
        @test stats["n_rounds"] == 5
    end

    @testset "Utilities" begin
        config = EmergentConfig()
        rng = MersenneTwister(42)

 # Test trait sampling
        traits = GlimpseABM.sample_all_traits(config; rng=rng)
        @test haskey(traits, "uncertainty_tolerance")
        @test traits["uncertainty_tolerance"] >= 0.0
        @test traits["uncertainty_tolerance"] <= 1.0

 # Test weighted choice
        items = ["a", "b", "c"]
        weights = [0.7, 0.2, 0.1]
        choice = GlimpseABM.weighted_choice(items, weights; rng=rng)
        @test choice in items

 # Test HHI computation
        shares = [0.5, 0.3, 0.2]
        hhi = GlimpseABM.compute_hhi(shares)
        @test hhi > 0.0
        @test hhi <= 1.0

 # Test normalize probs
        probs = [2.0, 3.0, 5.0]
        normalized = GlimpseABM.normalize_probs(probs)
        @test isapprox(sum(normalized), 1.0)
    end

 # Run external regression tests (each in its own @testset)
    include("test_tier_divergence.jl")
    include("test_diagnostics.jl")
    include("test_action_keys.jl")
    include("test_market_conditions.jl")
    include("test_crowding_convexity.jl")
    include("test_confidence_sizing.jl")
    include("test_knightian_pathway.jl")
    include("test_demand_clamp.jl")
    include("test_snapshot.jl")
    include("test_io_roundtrip.jl")
    include("test_opportunity_evaluation.jl")
    include("test_perception.jl")
    include("test_sham_ai_placebo.jl")
    include("test_private_discovery.jl")
    include("test_survival_threshold_invariance.jl")
    include("test_tier_label_bonus_regression.jl")
    include("test_decision_utility_ai_bonus_regression.jl")
    include("test_ai_assisted_innovation_regression.jl")
    include("test_competitive_recursion_behavioral.jl")
    include("test_uncertainty_crowding_behavioral.jl")
    include("test_derivative_novelty_regression.jl")
    include("test_ai_signal_uncertainty_regression.jl")
    include("test_observable_ai_disconfirmation_regression.jl")
    include("test_variance_wiring.jl")
    include("test_confidence_outcome_diagnostics.jl")
    include("test_emergent_uncertainty_diagnostics.jl")
    include("test_paradox_mechanism_scorecard.jl")
    include("test_engine_invariants.jl")
    include("test_fixed_tier_assignment.jl")
    include("test_strategy_ladder.jl")
    include("test_open_action_space.jl")
    include("test_emergence_audit.jl")
    include("test_robustness_addendum_knobs.jl")
    include("test_opportunity_tail_clamp.jl")

end

println("All tests passed!")
