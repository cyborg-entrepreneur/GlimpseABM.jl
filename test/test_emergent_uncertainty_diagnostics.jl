using Test
using GlimpseABM
using Random

@testset "Emergent uncertainty diagnostics" begin
    @testset "Actor ignorance preserves venture-scale error ordering" begin
        moderate = AgentUncertaintyMetrics()
        record_investment_outcome!(moderate, 2.0, 1.0, 0.0)
        moderate_actor = compute_emergent_uncertainty(moderate)["actor_ignorance"]

        extreme = AgentUncertaintyMetrics()
        record_investment_outcome!(extreme, 21.0, 1.0, 0.0)
        extreme_actor = compute_emergent_uncertainty(extreme)["actor_ignorance"]

        @test 0.0 < moderate_actor < extreme_actor < 1.0
    end

    @testset "Competitive recursion reflects observed crowding exposure" begin
        low = AgentUncertaintyMetrics()
        record_investment_outcome!(low, 1.0, 1.0, 0.5)

        high = AgentUncertaintyMetrics()
        record_investment_outcome!(high, 1.0, 1.0, 3.0)

        low_recursion = compute_emergent_uncertainty(low)["competitive_recursion"]
        high_recursion = compute_emergent_uncertainty(high)["competitive_recursion"]

        @test 0.0 < low_recursion < high_recursion < 1.0
        @test high_recursion > 0.5
    end

    @testset "Tier aggregation exposes survivor and all-agent evidence scopes" begin
        cfg = EmergentConfig(N_AGENTS=2, N_ROUNDS=1, RANDOM_SEED=123)
        GlimpseABM.initialize!(cfg)
        alive_agent = EmergentAgent(
            1,
            cfg;
            fixed_ai_level="premium",
            rng=MersenneTwister(123),
       )
        failed_agent = EmergentAgent(
            2,
            cfg;
            fixed_ai_level="premium",
            rng=MersenneTwister(124),
       )
        failed_agent.alive = false

        record_investment_outcome!(alive_agent.uncertainty_metrics, 1.2, 1.0, 0.5)
        record_investment_outcome!(failed_agent.uncertainty_metrics, 5.0, 1.0, 3.0)

        survivor = aggregate_emergent_uncertainty_by_tier([alive_agent, failed_agent])
        all_agents = aggregate_emergent_uncertainty_by_tier(
            [alive_agent, failed_agent];
            include_dead=true,
       )

        @test survivor["premium"]["n_agents"] == 1.0
        @test all_agents["premium"]["n_agents"] == 2.0
        @test all_agents["premium"]["n_failed"] == 1.0
        @test all_agents["premium"]["actor_ignorance_observations"] == 2.0
        @test all_agents["premium"]["competitive_recursion"] > survivor["premium"]["competitive_recursion"]
    end

    @testset "Matured investment path records entry and capacity crowding" begin
        cfg = EmergentConfig(N_AGENTS=2, N_ROUNDS=2, RANDOM_SEED=944)
        GlimpseABM.initialize!(cfg)
        market = MarketEnvironment(cfg; rng=MersenneTwister(944))
        market_conditions = GlimpseABM.get_market_conditions(market)
        agent = EmergentAgent(
            1,
            cfg;
            primary_sector="tech",
            initial_capital=5_000_000.0,
            fixed_ai_level="none",
            rng=MersenneTwister(945),
       )
        opp = Opportunity(
            id="crowding-exposure-path",
            latent_return_potential=1.2,
            latent_failure_potential=0.1,
            discovered=true,
            sector="tech",
            time_to_maturity=1,
            competition=0.50,
            total_invested=200_000.0,
            capacity=100_000.0,
            config=cfg,
            rng=MersenneTwister(946),
       )

        push!(agent.active_investments, Dict{String,Any}(
            "opportunity" => opp,
            "amount" => 100_000.0,
            "maturity_round" => 1,
            "ai_level" => "none",
            "ai_label" => "none",
            "estimated_return" => 1.1,
            "competition_at_entry" => 0.20,
            "capacity_saturation_at_entry" => 1.60,
       ))

        matured = GlimpseABM.process_matured_investments!(
            agent,
            market,
            1;
            market_conditions=market_conditions,
       )

        @test length(matured) == 1
        outcome = only(matured)
        @test outcome["competition_at_entry"] ≈ 0.20
        @test outcome["competition_at_maturity"] ≈ 0.50
        @test outcome["capacity_saturation_at_entry"] ≈ 1.60
        @test outcome["capacity_saturation_at_maturity"] ≈ 2.00
        @test outcome["crowding_exposure"] ≈ 2.00
        @test only(agent.uncertainty_metrics.competition_levels) ≈ 2.00
        @test compute_emergent_uncertainty(agent.uncertainty_metrics)["competitive_recursion"] > 0.7
    end
end
