using Test
using Random

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using GlimpseABM

@testset "Private discovery emerges from signal and search capacity" begin
    cfg = EmergentConfig(N_AGENTS=2, RANDOM_SEED=42)
    GlimpseABM.initialize!(cfg)
    market = MarketEnvironment(cfg; rng=MersenneTwister(1))

    data_rich = Opportunity(
        id="data_rich_hidden",
        latent_return_potential=2.0,
        latent_failure_potential=0.25,
        complexity=0.0,
        discovered=false,
        created_by=nothing,
        discovery_round=nothing,
        config=cfg,
        competition=10.0,
        sector="tech",
        capital_requirements=100_000.0,
        market_impact=25.0,
        total_invested=10_000_000.0,
        capacity=1_000_000.0,
        novelty_score=0.0,
        combination_uncertainty=0.0,
        required_discovery_threshold=0.0,
   )
    push!(market.opportunities, data_rich)
    market.opportunity_map[data_rich.id] = data_rich

    premium = EmergentAgent(1, cfg; primary_sector="tech", fixed_ai_level="premium",
                            rng=MersenneTwister(2))
    premium.traits["exploration_tendency"] = 1.0
    premium.traits["market_awareness"] = 1.0
    premium.resources.knowledge["tech"] = 1.0

    none = EmergentAgent(2, cfg; primary_sector="retail", fixed_ai_level="none",
                         rng=MersenneTwister(3))
    none.traits["exploration_tendency"] = 0.0
    none.traits["market_awareness"] = 0.0
    none.resources.knowledge["tech"] = 0.0

    p_premium = GlimpseABM.opportunity_discovery_probability(market, premium, data_rich, "premium")
    p_none = GlimpseABM.opportunity_discovery_probability(market, none, data_rich, "none")
    @test p_premium > 0.95
    @test p_premium < 1.0
    @test p_none == 0.0

    visible_to_premium = GlimpseABM.get_opportunities_for_agent(market, premium)
    @test data_rich.id in [opp.id for opp in visible_to_premium]
    @test data_rich.id in premium.resources.private_opportunity_ids
    @test data_rich.discovered == false
    @test data_rich.discovery_round === nothing

    visible_to_none = GlimpseABM.get_opportunities_for_agent(market, none)
    @test !(data_rich.id in [opp.id for opp in visible_to_none])
end

@testset "Low-signal unknown opportunities remain harder even for frontier AI" begin
    cfg = EmergentConfig(N_AGENTS=1, RANDOM_SEED=7)
    GlimpseABM.initialize!(cfg)
    market = MarketEnvironment(cfg; rng=MersenneTwister(7))
    agent = EmergentAgent(1, cfg; primary_sector="tech", fixed_ai_level="premium",
                          rng=MersenneTwister(8))
    agent.traits["exploration_tendency"] = 1.0
    agent.traits["market_awareness"] = 1.0
    agent.resources.knowledge["tech"] = 1.0

    data_rich = Opportunity(
        id="data_rich_probe",
        complexity=0.0,
        discovered=false,
        sector="tech",
        competition=10.0,
        market_impact=25.0,
        total_invested=10_000_000.0,
        capacity=1_000_000.0,
        novelty_score=0.0,
        combination_uncertainty=0.0,
        required_discovery_threshold=0.0,
   )
    tacit_unknown = Opportunity(
        id="tacit_unknown_probe",
        complexity=2.0,
        discovered=false,
        sector="tech",
        competition=0.0,
        market_impact=0.0,
        total_invested=0.0,
        capacity=1_000_000.0,
        novelty_score=1.0,
        combination_uncertainty=1.0,
        required_discovery_threshold=1.0,
        truly_unknown=true,
   )

    p_data_rich = GlimpseABM.opportunity_discovery_probability(market, agent, data_rich, "premium")
    p_tacit = GlimpseABM.opportunity_discovery_probability(market, agent, tacit_unknown, "premium")

    @test p_data_rich > 0.95
    @test p_tacit > 0.0
    @test p_tacit < p_data_rich / 25.0
end

@testset "Public discovery follows observable market traces" begin
    cfg = EmergentConfig(N_AGENTS=10, RANDOM_SEED=9)
    GlimpseABM.initialize!(cfg)

    no_trace = Opportunity(
        id="no_trace",
        discovered=false,
        sector="tech",
        competition=0.0,
        total_invested=0.0,
        market_impact=0.0,
        capacity=1_000_000.0,
   )
    visible_trace = Opportunity(
        id="visible_trace",
        discovered=false,
        sector="tech",
        competition=8.0,
        total_invested=5_000_000.0,
        market_impact=10.0,
        capacity=1_000_000.0,
   )

    @test GlimpseABM._public_diffusion_probability(no_trace, 0) == 0.0
    @test GlimpseABM._public_diffusion_probability(visible_trace, 10) > 0.20
end

println("Private discovery regression tests passed.")
