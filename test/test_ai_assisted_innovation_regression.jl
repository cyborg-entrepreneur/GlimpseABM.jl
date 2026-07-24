using Test
using Random

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using GlimpseABM
include("test_helpers.jl")

function innovation_engine_fixture()
    config = EmergentConfig()
    GlimpseABM.initialize!(config)
    kb = KnowledgeBase(config)
    return InnovationEngine(config, kb, CombinationTracker())
end

function innovation_probe_knowledge()
    return [
        Knowledge(id="probe_tech_low", domain="technology", level=0.20, discovered_round=0),
        Knowledge(id="probe_market_mid", domain="market", level=0.30, discovered_round=0),
        Knowledge(id="probe_process_high", domain="process", level=0.80, discovered_round=0),
        Knowledge(id="probe_business_high", domain="business_model", level=0.90, discovered_round=0),
    ]
end

@testset "Default AI assistance does not add hidden innovation quality" begin
    engine = innovation_engine_fixture()
    agent = EmergentAgent(1, engine.config; initial_capital=1_000_000.0,
                          primary_sector="tech", fixed_ai_level="premium",
                          rng=MersenneTwister(2))
    pieces = innovation_probe_knowledge()[1:2]

    human = GlimpseABM.create_innovation(
        engine,
        pieces,
        "incremental",
        agent,
        1;
        ai_assisted=false,
        ai_level_used="none",
        rng=MersenneTwister(7),
    )
    assisted = GlimpseABM.create_innovation(
        engine,
        pieces,
        "incremental",
        agent,
        1;
        ai_assisted=true,
        ai_domains_used=["technical_assessment"],
        ai_level_used="premium",
        rng=MersenneTwister(7),
    )

    @test assisted.quality == human.quality
end

@testset "AI assistance does not directly shift innovation type" begin
    engine = innovation_engine_fixture()
    pieces = innovation_probe_knowledge()[1:2]
    traits = Dict(
        "trait_momentum" => 0.1,
        "innovativeness" => 0.5,
        "exploration_tendency" => 0.5,
    )
    market_conditions = test_market_conditions(volatility=0.1, round=1)

    human_type = GlimpseABM.determine_innovation_type(
        engine,
        pieces,
        traits,
        market_conditions;
        ai_assisted=false,
        rng=MersenneTwister(13),
    )
    assisted_type = GlimpseABM.determine_innovation_type(
        engine,
        pieces,
        traits,
        market_conditions;
        ai_assisted=true,
        rng=MersenneTwister(13),
    )

    @test assisted_type == human_type
end

@testset "AI tier labels do not directly rewrite knowledge combinations" begin
    engine = innovation_engine_fixture()
    accessible = innovation_probe_knowledge()
    usage = engine.knowledge_base.knowledge_usage
    usage["probe_tech_low"] = 100
    usage["probe_market_mid"] = 80
    usage["probe_process_high"] = 0
    usage["probe_business_high"] = 1

    traits = Dict("innovativeness" => 0.5)

    none_combo = GlimpseABM.select_knowledge_combination(
        engine,
        accessible,
        2,
        traits,
        nothing,
        "none";
        rng=MersenneTwister(1),
    )
    premium_combo = GlimpseABM.select_knowledge_combination(
        engine,
        accessible,
        2,
        traits,
        nothing,
        "premium";
        rng=MersenneTwister(1),
    )

    @test [k.id for k in premium_combo] == [k.id for k in none_combo]
end

@testset "AI breadth does not directly suppress combination reuse" begin
    function reuse_probe(tier::String)
        config = EmergentConfig()
        GlimpseABM.initialize!(config)
        config.INNOVATION_PROBABILITY = 1.0
        config.INNOVATION_REUSE_PROBABILITY = 0.5

        kb = KnowledgeBase(config)
        kb.agent_knowledge[1] = Set([k.id for k in kb.knowledge_registry])

        tracker = CombinationTracker()
        reuse_signature = join(sort([k.id for k in kb.knowledge_registry[1:5]]), "||")
        GlimpseABM.record_combination!(tracker, reuse_signature, "prior_innovation")
        engine = InnovationEngine(config, kb, tracker)

        agent = EmergentAgent(
            1,
            config;
            initial_capital=1_000_000.0,
            primary_sector="tech",
            fixed_ai_level=tier,
            rng=MersenneTwister(11),
        )
        for key in keys(agent.traits)
            agent.traits[key] = 1.0
        end
        agent.competence = 1.0
        agent.innovativeness = 1.0

        innovation = GlimpseABM.attempt_innovation!(
            engine,
            agent,
            test_market_conditions(volatility=0.1, round=1),
            1;
            ai_level=tier,
            rng=MersenneTwister(11),
        )
        return innovation, reuse_signature
    end

    none_innovation, reuse_signature = reuse_probe("none")
    premium_innovation, _ = reuse_probe("premium")

    @test !isnothing(none_innovation)
    @test !isnothing(premium_innovation)
    @test none_innovation.combination_signature == reuse_signature
    @test premium_innovation.combination_signature == reuse_signature
end

@testset "AI assistance does not add hidden market-potential multiplier" begin
    market_conditions = test_market_conditions(volatility=0.1, round=1)
    human = Innovation(
        id="human_potential_probe",
        type="radical",
        knowledge_components=["a", "b"],
        novelty=0.60,
        quality=0.70,
        round_created=1,
        creator_id=1,
        ai_assisted=false,
    )
    assisted = Innovation(
        id="assisted_potential_probe",
        type="radical",
        knowledge_components=["a", "b"],
        novelty=0.60,
        quality=0.70,
        round_created=1,
        creator_id=1,
        ai_assisted=true,
        ai_domains_used=["technical_assessment"],
        ai_level_used="premium",
    )

    @test GlimpseABM.calculate_potential(assisted, market_conditions) ==
          GlimpseABM.calculate_potential(human, market_conditions)
end

println("AI-assisted innovation tests passed.")
