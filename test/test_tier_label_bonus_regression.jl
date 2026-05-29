using Test
using Random

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using GlimpseABM
include("test_helpers.jl")

function equalize_ai_primitives_to_none!(config::EmergentConfig)
    none_cfg = config.AI_LEVELS["none"]
    for tier in ["basic", "advanced", "premium"]
        old = config.AI_LEVELS[tier]
        config.AI_LEVELS[tier] = GlimpseABM.AILevelConfig(
            old.cost,
            old.cost_type,
            none_cfg.info_quality,
            none_cfg.info_breadth,
            old.per_use_cost,
        )
        config.AI_DOMAIN_CAPABILITIES[tier] = deepcopy(config.AI_DOMAIN_CAPABILITIES["none"])
        config.AI_QUALITY_BOOST[tier] = config.AI_QUALITY_BOOST["none"]
        config.AI_INFORMATION_QUALITY_BOOSTS[tier] = config.AI_INFORMATION_QUALITY_BOOSTS["none"]
        config.AI_EXECUTION_SUCCESS_MULTIPLIERS[tier] = config.AI_EXECUTION_SUCCESS_MULTIPLIERS["none"]
    end
    return config
end

function accessible_knowledge_ids_for(tier::String; seed::Int=1)
    config = EmergentConfig()
    GlimpseABM.initialize!(config)
    equalize_ai_primitives_to_none!(config)

    kb = KnowledgeBase(config)
    traits = Dict("exploration_tendency" => 0.0)
    accessible = get_accessible_knowledge(
        kb,
        1,
        tier;
        agent_traits=traits,
        rng=MersenneTwister(seed),
    )
    return sort([k.id for k in accessible])
end

function innovation_attempt_fixture(tier::String)
    config = EmergentConfig()
    GlimpseABM.initialize!(config)
    equalize_ai_primitives_to_none!(config)

    kb = KnowledgeBase(config)
    kb.agent_knowledge[1] = Set([k.id for k in kb.knowledge_registry])
    engine = InnovationEngine(config, kb, CombinationTracker())

    agent = EmergentAgent(
        1,
        config;
        initial_capital=1_000_000.0,
        primary_sector="tech",
        fixed_ai_level=tier,
        rng=MersenneTwister(99),
    )
    for key in keys(agent.traits)
        agent.traits[key] = 0.0
    end
    agent.traits["innovativeness"] = 0.0
    agent.traits["exploration_tendency"] = 0.0
    agent.traits["market_awareness"] = 0.0
    agent.competence = 0.0

    return engine, agent
end

@testset "Tier labels do not create hidden knowledge access bonuses" begin
    none_ids = accessible_knowledge_ids_for("none"; seed=1)
    premium_ids = accessible_knowledge_ids_for("premium"; seed=1)

    @test premium_ids == none_ids
end

@testset "Tier labels do not create hidden innovation attempt bonuses" begin
    market_conditions = test_market_conditions(volatility=0.1, round=1)

    none_engine, none_agent = innovation_attempt_fixture("none")
    premium_engine, premium_agent = innovation_attempt_fixture("premium")

    none_innovation = attempt_innovation!(
        none_engine,
        none_agent,
        market_conditions,
        1;
        ai_level="none",
        rng=MersenneTwister(1),
    )
    premium_innovation = attempt_innovation!(
        premium_engine,
        premium_agent,
        market_conditions,
        1;
        ai_level="premium",
        rng=MersenneTwister(1),
    )

    @test isnothing(premium_innovation) == isnothing(none_innovation)
end

println("Tier-label bonus regression tests passed.")
