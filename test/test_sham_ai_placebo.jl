using Test
using Random

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using GlimpseABM
include("test_helpers.jl")

@testset "Sham AI labels do not leak behavioral AI effects" begin
    config = EmergentConfig()
    GlimpseABM.initialize!(config)
    config.SHAM_AI_LABELS_USE_NONE = true
    config.AI_EXECUTION_SUCCESS_MULTIPLIERS["premium"] = 10.0
    config.AI_COST_INTENSITY = 0.0

    opp = Opportunity(
        id="sham_placebo_probe",
        latent_return_potential=2.0,
        latent_failure_potential=0.35,
        complexity=0.4,
        sector="tech",
        competition=2.0,
        capital_requirements=10_000.0,
        time_to_maturity=1,
        config=config,
    )
    market = MarketEnvironment(config; rng=MersenneTwister(11))
    agent = EmergentAgent(1, config; initial_capital=1_000_000.0,
                          fixed_ai_level="premium", rng=MersenneTwister(7))
    market_conditions = test_market_conditions(volatility=0.1)

    premium_label_score = GlimpseABM.evaluate_opportunity_basic(
        agent,
        opp,
        market_conditions;
        estimated_return=5.0,
        estimated_uncertainty=0.15,
        confidence=0.95,
    )
    explicit_premium_score = GlimpseABM.evaluate_opportunity_basic(
        agent,
        opp,
        market_conditions;
        estimated_return=5.0,
        estimated_uncertainty=0.15,
        confidence=0.95,
        ai_level="premium",
    )
    agent.fixed_ai_level = "none"
    none_label_score = GlimpseABM.evaluate_opportunity_basic(
        agent,
        opp,
        market_conditions;
        estimated_return=5.0,
        estimated_uncertainty=0.15,
        confidence=0.95,
    )
    @test premium_label_score == none_label_score
    @test explicit_premium_score == none_label_score

    agent.fixed_ai_level = "premium"
    perception = GlimpseABM.empty_perception()
    agent.rng = MersenneTwister(13)
    premium_evals = GlimpseABM.evaluate_portfolio_opportunities(
        agent,
        [opp],
        market_conditions,
        perception;
        ai_level="premium",
    )
    agent.rng = MersenneTwister(13)
    none_evals = GlimpseABM.evaluate_portfolio_opportunities(
        agent,
        [opp],
        market_conditions,
        perception;
        ai_level="none",
    )
    @test only(premium_evals).ai_level_used == "none"
    @test only(premium_evals).final_score == only(none_evals).final_score

    outcome = GlimpseABM.execute_action!(
        agent,
        "invest",
        market,
        1;
        opportunity=opp,
        estimated_return=1.2,
        confidence=0.8,
        signal_score=1.0,
    )

    @test outcome["ai_level_used"] == "premium"
    @test outcome["ai_used"] === false
    @test only(agent.active_investments)["ai_level"] == "none"

    amount = Float64(outcome["amount"])
    expected_multiple = GlimpseABM.realized_return(
        opp,
        market_conditions,
        "none";
        rng=MersenneTwister(99),
    )
    agent.rng = MersenneTwister(99)
    before_maturity = GlimpseABM.get_capital(agent)
    matured = GlimpseABM.process_matured_investments!(
        agent,
        market,
        2;
        market_conditions=market_conditions,
    )
    @test only(matured)["ai_level"] == "none"
    @test only(matured)["ai_used"] === false
    @test isapprox(GlimpseABM.get_capital(agent) - before_maturity,
                   amount * expected_multiple; atol=1e-8)

    actions = [
        merge(copy(outcome), Dict{String,Any}(
            "chosen_opportunity_obj" => opp,
            "amount" => amount,
            "ai_level_used" => "premium",
            "ai_used" => false,
        )),
        Dict{String,Any}(
            "action" => "invest",
            "agent_id" => 2,
            "round" => 1,
            "ai_level_used" => "premium",
            "ai_used" => false,
            "amount" => amount,
            "opportunity_id" => opp.id,
            "chosen_opportunity_obj" => opp,
        ),
    ]

    GlimpseABM.update_clearing_metrics!(market, actions)
    @test market.tier_invest_share["none"] == 1.0
    @test market.tier_invest_share["premium"] == 0.0

    env = KnightianUncertaintyEnvironment(config; rng=MersenneTwister(12))
    state = GlimpseABM.measure_uncertainty_state!(env, market, actions, Innovation[], 1)
    @test state["competitive_recursion"]["ai_premium_share"] == 0.0
    @test env._volatility_state["last_ai_shares"] == [1.0, 0.0, 0.0, 0.0]
end

println("Sham AI placebo regression tests passed.")
