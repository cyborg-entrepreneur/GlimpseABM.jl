using Test
using Random

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using GlimpseABM

function behavioral_uncertainty_state(
    actions::Vector{Dict{String,Any}};
    ai_novelty_uplift::Float64 = 0.0,
    recursion_volatility_weight::Union{Nothing,Float64} = nothing,
    seed::Int = 91,
)
    config = EmergentConfig()
    config.N_AGENTS = length(actions)
    config.N_ROUNDS = 1
    config.AI_NOVELTY_UPLIFT = ai_novelty_uplift
    if !isnothing(recursion_volatility_weight)
        config.KNIGHTIAN_PRODUCER_WEIGHTS = KnightianProducerWeights(
            competitive_recursion=CompetitiveRecursionProducerWeights(
                volatility_weight=recursion_volatility_weight,
            ),
        )
    end
    config.enable_round_logging = false
    GlimpseABM.initialize!(config)

    market = MarketEnvironment(config; rng=MersenneTwister(seed))
    env = KnightianUncertaintyEnvironment(config; rng=MersenneTwister(seed + 1))

    GlimpseABM.step!(market, 1, actions, Innovation[])
    GlimpseABM.update_clearing_metrics!(market, actions)
    return GlimpseABM.measure_uncertainty_state!(env, market, actions, Innovation[], 1)
end

function crowding_maintain_actions(tier::String, n::Int; start_id::Int = 1)
    return [
        Dict{String,Any}(
            "action" => "maintain",
            "agent_id" => start_id + i - 1,
            "round" => 1,
            "ai_level_used" => tier,
            "ai_used" => tier != "none",
        )
        for i in 1:n
    ]
end

function crowding_invest_actions(
    tier::String,
    opp_ids::Vector{String};
    amount::Float64 = 1_000_000.0,
    start_id::Int = 1,
)
    return [
        Dict{String,Any}(
            "action" => "invest",
            "agent_id" => start_id + i - 1,
            "round" => 1,
            "ai_level_used" => tier,
            "ai_used" => tier != "none",
            "opportunity_id" => opp_id,
            "amount" => amount,
            "chosen_opportunity_details" => Dict{String,Any}(
                "id" => opp_id,
                "sector" => "tech",
            ),
        )
        for (i, opp_id) in enumerate(opp_ids)
    ]
end

@testset "Uncertainty crowding and novelty use behavioral channels" begin
    spread_state = behavioral_uncertainty_state(
        crowding_invest_actions("premium", ["opp_$(i)" for i in 1:24])
    )
    clustered_state = behavioral_uncertainty_state(
        crowding_invest_actions("premium", fill("opp_shared", 24))
    )

    @test clustered_state["practical_indeterminism"]["opportunity_overlap"] >
          spread_state["practical_indeterminism"]["opportunity_overlap"]
    @test clustered_state["practical_indeterminism"]["crowding_pressure"] >
          spread_state["practical_indeterminism"]["crowding_pressure"]
    # Crowding is opportunity-level: clustering capital into one opportunity
    # raises its competition/saturation, so opportunity_competition is at least
    # as high as the spread case (was: a sector_clearing_pressure check). Also
    # assert the new metric is well-formed (bounded [0,1]) so the comparison
    # cannot pass vacuously on a missing/NaN value.
    clustered_oc = clustered_state["practical_indeterminism"]["opportunity_competition"]
    spread_oc = spread_state["practical_indeterminism"]["opportunity_competition"]
    @test 0.0 <= clustered_oc <= 1.0
    @test 0.0 <= spread_oc <= 1.0
    @test clustered_oc >= spread_oc

    sparse_spread_state = behavioral_uncertainty_state(vcat(
        crowding_invest_actions("premium", ["opp_sparse_1", "opp_sparse_2"]; amount=1_000.0, start_id=1),
        crowding_maintain_actions("premium", 22; start_id=3),
    ))

    @test sparse_spread_state["practical_indeterminism"]["opportunity_overlap"] == 0.0
    @test sparse_spread_state["practical_indeterminism"]["investment_concentration"] > 0.0
    @test sparse_spread_state["practical_indeterminism"]["investment_volume"] < 0.10
    @test sparse_spread_state["practical_indeterminism"]["crowding_pressure"] < 0.15

    no_ai_state = behavioral_uncertainty_state(crowding_maintain_actions("none", 24))
    premium_state = behavioral_uncertainty_state(crowding_maintain_actions("premium", 24))

    @test premium_state["agentic_novelty"]["ai_usage_share"] == 1.0
    @test premium_state["agentic_novelty"]["ai_direct_novelty_uplift"] == 0.0
    @test isapprox(
        premium_state["agentic_novelty"]["level"],
        no_ai_state["agentic_novelty"]["level"];
        atol=1e-12,
    )

    uplift_state = behavioral_uncertainty_state(
        crowding_maintain_actions("premium", 24);
        ai_novelty_uplift=0.20,
    )
    @test uplift_state["agentic_novelty"]["ai_direct_novelty_uplift"] > 0.0
    @test uplift_state["agentic_novelty"]["level"] > premium_state["agentic_novelty"]["level"]

    @test hasproperty(EmergentConfig(), :UNCERTAINTY_AI_HERDING_WEIGHT)

    no_ai_recursion = behavioral_uncertainty_state(crowding_maintain_actions("none", 24))
    @test isapprox(no_ai_recursion["competitive_recursion"]["ai_delta"], 0.0; atol=1e-12)
    @test haskey(no_ai_recursion["competitive_recursion"], "components")
    @test haskey(no_ai_recursion["competitive_recursion"], "raw_score")

    no_ai_zero_vol_recursion = behavioral_uncertainty_state(
        crowding_maintain_actions("none", 24);
        recursion_volatility_weight=0.0,
    )
    no_ai_strong_vol_recursion = behavioral_uncertainty_state(
        crowding_maintain_actions("none", 24);
        recursion_volatility_weight=1.0,
    )
    @test isapprox(no_ai_zero_vol_recursion["competitive_recursion"]["ai_delta"], 0.0; atol=1e-12)
    @test isapprox(no_ai_strong_vol_recursion["competitive_recursion"]["ai_delta"], 0.0; atol=1e-12)
    @test no_ai_strong_vol_recursion["competitive_recursion"]["components"]["volatility"] >
          no_ai_zero_vol_recursion["competitive_recursion"]["components"]["volatility"]
    @test no_ai_strong_vol_recursion["competitive_recursion"]["level"] >
          no_ai_zero_vol_recursion["competitive_recursion"]["level"]
end

println("Uncertainty crowding behavioral regression tests passed.")
