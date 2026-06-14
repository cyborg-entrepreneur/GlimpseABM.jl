using Test
using Random

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using GlimpseABM

function recursion_probe_state(actions::Vector{Dict{String,Any}}; seed::Int=77)
    config = EmergentConfig()
    config.N_AGENTS = length(actions)
    config.N_ROUNDS = 1
    config.enable_round_logging = false
    GlimpseABM.initialize!(config)

    market = MarketEnvironment(config; rng=MersenneTwister(seed))
    env = KnightianUncertaintyEnvironment(config; rng=MersenneTwister(seed + 1))
    return GlimpseABM.measure_uncertainty_state!(env, market, actions, Innovation[], 1)
end

function maintain_actions(tier::String, n::Int)
    return [
        Dict{String,Any}(
            "action" => "maintain",
            "agent_id" => i,
            "round" => 1,
            "ai_level_used" => tier,
            "ai_used" => tier != "none",
       )
        for i in 1:n
    ]
end

function reuse_actions(tier::String, n::Int)
    return [
        Dict{String,Any}(
            "action" => "innovate",
            "agent_id" => i,
            "round" => 1,
            "ai_level_used" => tier,
            "ai_used" => tier != "none",
            "combination_signature" => "shared_combo",
            "is_new_combination" => false,
       )
        for i in 1:n
    ]
end

function invest_actions(tier::String, opp_ids::Vector{String})
    return [
        Dict{String,Any}(
            "action" => "invest",
            "agent_id" => i,
            "round" => 1,
            "ai_level_used" => tier,
            "ai_used" => tier != "none",
            "opportunity_id" => opp_id,
            "amount" => 1000.0,
       )
        for (i, opp_id) in enumerate(opp_ids)
    ]
end

@testset "Competitive recursion emerges from behavior, not premium labels" begin
    no_ai_state = recursion_probe_state(maintain_actions("none", 24))
    premium_state = recursion_probe_state(maintain_actions("premium", 24))

    @test isapprox(
        premium_state["competitive_recursion"]["level"],
        no_ai_state["competitive_recursion"]["level"];
        atol=1e-12,
   )

    basic_reuse = recursion_probe_state(reuse_actions("basic", 24))
    premium_reuse = recursion_probe_state(reuse_actions("premium", 24))

    @test isapprox(
        premium_reuse["competitive_recursion"]["level"],
        basic_reuse["competitive_recursion"]["level"];
        atol=1e-12,
   )

    spread_ids = ["opp_$(i)" for i in 1:24]
    clustered_ids = fill("opp_shared", 24)
    spread_state = recursion_probe_state(invest_actions("premium", spread_ids))
    clustered_state = recursion_probe_state(invest_actions("premium", clustered_ids))

    @test clustered_state["competitive_recursion"]["level"] >
          spread_state["competitive_recursion"]["level"]
    @test clustered_state["competitive_recursion"]["ai_action_correlation"] >
          spread_state["competitive_recursion"]["ai_action_correlation"]
end

println("Competitive recursion behavioral regression tests passed.")
