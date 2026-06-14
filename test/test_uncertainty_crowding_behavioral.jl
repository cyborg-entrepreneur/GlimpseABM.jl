using Test
using Random

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using GlimpseABM

function behavioral_uncertainty_state(
    actions_or_builder;
    n_agents::Union{Nothing,Int} = nothing,
    ai_novelty_uplift::Float64 = 0.0,
    recursion_volatility_weight::Union{Nothing,Float64} = nothing,
    seed::Int = 91,
)
    config = EmergentConfig()
    config.N_AGENTS = isnothing(n_agents) ?
        (actions_or_builder isa Vector ? length(actions_or_builder) : 24) : n_agents
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

 # A builder closure receives the live market so synthetic actions can
 # reference REAL Opportunity objects. The opportunity_competition signal is
 # an investment-weighted aggregate over market.opportunities (competition
 # trace + total_invested/capacity saturation), and market.step! bumps
 # competition only via action["chosen_opportunity_obj"] — fabricated
 # string IDs never touch it. The pre-fix harness used fabricated IDs, so
 # the signal evaluated 0.0 on BOTH sides of every comparison and the
 # crowding assertions passed vacuously (review finding, 2026-06-09).
    actions = actions_or_builder isa Vector ? actions_or_builder :
              actions_or_builder(market)

    GlimpseABM.step!(market, 1, actions, Innovation[])
    GlimpseABM.update_clearing_metrics!(market, actions)
    return GlimpseABM.measure_uncertainty_state!(env, market, actions, Innovation[], 1)
end

# Build invest actions against the market's real opportunities. `cluster=true`
# pours every agent's capital into one opportunity; otherwise capital is spread
# across distinct opportunities (cycling if fewer exist than agents). Funds
# opp.total_invested directly — the harness has no full agents to route
# through _execute_invest!, and the signal under test reads the funded state,
# not the execution path.
function crowding_invest_actions_real(
    market, tier::String, n::Int;
    cluster::Bool, amount::Float64 = 1_000_000.0, start_id::Int = 1,
)
    opps = market.opportunities
    @assert !isempty(opps) "market generated no opportunities"
    actions = Dict{String,Any}[]
    for i in 1:n
        opp = cluster ? opps[1] : opps[mod1(i, length(opps))]
        opp.total_invested += amount
        push!(actions, Dict{String,Any}(
            "action" => "invest",
            "agent_id" => start_id + i - 1,
            "round" => 1,
            "ai_level_used" => tier,
            "ai_used" => tier != "none",
            "opportunity_id" => opp.id,
            "amount" => amount,
            "chosen_opportunity_obj" => opp,
            "chosen_opportunity_details" => Dict{String,Any}(
                "id" => opp.id,
                "sector" => hasfield(typeof(opp), :sector) ? opp.sector : "tech",
           ),
       ))
    end
    return actions
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
 # Amount sized against the N=24 market: ~5 opportunities at ~$2.0–2.8M
 # capacity. Spread (24 × $100k over 5 opps ≈ $480k each) stays well under
 # capacity; clustered (24 × $100k = $2.4M into one) saturates it. Both the
 # saturation leg and the competition-trace leg of the signal then separate
 # the two cases instead of clamping to 1.0 on both sides.
    spread_state = behavioral_uncertainty_state(
        market -> crowding_invest_actions_real(market, "premium", 24;
                                               cluster=false, amount=100_000.0);
        n_agents=24,
   )
    clustered_state = behavioral_uncertainty_state(
        market -> crowding_invest_actions_real(market, "premium", 24;
                                               cluster=true, amount=100_000.0);
        n_agents=24,
   )

    @test clustered_state["practical_indeterminism"]["opportunity_overlap"] >
          spread_state["practical_indeterminism"]["opportunity_overlap"]
    @test clustered_state["practical_indeterminism"]["crowding_pressure"] >
          spread_state["practical_indeterminism"]["crowding_pressure"]
 # Crowding is opportunity-level: clustering capital into one opportunity
 # saturates it (24× the per-opportunity capital of the spread case), so
 # opportunity_competition must be STRICTLY higher — and alive on both
 # sides. The non-degeneracy floors are the real guard: a regression that
 # zeroes the signal (the pre-fix harness state) fails them immediately
 # instead of passing 0.0 >= 0.0. This channel carries the full
 # crowding_opportunity_competition_weight producer weight inherited from
 # the deleted sector-level terms.
    clustered_oc = clustered_state["practical_indeterminism"]["opportunity_competition"]
    spread_oc = spread_state["practical_indeterminism"]["opportunity_competition"]
    @test 0.0 <= clustered_oc <= 1.0
    @test 0.0 <= spread_oc <= 1.0
    @test clustered_oc > spread_oc          # strict: clustering must register
    @test clustered_oc > 0.25               # clustered case is genuinely crowded
    @test spread_oc > 0.0                   # signal alive in the spread case too

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
