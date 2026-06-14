using Test
using Random
using Statistics

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using GlimpseABM

include(joinpath(@__DIR__, "..", "scripts", "run_mixed_tier_analysis_full.jl"))

@testset "Mixed-tier treatment effects use paired run contrasts" begin
    all_results = [
        Dict{String,Any}(
            "status" => "completed",
            "run_idx" => 1,
            "tier_stats" => Dict(
                "none" => Dict("survival_rate" => 0.50),
                "basic" => Dict("survival_rate" => 0.52),
                "advanced" => Dict("survival_rate" => 0.48),
                "premium" => Dict("survival_rate" => 0.42),
           ),
       ),
        Dict{String,Any}(
            "status" => "completed",
            "run_idx" => 2,
            "tier_stats" => Dict(
                "none" => Dict("survival_rate" => 0.60),
                "basic" => Dict("survival_rate" => 0.61),
                "advanced" => Dict("survival_rate" => 0.56),
                "premium" => Dict("survival_rate" => 0.50),
           ),
       ),
    ]

    paired = paired_treatment_effects(all_results)
    @test paired["premium"]["mean_pp"] ≈ mean([-8.0, -10.0])
    @test paired["premium"]["sd_pp"] ≈ std([-8.0, -10.0])
    @test paired["premium"]["n_runs"] == 2
    @test paired["none"]["mean_pp"] == 0.0
end

@testset "Persistent action biases affect production action probabilities" begin
    cfg = EmergentConfig(
        N_AGENTS=2,
        N_ROUNDS=1,
        RANDOM_SEED=931,
        ACTION_BIAS_SIGMA=0.0,
        ACTION_SELECTION_NOISE=0.0,
   )
    GlimpseABM.initialize!(cfg)
    market = MarketEnvironment(cfg; rng=MersenneTwister(931))
    mc = GlimpseABM.get_market_conditions(market)
    opps = GlimpseABM.get_available_opportunities(market)

    neutral = EmergentAgent(1, cfg; primary_sector="tech", fixed_ai_level="none",
                            initial_capital=5_000_000.0, rng=MersenneTwister(932))
    biased = EmergentAgent(2, cfg; primary_sector="tech", fixed_ai_level="none",
                           initial_capital=5_000_000.0, rng=MersenneTwister(932))
    biased.traits = copy(neutral.traits)
    biased.resources.knowledge = copy(neutral.resources.knowledge)
    biased.action_biases = Dict("invest" => 0.45, "innovate" => -0.15,
                                "explore" => -0.15, "maintain" => -0.15)

    neutral_outcome = GlimpseABM.make_decision!(
        neutral, opps, mc, market, 1; info_system=nothing
   )
    biased_outcome = GlimpseABM.make_decision!(
        biased, opps, mc, market, 1; info_system=nothing
   )

    @test haskey(biased_outcome, "action_probabilities")
    @test biased_outcome["action_probabilities"]["invest"] >
          neutral_outcome["action_probabilities"]["invest"]
end

@testset "Uncertainty response profile changes production utility sensitivity" begin
    cfg = EmergentConfig(N_AGENTS=2, N_ROUNDS=1, RANDOM_SEED=932)
    GlimpseABM.initialize!(cfg)
    agent = EmergentAgent(1, cfg; primary_sector="tech", fixed_ai_level="none",
                          initial_capital=5_000_000.0, rng=MersenneTwister(933))
    perception = empty_perception()

    agent.uncertainty_response.response_weights["actor_ignorance"] = 0.1
    low_response = GlimpseABM.calculate_exploration_utility(agent, perception; ai_level="none")

    agent.uncertainty_response.response_weights["actor_ignorance"] = 0.9
    high_response = GlimpseABM.calculate_exploration_utility(agent, perception; ai_level="none")

    @test high_response > low_response
end

@testset "Public opportunity visibility is stochastic, not deterministic top-k" begin
    cfg = EmergentConfig(N_AGENTS=2, RANDOM_SEED=933)
    GlimpseABM.initialize!(cfg)
    market = MarketEnvironment(cfg; rng=MersenneTwister(934))
    empty!(market.opportunities)
    empty!(market.opportunity_map)

    for i in 1:80
        opp = Opportunity(
            id="public_$(i)",
            discovered=true,
            sector=i <= 40 ? "tech" : "retail",
            config=cfg,
            capacity=1_000_000.0,
       )
        push!(market.opportunities, opp)
        market.opportunity_map[opp.id] = opp
    end

    a = EmergentAgent(1, cfg; primary_sector="tech", fixed_ai_level="premium",
                      rng=MersenneTwister(935))
    b = EmergentAgent(2, cfg; primary_sector="tech", fixed_ai_level="premium",
                      rng=MersenneTwister(936))
    a.resources.knowledge["tech"] = 1.0
    b.resources.knowledge["tech"] = 1.0

    visible_a = GlimpseABM.get_opportunities_for_agent(market, a)
    visible_b = GlimpseABM.get_opportunities_for_agent(market, b)
    ids_a = Set(opp.id for opp in visible_a)
    ids_b = Set(opp.id for opp in visible_b)

    @test 0 < length(ids_a) < length(market.opportunities)
    @test 0 < length(ids_b) < length(market.opportunities)
    @test length(intersect(ids_a, ids_b)) < length(union(ids_a, ids_b))
end

@testset "RETURN_NOISE_SCALE controls realized-return dispersion" begin
    base_cfg = EmergentConfig(N_AGENTS=1, RANDOM_SEED=934, RETURN_NOISE_SCALE=0.0)
    noisy_cfg = EmergentConfig(N_AGENTS=1, RANDOM_SEED=934, RETURN_NOISE_SCALE=0.8)
    GlimpseABM.initialize!(base_cfg)
    GlimpseABM.initialize!(noisy_cfg)
    market = MarketEnvironment(base_cfg; rng=MersenneTwister(937))
    mc = GlimpseABM.get_market_conditions(market)

    base_opp = Opportunity(
        id="return-noise-probe",
        latent_return_potential=2.0,
        latent_failure_potential=0.25,
        sector="tech",
        config=base_cfg,
        capacity=10_000_000.0,
   )
    noisy_opp = Opportunity(
        id="return-noise-probe-noisy",
        latent_return_potential=2.0,
        latent_failure_potential=0.25,
        sector="tech",
        config=noisy_cfg,
        capacity=10_000_000.0,
   )

    base_draws = [GlimpseABM.realized_return(base_opp, mc, "none"; rng=MersenneTwister(10_000 + i)) for i in 1:250]
    noisy_draws = [GlimpseABM.realized_return(noisy_opp, mc, "none"; rng=MersenneTwister(10_000 + i)) for i in 1:250]

    @test std(noisy_draws) > std(base_draws)
end

@testset "Venture-tail returns feed confidence through transformed evidence" begin
    cfg = EmergentConfig(
        N_AGENTS=2,
        N_ROUNDS=1,
        RANDOM_SEED=938,
        ACTION_BIAS_SIGMA=0.0,
        ACTION_SELECTION_NOISE=0.0,
   )
    GlimpseABM.initialize!(cfg)
    market = MarketEnvironment(cfg; rng=MersenneTwister(938))
    mc = GlimpseABM.get_market_conditions(market)
    env = KnightianUncertaintyEnvironment(cfg)

    function outcome_after_recent_multiple(multiple::Float64)
        agent = EmergentAgent(1, cfg; primary_sector="tech", fixed_ai_level="none",
                              initial_capital=5_000_000.0, rng=MersenneTwister(939))
        agent.recent_outcomes = Dict{String,Any}[
            Dict{String,Any}(
                "action" => "invest",
                "success" => multiple >= 1.0,
                "return_multiple" => multiple,
                "investment_amount" => 1.0,
                "capital_returned" => multiple,
                "ai_used" => false,
           )
        ]
        return GlimpseABM.make_decision!(
            agent, Opportunity[], mc, market, 1;
            uncertainty_env=env,
            info_system=nothing
       )
    end

    five_x = outcome_after_recent_multiple(5.0)
    fifty_x = outcome_after_recent_multiple(50.0)
    five_stats = five_x["experience_stats"]
    fifty_stats = fifty_x["experience_stats"]
    five_conf = five_x["confidence_diagnostics"]
    fifty_conf = fifty_x["confidence_diagnostics"]

    @test haskey(fifty_stats, "return_evidence")
    @test five_stats["raw_mean_roi"] ≈ 4.0
    @test fifty_stats["raw_mean_roi"] ≈ 49.0
    @test five_stats["roi_clamp_hit"] == false
    @test fifty_stats["roi_clamp_hit"] == false
    @test fifty_stats["return_evidence"] > five_stats["return_evidence"]
    @test fifty_conf["decision_confidence"] > five_conf["decision_confidence"]
    @test haskey(fifty_conf, "confidence_saturation_hit")
end

println("Variance wiring regression tests passed.")
