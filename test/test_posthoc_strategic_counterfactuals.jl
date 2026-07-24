using Test
using Random
using DataFrames
using GlimpseABM

module PosthocStrategicRunnerHarness
include(joinpath(@__DIR__, "..", "scripts",
                 "run_posthoc_strategic_counterfactuals.jl"))
end

const PSRH = PosthocStrategicRunnerHarness

@testset "post hoc signals reach choice and use a matched sequential control" begin
    signal_cfg = EmergentConfig(
        N_AGENTS=2,
        N_ROUNDS=1,
        ENABLE_POPULATION_SCALING=false,
    )
    signal_market = MarketEnvironment(signal_cfg; rng=MersenneTwister(7201))
    high = Opportunity(
        id="high_unsignaled",
        discovered=true,
        sector="tech",
        latent_return_potential=5.0,
        capacity=1_000_000.0,
        rng=MersenneTwister(7202),
    )
    low = Opportunity(
        id="low_signaled",
        discovered=true,
        sector="tech",
        latent_return_potential=1.0,
        capacity=1_000_000.0,
        rng=MersenneTwister(7203),
    )
    baseline_agent = EmergentAgent(
        1, signal_cfg; fixed_ai_level="premium", rng=MersenneTwister(7204))
    signaled_agent = EmergentAgent(
        2, signal_cfg; fixed_ai_level="premium", rng=MersenneTwister(7204))
    mc = GlimpseABM.get_market_conditions(signal_market)
    baseline_evals = GlimpseABM.evaluate_portfolio_opportunities(
        baseline_agent, [high, low], mc, empty_perception();
        ai_level="premium", market=signal_market)
    signaled_evals = GlimpseABM.evaluate_portfolio_opportunities(
        signaled_agent, [high, low], mc, empty_perception();
        ai_level="premium", market=signal_market,
        early_signals=Dict("low_signaled" => 100), signal_weight=1.0)
    @test first(baseline_evals).opportunity.id == "high_unsignaled"
    @test first(signaled_evals).opportunity.id == "low_signaled"

    by_name = Dict(c.name => c for c in PSRH.all_conditions())
    @test haskey(by_name, "SEQUENTIAL_SIGNAL_BASELINE")
    sequential_cfg = PSRH.build_config(by_name["SEQUENTIAL_SIGNAL_BASELINE"], 7301)
    amplify_cfg = PSRH.build_config(by_name["MANIPULATION_AMPLIFY"], 7301)
    @test sequential_cfg.SEQUENTIAL_DECISIONS_ENABLED
    @test !sequential_cfg.POSTHOC_MANIPULATION_ENABLED
    @test amplify_cfg.SEQUENTIAL_DECISIONS_ENABLED
    @test amplify_cfg.POSTHOC_MANIPULATION_ENABLED
    manifest = PSRH.condition_manifest_rows([
        by_name["SEQUENTIAL_SIGNAL_BASELINE"],
        by_name["MANIPULATION_AMPLIFY"],
    ])
    @test :manipulation_signal_strength in propertynames(manifest)
    @test :sequential_decisions_enabled in propertynames(manifest)
    @test :reference_condition in propertynames(manifest)
    @test !only(manifest[manifest.condition .==
        "SEQUENTIAL_SIGNAL_BASELINE", :manipulation_enabled])
    @test only(manifest[manifest.condition .==
        "MANIPULATION_AMPLIFY", :manipulation_enabled])
    @test only(manifest[manifest.condition .==
        "MANIPULATION_AMPLIFY", :reference_condition]) ==
        "SEQUENTIAL_SIGNAL_BASELINE"

    selected = withenv("CONDITIONS" => "MANIPULATION_AMPLIFY") do
        PSRH.selected_conditions()
    end
    @test Set(c.name for c in selected) == Set([
        "POSTHOC_BASELINE",
        "SEQUENTIAL_SIGNAL_BASELINE",
        "MANIPULATION_AMPLIFY",
    ])
    @test PSRH.preflight_conditions(selected) === nothing

    rows = NamedTuple[]
    for run_idx in 1:2
        for (condition, family, survival, capital, manipulation_added,
             mean_manipulation_available, mean_manipulation_on_choice,
             manipulation_exposure) in [
                ("POSTHOC_BASELINE", "baseline", 0.50, 100.0, 0.0,
                 NaN, NaN, NaN),
                ("SEQUENTIAL_SIGNAL_BASELINE", "sequential_control", 0.60,
                 110.0, 0.0, 0.0, 0.0, 0.0),
                ("MANIPULATION_AMPLIFY", "manipulation", 0.70, 130.0,
                 6.0, 4.0, 2.0, 0.50),
                ("WEAK_TARGETING_LOW", "weak_targeting", 0.55, 105.0,
                 0.0, NaN, NaN, NaN),
            ]
            push!(rows, (
                condition=condition,
                family=family,
                run_idx=run_idx,
                seed=20260425 + run_idx,
                tier="premium",
                survival_rate=survival,
                mean_final_capital=capital,
                mean_weak_exposure=0.0,
                mean_targeting_multiplier=1.0,
                weak_agent_targeting_rate=0.0,
                mean_weak_agent_target_pool_size=NaN,
                mean_weak_agent_visible_pool_size=NaN,
                mean_weak_agent_foregone_latent_gap=NaN,
                manipulation_signals_added=manipulation_added,
                late_invest_events=4,
                mean_sequential_signals_available=2.0,
                mean_manipulation_signals_available=
                    mean_manipulation_available,
                mean_manipulation_signal_count_on_choice=
                    mean_manipulation_on_choice,
                manipulation_signal_choice_exposure_rate=
                    manipulation_exposure,
            ))
        end
    end
    effects = PSRH.paired_effects(DataFrame(rows))
    amplify = only(eachrow(effects[effects.condition .==
        "MANIPULATION_AMPLIFY", :]))
    weak = only(eachrow(effects[effects.condition .==
        "WEAK_TARGETING_LOW", :]))
    @test amplify.reference_condition == "SEQUENTIAL_SIGNAL_BASELINE"
    @test amplify.survival_delta_pp ≈ 10.0
    @test amplify.survival_delta_ci95_low_pp ≈ 10.0
    @test amplify.survival_delta_ci95_high_pp ≈ 10.0
    @test amplify.mean_final_capital_delta ≈ 20.0
    @test amplify.mean_final_capital_delta_ci95_low ≈ 20.0
    @test amplify.mean_final_capital_delta_ci95_high ≈ 20.0
    @test amplify.manipulation_signal_choice_exposure_rate == 0.50
    @test weak.reference_condition == "POSTHOC_BASELINE"
    @test weak.survival_delta_pp ≈ 5.0
    @test weak.mean_final_capital_delta ≈ 5.0
end

@testset "post hoc DRY_RUN is a no-write config preflight" begin
    rendered = mktemp() do _, output
        withenv(
            "DRY_RUN" => "1",
            "CONDITIONS" => "MANIPULATION_AMPLIFY",
        ) do
            redirect_stdout(output) do
                @test PSRH.main() === nothing
            end
        end
        flush(output)
        seekstart(output)
        read(output, String)
    end
    @test occursin("Config preflight: all 3 post hoc condition configs build cleanly.",
                   rendered)
    @test !occursin("Running 3 tasks", rendered)
    @test !occursin("Wrote post hoc strategic outputs", rendered)
end

@testset "post hoc weak-targeting exposure and manipulation probes" begin
    cfg = EmergentConfig(
        N_AGENTS=4,
        N_ROUNDS=2,
        ENABLE_POPULATION_SCALING=false,
        POSTHOC_WEAK_TARGETING_ENABLED=true,
        POSTHOC_WEAK_TARGETING_STRENGTH=2.0,
        POSTHOC_WEAK_TARGETING_MAX_MULTIPLIER=3.0,
    )
    market = MarketEnvironment(cfg; rng=MersenneTwister(7101))
    opp = Opportunity(
        id="posthoc_weak_target",
        discovered=true,
        capacity=1_000_000.0,
        total_invested=0.0,
        competition=0.0,
        rng=MersenneTwister(7102),
    )

    GlimpseABM.record_opportunity_tier_investment!(market, opp.id, "none", 500_000.0)
    GlimpseABM.record_opportunity_tier_investment!(market, opp.id, "advanced", 500_000.0)
    exposure = GlimpseABM.opportunity_weak_exposure(market, opp, ["none", "basic"])
    @test exposure ≈ 0.5 * (1.0 - exp(-0.5))

    premium = EmergentAgent(1, cfg; fixed_ai_level="premium", rng=MersenneTwister(7103))
    weak_exposure, multiplier =
        GlimpseABM.posthoc_weak_targeting_multiplier(premium, market, opp, "premium")
    @test weak_exposure ≈ exposure
    @test 1.0 < multiplier <= cfg.POSTHOC_WEAK_TARGETING_MAX_MULTIPLIER

    none_agent = EmergentAgent(2, cfg; fixed_ai_level="none", rng=MersenneTwister(7104))
    none_exposure, none_multiplier =
        GlimpseABM.posthoc_weak_targeting_multiplier(none_agent, market, opp, "none")
    @test none_exposure ≈ exposure
    @test none_multiplier == 1.0

    target_cfg = EmergentConfig(
        N_AGENTS=4,
        N_ROUNDS=2,
        ENABLE_POPULATION_SCALING=false,
        POSTHOC_WEAK_AGENT_TARGETING_ENABLED=true,
    )
    target_market = MarketEnvironment(target_cfg; rng=MersenneTwister(7111))
    weak_opp = Opportunity(
        id="weak_occupied",
        discovered=true,
        sector="tech",
        latent_return_potential=1.0,
        capacity=1_000_000.0,
        rng=MersenneTwister(7112),
    )
    high_opp = Opportunity(
        id="high_quality_empty",
        discovered=true,
        sector="tech",
        latent_return_potential=8.0,
        capacity=1_000_000.0,
        rng=MersenneTwister(7113),
    )
    GlimpseABM.record_opportunity_tier_investment!(
        target_market, weak_opp.id, "none", 250_000.0)
    target_agent = EmergentAgent(
        5, target_cfg; fixed_ai_level="premium", rng=MersenneTwister(7114))
    target_evals = GlimpseABM.evaluate_portfolio_opportunities(
        target_agent,
        [high_opp, weak_opp],
        GlimpseABM.get_market_conditions(target_market),
        empty_perception();
        ai_level="premium",
        market=target_market,
    )
    @test length(target_evals) == 1
    @test target_evals[1].opportunity.id == "weak_occupied"
    @test target_evals[1].posthoc_weak_agent_targeting_applied
    @test target_evals[1].posthoc_weak_agent_target_pool_size == 1
    @test target_evals[1].posthoc_weak_agent_visible_pool_size == 2

    fallback_market = MarketEnvironment(target_cfg; rng=MersenneTwister(7115))
    fallback_agent = EmergentAgent(
        6, target_cfg; fixed_ai_level="premium", rng=MersenneTwister(7116))
    fallback_evals = GlimpseABM.evaluate_portfolio_opportunities(
        fallback_agent,
        [high_opp, weak_opp],
        GlimpseABM.get_market_conditions(fallback_market),
        empty_perception();
        ai_level="premium",
        market=fallback_market,
    )
    @test length(fallback_evals) == 2
    @test !fallback_evals[1].posthoc_weak_agent_targeting_applied

    GlimpseABM.release_opportunity_tier_investment!(market, opp.id, "none", 500_000.0)
    @test GlimpseABM.opportunity_tier_investment(market, opp.id, "none") == 0.0

    manip_cfg = EmergentConfig(
        N_AGENTS=4,
        N_ROUNDS=2,
        ENABLE_POPULATION_SCALING=false,
        POSTHOC_MANIPULATION_ENABLED=true,
        POSTHOC_MANIPULATION_MODE="amplify",
        POSTHOC_MANIPULATION_SIGNAL_STRENGTH=3,
    )
    manip_agent = EmergentAgent(3, manip_cfg; fixed_ai_level="premium", rng=MersenneTwister(7105))
    signals = Dict{String,Int}()
    manipulation_signals = Dict{String,Int}()
    outcome = Dict{String,Any}(
        "action" => "invest",
        "opportunity_id" => "posthoc_signal_target",
        "ai_behavior_level" => "premium",
    )
    signal_opp = Opportunity(id="posthoc_signal_target", discovered=true,
                             rng=MersenneTwister(7106))
    GlimpseABM._posthoc_apply_manipulation_signal!(
        manip_cfg, signals, manip_agent, outcome, [signal_opp];
        manipulation_signals=manipulation_signals)
    @test signals["posthoc_signal_target"] == 3
    @test manipulation_signals["posthoc_signal_target"] == 3
    @test outcome["posthoc_manipulation_signals_added"] == 3

    consumed = Dict{String,Any}(
        "action" => "invest",
        "opportunity_id" => "posthoc_signal_target",
    )
    GlimpseABM._annotate_sequential_signal_exposure!(
        consumed, signals, manipulation_signals, "late")
    @test consumed["posthoc_manipulation_signals_available"] == 3
    @test consumed["posthoc_manipulation_signal_count_on_choice"] == 3

    manip_cfg.POSTHOC_MANIPULATION_MODE = "decoy"
    signals = Dict{String,Int}()
    chosen = Opportunity(id="chosen", discovered=true, competition=0.0,
                         rng=MersenneTwister(7107))
    decoy = Opportunity(id="decoy", discovered=true, competition=5.0,
                        rng=MersenneTwister(7108))
    outcome = Dict{String,Any}(
        "action" => "invest",
        "opportunity_id" => "chosen",
        "ai_behavior_level" => "premium",
    )
    GlimpseABM._posthoc_apply_manipulation_signal!(
        manip_cfg, signals, manip_agent, outcome, [chosen, decoy])
    @test haskey(signals, "decoy")
    @test outcome["posthoc_manipulation_signal_target"] == "decoy"

    @test_throws ErrorException initialize!(
        EmergentConfig(POSTHOC_MANIPULATION_MODE="not_a_mode"))
    @test_throws ErrorException initialize!(
        EmergentConfig(POSTHOC_WEAK_TARGETING_STRENGTH=-0.1))
    @test_throws ErrorException initialize!(
        EmergentConfig(POSTHOC_WEAK_AGENT_TARGETING_MIN_EXPOSURE=-0.1))
end
