using Test

include(joinpath(@__DIR__, "..", "scripts", "run_mixed_tier_analysis_full.jl"))

@testset "Paradox mechanism scorecard" begin
    function tier_summary(;
        survival=0.60,
        actor=0.40,
        practical=0.30,
        novelty=0.25,
        recursion=0.10,
        visible=20.0,
        info_quality=0.25,
        abs_gap=0.10,
        roi_std=0.20,
        niches=0.20,
        combos=1.00,
        survivor_actor=nothing,
        actor_obs=10.0,
        perceived_actor=0.45,
        perceived_practical=0.50,
        perceived_novelty=0.40,
        perceived_recursion=0.30,
    )
        survivor_actor_level = isnothing(survivor_actor) ? actor : survivor_actor
        return Dict{String,Any}(
            "mean_survival_rate" => survival,
            "emergent_actor_ignorance" => actor,
            "emergent_practical_indeterminism" => practical,
            "emergent_agentic_novelty" => novelty,
            "emergent_competitive_recursion" => recursion,
            "all_agent_emergent_actor_ignorance" => actor,
            "all_agent_emergent_practical_indeterminism" => practical,
            "all_agent_emergent_agentic_novelty" => novelty,
            "all_agent_emergent_competitive_recursion" => recursion,
            "survivor_emergent_actor_ignorance" => survivor_actor_level,
            "survivor_emergent_practical_indeterminism" => practical,
            "survivor_emergent_agentic_novelty" => novelty,
            "survivor_emergent_competitive_recursion" => recursion,
            "all_agent_emergent_actor_ignorance_observations" => actor_obs,
            "mean_visible_opportunities" => visible,
            "mean_info_quality_used" => info_quality,
            "perceived_actor_ignorance_bounded_level" => perceived_actor,
            "perceived_actor_ignorance_gap_vs_prior" => -0.05,
            "perceived_practical_indeterminism_bounded_level" => perceived_practical,
            "perceived_practical_indeterminism_gap_vs_prior" => 0.04,
            "perceived_agentic_novelty_bounded_level" => perceived_novelty,
            "perceived_agentic_novelty_gap_vs_prior" => 0.03,
            "perceived_competitive_recursion_bounded_level" => perceived_recursion,
            "perceived_competitive_recursion_gap_vs_prior" => 0.02,
            "confidence_outcome_abs_gap_mean" => abs_gap,
            "confidence_outcome_realized_multiple_std" => roi_std,
            "mean_niches_per_agent" => niches,
            "mean_combinations_per_agent" => combos,
        )
    end

    summary = Dict{String,Any}(
        "summary_stats" => Dict{String,Dict{String,Any}}(
            "none" => tier_summary(),
            "basic" => tier_summary(actor=0.35, practical=0.32, visible=30.0, info_quality=0.55),
            "advanced" => tier_summary(actor=0.32, practical=0.34, novelty=0.30, visible=38.0, info_quality=0.70),
            "premium" => tier_summary(
                survival=0.50,
                actor=0.25,
                practical=0.42,
                novelty=0.45,
                recursion=0.25,
                visible=50.0,
                info_quality=0.90,
                abs_gap=0.18,
                roi_std=0.40,
                niches=0.70,
                combos=2.20,
                survivor_actor=0.31,
                actor_obs=42.0,
                perceived_actor=0.38,
                perceived_practical=0.66,
                perceived_novelty=0.58,
                perceived_recursion=0.46,
            ),
        ),
        "decision_diagnostics_summary" => Dict{String,Float64}(
            "mean_market_crowding_pressure" => 0.21,
            "mean_market_opportunity_overlap" => 0.34,
            "mean_market_investment_concentration" => 0.13,
            "mean_market_opportunity_competition" => 0.18,
            "mean_market_ai_herding_intensity" => 0.27,
            "mean_market_ai_action_correlation" => 0.46,
            "mean_market_combo_reuse_pressure" => 0.31,
        ),
    )

    scorecard = build_paradox_mechanism_scorecard(summary)
    premium = scorecard["premium"]

    @test premium["first_order_knowledge_gain_vs_none"] ≈ 0.15
    @test premium["all_agent_actor_ignorance_level"] ≈ 0.25
    @test premium["survivor_actor_ignorance_level"] ≈ 0.31
    @test premium["actor_ignorance_survivorship_gap"] ≈ 0.06
    @test premium["actor_ignorance_observations"] ≈ 42.0
    @test premium["visible_opportunity_gain_vs_none"] ≈ 30.0
    @test premium["info_quality_gain_vs_none"] ≈ 0.65
    @test premium["perceived_actor_ignorance_delta_vs_none"] ≈ -0.07
    @test premium["perceived_practical_indeterminism_delta_vs_none"] ≈ 0.16
    @test premium["epistemic_horizon_delta_vs_none"] > 0.0
    @test premium["knowledge_horizon_tradeoff"] > 0.0
    @test premium["first_order_benefit_observed"] == true
    @test premium["horizon_recession_observed"] == true
    @test premium["paradox_logic_supported"] == true
    @test premium["dominant_second_order_channel"] == "agentic_novelty"
    @test premium["first_order_evidence_count"] == 3
    @test premium["second_order_evidence_count"] == 3
    @test premium["paradox_alignment_score"] > 0.0
    @test premium["survival_effect_pp_vs_none"] ≈ -10.0
    @test premium["market_crowding_pressure"] ≈ 0.21
    @test premium["market_ai_action_correlation"] ≈ 0.46

    none = scorecard["none"]
    @test none["first_order_knowledge_gain_vs_none"] == 0.0
    @test none["epistemic_horizon_delta_vs_none"] == 0.0
    @test none["paradox_logic_supported"] == false
end
