using Test
using Random
using DataFrames

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using GlimpseABM

module RobustnessSuiteHarness
include(joinpath(@__DIR__, "..", "scripts", "run_robustness_suite.jl"))
end

const RRH = RobustnessSuiteHarness

function _robustness_condition(name::AbstractString)
    by_name = Dict(c.name => c for c in RRH.all_conditions())
    @test haskey(by_name, name)
    return by_name[name]
end

function _robustness_config(name::AbstractString; seed::Int=20260425)
    return RRH.build_config(_robustness_condition(name), seed)
end

@testset "Robustness suite wiring" begin
    @testset "canonical tail and cost cells are live" begin
        baseline = _robustness_config("BASELINE")
        truncated = _robustness_config("TRUNCATED_TAIL")
        moderate = _robustness_config("MODERATE_TAIL")
        cost_low = _robustness_config("OPS_COST_060")
        cost_mid = _robustness_config("OPS_COST_075")
        cost_high = _robustness_config("OPS_COST_100")

        @test baseline.NICHE_SIZE_LOG_SIGMA == RRH.NICHE_SIGMA
        @test truncated.NICHE_SIZE_LOG_SIGMA == 0.0
        @test 0.0 < moderate.NICHE_SIZE_LOG_SIGMA < baseline.NICHE_SIZE_LOG_SIGMA
        @test truncated.LOG_SIGMA_MULT == 1.0
        # Corrected canonical: the truncated/moderate/baseline sweep holds the latent
        # RETURN tail realistic (LOG_SIGMA_MULT == 1.0 for all three) and varies ONLY the
        # niche-size tail (checked above), so MODERATE is a clean niche-only variant.
        @test moderate.LOG_SIGMA_MULT == truncated.LOG_SIGMA_MULT == baseline.LOG_SIGMA_MULT == 1.0

        @test cost_low.OPS_COST_INTENSITY == 0.60
        @test cost_mid.OPS_COST_INTENSITY == 0.75
        @test cost_high.OPS_COST_INTENSITY == 1.00
        @test all(!=(baseline.OPS_COST_INTENSITY),
            [cost_low.OPS_COST_INTENSITY, cost_mid.OPS_COST_INTENSITY, cost_high.OPS_COST_INTENSITY])
    end

    @testset "canonical payoff crowding is pure outstanding-capital load" begin
        baseline = _robustness_config("BASELINE")
        action_hhi = _robustness_config("CROWDING_ACTION_HHI_BLEND")

        # Canonical economics: local return dilution depends on the outstanding
        # investment/capacity ratio only. The market-wide action-category HHI is
        # retained as an explicit robustness specification, not silently mixed
        # into every opportunity's productive-capacity constraint.
        @test EmergentConfig().CROWDING_INDEX_BLEND == 0.0
        @test baseline.CROWDING_INDEX_BLEND == 0.0
        @test action_hhi.CROWDING_INDEX_BLEND == 0.30

        changed = Symbol[
            field for field in fieldnames(EmergentConfig)
            if field != :TRAIT_DISTRIBUTIONS &&
               !isequal(getfield(baseline, field), getfield(action_hhi, field))
        ]
        # TraitDistribution has identity rather than structural equality; its
        # printed values verify the independently constructed defaults match.
        @test repr(baseline.TRAIT_DISTRIBUTIONS) ==
            repr(action_hhi.TRAIT_DISTRIBUTIONS)
        @test changed == [:CROWDING_INDEX_BLEND]
    end

    @testset "capacity and open-futures cells isolate one mechanism" begin
        baseline = _robustness_config("BASELINE")
        cap_low = _robustness_config("CAPACITY_MEAN_HALF")
        cap_high = _robustness_config("CAPACITY_MEAN_DOUBLE")
        capacity_aware = _robustness_config("CAPACITY_AWARE_SELECTION")
        component_fit = _robustness_config("OPPORTUNITY_COMPONENT_FIT_ON")
        endogenous_only = _robustness_config("ENDOGENOUS_OPPORTUNITIES_ONLY")
        one_niche = _robustness_config("NICHE_MULTIPLICITY_ONE")
        fixed_two = _robustness_config("NICHE_MULTIPLICITY_FIXED_TWO")
        three_niches = _robustness_config("NICHE_MULTIPLICITY_THREE")
        performative_common = _robustness_config("PERFORMATIVE_COMMON_050")
        performative_appropriable =
            _robustness_config("PERFORMATIVE_APPROPRIABLE_050")

        changed(a, b) = Symbol[
            field for field in fieldnames(EmergentConfig)
            if field != :TRAIT_DISTRIBUTIONS &&
               !isequal(getfield(a, field), getfield(b, field))
        ]

        @test cap_low.OPPORTUNITY_BASE_CAPACITY ==
            0.5 * baseline.OPPORTUNITY_BASE_CAPACITY
        @test cap_high.OPPORTUNITY_BASE_CAPACITY ==
            2.0 * baseline.OPPORTUNITY_BASE_CAPACITY
        @test changed(baseline, cap_low) == [:OPPORTUNITY_BASE_CAPACITY]
        @test changed(baseline, cap_high) == [:OPPORTUNITY_BASE_CAPACITY]
        @test capacity_aware.DECISION_CROWDING_AVERSION_WEIGHT == 1.0
        @test changed(baseline, capacity_aware) ==
            [:DECISION_CROWDING_AVERSION_WEIGHT]

        @test component_fit.ENABLE_OPPORTUNITY_COMPONENTS
        @test changed(baseline, component_fit) == [:ENABLE_OPPORTUNITY_COMPONENTS]
        @test !endogenous_only.ENABLE_BACKGROUND_OPPORTUNITY_REPLENISHMENT
        @test changed(baseline, endogenous_only) ==
            [:ENABLE_BACKGROUND_OPPORTUNITY_REPLENISHMENT]

        @test (one_niche.NICHE_OPPORTUNITIES_PER_DISCOVERY_MIN,
               one_niche.NICHE_OPPORTUNITIES_PER_DISCOVERY_MAX) == (1, 1)
        @test (fixed_two.NICHE_OPPORTUNITIES_PER_DISCOVERY_MIN,
               fixed_two.NICHE_OPPORTUNITIES_PER_DISCOVERY_MAX) == (2, 2)
        @test (three_niches.NICHE_OPPORTUNITIES_PER_DISCOVERY_MIN,
               three_niches.NICHE_OPPORTUNITIES_PER_DISCOVERY_MAX) == (3, 3)
        @test performative_common.PERFORMATIVE_DEMAND_ELASTICITY == 0.50
        @test !performative_common.PERFORMATIVE_DEMAND_APPROPRIABLE
        @test changed(baseline, performative_common) ==
            [:PERFORMATIVE_DEMAND_ELASTICITY]
        @test performative_appropriable.PERFORMATIVE_DEMAND_ELASTICITY == 0.50
        @test performative_appropriable.PERFORMATIVE_DEMAND_APPROPRIABLE
        @test changed(baseline, performative_appropriable) ==
            [:PERFORMATIVE_DEMAND_ELASTICITY,
             :PERFORMATIVE_DEMAND_APPROPRIABLE]
    end

    @testset "effective-config export carries canonical tail fields" begin
        rows = RRH.effective_config_rows([
            _robustness_condition("BASELINE"),
            _robustness_condition("TRUNCATED_TAIL"),
            _robustness_condition("MODERATE_TAIL"),
            _robustness_condition("OPS_COST_075"),
            _robustness_condition("PERFORMATIVE_APPROPRIABLE_050"),
        ], 20260425)

        @test :niche_size_log_sigma in propertynames(rows)
        @test :ops_cost_intensity in propertynames(rows)
        @test :heavy_tail_returns in propertynames(rows)
        @test :option_b_tail_env in propertynames(rows)
        @test :crowding_index_blend in propertynames(rows)
        @test :decision_crowding_aversion_weight in propertynames(rows)
        @test :performative_demand_elasticity in propertynames(rows)
        @test :performative_demand_appropriable in propertynames(rows)
        @test rows[rows.condition .== "BASELINE", :niche_size_log_sigma][1] == RRH.NICHE_SIGMA
        @test rows[rows.condition .== "TRUNCATED_TAIL", :niche_size_log_sigma][1] == 0.0
        @test rows[rows.condition .== "BASELINE", :crowding_index_blend][1] == 0.0
        performative = rows[rows.condition .==
            "PERFORMATIVE_APPROPRIABLE_050", :]
        @test only(performative.performative_demand_elasticity) == 0.50
        @test only(performative.performative_demand_appropriable)
    end

    @testset "venture ledger records canonical capacity" begin
        cfg = EmergentConfig(N_AGENTS=4, N_ROUNDS=1, RANDOM_SEED=13)
        GlimpseABM.initialize!(cfg)
        agent = EmergentAgent(1, cfg; rng=MersenneTwister(13))
        # The append-only resolution ledger now carries the complete proximate
        # causal trace through exact applied crowding and value capture.
        @test eltype(agent.venture_ledger) == NTuple{20,Float64}
    end

    @testset "filtered performativity runs retain their paired baseline" begin
        picked = RRH.filter_conditions(
            RRH.all_conditions(), "PERFORMATIVE_APPROPRIABLE_100")
        @test [condition.name for condition in picked] ==
            ["BASELINE", "PERFORMATIVE_APPROPRIABLE_100"]
        @test [condition.name for condition in RRH.filter_conditions(
            RRH.all_conditions(), "BASELINE")] == ["BASELINE"]
        @test length(RRH.filter_conditions(RRH.all_conditions(), "")) ==
            length(RRH.all_conditions())
        @test_throws ErrorException RRH.filter_conditions(
            RRH.all_conditions(), "NOT_A_REAL_CONDITION")
    end

    @testset "performed demand is live, validated, and maturity-order invariant" begin
        @test_throws ErrorException GlimpseABM.initialize!(
            EmergentConfig(PERFORMATIVE_DEMAND_ELASTICITY=-0.1))
        @test_throws ErrorException GlimpseABM.initialize!(
            EmergentConfig(PERFORMATIVE_DEMAND_ELASTICITY=NaN))

        cfg = EmergentConfig(
            N_AGENTS=2,
            N_ROUNDS=1,
            RANDOM_SEED=8101,
            ENABLE_POPULATION_SCALING=false,
            PERFORMATIVE_DEMAND_ELASTICITY=1.0,
            PERFORMATIVE_DEMAND_APPROPRIABLE=true,
            enable_round_logging=false,
        )
        GlimpseABM.initialize!(cfg)
        market = MarketEnvironment(cfg; rng=MersenneTwister(8101))
        opp = Opportunity(
            id="performative_cohort_probe",
            latent_return_potential=2.0,
            latent_failure_potential=0.1,
            discovered=true,
            sector="tech",
            capital_requirements=250.0,
            time_to_maturity=1,
            capacity=100.0,
            total_invested=500.0,
            committed_prev_round=500.0,
            config=cfg,
            rng=MersenneTwister(8102),
        )
        GlimpseABM.add_opportunity!(market, opp)
        mc = GlimpseABM.get_market_conditions(market)

        baseline_cfg = deepcopy(cfg)
        baseline_cfg.PERFORMATIVE_DEMAND_ELASTICITY = 0.0
        baseline_terms = GlimpseABM.capacity_crowding_terms(
            opp, mc, baseline_cfg; commitment_share=0.5,
            current_total_invested=500.0)
        appropriable_terms = GlimpseABM.capacity_crowding_terms(
            opp, mc, cfg; commitment_share=0.5,
            current_total_invested=500.0)
        common_terms = GlimpseABM.capacity_crowding_terms(
            opp, mc, cfg; commitment_share=1.0,
            current_total_invested=500.0)
        @test baseline_terms.effective_capacity == 100.0
        @test baseline_terms.performative_uplift_fraction == 0.0
        @test baseline_terms.effective_capacity <
            appropriable_terms.effective_capacity < common_terms.effective_capacity
        @test baseline_terms.multiplier <
            appropriable_terms.multiplier < common_terms.multiplier

        agents = [
            EmergentAgent(i, cfg; fixed_ai_level="none",
                          rng=MersenneTwister(8110)) for i in 1:2
        ]
        for agent in agents
            push!(agent.active_investments, Dict{String,Any}(
                "amount" => 250.0,
                "maturity_round" => 1,
                "opportunity" => opp,
                "opportunity_id" => opp.id,
                "ai_level" => "none",
                "ai_label" => "none",
                "estimated_return" => 2.0,
            ))
        end
        maturity_load = Dict(opp.id => 500.0)
        first_outcome = only(GlimpseABM.process_matured_investments!(
            agents[1], market, 1;
            market_conditions=mc,
            maturity_total_invested=maturity_load,
        ))
        second_outcome = only(GlimpseABM.process_matured_investments!(
            agents[2], market, 1;
            market_conditions=mc,
            maturity_total_invested=maturity_load,
        ))
        @test first_outcome["performative_commitment_share"] == 0.5
        @test second_outcome["performative_commitment_share"] == 0.5
        @test first_outcome["performative_effective_capacity"] ==
            second_outcome["performative_effective_capacity"]
        @test first_outcome["crowding_return_multiplier"] ==
            second_outcome["crowding_return_multiplier"]
        @test first_outcome["effective_capacity_saturation_at_maturity"] ==
            second_outcome["effective_capacity_saturation_at_maturity"]
        @test first_outcome["return_multiple"] == second_outcome["return_multiple"]
    end

    @testset "performed demand covers culled cohorts and founder-level stakes" begin
        function prepared_stake(opp::Opportunity, amount::Float64,
                                maturity_round::Int)
            return Dict{String,Any}(
                "amount" => amount,
                "maturity_round" => maturity_round,
                "opportunity" => opp,
                "opportunity_id" => opp.id,
                "ai_level" => "none",
                "ai_label" => "none",
                "estimated_return" => 2.0,
                "instrument_estimated_return" => 2.0,
                "sector" => "tech",
            )
        end

        culled_cfg = EmergentConfig(
            N_AGENTS=2,
            N_ROUNDS=1,
            RANDOM_SEED=8201,
            ENABLE_POPULATION_SCALING=false,
            ENABLE_BACKGROUND_OPPORTUNITY_REPLENISHMENT=false,
            USE_UNIFORM_INITIAL_CAPITAL=true,
            INITIAL_CAPITAL=10_000_000.0,
            OPS_COST_INTENSITY=0.0,
            PERFORMATIVE_DEMAND_ELASTICITY=1.0,
            PERFORMATIVE_DEMAND_APPROPRIABLE=true,
            enable_round_logging=false,
        )
        culled_sim = EmergentSimulation(config=culled_cfg, seed=8201)
        culled_opp = Opportunity(
            id="culled_performative_cohort",
            latent_return_potential=2.0,
            latent_failure_potential=0.1,
            discovered=true,
            sector="tech",
            capital_requirements=250.0,
            time_to_maturity=1,
            capacity=100.0,
            total_invested=500.0,
            committed_prev_round=500.0,
            config=culled_cfg,
            age=11,
            rng=MersenneTwister(8202),
        )
        GlimpseABM.add_opportunity!(culled_sim.market, culled_opp)
        for agent in culled_sim.agents
            agent.fixed_ai_level = "none"
            agent.current_ai_level = "none"
            agent.rng = MersenneTwister(8210)
            agent.active_investments = [prepared_stake(culled_opp, 250.0, 1)]
        end
        GlimpseABM.manage_opportunities!(
            culled_sim.market, 0, Dict{String,Int}(), 0.0)
        @test !(culled_opp in culled_sim.market.opportunities)

        # Use the full simulation step: the phase-start maturity snapshot must
        # include opportunities retained only by active investment records.
        GlimpseABM.step!(culled_sim, 1)
        first_event = only(culled_sim.agents[1].venture_ledger)
        second_event = only(culled_sim.agents[2].venture_ledger)
        @test first_event[11] == second_event[11]
        @test first_event[15] == second_event[15]
        @test first_event[17] == second_event[17]

        # A partial maturity in a culled opportunity must refresh the lagged
        # cohort for the following round, even though the opportunity is no
        # longer present in market.opportunities.
        lag_cfg = deepcopy(culled_cfg)
        lag_cfg.N_AGENTS = 1
        lag_cfg.RANDOM_SEED = 8221
        lag_sim = EmergentSimulation(config=lag_cfg, seed=8221)
        lag_opp = Opportunity(
            id="culled_lag_refresh",
            latent_return_potential=2.0,
            latent_failure_potential=0.1,
            discovered=true,
            sector="tech",
            capital_requirements=250.0,
            time_to_maturity=2,
            capacity=100.0,
            total_invested=500.0,
            committed_prev_round=500.0,
            config=lag_cfg,
            age=11,
            rng=MersenneTwister(8222),
        )
        GlimpseABM.add_opportunity!(lag_sim.market, lag_opp)
        lag_agent = only(lag_sim.agents)
        lag_agent.fixed_ai_level = "none"
        lag_agent.current_ai_level = "none"
        lag_agent.active_investments = [
            prepared_stake(lag_opp, 250.0, 1),
            prepared_stake(lag_opp, 250.0, 2),
        ]
        GlimpseABM.manage_opportunities!(
            lag_sim.market, 0, Dict{String,Int}(), 0.0)
        GlimpseABM.step!(lag_sim, 1)
        @test lag_opp.total_invested == 250.0
        @test lag_opp.committed_prev_round == 250.0

        # Appropriability is founder-level, not record-level: splitting one
        # founder's 200-unit position into two records cannot halve the share.
        share_cfg = deepcopy(culled_cfg)
        share_cfg.RANDOM_SEED = 8241
        share_sim = EmergentSimulation(config=share_cfg, seed=8241)
        share_opp = Opportunity(
            id="founder_aggregate_share",
            latent_return_potential=2.0,
            latent_failure_potential=0.1,
            discovered=true,
            sector="tech",
            capital_requirements=100.0,
            time_to_maturity=2,
            capacity=100.0,
            total_invested=400.0,
            committed_prev_round=400.0,
            config=share_cfg,
            rng=MersenneTwister(8242),
        )
        GlimpseABM.add_opportunity!(share_sim.market, share_opp)
        focal, other = share_sim.agents
        for agent in share_sim.agents
            agent.fixed_ai_level = "none"
            agent.current_ai_level = "none"
        end
        focal.active_investments = [
            prepared_stake(share_opp, 100.0, 1),
            prepared_stake(share_opp, 100.0, 1),
        ]
        other.active_investments = [prepared_stake(share_opp, 200.0, 2)]
        GlimpseABM.step!(share_sim, 1)
        @test [event[19] for event in focal.venture_ledger] == [0.5, 0.5]
        @test all(event[17] == focal.venture_ledger[1][17]
                  for event in focal.venture_ledger)
    end

    @testset "proximate causal chain and opportunity flows reach per-run output" begin
        cfg = EmergentConfig(
            N_AGENTS=4,
            N_ROUNDS=2,
            RANDOM_SEED=731,
            USE_UNIFORM_INITIAL_CAPITAL=true,
            INITIAL_CAPITAL=1_000_000.0,
            NICHE_OPPORTUNITIES_PER_DISCOVERY_MIN=2,
            NICHE_OPPORTUNITIES_PER_DISCOVERY_MAX=2,
            enable_round_logging=false,
        )
        GlimpseABM.initialize!(cfg)
        sim = EmergentSimulation(config=cfg, seed=731, run_id="telemetry_export_probe")
        a1, a2 = sim.agents[1], sim.agents[2]
        a1.fixed_ai_level = "none"
        a1.current_ai_level = "none"
        a2.fixed_ai_level = "premium"
        a2.current_ai_level = "premium"

        opp = Opportunity(
            id="commitment_export_probe",
            latent_return_potential=2.0,
            latent_failure_potential=0.0,
            discovered=true,
            sector="tech",
            capital_requirements=10_000.0,
            time_to_maturity=1,
            capacity=100_000.0,
            config=cfg,
            rng=MersenneTwister(732),
        )
        GlimpseABM.add_opportunity!(sim.market, opp)

        first_outcome = GlimpseABM.execute_action!(
            a1, "invest", sim.market, 1;
            opportunity=opp,
            estimated_return=1.8,
            instrument_estimated_return=1.9,
        )
        second_outcome = GlimpseABM.execute_action!(
            a2, "invest", sim.market, 1;
            opportunity=opp,
            estimated_return=1.9,
            instrument_estimated_return=2.0,
        )
        @test first_outcome["success"]
        @test second_outcome["success"]
        @test length(sim.market.opportunity_commitment_events[opp.id]) == 2

        matured = GlimpseABM.process_matured_investments!(a1, sim.market, 2)
        @test length(matured) == 1
        @test matured[1]["post_commitment_rival_capital"] ≈
            second_outcome["amount"]
        @test matured[1]["capacity_saturation_change"] ≈
            matured[1]["capacity_saturation_at_maturity"] -
            matured[1]["capacity_saturation_at_entry"]
        @test matured[1]["crowding_return_multiplier"] ≈
            a1.venture_ledger[1][15]
        @test matured[1]["realized_latent_capture"] ≈
            matured[1]["return_multiple"] / opp.latent_return_potential
        @test matured[1]["performative_effective_capacity"] == opp.capacity
        @test matured[1]["performative_uplift_fraction"] == 0.0
        @test matured[1]["performative_effective_capacity"] ≈
            a1.venture_ledger[1][17]

        niche_action = Dict{String,Any}(
            "action" => "explore",
            "exploration_type" => "niche_discovery",
            "agent_id" => a1.id,
            "niche_sector" => "tech_specialized",
        )
        @test length(GlimpseABM._create_niche_opportunities_from_action!(
            sim, niche_action, 2)) == 2

        condition = _robustness_condition("BASELINE")
        per_run = DataFrame(RRH.summarize_simulation(condition, sim, 1, 731))
        required = [
            :mean_capacity_saturation_at_entry,
            :mean_capacity_saturation_at_maturity,
            :mean_effective_capacity_saturation_at_maturity,
            :mean_capacity_saturation_change,
            :mean_post_commitment_rival_capital,
            :mean_crowding_return_multiplier,
            :mean_realized_latent_capture,
            :mean_performative_effective_capacity,
            :mean_performative_lagged_commitment,
            :mean_performative_commitment_share,
            :mean_performative_uplift_fraction,
            :exploration_opportunities_created_total,
            :innovation_opportunities_spawned_total,
            :background_opportunities_created_total,
            :opportunities_publicized_total,
            :opportunities_culled_total,
        ]
        @test isempty(setdiff(required, propertynames(per_run)))
        summary = RRH.summarize_rows(per_run)
        summary_required = [
            :mean_effective_capacity_saturation_at_maturity,
            :mean_performative_effective_capacity,
            :mean_performative_lagged_commitment,
            :mean_performative_commitment_share,
            :mean_performative_uplift_fraction,
        ]
        @test isempty(setdiff(summary_required, propertynames(summary)))
        none_row = only(eachrow(per_run[per_run.tier .== "none", :]))
        none_summary = only(eachrow(summary[summary.tier .== "none", :]))
        for metric in summary_required
            @test isequal(none_summary[metric], none_row[metric])
        end
        @test none_row.mean_post_commitment_rival_capital ≈
            second_outcome["amount"]
        @test first(per_run.exploration_opportunities_created_total) == 2.0
    end

    @testset "paired ablation output spans economic, innovation, and survival outcomes" begin
        rows = NamedTuple[]
        for (run_idx, none_survival, premium_survival, none_wealth, premium_wealth,
             none_combinations, premium_combinations) in [
                (1, 0.70, 0.60, 1_000_000.0, 900_000.0, 1.0, 3.0),
                (2, 0.70, 0.55, 1_000_000.0, 880_000.0, 1.5, 4.0),
                (3, 0.70, 0.65, 1_000_000.0, 950_000.0, 2.0, missing),
            ]
            for (tier, survival, wealth, combinations) in [
                ("none", none_survival, none_wealth, none_combinations),
                ("premium", premium_survival, premium_wealth, premium_combinations),
            ]
                push!(rows, (
                    condition="SYNTHETIC_ABLATION",
                    category="test",
                    description="Synthetic paired-outcome export probe.",
                    theoretical_role="Test multivariate Figure 4 contract.",
                    design="fixed_mixed",
                    run_idx=run_idx,
                    tier=tier,
                    survival_rate=survival,
                    mean_net_worth=wealth,
                    combinations_per_agent=combinations,
                ))
            end
        end
        per_run = DataFrame(rows)
        effects = RRH.paired_outcome_effects(per_run)
        frontier = effects[effects.tier .== "premium", :]

        @test Set(frontier.outcome_family) == Set(["economic", "innovation", "survival"])
        @test only(frontier[frontier.outcome .== "terminal_net_worth", :n_runs]) == 3
        @test only(frontier[frontier.outcome .== "knowledge_combinations", :n_runs]) == 2
        @test only(frontier[frontier.outcome .== "five_year_survival", :n_runs]) == 3
        @test only(frontier[frontier.outcome .== "terminal_net_worth", :mean_effect]) ≈ -90.0
        @test only(frontier[frontier.outcome .== "knowledge_combinations", :mean_effect]) ≈ 2.25
        @test only(frontier[frontier.outcome .== "five_year_survival", :mean_effect]) ≈ -10.0

        # The new generic survival estimate must remain byte-for-byte compatible
        # in estimand and scaling with the legacy survival-only output.
        legacy = RRH.paired_treatment_effects(per_run)
        legacy_frontier = only(legacy[legacy.tier .== "premium", :mean_te_pp])
        @test legacy_frontier ==
            only(frontier[frontier.outcome .== "five_year_survival", :mean_effect])
    end

    @testset "innovation-spawned opportunities use canonical capacity tail" begin
        cfg = EmergentConfig(N_AGENTS=4, N_ROUNDS=1, RANDOM_SEED=43)
        cfg.NICHE_SIZE_LOG_SIGMA = 3.0
        GlimpseABM.initialize!(cfg)
        market = MarketEnvironment(cfg; rng=MersenneTwister(43))
        base = cfg.OPPORTUNITY_BASE_CAPACITY
        variance = cfg.OPPORTUNITY_CAPACITY_VARIANCE

        caps = Float64[]
        for i in 1:20
            innovation = Innovation(
                id="spawn_cap_probe_$i",
                type="radical",
                knowledge_components=["k1", "k2"],
                novelty=0.85,
                quality=0.75,
                round_created=i,
                creator_id=1,
                success=true,
                ai_assisted=true,
                sector="tech",
                combination_signature="k1||k2||$i",
                cash_multiple=3.0,
                scarcity=0.90,
                is_new_combination=true,
                ai_level_used="premium",
            )
            spawned = GlimpseABM.spawn_opportunity_from_innovation!(market, innovation, 3.0)
            push!(caps, spawned.capacity)
        end

        uniform_lo = base * (1.0 - variance)
        uniform_hi = base * (1.0 + variance)
        @test any(c -> c < uniform_lo || c > uniform_hi, caps)
    end
end
