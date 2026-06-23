using Test
using Random

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using GlimpseABM

module ReviewerRobustnessHarness
include(joinpath(@__DIR__, "..", "scripts", "run_robustness_suite.jl"))
end

const RRH = ReviewerRobustnessHarness

function _reviewer_condition(name::AbstractString)
    by_name = Dict(c.name => c for c in RRH.all_conditions())
    @test haskey(by_name, name)
    return by_name[name]
end

function _reviewer_config(name::AbstractString; seed::Int=20260425)
    return RRH.build_config(_reviewer_condition(name), seed)
end

@testset "Reviewer robustness suite wiring" begin
    @testset "canonical tail and cost cells are live" begin
        baseline = _reviewer_config("BASELINE")
        truncated = _reviewer_config("TRUNCATED_TAIL")
        moderate = _reviewer_config("MODERATE_TAIL")
        cost_low = _reviewer_config("OPS_COST_060")
        cost_mid = _reviewer_config("OPS_COST_075")
        cost_high = _reviewer_config("OPS_COST_100")

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

    @testset "effective-config export carries canonical tail fields" begin
        rows = RRH.effective_config_rows([
            _reviewer_condition("BASELINE"),
            _reviewer_condition("TRUNCATED_TAIL"),
            _reviewer_condition("MODERATE_TAIL"),
            _reviewer_condition("OPS_COST_075"),
        ], 20260425)

        @test :niche_size_log_sigma in propertynames(rows)
        @test :ops_cost_intensity in propertynames(rows)
        @test :heavy_tail_returns in propertynames(rows)
        @test :unicorn_tail_env in propertynames(rows)
        @test rows[rows.condition .== "BASELINE", :niche_size_log_sigma][1] == RRH.NICHE_SIGMA
        @test rows[rows.condition .== "TRUNCATED_TAIL", :niche_size_log_sigma][1] == 0.0
    end

    @testset "venture ledger records canonical capacity" begin
        cfg = EmergentConfig(N_AGENTS=4, N_ROUNDS=1, RANDOM_SEED=13)
        GlimpseABM.initialize!(cfg)
        agent = EmergentAgent(1, cfg; rng=MersenneTwister(13))
        @test eltype(agent.venture_ledger) == NTuple{8,Float64}
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
