using Test
using DataFrames
using CSV

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using GlimpseABM

@testset "Partial override regressions" begin
    cfg = EmergentConfig(N_AGENTS=1000)
    GlimpseABM.initialize!(cfg)

 # Override a single nested field inside a Dict{String,SectorProfile} entry.
 # Without struct-aware deep merge this errored with
 # "Missing field 'return_range' while reconstructing SectorProfile".
    original_tech = cfg.SECTOR_PROFILES["tech"]
    original_retail_ops = cfg.SECTOR_PROFILES["retail"].operational_cost_range
    GlimpseABM.apply_overrides!(cfg, Dict{String,Any}(
        "SECTOR_PROFILES" => Dict{String,Any}(
            "tech" => Dict{String,Any}(
                "operational_cost_range" => [22_000.0, 33_000.0],
           ),
       ),
   ))
    @test cfg.SECTOR_PROFILES["tech"].operational_cost_range == (22_000.0, 33_000.0)
    @test cfg.SECTOR_PROFILES["tech"].return_range == original_tech.return_range
    @test cfg.SECTOR_PROFILES["tech"].survival_threshold == original_tech.survival_threshold
 # Untouched sectors retain all fields.
    @test cfg.SECTOR_PROFILES["retail"].operational_cost_range == original_retail_ops

 # Same fix exercised through Dict{String,AILevelConfig}.
    original_premium = cfg.AI_LEVELS["premium"]
    GlimpseABM.apply_overrides!(cfg, Dict{String,Any}(
        "AI_LEVELS" => Dict{String,Any}(
            "premium" => Dict{String,Any}("cost" => 3700.0),
       ),
   ))
    @test cfg.AI_LEVELS["premium"].cost == 3700.0
    @test cfg.AI_LEVELS["premium"].info_quality == original_premium.info_quality
    @test cfg.AI_LEVELS["basic"].cost == 30.0  # untouched tier unchanged

 # And through nested Dict inside Dict{String,TraitDistribution}.params.
    original_competence = cfg.TRAIT_DISTRIBUTIONS["competence"]
    GlimpseABM.apply_overrides!(cfg, Dict{String,Any}(
        "TRAIT_DISTRIBUTIONS" => Dict{String,Any}(
            "competence" => Dict{String,Any}(
                "params" => Dict{String,Any}("low" => 0.15),
           ),
       ),
   ))
    @test cfg.TRAIT_DISTRIBUTIONS["competence"].params["low"] == 0.15
    @test cfg.TRAIT_DISTRIBUTIONS["competence"].params["high"] == original_competence.params["high"]
    @test cfg.TRAIT_DISTRIBUTIONS["competence"].dist == original_competence.dist

 # Top-level calibration structs also support partial overrides, which keeps
 # sensitivity sweeps from restating every weight in a family.
    original_recursion = cfg.KNIGHTIAN_PRODUCER_WEIGHTS.competitive_recursion
    GlimpseABM.apply_overrides!(cfg, Dict{String,Any}(
        "KNIGHTIAN_PRODUCER_WEIGHTS" => Dict{String,Any}(
            "competitive_recursion" => Dict{String,Any}(
                "volatility_weight" => 0.44,
           ),
       ),
   ))
    @test cfg.KNIGHTIAN_PRODUCER_WEIGHTS.competitive_recursion.volatility_weight == 0.44
    @test cfg.KNIGHTIAN_PRODUCER_WEIGHTS.competitive_recursion.ai_herd_weight ==
          original_recursion.ai_herd_weight

    original_perception = cfg.UNCERTAINTY_PERCEPTION_WEIGHTS
    GlimpseABM.apply_overrides!(cfg, Dict{String,Any}(
        "UNCERTAINTY_PERCEPTION_WEIGHTS" => Dict{String,Any}(
            "actor_prior_weight" => 0.61,
       ),
   ))
    @test cfg.UNCERTAINTY_PERCEPTION_WEIGHTS.actor_prior_weight == 0.61
    @test cfg.UNCERTAINTY_PERCEPTION_WEIGHTS.recursion_prior_weight ==
          original_perception.recursion_prior_weight
end

@testset "I/O roundtrip regressions" begin
    mktempdir() do dir
        cfg = EmergentConfig(N_AGENTS=2, N_ROUNDS=1, RANDOM_SEED=123)
        GlimpseABM.initialize!(cfg)

        config_path = joinpath(dir, "config_snapshot.json")
        GlimpseABM.save_config_snapshot(cfg, config_path)
        loaded = GlimpseABM.load_config_snapshot(config_path)

        @test loaded.N_AGENTS == cfg.N_AGENTS
        @test loaded.N_ROUNDS == cfg.N_ROUNDS
        @test loaded.OPPORTUNITY_BASE_CAPACITY == cfg.OPPORTUNITY_BASE_CAPACITY
        @test loaded.AI_LEVELS["premium"].cost == cfg.AI_LEVELS["premium"].cost
        @test loaded.SECTOR_PROFILES["tech"].operational_cost_range ==
              cfg.SECTOR_PROFILES["tech"].operational_cost_range
        @test loaded.TRAIT_DISTRIBUTIONS["competence"].params["low"] ==
              cfg.TRAIT_DISTRIBUTIONS["competence"].params["low"]
    end

    mktempdir() do dir
        cfg = EmergentConfig(N_AGENTS=2, N_ROUNDS=1, RANDOM_SEED=456)
        GlimpseABM.initialize!(cfg)
        GlimpseABM.save_config_snapshot(cfg, joinpath(dir, "config_snapshot.json"))

        history = DataFrame(round=[1], note=["alpha,beta"], value=[1.25])
        CSV.write(joinpath(dir, "history.csv"), history)
        open(joinpath(dir, "metadata.json"), "w") do io
            write(io, "{\"run_id\":\"csv_fallback_probe\",\"timestamp\":\"2026-05-13T00:00:00\"}")
        end

        loaded = GlimpseABM.load_results(dir)
        @test nrow(loaded.history) == 1
        @test loaded.history.note[1] == "alpha,beta"
        @test loaded.history.value[1] == 1.25
        @test loaded.run_id == "csv_fallback_probe"
    end
end

println("I/O roundtrip regression tests passed.")
