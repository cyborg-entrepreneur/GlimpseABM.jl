# Regression test for the S2 per-opportunity knowledge edge
# (STRATEGY_MODE="comparative_advantage" with ENABLE_OPPORTUNITY_COMPONENTS).
#
# Guards:
#   1. Overlap differentiates across agents AND opportunities, and reduces to the
#      sector-familiarity fallback when the agent has no component knowledge
#      (function-level byte-identity when the flag is off).
#   2. strategy_edge_context centers on the choice-set mean and is off-neutral
#      (empty agent components → sector-familiarity fallback, no differentiation).
#   3. LIVE ONTOLOGY MATCH: in a real run the agent-knowledge IDs and the
#      opportunity-component IDs share the same namespace — the exact
#      sector-name-vs-component-id mis-intersection the _component_overlap
#      docstring (strategy.jl) warns about.
#   4. Population is gated: opportunities carry no components when the flag is off.

using Test
using Random
using Statistics
using GlimpseABM

_comp_opp(ids) = Opportunity(id = "o_" * join(ids, "_"), sector = "tech",
                             discovered = true, knowledge_components = collect(ids))

@testset "S2 component edge: overlap differentiates + off-fallback" begin
    opp_ab = _comp_opp(["c1", "c2"])
    opp_cd = _comp_opp(["c3", "c4"])
    opp_c2 = _comp_opp(["c2", "c5"])
    opp_empty = Opportunity(id = "o_empty", sector = "tech", discovered = true)  # no components

    A = Set(["c1", "c2", "c9"])   # both of opp_ab, one of opp_c2, none of opp_cd
    B = Set(["c3", "c4"])
    fb = 0.1

    @test GlimpseABM._component_overlap(opp_ab, A, fb) == 1.0
    @test GlimpseABM._component_overlap(opp_cd, A, fb) == 0.0
    @test GlimpseABM._component_overlap(opp_c2, A, fb) == 0.5
    # Same opportunity, different agents → different edge (heterogeneity does work).
    @test GlimpseABM._component_overlap(opp_ab, B, fb) == 0.0
    @test GlimpseABM._component_overlap(opp_ab, A, fb) !=
          GlimpseABM._component_overlap(opp_ab, B, fb)
    # Empty agent components (flag off / no knowledge) → sector-familiarity fallback.
    @test GlimpseABM._component_overlap(opp_ab, Set{String}(), fb) == fb
    # Opportunity with no components → fallback regardless of the agent.
    @test GlimpseABM._component_overlap(opp_empty, A, fb) == fb

    # private_edge: agent_components drives the edge; default-empty → fallback.
    @test GlimpseABM.private_edge(opp_ab, nothing, 0.3, A) == 1.0
    @test GlimpseABM.private_edge(opp_ab, nothing, 0.3) == 0.3
end

@testset "S2 component edge: edge-context centering + off-neutrality" begin
    config = EmergentConfig(STRATEGY_MODE = "comparative_advantage")
    GlimpseABM.initialize!(config)
    sk = Dict("tech" => 0.5)
    opps = [_comp_opp(["c1", "c2"]), _comp_opp(["c3", "c4"]), _comp_opp(["c2", "c5"])]
    A = Set(["c1", "c2", "c9"])

    ctx = GlimpseABM.strategy_edge_context(config, sk, opps, A)
    @test ctx !== nothing
    edges = [ctx.edges[o.id] for o in opps]
    @test edges == [1.0, 0.0, 0.5]
    @test ctx.edge_mean ≈ mean(edges)     # centering anchor is the choice-set mean
    @test !allequal(edges)                 # component edge differentiates within a sector

    # Flag off (empty agent components): edges fall back to sector familiarity,
    # identical across same-sector opportunities → no differentiation.
    ctx0 = GlimpseABM.strategy_edge_context(config, sk, opps, Set{String}())
    edges0 = [ctx0.edges[o.id] for o in opps]
    @test allequal(edges0)
    @test edges0 != edges                  # the component set genuinely drives the edge
end

@testset "S2 component edge: live ontology match (agent IDs ∩ opp IDs ≠ ∅)" begin
    config = EmergentConfig(N_AGENTS = 120, N_ROUNDS = 30, RANDOM_SEED = 4242,
                            STRATEGY_MODE = "comparative_advantage",
                            ENABLE_OPPORTUNITY_COMPONENTS = true,
                            DERIVED_KNOWLEDGE_QUALITY_GATE = 0.55,
                            enable_round_logging = false)
    GlimpseABM.initialize!(config)
    sim = EmergentSimulation(config = config, seed = 4242)
    for r in 1:config.N_ROUNDS
        GlimpseABM.step!(sim, r)
    end

    opp_comp_ids = Set{String}()
    n_opps_with = 0
    for o in sim.market.opportunities
        if !isempty(o.knowledge_components)
            n_opps_with += 1
            union!(opp_comp_ids, o.knowledge_components)
        end
    end
    agent_comp_ids = isempty(sim.knowledge_base.agent_knowledge) ? Set{String}() :
        reduce(union, values(sim.knowledge_base.agent_knowledge))

    @test n_opps_with > 0                  # population fires (created niches carry components)
    @test !isempty(agent_comp_ids)         # recombinant engine fires (agents accumulate components)
    # THE GUARD: the two namespaces share IDs — agents and opportunities draw on the
    # SAME component ontology, so the overlap edge is real, not a silent
    # sector-name-vs-component-id mis-intersection.
    @test !isempty(intersect(agent_comp_ids, opp_comp_ids))
end

@testset "S2 component edge: population gated off" begin
    config = EmergentConfig(N_AGENTS = 60, N_ROUNDS = 15, RANDOM_SEED = 4242,
                            STRATEGY_MODE = "comparative_advantage",
                            ENABLE_OPPORTUNITY_COMPONENTS = false,
                            DERIVED_KNOWLEDGE_QUALITY_GATE = 0.55,
                            enable_round_logging = false)
    GlimpseABM.initialize!(config)
    sim = EmergentSimulation(config = config, seed = 4242)
    for r in 1:config.N_ROUNDS
        GlimpseABM.step!(sim, r)
    end
    @test all(isempty(o.knowledge_components) for o in sim.market.opportunities)
end

println("S2 component-edge regression tests passed.")
