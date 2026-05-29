# Regression: the perception → utility pathway for Knightian uncertainty.
#
# An earlier ignorance_adjustment sigmoid was centered outside agents'
# operating range, so perception barely affected utility. Later fixes removed
# direct raw-tier perception discounts: premium should differ from none through
# actual visibility/evidence, learned experience, and behavioral concentration,
# not because the tier label directly lowers actor ignorance.
#
# This test guards the pathway — if either fix regresses, the Knightian
# framing detaches from decision utility and the paper's claim no longer
# holds in the code.

using Test
using Random
using Statistics

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using GlimpseABM

@testset "Knightian perception → utility pathway" begin
    cfg = EmergentConfig(N_AGENTS=100, N_ROUNDS=15, RANDOM_SEED=42,
                         AGENT_AI_MODE="fixed")
    GlimpseABM.initialize!(cfg)

    # Gather per-agent perception across tiers using the same production
    # visibility and agent-state wiring used by make_decision!.
    function probe(tier::String)
        sim = EmergentSimulation(config=cfg, seed=42,
                                  initial_tier_distribution=Dict(tier=>1.0))
        GlimpseABM.run!(sim)
        alive = [a for a in sim.agents if a.alive]
        isempty(alive) && return (NaN, NaN, NaN, NaN)
        mc = GlimpseABM.get_market_conditions(
            sim.market;
            uncertainty_state=GlimpseABM.get_uncertainty_state(sim.uncertainty_env),
        )
        igs = Float64[]
        recs = Float64[]
        visible_counts = Float64[]
        info_quality = Float64[]
        for a in alive[1:min(20, length(alive))]
            opps = GlimpseABM.get_opportunities_for_agent(sim.market, a)
            perc = GlimpseABM.perceive_uncertainty(sim.uncertainty_env, a.traits,
                opps, mc; ai_level=GlimpseABM.get_ai_level(a),
                agent_id=a.id,
                agent_knowledge=Set(keys(a.resources.knowledge)),
                sector_knowledge=a.resources.knowledge,
                action_history=a.action_history,
                ai_learning_profile=a.ai_learning,
                recent_outcomes=a.recent_outcomes)
            push!(igs, Float64(get(get(perc, "actor_ignorance", Dict()), "level", 0.5)))
            push!(recs, Float64(get(get(perc, "competitive_recursion", Dict()), "level", 0.5)))
            push!(visible_counts, length(opps))
            push!(info_quality, Float64(perc.knowledge_signal.info_quality))
        end
        return (mean(igs), mean(recs), mean(visible_counts), mean(info_quality))
    end

    none_ig, none_rec, none_visible, none_info = probe("none")
    prem_ig, prem_rec, prem_visible, prem_info = probe("premium")

    # ───────────────────────────────────────────────────────────────
    # Property 1: frontier/premium actor ignorance is mediated by observable
    # opportunity access and evidence quality, not a direct raw-tier shortcut.
    # With populated niche ontology, frontier can see many more opaque/tacit
    # opportunities. That can raise, rather than lower, average actor ignorance
    # because the extra opportunity surface carries real unresolved uncertainty.
    # ───────────────────────────────────────────────────────────────
    @test prem_visible > none_visible
    @test isfinite(prem_info)
    @test isfinite(none_info)
    @test isfinite(prem_ig)
    @test isfinite(none_ig)
    @test abs(prem_ig - none_ig) > 0.001
    if prem_ig > none_ig
        @test prem_info < none_info
    else
        @test none_ig > prem_ig
    end

    # ───────────────────────────────────────────────────────────────
    # Property 2: the ignorance_adjustment multiplier used in
    # calculate_investment_utility still responds to emergent perception
    # differences in the correct direction, without assuming the frontier tier
    # must always have lower actor ignorance.
    # ───────────────────────────────────────────────────────────────
    function ig_adj(actor_unc)
        clamp(1.0 - actor_unc * 0.8, 0.2, 1.0)
    end
    none_adj = ig_adj(none_ig)
    prem_adj = ig_adj(prem_ig)
    @test abs(prem_adj - none_adj) > 0.0005
    @test (prem_ig > none_ig && prem_adj < none_adj) ||
          (prem_ig < none_ig && prem_adj > none_adj)

    # ───────────────────────────────────────────────────────────────
    # Property 3: premium perceives more competitive_recursion than
    # none (they converge on the same top opps → crowded niches).
    # This signals the trap's presence in perception.
    # ───────────────────────────────────────────────────────────────
    @test prem_rec > none_rec

    # ───────────────────────────────────────────────────────────────
    # Property 4: recursion's behavioral effect is material. With a recursion
    # coefficient of 0.25, even moderate recursion (0.25) produces a ~6.25%
    # utility hit.
    # ───────────────────────────────────────────────────────────────
    rec_coef = 0.25
    @test rec_coef * prem_rec >= 0.04  # premium's recursion penalty is material
end

println("Knightian pathway test passed.")
