# v3.3/v3.5 regression: the perception → utility pathway for Knightian uncertainty.
#
# Before v3.3 the ignorance_adjustment sigmoid was centered outside agents'
# operating range, so perception barely affected utility. Later fixes removed
# direct raw-tier perception discounts: premium should differ from none through
# actual visibility/evidence, learned experience, and behavioral concentration,
# not because the tier label directly lowers actor ignorance.
#
# This test guards the pathway — if either fix regresses, the Knightian
# framing detaches from decision utility and the paper's claim no longer
# holds in the code.
#
# Restoration: the guards below previously (a)
# asserted |Δ| > 0.001 (a vacuous floor ~80× weaker than the original 0.08
# directional spread) and (b) mirrored a stale copy of the utility formula
# (clamp(1 − unc·0.8, 0.2, 1.0) on RAW perception) instead of exercising the
# production path, which routes perception through
# _response_adjusted_uncertainty before the 0.8 mapping. Both are fixed:
# magnitudes are material again, and every pathway assertion now calls the
# production functions (no local re-implementation of any coefficient).

using Test
using Random
using Statistics

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using GlimpseABM

# Rebuild an immutable perception-dimension struct with its `level` replaced,
# and a Perception with one dimension swapped. Uses positional constructors so
# no production constant is duplicated here.
function _with_level(dim_struct, new_level::Float64)
    T = typeof(dim_struct)
    names = fieldnames(T)
    vals = Any[getfield(dim_struct, f) for f in names]
    i = findfirst(==(:level), collect(names))
    @assert !isnothing(i) "$(T) has no :level field"
    vals[i] = new_level
    return T(vals...)
end

function _with_dimension(p::GlimpseABM.Perception, dim::Symbol, new_level::Float64)
    names = fieldnames(GlimpseABM.Perception)
    vals = Any[getfield(p, f) for f in names]
    i = findfirst(==(dim), collect(names))
    @assert !isnothing(i) "Perception has no field $(dim)"
    vals[i] = _with_level(vals[i], new_level)
    return GlimpseABM.Perception(vals...)
end

@testset "Knightian perception → utility pathway" begin
    cfg = EmergentConfig(N_AGENTS=100, N_ROUNDS=15, RANDOM_SEED=42,
                         AGENT_AI_MODE="fixed")
    GlimpseABM.initialize!(cfg)

    # Gather per-agent perception across tiers using the same production
    # visibility and agent-state wiring used by make_decision!. Also keep one
    # live (agent, opportunities, market_conditions, perception) tuple per tier
    # for the production utility probe below.
    function probe(tier::String)
        sim = EmergentSimulation(config=cfg, seed=42,
                                  initial_tier_distribution=Dict(tier=>1.0))
        GlimpseABM.run!(sim)
        alive = [a for a in sim.agents if a.alive]
        isempty(alive) && return nothing
        mc = GlimpseABM.get_market_conditions(
            sim.market;
            uncertainty_state=GlimpseABM.get_uncertainty_state(sim.uncertainty_env),
        )
        igs = Float64[]
        recs = Float64[]
        actor_pressures = Float64[]
        rec_pressures = Float64[]
        visible_counts = Float64[]
        info_quality = Float64[]
        utility_ctx = nothing
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
            ig = Float64(get(get(perc, "actor_ignorance", Dict()), "level", 0.5))
            rec = Float64(get(get(perc, "competitive_recursion", Dict()), "level", 0.5))
            push!(igs, ig)
            push!(recs, rec)
            # PRODUCTION transform — the same function calculate_investment_utility
            # applies before the ignorance/recursion coefficients. Guarded here so
            # a saturating or constant response remap is caught even though the
            # perception levels themselves still vary.
            push!(actor_pressures,
                  GlimpseABM._response_adjusted_uncertainty(a, "actor_ignorance", ig))
            push!(rec_pressures,
                  GlimpseABM._response_adjusted_uncertainty(a, "competitive_recursion", rec))
            push!(visible_counts, length(opps))
            push!(info_quality, Float64(perc.knowledge_signal.info_quality))
            if isnothing(utility_ctx) && !isempty(opps)
                utility_ctx = (agent=a, opps=opps, mc=mc, perc=perc)
            end
        end
        return (ig=mean(igs), rec=mean(recs),
                actor_pressure=mean(actor_pressures),
                rec_pressure=mean(rec_pressures),
                visible=mean(visible_counts), info=mean(info_quality),
                ctx=utility_ctx)
    end

    none_p = probe("none")
    prem_p = probe("premium")
    @test !isnothing(none_p)
    @test !isnothing(prem_p)

    # ───────────────────────────────────────────────────────────────
    # Property 1: frontier/premium actor ignorance is mediated by observable
    # opportunity access and evidence quality, not a direct raw-tier shortcut.
    # With populated niche ontology, frontier can see many more opaque/tacit
    # opportunities. That can raise, rather than lower, average actor ignorance
    # because the extra opportunity surface carries real unresolved uncertainty.
    # Direction is design-ambiguous in v3.5; MAGNITUDE is not — the tiers must
    # perceive materially different ignorance or the Knightian framing detaches
    # from the tier comparison entirely.
    # ───────────────────────────────────────────────────────────────
    @test prem_p.visible > none_p.visible
    @test isfinite(prem_p.info) && isfinite(none_p.info)
    @test isfinite(prem_p.ig) && isfinite(none_p.ig)
    @test abs(prem_p.ig - none_p.ig) > 0.02   # material spread (was vacuous 0.001)
    if prem_p.ig > none_p.ig
        @test prem_p.info < none_p.info
    else
        @test none_p.ig > prem_p.ig
    end

    # ───────────────────────────────────────────────────────────────
    # Property 2: the PRODUCTION response transform still (a) discriminates the
    # operating range — i.e., has not saturated into a constant — and (b)
    # propagates the tier perception spread into materially different pressures.
    # Uses _response_adjusted_uncertainty directly on live agents; no local
    # formula mirror.
    # ───────────────────────────────────────────────────────────────
    probe_agent = prem_p.ctx.agent
    lo = GlimpseABM._response_adjusted_uncertainty(probe_agent, "actor_ignorance", 0.2)
    hi = GlimpseABM._response_adjusted_uncertainty(probe_agent, "actor_ignorance", 0.8)
    @test hi > lo                       # monotone in the level
    @test hi - lo > 0.10                # not flattened/saturated on the range
    lo_rec = GlimpseABM._response_adjusted_uncertainty(probe_agent, "competitive_recursion", 0.2)
    hi_rec = GlimpseABM._response_adjusted_uncertainty(probe_agent, "competitive_recursion", 0.8)
    @test hi_rec > lo_rec
    @test hi_rec - lo_rec > 0.10

    @test abs(prem_p.actor_pressure - none_p.actor_pressure) > 0.01
    # Sign consistency: the transform is monotone (asserted above), so the
    # pressure ordering must match the raw perception ordering.
    @test sign(prem_p.actor_pressure - none_p.actor_pressure) ==
          sign(prem_p.ig - none_p.ig)

    # ───────────────────────────────────────────────────────────────
    # Property 3: premium perceives more competitive_recursion than
    # none (they converge on the same top opps → crowded niches).
    # This signals the trap's presence in perception — directional AND material.
    # ───────────────────────────────────────────────────────────────
    @test prem_p.rec > none_p.rec
    @test prem_p.rec - none_p.rec > 0.02

    # ───────────────────────────────────────────────────────────────
    # Property 4: END-TO-END — calculate_investment_utility (the real function,
    # production constants, no mirror) responds to perceived ignorance and
    # recursion in the correct direction and at material magnitude. Perception
    # variants are built by struct-rebuild on the live perception; agents are
    # deep-copied per call so RNG/capital state cannot leak between calls.
    # Catches: zeroed coefficients, clamp dead zones, transform saturation,
    # and formula drift — the exact regressions the old mirrored-formula
    # version could not see.
    # ───────────────────────────────────────────────────────────────
    ctx = prem_p.ctx
    @test !isnothing(ctx)
    ai_lvl = GlimpseABM.get_ai_level(ctx.agent)

    function utility_at(dim::Symbol, level::Float64)
        agent_copy = deepcopy(ctx.agent)
        perc_var = _with_dimension(ctx.perc, dim, level)
        return GlimpseABM.calculate_investment_utility(
            agent_copy, ctx.opps, ctx.mc, perc_var;
            ai_level=ai_lvl, info_system=nothing)
    end

    u_ig_low  = utility_at(:actor_ignorance, 0.15)
    u_ig_high = utility_at(:actor_ignorance, 0.85)
    @test u_ig_low > u_ig_high              # ignorance suppresses investment utility
    # Material, not epsilon. The ignorance adjustment is multiplicative on the
    # neutral-information scaled score (info_system=nothing here), so the
    # absolute gap is smaller than the subtractive recursion penalty below.
    # Threshold recalibrated 2026-06-09: the innovativeness respecification
    # (lognormal → Beta(6,2)) shifted seed-42 trait draws, moving the observed
    # healthy gap from ≈ 0.016 to ≈ 0.0092 (verified unaffected by the
    # dormant-mechanism excisions, which are value-identical at this seed).
    # 0.005 still sits well above any flattened-pathway regression (≈ 0).
    @test u_ig_low - u_ig_high > 0.005

    u_rec_low  = utility_at(:competitive_recursion, 0.15)
    u_rec_high = utility_at(:competitive_recursion, 0.85)
    @test u_rec_low > u_rec_high            # perceived recursion deters piling in
    @test u_rec_low - u_rec_high > 0.02     # the trap is behaviorally material
end

println("Knightian pathway test passed.")
