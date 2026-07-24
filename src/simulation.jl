"""
Main simulation orchestrator for GlimpseABM.jl

This module implements the EmergentSimulation that coordinates agents,
market, and uncertainty environments.
"""

using Random
using Statistics
using DataFrames
using Dates
using SHA

const PERCEPTION_TELEMETRY_DIMENSIONS = (
    "actor_ignorance",
    "practical_indeterminism",
    "agentic_novelty",
    "competitive_recursion",
)

const PERCEPTION_TELEMETRY_FIELDS = (
    "bounded_level",
    "normalized_score",
    "raw_score",
    "local_raw",
    "prior",
    "gap_vs_prior",
    "bound_hit",
)

const KNIGHTIAN_COMPONENT_DIMENSIONS = (
    "actor_ignorance",
    "practical_indeterminism",
    "agentic_novelty",
    "competitive_recursion",
)

function _flatten_knightian_component_telemetry(
    uncertainty_state::Dict{String,Dict{String,Any}}
)::Dict{String,Float64}
    telemetry = Dict{String,Float64}()
    for dim in KNIGHTIAN_COMPONENT_DIMENSIONS
        dim_state = get(uncertainty_state, dim, Dict{String,Any}())
        for component_field in ("components", "crowding_components", "behavioral_crowding_components")
            components = get(dim_state, component_field, nothing)
            components isa AbstractDict || continue
            label = component_field == "components" ? "component" :
                component_field == "crowding_components" ? "crowding_component" :
                "behavioral_crowding_component"
            for (name, value) in components
                value isa Number || continue
                telemetry["knightian_$(dim)_$(label)_$(name)"] = Float64(value)
            end
        end
        raw = get(dim_state, "raw_score", nothing)
        if raw isa Number
            telemetry["knightian_$(dim)_raw_score"] = Float64(raw)
        end
    end
    return telemetry
end

# ============================================================================
# EMERGENT SIMULATION
# ============================================================================

"""
Main simulation class that orchestrates the agent-based model.
"""
mutable struct EmergentSimulation
    config::EmergentConfig
    agents::Vector{EmergentAgent}
    market::MarketEnvironment
    uncertainty_env::KnightianUncertaintyEnvironment
    knowledge_base::KnowledgeBase
    innovation_engine::InnovationEngine
    info_system::InformationSystem  # AI information generation system
    current_round::Int
    history::Vector{Dict{String,Any}}
    run_id::String
    output_dir::String
    rng::Random.AbstractRNG
    start_time::DateTime
    # Uncertainty transformation tracking
    previous_uncertainty_levels::Dict{String,Float64}
    baseline_uncertainty_levels::Dict{String,Float64}
end

function _action_decision_confidence(action::Dict{String,Any})::Float64
    perception = get(action, "perception", nothing)
    if perception isa Perception
        return Float64(perception.decision_confidence)
    elseif perception isa Dict
        return Float64(get(perception, "decision_confidence", get(action, "ai_confidence", 0.5)))
    else
        return Float64(get(action, "decision_confidence", get(action, "ai_confidence", 0.5)))
    end
end

function _action_float(action::Dict{String,Any}, key::String, default::Float64)::Float64
    value = get(action, key, default)
    value isa Number || return default
    parsed = Float64(value)
    return isfinite(parsed) ? parsed : default
end

function _action_niche_sector(action::Dict{String,Any})::Union{String,Nothing}
    explicit_niche = get(action, "niche_sector", nothing)
    if !isnothing(explicit_niche)
        return String(explicit_niche)
    end

    discovered_sector = get(action, "discovered_sector", nothing)
    isnothing(discovered_sector) && return nothing

    modifier = get(action, "niche_modifier", nothing)
    if !isnothing(modifier)
        return "$(_base_sector_name(String(discovered_sector)))_$(String(modifier))"
    end

    return String(discovered_sector)
end

function _posthoc_public_trace_score(opp::Opportunity)::Float64
    capacity = hasfield(typeof(opp), :capacity) ? max(Float64(opp.capacity), 1.0) : 1.0
    invested = hasfield(typeof(opp), :total_invested) ? Float64(opp.total_invested) : 0.0
    competition = hasfield(typeof(opp), :competition) ? Float64(opp.competition) : 0.0
    impact = hasfield(typeof(opp), :market_impact) ? Float64(opp.market_impact) : 0.0
    invested_trace = 1.0 - exp(-max(0.0, invested) / capacity)
    competition_trace = 1.0 - exp(-max(0.0, competition) / 1.5)
    impact_trace = 1.0 - exp(-max(0.0, impact) / 3.0)
    return 0.45 * invested_trace + 0.35 * competition_trace + 0.20 * impact_trace
end

function _posthoc_decoy_target(
    opportunities::Vector{Opportunity},
    chosen_id::AbstractString,
    top_k::Int,
)::Union{String,Nothing}
    candidates = [opp for opp in opportunities if opp.id != String(chosen_id)]
    isempty(candidates) && return nothing
    sort!(candidates, by=_posthoc_public_trace_score, rev=true)
    n = min(max(1, top_k), length(candidates))
    return candidates[n].id
end

function _posthoc_apply_manipulation_signal!(
    config::EmergentConfig,
    early_signals::Dict{String,Int},
    agent::EmergentAgent,
    outcome::Dict{String,Any},
    opportunities::Vector{Opportunity},
    ;
    manipulation_signals::Union{Dict{String,Int},Nothing}=nothing,
)::Nothing
    if !config.POSTHOC_MANIPULATION_ENABLED ||
       config.POSTHOC_MANIPULATION_SIGNAL_STRENGTH <= 0 ||
       get(outcome, "action", "") != "invest"
        return nothing
    end

    tier = String(get(outcome, "ai_behavior_level", get_ai_level(agent)))
    tier in config.POSTHOC_MANIPULATION_TIERS || return nothing

    chosen_id = String(get(outcome, "opportunity_id", ""))
    isempty(chosen_id) && return nothing
    mode = lowercase(strip(config.POSTHOC_MANIPULATION_MODE))
    target_id = if mode == "amplify"
        chosen_id
    else
        decoy = _posthoc_decoy_target(
            opportunities, chosen_id, config.POSTHOC_MANIPULATION_DECOY_TOP_K)
        isnothing(decoy) ? chosen_id : decoy
    end

    added = config.POSTHOC_MANIPULATION_SIGNAL_STRENGTH
    early_signals[target_id] = get(early_signals, target_id, 0) + added
    if !isnothing(manipulation_signals)
        manipulation_signals[target_id] =
            get(manipulation_signals, target_id, 0) + added
    end
    outcome["posthoc_manipulation_mode"] = mode
    outcome["posthoc_manipulation_signal_target"] = target_id
    outcome["posthoc_manipulation_signals_added"] = added
    return nothing
end

"""Attach inert, consumer-side telemetry for the sequential signal channel."""
function _annotate_sequential_signal_exposure!(
    outcome::Dict{String,Any},
    early_signals::Dict{String,Int},
    manipulation_signals::Dict{String,Int},
    role::String,
)::Nothing
    is_late = role == "late"
    chosen_id = get(outcome, "action", "") == "invest" ?
        String(get(outcome, "opportunity_id", "")) : ""
    outcome["sequential_decision_role"] = role
    outcome["sequential_signals_available"] =
        is_late ? sum(values(early_signals); init=0) : 0
    outcome["sequential_signal_count_on_choice"] =
        is_late && !isempty(chosen_id) ? get(early_signals, chosen_id, 0) : 0
    outcome["posthoc_manipulation_signals_available"] =
        is_late ? sum(values(manipulation_signals); init=0) : 0
    outcome["posthoc_manipulation_signal_count_on_choice"] =
        is_late && !isempty(chosen_id) ?
        get(manipulation_signals, chosen_id, 0) : 0
    return nothing
end

"""
Initialize a new simulation.
"""
function EmergentSimulation(;
    config::EmergentConfig = EmergentConfig(),
    output_dir::String = "results",
    run_id::String = "run_$(Dates.format(now(), "yyyymmdd_HHMMSS"))",
    seed::Union{Int,Nothing} = nothing,
    initial_tier_distribution::Union{Dict{String,Float64},Nothing} = nothing
)
    # Initialize configuration
    initialize!(config)

    # Derive the seed by hashing run_id into the base seed, so each run_id gets
    # a distinct but reproducible seed:
    #   seed = (base_seed + SHA256(run_id) % 1_000_000) % (2^32 - 1)
    actual_seed = if !isnothing(seed)
        seed
    elseif !isnothing(config.RANDOM_SEED) && config.RANDOM_SEED > 0
        # Deterministic hash (avoid Julia's randomized hash)
        run_hash = reinterpret(UInt64, sha256(run_id)[1:8])[1] % 1_000_000
        mod(config.RANDOM_SEED + run_hash, 2^32 - 1)
    else
        # Fallback to random seed
        rand(1:2^31-1)
    end

    # Create RNG — optionally the deterministic MT19937 (NumpyRNG) stream
    rng = if config.USE_NUMPY_RNG
        NumpyRNG(actual_seed)
    else
        MersenneTwister(actual_seed)
    end

    # Create market environment
    market = MarketEnvironment(config; rng=rng)

    # Create knowledge base and innovation engine
    knowledge_base = KnowledgeBase(config)
    combination_tracker = CombinationTracker()
    innovation_engine = InnovationEngine(config, knowledge_base, combination_tracker)

    # Attach knowledge_base to the uncertainty env so agentic-scarcity
    # perception reads live component-usage state via
    # get_component_scarcity_metric.
    uncertainty_env = KnightianUncertaintyEnvironment(config;
                                                      knowledge_base=knowledge_base,
                                                      rng=rng)

    # Create information system for AI-assisted analysis. common_error_seed
    # seeds the dedicated common-error RNG stream used when
    # AI_ERROR_CORRELATION > 0.
    info_system = InformationSystem(config; common_error_seed=actual_seed)

    # Determine initial AI tier distribution
    # Default: 100% none. Can specify e.g. Dict("none"=>0.25, "basic"=>0.25, "advanced"=>0.25, "premium"=>0.25)
    tier_order = ["none", "basic", "advanced", "premium"]
    tier_probs = if isnothing(initial_tier_distribution)
        [1.0, 0.0, 0.0, 0.0]  # Default: all start at none
    else
        [get(initial_tier_distribution, t, 0.0) for t in tier_order]
    end
    # Normalize probabilities
    total_prob = sum(tier_probs)
    if total_prob > 0
        tier_probs = tier_probs ./ total_prob
    else
        tier_probs = [1.0, 0.0, 0.0, 0.0]
    end

    # Create agents with distributed initial tiers. Two modes:
    # - AGENT_AI_MODE="fixed" (main paper analyses): lock each agent at its
    #   sampled tier by setting fixed_ai_level=<tier>. get_ai_level() returns
    #   this, choose_ai_level is never called, and the tier is permanent.
    # - AGENT_AI_MODE="emergent" (robustness checks): leave fixed_ai_level=nothing
    #   so make_decision! calls choose_ai_level each round. current_ai_level
    #   starts at the sampled initial_tier but then evolves dynamically.
    agent_ai_mode = getfield_default(config, :AGENT_AI_MODE, "fixed")
    agents = EmergentAgent[]
    for i in 1:config.N_AGENTS
        # Sample initial tier based on distribution
        r = rand(rng)
        cumsum = 0.0
        initial_tier = "none"
        for (j, tier) in enumerate(tier_order)
            cumsum += tier_probs[j]
            if r <= cumsum
                initial_tier = tier
                break
            end
        end
        fixed_kw = agent_ai_mode == "emergent" ? nothing : initial_tier
        agent = EmergentAgent(i, config; rng=rng, fixed_ai_level=fixed_kw)
        # Emergent agents start at the sampled tier but can switch; fixed
        # agents stay at initial_tier permanently.
        if agent_ai_mode == "emergent" && initial_tier != "none"
            agent.current_ai_level = initial_tier
        end
        # Initialize subscription schedule for the starting tier (fixed-tier
        # agents will use this tier all run; emergent agents may later cancel
        # and start different ones via ensure_subscription_schedule!).
        if initial_tier != "none"
            ensure_subscription_schedule!(agent, initial_tier)
        end
        push!(agents, agent)
    end

    return EmergentSimulation(
        config,
        agents,
        market,
        uncertainty_env,
        knowledge_base,
        innovation_engine,
        info_system,
        0,
        Dict{String,Any}[],
        run_id,
        output_dir,
        rng,
        now(),
        Dict{String,Float64}(),  # previous_uncertainty_levels
        Dict{String,Float64}()   # baseline_uncertainty_levels
    )
end

"""
Initialize agents with a fixed AI level (for causal analysis).
"""
function initialize_agents!(sim::EmergentSimulation; fixed_ai_level::Union{String,Nothing} = nothing)
    for agent in sim.agents
        if !isnothing(fixed_ai_level)
            agent.fixed_ai_level = fixed_ai_level
            agent.current_ai_level = fixed_ai_level
            # Start subscription schedule if this is a subscription tier
            ensure_subscription_schedule!(agent, fixed_ai_level)
        end
    end
end

"""
Run the full simulation.
"""
function run!(sim::EmergentSimulation)
    println("[$(sim.run_id)] Starting simulation...")

    for round in 1:sim.config.N_ROUNDS
        step!(sim, round)

        # Log progress periodically
        if sim.config.enable_round_logging && round % sim.config.round_log_interval == 0
            alive_count = count(a -> a.alive, sim.agents)
            capitals = [get_capital(a) for a in sim.agents if a.alive]
            mean_capital = isempty(capitals) ? 0.0 : mean(capitals)
            # println("[$(sim.run_id)] Round $round: $(alive_count)/$(sim.config.N_AGENTS) alive, mean capital: \$$(round(Int, mean_capital))")
        end
    end

    println("[$(sim.run_id)] Simulation finished.")
    return sim
end

function _create_niche_opportunities_from_action!(
    sim::EmergentSimulation,
    action::Dict{String,Any},
    round::Int,
)::Vector{String}
    get(action, "action", "") == "explore" || return String[]
    get(action, "exploration_type", "") == "niche_discovery" || return String[]

    niche_id = _action_niche_sector(action)
    isnothing(niche_id) && return String[]

    agent_id = get(action, "agent_id", 0)
    n_min = sim.config.NICHE_OPPORTUNITIES_PER_DISCOVERY_MIN
    n_max = sim.config.NICHE_OPPORTUNITIES_PER_DISCOVERY_MAX
    n_niche_opps = rand(sim.rng, n_min:n_max)
    created_ids = String[]

    for _ in 1:n_niche_opps
        new_opp = create_niche_opportunity(
            sim.market,
            string(niche_id),
            agent_id,
            round;
            sector_novelty_context=_action_float(action, "sector_novelty_context", 0.35),
            sector_uncertainty_context=_action_float(action, "sector_uncertainty_context", 0.35),
            search_effort=_action_float(action, "ai_search_effort", 0.0),
            niche_creation_evidence=_action_float(action, "niche_creation_evidence", 0.0),
        )
        push!(created_ids, new_opp.id)
    end

    if !isempty(created_ids)
        action["new_opportunity_id"] = first(created_ids)
        action["new_opportunity_ids"] = created_ids
    end

    return created_ids
end

function _apply_matured_ai_learning!(
    sim::EmergentSimulation,
    agent::EmergentAgent,
    matured_outcome::Dict{String,Any}
)
    ai_tier = String(get(
        matured_outcome,
        "ai_level",
        get(matured_outcome, "ai_label", "none"),
    ))
    ai_assisted = counts_as_ai_use(sim.config, ai_tier)

    domain = String(get(matured_outcome, "ai_analysis_domain", "market_analysis"))
    inv_amount = Float64(get(
        matured_outcome,
        "investment_amount",
        get(matured_outcome, "investment", Dict()) isa Dict ?
            get(get(matured_outcome, "investment", Dict()), "amount", 1.0) : 1.0,
    ))
    cap_returned = Float64(get(
        matured_outcome,
        "capital_returned",
        inv_amount * Float64(get(matured_outcome, "return_multiple", 1.0)),
    ))
    # Raw/decision split: the decision-time PREDICTION scored here is the RAW
    # instrument estimate —
    # what the InformationSystem actually said ("instrument_estimated_return")
    # — never the decision-basis "estimated_return", which S1 strategy
    # shading may have transformed (judging the instrument against its own
    # shaded transform manufactures forecast error the instrument never
    # made). The prediction is symmetric across populations: the
    # InformationSystem produces estimated_return (and a confidence) for tier
    # "none" too, so no-AI agents are scored against THEIR OWN forecast
    # exactly as AI users are scored against the AI-informed one. Compatibility
    # records without the raw key fall back to "estimated_return" (identical
    # unless S1 was active); records with NO estimate at all are EXCLUDED
    # from exposure recording (pred_return === nothing → the shared scorer
    # returns nothing) — never scored against the realized multiple.
    has_estimate = Bool(get(matured_outcome, "has_return_estimate", true))
    pred_return = has_estimate ? _stored_instrument_estimate(matured_outcome) : nothing
    conf = Float64(get(matured_outcome, "ai_confidence", 0.5))
    success = Bool(get(matured_outcome, "success", false))

    # Market-level uncertainty observes realized forecast disconfirmation, not
    # the hidden hallucination flag. EVERY matured outcome (with a stored
    # estimate) is recorded as an exposure — AI-assisted or not — so the
    # rolling stats carry a human baseline alongside the AI population.
    # Pareto-tailed venture returns make large forecast errors ubiquitous for
    # everyone. AI-specific consumers read only the EXCESS of the AI rate over
    # the baseline. Scoring and recording are shared with the pivot path
    # (_score_and_record_forecast_disconfirmation!, agents.jl).
    disconfirmation = _score_and_record_forecast_disconfirmation!(
        sim.uncertainty_env, agent, sim.current_round;
        ai_tier=ai_tier,
        domain=domain,
        investment_amount=inv_amount,
        capital_returned=cap_returned,
        predicted_return=pred_return,
        ai_confidence=conf,
        success=success,
    )
    if isnothing(disconfirmation)
        # No estimate was ever stored for this investment: excluded from
        # disconfirmation exposure recording AND from AI-trust learning
        # (there is no forecast to attribute accuracy or failure to).
        return (
            applied=false,
            was_accurate=nothing,
            suspected_ai_failure=false,
            disconfirmation_score=0.0,
            hidden_hallucination=Bool(get(matured_outcome, "ai_contains_hallucination", false)),
        )
    end

    if !ai_assisted
        # No-AI outcomes contribute to the population-level baseline above but
        # do not drive AI-trust learning or hallucination-penalty mechanics —
        # there is no AI to attribute the miss to. Return shape matches the
        # no-AI output contract.
        return (
            applied=false,
            was_accurate=nothing,
            suspected_ai_failure=false,
            disconfirmation_score=0.0,
            hidden_hallucination=Bool(get(matured_outcome, "ai_contains_hallucination", false)),
        )
    end

    was_accurate = update_agent_learning!(
        agent.ai_learning,
        domain,
        inv_amount,
        cap_returned,
        pred_return,
        disconfirmation.suspected_ai_failure,
        conf,
        success,
    )

    update_domain_belief!(sim.knowledge_base, agent.id, domain, disconfirmation.accuracy_score)

    # Hidden hallucination truth remains telemetry only. Knowledge/trust damage
    # follows observable disconfirmation: forecast error, confidence, realized
    # return, repeated domain misses, and analytical inference.
    if disconfirmation.suspected_ai_failure
        apply_hallucination_penalty!(
            sim.knowledge_base,
            agent,
            domain,
            disconfirmation.severity,
        )
    end

    return (
        applied=true,
        was_accurate=was_accurate,
        suspected_ai_failure=disconfirmation.suspected_ai_failure,
        disconfirmation_score=disconfirmation.disconfirmation_score,
        severity=disconfirmation.severity,
        hidden_hallucination=Bool(get(matured_outcome, "ai_contains_hallucination", false)),
    )
end

"""
Build the opportunity cohort whose outstanding capital can affect Phase-2
realizations. Live market opportunities are included for continuity, and active
investment records retain culled opportunities until their capital resolves.
"""
function _maturity_opportunity_index(
    sim::EmergentSimulation,
)::Dict{String,Opportunity}
    opportunities = Dict{String,Opportunity}(
        opp.id => opp for opp in sim.market.opportunities
    )
    for agent in sim.agents
        agent.alive || continue
        for investment in agent.active_investments
            get(investment, "capital_released_at_death", false) && continue
            opp = get(investment, "opportunity", nothing)
            opp isa Opportunity || continue
            if haskey(opportunities, opp.id) && opportunities[opp.id] !== opp
                error("Conflicting Opportunity objects share id=$(opp.id) " *
                      "across the market and active investments.")
            end
            opportunities[opp.id] = opp
        end
    end
    return opportunities
end

"""
Snapshot the prior-round outstanding commitment for every opportunity that can
still realize next round, including culled opportunities retained by an active
investment. This is called at the same end-of-round boundary as the live-market
snapshot in `manage_opportunities!`.
"""
function _snapshot_performative_commitments!(sim::EmergentSimulation)::Nothing
    for opp in values(_maturity_opportunity_index(sim))
        opp.committed_prev_round = Float64(opp.total_invested)
    end
    return nothing
end

"""
Execute a single simulation round.

Phase order: operating costs -> matured investments -> decisions -> survival
checks -> market update -> clearing/dynamics -> uncertainty measurement -> history.
"""
function step!(sim::EmergentSimulation, round::Int)
    sim.current_round = round
    # Stamp the market with the current round BEFORE discovery runs so that
    # opportunities discovered this round get the correct discovery_round.
    sim.market.current_round = round

    # Invalidate the InformationSystem caches at the start of each round.
    # The info-cache key is (opp.id, ai_level, agent_id) but opportunity state
    # (competition, total_invested, market_impact) evolves across rounds.
    # Clearing here forces a fresh Information draw per round per agent and
    # resets the common-error cache for the current round.
    clear_cache!(sim.info_system)

    # Get current uncertainty state and market conditions. MarketConditions is
    # an immutable typed struct, so uncertainty_state is injected at construction.
    uncertainty_state = get_uncertainty_state(sim.uncertainty_env)
    market_conditions = get_market_conditions(sim.market; uncertainty_state=uncertainty_state)

    # Get alive agents
    alive_agents = filter(a -> a.alive, sim.agents)

    # Phase 1: Apply operational costs FIRST, scaled by a severity multiplier,
    # before matured investments and decisions.
    avg_comp = market_conditions.avg_competition
    volatility = market_conditions.volatility
    base_vol = Float64(sim.config.MARKET_VOLATILITY)
    severity = 1.0 + avg_comp * 0.35 + max(0.0, volatility - base_vol) * 0.45
    severity = clamp(severity, 0.7, 1.9)

    for agent in sim.agents
        if !agent.alive
            continue
        end
        # Calculate base cost (sector-specific via agent.operating_cost_estimate
        # in estimate_operational_costs)
        estimated_cost = estimate_operational_costs(agent, sim.market)
        # Apply the round-local severity multiplier.
        operating_cost = max(0.0, estimated_cost * severity)
        # Keep `agent.operating_cost_estimate` as the sector base. Severity is a
        # round-local multiplier, not a persistent update to the base estimate.
        if operating_cost > 0.0
            set_capital!(agent, get_capital(agent) - operating_cost)
        end
    end

    # Phase 1.5: Charge AI subscription installments.
    for agent in sim.agents
        if !agent.alive
            continue
        end
        apply_subscription_carry!(agent, round)
    end

    # Solvency strike accrual is consolidated at end-of-round (Phase 4 below)
    # so each agent receives one solvency observation per round.

    # Phase 2: Process matured investments, passing market_conditions with the
    # current uncertainty_state. Snapshot each niche's outstanding capital once
    # at phase start so co-maturing investments face the same crowding load,
    # independent of agent iteration order.
    all_matured = Dict{String,Any}[]
    maturity_total_invested = Dict{String,Float64}(
        opp_id => Float64(opp.total_invested)
        for (opp_id, opp) in _maturity_opportunity_index(sim)
    )
    for agent in sim.agents
        if !agent.alive
            continue
        end
        matured = process_matured_investments!(
            agent,
            sim.market,
            round;
            market_conditions=market_conditions,
            maturity_total_invested=maturity_total_invested,
        )
        for m in matured
            # Update tier beliefs from investment outcomes. Pass the realized
            # return_multiple to the continuous overload so a 3× return updates
            # more strongly than a 1.1× barely-success (the Boolean overload
            # would collapse all positive outcomes to multiplier=1.5).
            ai_tier = get(m, "ai_level", get_ai_level(agent))
            return_mult = Float64(get(m, "return_multiple", get(m, "success", false) ? 1.5 : 0.5))
            update_tier_belief!(agent.ai_learning, ai_tier, return_mult)
            # Drive AI-trust and experience-based trait evolution. An outcome
            # is "AI-accurate" when the RAW instrument estimate at investment
            # (F1: instrument_estimated_return — what the AI actually said,
            # not the possibly-S1-shaded decision basis) was within 25% of
            # the realized return_multiple. Legacy records fall back to
            # "estimated_return"; records with no estimate at all are not
            # scored as accurate (ai_was_accurate=nothing).
            if counts_as_ai_use(sim.config, String(ai_tier))
                est = Bool(get(m, "has_return_estimate", true)) ?
                    _stored_instrument_estimate(m) : nothing
                if isnothing(est)
                    update_state_from_outcome!(agent, m; ai_was_accurate=nothing)
                else
                    actual = Float64(get(m, "return_multiple", 1.0))
                    ai_accurate = est > 0 && abs(actual - est) / max(abs(est), 0.1) < 0.25
                    update_state_from_outcome!(agent, m; ai_was_accurate=ai_accurate)
                end
            else
                update_state_from_outcome!(agent, m; ai_was_accurate=nothing)
            end

            # Pure telemetry: record confidence calibration against realized
            # investment outcomes. No downstream decision path reads this.
            decision_conf = Float64(get(m, "decision_confidence", get(m, "ai_confidence", 0.5)))
            realized_roi = Float64(get(m, "return_multiple", 1.0))
            record_confidence_outcome_observation!(agent, decision_conf, realized_roi;
                                                   ai_used=counts_as_ai_use(sim.config, String(ai_tier)),
                                                   channel="investment")

            _apply_matured_ai_learning!(sim, agent, m)
        end
        append!(all_matured, matured)
        # Solvency check deferred to end-of-round (see comment above Phase 1.5).
    end

    # Apply branch (sector) feedback from aggregated ROI of this round's
    # matured investments. Each sector's branch parameters drift toward the
    # realized mean ROI, creating a feedback loop where well-performing sectors
    # gradually produce better returns and poorly-performing sectors degrade.
    if !isempty(all_matured)
        sector_rois = Dict{String,Vector{Float64}}()
        for m in all_matured
            # "unknown" fallback (rather than a real sector) surfaces missing
            # metadata; the skip-if-unknown below avoids misattributing branch
            # feedback when the source record is incomplete.
            sector = String(get(m, "sector", "unknown"))
            sector == "unknown" && continue
            ret = Float64(get(m, "return_multiple", 1.0))
            roi = ret - 1.0  # ROI as fractional change
            push!(get!(sector_rois, sector, Float64[]), roi)
        end
        for (sector, rois) in sector_rois
            if !isempty(rois)
                apply_branch_feedback!(sim.market, sector, mean(rois))
            end
        end
    end

    # Get available opportunities (after applying costs).
    available_opportunities = get_available_opportunities(sim.market)

    # Refresh alive agents after survival checks
    alive_agents = filter(a -> a.alive, sim.agents)

    # Phase 3: AI level selection and action decisions.
    sequential_enabled = hasfield(typeof(sim.config), :SEQUENTIAL_DECISIONS_ENABLED) ?
        sim.config.SEQUENTIAL_DECISIONS_ENABLED : false

    agent_actions = Dict{String,Any}[]

    if sequential_enabled && length(alive_agents) > 1
        # Sequential decision making: early agents decide first, their choices become visible signals
        early_fraction = hasfield(typeof(sim.config), :EARLY_DECISION_FRACTION) ?
            sim.config.EARLY_DECISION_FRACTION : 0.3
        signal_weight = hasfield(typeof(sim.config), :SIGNAL_VISIBILITY_WEIGHT) ?
            sim.config.SIGNAL_VISIBILITY_WEIGHT : 0.15

        n_early = max(1, Int(floor(length(alive_agents) * early_fraction)))

        # Shuffle and split into early and late deciders
        shuffled_agents = shuffle(sim.rng, collect(alive_agents))
        early_agents = shuffled_agents[1:n_early]
        late_agents = shuffled_agents[n_early+1:end]

        # Phase 3a: Early agents decide (no visibility signals)
        early_signals = Dict{String,Int}()  # Track which opportunities early agents invested in
        manipulation_signals = Dict{String,Int}()  # Subset added by the post-hoc treatment

        for agent in early_agents
            # USE_NETWORK_EFFECTS=false leaves the neighbor list empty, which
            # downstream (collect_neighbor_signals) treats as "no peers visible".
            neighbor_agents = EmergentAgent[]
            if sim.config.USE_NETWORK_EFFECTS
                other_agents = filter(a -> a.id != agent.id && a.alive, alive_agents)
                if !isempty(other_agents)
                    n_neighbors = min(Int(sim.config.NETWORK_N_NEIGHBORS), length(other_agents))
                    neighbor_agents = rand(sim.rng, other_agents, n_neighbors)
                end
            end

            # Per-agent opportunity filter: use the AI-tier-aware visibility
            # set rather than the global pool, falling back to the global pool
            # if the filter returns empty.
            agent_opportunities = get_opportunities_for_agent(sim.market, agent)
            if isempty(agent_opportunities)
                agent_opportunities = available_opportunities
            end

            outcome = make_decision!(
                agent,
                agent_opportunities,
                market_conditions,
                sim.market,
                round;
                uncertainty_env=sim.uncertainty_env,
                neighbor_agents=neighbor_agents,
                innovation_engine=sim.innovation_engine,
                info_system=sim.info_system
            )

            push!(agent_actions, outcome)

            # Record visible signal for invest actions
            if get(outcome, "action", "") == "invest"
                opp_id = string(get(outcome, "opportunity_id", ""))
                if !isempty(opp_id)
                    early_signals[opp_id] = get(early_signals, opp_id, 0) + 1
                end
                _posthoc_apply_manipulation_signal!(
                    sim.config,
                    early_signals,
                    agent,
                    outcome,
                    agent_opportunities;
                    manipulation_signals=manipulation_signals,
                )
            end
            _annotate_sequential_signal_exposure!(
                outcome, early_signals, manipulation_signals, "early")
        end

        # Phase 3b: Late agents decide with visible signals
        for agent in late_agents
            # USE_NETWORK_EFFECTS=false leaves the neighbor list empty (no peers visible).
            neighbor_agents = EmergentAgent[]
            if sim.config.USE_NETWORK_EFFECTS
                other_agents = filter(a -> a.id != agent.id && a.alive, alive_agents)
                if !isempty(other_agents)
                    n_neighbors = min(Int(sim.config.NETWORK_N_NEIGHBORS), length(other_agents))
                    neighbor_agents = rand(sim.rng, other_agents, n_neighbors)
                end
            end

            # Per-agent opportunity filter: tier-aware visibility.
            agent_opportunities = get_opportunities_for_agent(sim.market, agent)
            if isempty(agent_opportunities)
                agent_opportunities = available_opportunities
            end

            outcome = make_decision!(
                agent,
                agent_opportunities,
                market_conditions,
                sim.market,
                round;
                uncertainty_env=sim.uncertainty_env,
                neighbor_agents=neighbor_agents,
                innovation_engine=sim.innovation_engine,
                info_system=sim.info_system,
                early_signals=early_signals,
                signal_weight=signal_weight
            )

            _annotate_sequential_signal_exposure!(
                outcome, early_signals, manipulation_signals, "late")
            push!(agent_actions, outcome)
        end
    else
        # Original simultaneous decision logic
        for agent in sim.agents
            if !agent.alive
                continue
            end

            # Collect neighbor agents for social influence.
            # USE_NETWORK_EFFECTS=false leaves the neighbor list empty (no peers visible).
            neighbor_agents = EmergentAgent[]
            if sim.config.USE_NETWORK_EFFECTS && length(alive_agents) > 1
                other_agents = filter(a -> a.id != agent.id && a.alive, alive_agents)
                if !isempty(other_agents)
                    n_neighbors = min(Int(sim.config.NETWORK_N_NEIGHBORS), length(other_agents))
                    neighbor_agents = rand(sim.rng, other_agents, n_neighbors)
                end
            end

            # Per-agent opportunity filter: tier-aware visibility.
            agent_opportunities = get_opportunities_for_agent(sim.market, agent)
            if isempty(agent_opportunities)
                agent_opportunities = available_opportunities
            end

            # Use make_decision! which properly integrates AI level effects
            outcome = make_decision!(
                agent,
                agent_opportunities,
                market_conditions,
                sim.market,
                round;
                uncertainty_env=sim.uncertainty_env,
                neighbor_agents=neighbor_agents,
                innovation_engine=sim.innovation_engine,
                info_system=sim.info_system
            )

            push!(agent_actions, outcome)
        end
    end

    # Update tier beliefs from immediate INNOVATE outcomes only, using the
    # realized cash multiple (cash_multiple, or returned/rd_spend through the
    # compatibility path). Explore is deliberately excluded: discovery is a
    # tier-independent draw (DISCOVERY_PROBABILITY) whose Boolean success
    # carries no information about tier effectiveness — converting it to a
    # 1.5/0.5 pseudo-multiple would inject Bernoulli(p) noise into whichever
    # tier the explorer happened to hold.
    for action in agent_actions
        agent_id = get(action, "agent_id", 0)
        if agent_id < 1 || agent_id > length(sim.agents)
            continue
        end
        agent = sim.agents[agent_id]
        if !agent.alive
            continue
        end
        action_type = get(action, "action", "maintain")
        if action_type == "innovate"
            ai_tier = action_behavior_ai_level(action)
            return_mult = if haskey(action, "cash_multiple")
                Float64(action["cash_multiple"])
            else
                rd_spend = Float64(get(action, "rd_spend", 0.0))
                returned = Float64(get(action, "innovation_return",
                                       get(action, "recovery", 0.0)))
                rd_spend > 0 ? returned / rd_spend :
                    (get(action, "success", false) ? 1.5 : 0.5)
            end
            update_tier_belief!(agent.ai_learning, ai_tier, return_mult)

            # Record immediate-action confidence calibration. make_decision!
            # stores decision_confidence inside outcome["perception"], not at
            # the top level of the action dict.
            decision_conf = _action_decision_confidence(action)
            record_confidence_outcome_observation!(agent, decision_conf, return_mult;
                                                   ai_used=counts_as_ai_use(sim.config, String(ai_tier)),
                                                   channel=action_type)
            update_state_from_outcome!(agent, action; ai_was_accurate=nothing)
        elseif action_type == "explore"
            # Trait/experience evolution still applies; only the tier-belief
            # and pseudo-multiple confidence channels are gated off.
            update_state_from_outcome!(agent, action; ai_was_accurate=nothing)
        end
    end

    # End-of-round knowledge maintenance: cull rarely-used knowledge and prune
    # sectors below the strength threshold. Bounds knowledge-portfolio growth
    # across long runs.
    max_knowledge = hasproperty(sim.config, :MAX_AGENT_KNOWLEDGE) ?
        sim.config.MAX_AGENT_KNOWLEDGE : nothing
    prune_threshold = hasproperty(sim.config, :SECTOR_STRENGTH_PRUNE_THRESHOLD) ?
        sim.config.SECTOR_STRENGTH_PRUNE_THRESHOLD : 0.05
    for agent in sim.agents
        agent.alive || continue
        forget_stale_knowledge!(sim.knowledge_base, agent, round;
                                max_size=max_knowledge, drop_fraction=0.1)
        prune_by_sector_strength!(sim.knowledge_base, agent,
                                  agent.resources.knowledge;
                                  threshold=prune_threshold)
    end

    # Phase 4: Final survival check for all agents.
    for agent in sim.agents
        check_survival!(agent, round; market=sim.market)
    end

    # Phase 5: Update market.
    # Build Innovation objects from the real innovation-outcome fields stored
    # by attempt_innovation! on the action dict. Fabricating innovations with
    # random novelty/quality here would discard the agent's actual outcome, so
    # all downstream effects (opportunity spawning) would
    # see random values regardless of which tier created them.
    innovations = Innovation[]
    for action in agent_actions
        if get(action, "action", "") == "innovate" && get(action, "success", false)
            innov_id = string(get(action, "innovation_id",
                generate_innovation_id(round, get(action, "agent_id", 0), 0)))
            innov_type = String(get(action, "innovation_type", "incremental"))
            knowledge_components = let kc = get(action, "knowledge_components", String[])
                kc isa Vector{String} ? kc : String[string(x) for x in kc]
            end
            novelty = Float64(get(action, "innovation_novelty", 0.5))
            quality = Float64(get(action, "innovation_quality", 0.5))
            ai_assisted = Bool(get(action, "ai_assisted", false))
            ai_domains_used = let dd = get(action, "ai_domains_used", String[])
                dd isa Vector{String} ? dd : String[string(x) for x in dd]
            end
            scarcity_v = get(action, "innovation_scarcity", nothing)
            impact_v = get(action, "market_impact", nothing)
            sector_v = get(action, "innovation_sector", nothing)
            combo_sig_v = get(action, "combination_signature", nothing)
            success_v = get(action, "success", nothing)

            innov = Innovation(
                id=innov_id,
                type=innov_type,
                knowledge_components=knowledge_components,
                novelty=novelty,
                quality=quality,
                round_created=round,
                creator_id=get(action, "agent_id", 0),
                ai_level_used=String(get(action, "ai_behavior_level", action_behavior_ai_level(action))),
                ai_assisted=ai_assisted,
                ai_domains_used=ai_domains_used,
                sector=isnothing(sector_v) ? nothing : String(sector_v),
                combination_signature=isnothing(combo_sig_v) ? nothing : String(combo_sig_v),
                cash_multiple=Float64(get(action, "cash_multiple", 1.5)),
                market_impact=isnothing(impact_v) ? nothing : Float64(impact_v),
                success=isnothing(success_v) ? nothing : Bool(success_v),
                scarcity=isnothing(scarcity_v) ? nothing : Float64(scarcity_v),
                is_new_combination=Bool(get(action, "is_new_combination", false)),
            )
            push!(innovations, innov)
            cash_multiple = Float64(get(action, "cash_multiple", 1.5))
            new_opp = spawn_opportunity_from_innovation!(sim.market, innov, cash_multiple)

            # Record the innovation outcome on the spawned opp and on the
            # combination tracker. Side-effects: the spawned opp.market_impact
            # reflects the innovation's cash_multiple, and combination_tracker
            # accumulates success scores per knowledge-combination signature.
            return_achieved = Bool(get(action, "success", false)) ? cash_multiple : 0.0
            if !isnothing(new_opp) && new_opp isa Opportunity
                record_innovation_outcome!(sim.market, new_opp,
                                           Bool(get(action, "success", false)),
                                           return_achieved)
            end
            sig = innov.combination_signature
            if !isnothing(sig)
                record_outcome!(sim.innovation_engine.combination_tracker,
                                String(sig),
                                Bool(get(action, "success", false)) ? 1.0 : 0.0)
            end
        end
    end

    # Create niche opportunities from exploration discoveries.
    for action in agent_actions
        if get(action, "action", "") == "explore" && get(action, "exploration_type", "") == "niche_discovery"
            _create_niche_opportunities_from_action!(sim, action, round)
        end
    end

    market_state = step!(sim.market, round, agent_actions, innovations; matured_outcomes=all_matured)

    # Phase 5.5: Market clearing + dynamics (sector demand/supply imbalance).
    # update_clearing_metrics! populates market.sector_clearing_index, which
    # realized_return consults for the demand-shortfall multiplier.
    # update_market_dynamics! advances market_momentum and the boom-streak
    # counter.
    update_clearing_metrics!(sim.market, agent_actions)
    invest_actions = filter(a -> get(a, "action", "") == "invest", agent_actions)
    # Actions emit the invested amount under "amount"; fall back to
    # "investment_amount" for matured-outcome records that use that key.
    total_investment_for_dynamics = sum(
        Float64(get(a, "amount", get(a, "investment_amount", 0.0))) for a in invest_actions;
        init=0.0
    )
    n_ai_invest = count(action_counts_as_ai_use, invest_actions)
    ai_invest_share = isempty(invest_actions) ? 0.0 : n_ai_invest / length(invest_actions)
    update_market_dynamics!(sim.market, agent_actions, total_investment_for_dynamics,
                            ai_invest_share)

    # Phase 5.6: Opportunity lifecycle management.
    # Build per-round opportunity demand from agent invest actions and call
    # manage_opportunities! to age opportunities, decay competition (× 0.9),
    # and remove inactive opportunities. Per-round decay keeps crowding pressure
    # from accumulating monotonically across rounds.
    opportunity_demand = Dict{String,Int}()
    total_investment = 0.0
    for action in agent_actions
        if get(action, "action", "") == "invest"
            opp_id = string(get(action, "opportunity_id", ""))
            if !isempty(opp_id)
                opportunity_demand[opp_id] = get(opportunity_demand, opp_id, 0) + 1
                # Actions emit invested capital under "amount".
                total_investment += Float64(get(action, "amount", get(action, "investment_amount", 0.0)))
            end
        end
    end
    manage_opportunities!(sim.market, round, opportunity_demand, total_investment)
    # `manage_opportunities!` snapshots the live market before culling. Refresh
    # the full realizable cohort as well so culled-but-outstanding opportunities
    # carry a genuine one-round lag instead of freezing at their cull-round load.
    _snapshot_performative_commitments!(sim)

    # Phase 6: Update uncertainty measurements
    # Keep the environment's alive-population count current so the recursion
    # dimension's population scaling tracks the live population as agents exit.
    sim.uncertainty_env._last_alive_agents = count(a -> a.alive, sim.agents)
    record_ai_signals!(sim.uncertainty_env, round, agent_actions)
    uncertainty_state = measure_uncertainty_state!(
        sim.uncertainty_env,
        sim.market,
        agent_actions,
        innovations,
        round
    )

    # Phase 7: Record history
    round_stats = compile_round_stats(sim, round, agent_actions, all_matured, uncertainty_state)
    push!(sim.history, round_stats)

    return round_stats
end

"""
Observation-weighted aggregate of per-tier emergent uncertainty metrics into a
single population-level dict. Tiers with zero observations contribute nothing;
with a single populated tier (fixed-tier runs) this reduces to that tier's
values exactly.
"""
function _population_emergent_aggregate(by_tier::Dict)::Dict{String,Float64}
    defaults = Dict{String,Float64}(
        "actor_ignorance" => 0.5,
        "practical_indeterminism" => 0.5,
        "agentic_novelty" => 0.5,
        "competitive_recursion" => 0.0,
    )
    out = Dict{String,Float64}()
    for (dim, default) in defaults
        weighted_sum = 0.0
        total_obs = 0.0
        for stats in values(by_tier)
            obs = Float64(get(stats, "$(dim)_observations", 0.0))
            obs > 0.0 || continue
            weighted_sum += obs * Float64(get(stats, dim, default))
            total_obs += obs
        end
        out[dim] = total_obs > 0.0 ? weighted_sum / total_obs : default
        out["$(dim)_observations"] = total_obs
    end
    return out
end

"""
Compile statistics for a round.
"""
function compile_round_stats(
    sim::EmergentSimulation,
    round::Int,
    agent_actions::Vector{Dict{String,Any}},
    matured_outcomes::Vector{Dict{String,Any}},
    uncertainty_state::Dict{String,Dict{String,Any}}
)::Dict{String,Any}
    alive_agents = [a for a in sim.agents if a.alive]
    n_alive = length(alive_agents)
    n_total = length(sim.agents)

    # Capital statistics
    capitals = [get_capital(a) for a in alive_agents]
    mean_capital = isempty(capitals) ? 0.0 : mean(capitals)
    std_capital = isempty(capitals) || length(capitals) < 2 ? 0.0 : std(capitals)
    median_capital = isempty(capitals) ? 0.0 : median(capitals)

    # Action counts and capital tracking by action type
    action_counts = Dict{String,Int}()
    ai_usage = Dict{String,Int}("none" => 0, "basic" => 0, "advanced" => 0, "premium" => 0)

    # Track capital deployed/returned by action type for ROIC calculation
    capital_deployed = Dict{String,Float64}("invest" => 0.0, "innovate" => 0.0, "explore" => 0.0)
    capital_returned = Dict{String,Float64}("invest" => 0.0, "innovate" => 0.0, "explore" => 0.0)
    ai_analysis_cost_total = 0.0
    ai_decision_analysis_cost_total = 0.0
    ai_action_analysis_cost_total = 0.0
    ai_analysis_cost_by_tier = Dict{String,Float64}(
        "none" => 0.0,
        "basic" => 0.0,
        "advanced" => 0.0,
        "premium" => 0.0,
    )

    # Track opportunity IDs for HHI calculation
    opportunity_ids = String[]
    invest_confidences = Float64[]
    action_probability_entropies = Float64[]
    visible_opportunity_counts = Float64[]
    utility_floor_hits = 0
    utility_ceiling_hits = 0
    utility_value_count = 0
    # A1 open-action pivot telemetry: per-decision counts attached by make_decision!
    # only when ENABLE_PIVOT is on (defaults below keep disabled runs at 0).
    pivot_count_total = 0
    pivot_recovered_total = 0.0
    pivot_committed_total = 0.0
    roi_clamp_hits = 0
    experience_stats_count = 0
    raw_mean_roi_values = Float64[]
    return_evidence_values = Float64[]
    confidence_saturation_hits = 0
    confidence_diagnostics_count = 0
    visible_exposures_by_tier = Dict{String,Vector{String}}(
        "none" => String[],
        "basic" => String[],
        "advanced" => String[],
        "premium" => String[],
    )
    visible_opportunity_counts_by_tier = Dict{String,Vector{Float64}}(
        "none" => Float64[],
        "basic" => Float64[],
        "advanced" => Float64[],
        "premium" => Float64[],
    )
    info_quality_values_by_tier = Dict{String,Vector{Float64}}(
        "none" => Float64[],
        "basic" => Float64[],
        "advanced" => Float64[],
        "premium" => Float64[],
    )
    perception_telemetry_by_tier = Dict{String,Dict{String,Vector{Float64}}}(
        tier => Dict{String,Vector{Float64}}(
            "perceived_$(dim)_$(field)" => Float64[]
            for dim in PERCEPTION_TELEMETRY_DIMENSIONS
            for field in PERCEPTION_TELEMETRY_FIELDS
        )
        for tier in ("none", "basic", "advanced", "premium")
    )

    for action in agent_actions
        act_type = get(action, "action", "maintain")
        action_counts[act_type] = get(action_counts, act_type, 0) + 1

        ai_level = action_behavior_ai_level(action)
        if haskey(ai_usage, ai_level)
            ai_usage[ai_level] += 1
        end

        decision_ai_cost = Float64(get(action, "ai_decision_analysis_cost", 0.0))
        action_ai_cost = Float64(get(action, "ai_action_analysis_cost", get(action, "ai_analysis_cost", 0.0)))
        total_ai_cost = Float64(get(action, "ai_total_analysis_cost", decision_ai_cost + action_ai_cost))
        ai_analysis_cost_total += total_ai_cost
        ai_decision_analysis_cost_total += decision_ai_cost
        ai_action_analysis_cost_total += action_ai_cost
        if haskey(ai_analysis_cost_by_tier, ai_level)
            ai_analysis_cost_by_tier[ai_level] += total_ai_cost
        end

        probs = get(action, "action_probabilities", nothing)
        if probs isa Dict
            prob_values = Float64[]
            for v in values(probs)
                if v isa Number
                    push!(prob_values, Float64(v))
                end
            end
            if !isempty(prob_values)
                denom = log(max(2, length(prob_values)))
                entropy = -sum(p > 0 ? p * log(p) : 0.0 for p in prob_values) / denom
                push!(action_probability_entropies, entropy)
            end
        end

        pivot_count_total += Int(get(action, "pivot_count", 0))
        pivot_recovered_total += Float64(get(action, "pivot_recovered", 0.0))
        pivot_committed_total += Float64(get(action, "pivot_committed", 0.0))

        utils = get(action, "utilities", nothing)
        if utils isa Dict
            for v in values(utils)
                v isa Number || continue
                u = Float64(v)
                utility_floor_hits += u <= 1e-9 ? 1 : 0
                utility_ceiling_hits += u >= 1.0 - 1e-9 ? 1 : 0
                utility_value_count += 1
            end
        end

        n_visible = get(action, "n_visible_opportunities", nothing)
        if n_visible isa Number
            push!(visible_opportunity_counts, Float64(n_visible))
            if haskey(visible_opportunity_counts_by_tier, ai_level)
                push!(visible_opportunity_counts_by_tier[ai_level], Float64(n_visible))
            end
        end
        visible_ids = get(action, "visible_opportunity_ids", nothing)
        if visible_ids isa AbstractVector && haskey(visible_exposures_by_tier, ai_level)
            append!(visible_exposures_by_tier[ai_level], string.(visible_ids))
        end
        info_quality = get(action, "info_quality_used", nothing)
        if info_quality isa Number && haskey(info_quality_values_by_tier, ai_level)
            push!(info_quality_values_by_tier[ai_level], Float64(info_quality))
        end

        experience_stats = get(action, "experience_stats", nothing)
        if experience_stats isa Dict
            experience_stats_count += 1
            roi_clamp_hits += Bool(get(experience_stats, "roi_clamp_hit", false)) ? 1 : 0
            raw_roi = get(experience_stats, "raw_mean_roi", nothing)
            if raw_roi isa Number
                push!(raw_mean_roi_values, Float64(raw_roi))
            end
            evidence = get(experience_stats, "return_evidence", nothing)
            if evidence isa Number
                push!(return_evidence_values, Float64(evidence))
            end
        end

        confidence_diagnostics = get(action, "confidence_diagnostics", nothing)
        if confidence_diagnostics isa Dict
            confidence_diagnostics_count += 1
            confidence_saturation_hits += Float64(get(confidence_diagnostics, "confidence_saturation_hit", 0.0)) > 0.0 ? 1 : 0
            if haskey(perception_telemetry_by_tier, ai_level)
                tier_telemetry = perception_telemetry_by_tier[ai_level]
                for dim in PERCEPTION_TELEMETRY_DIMENSIONS
                    for field in PERCEPTION_TELEMETRY_FIELDS
                        src_key = "perception_$(dim)_$(field)"
                        dest_key = "perceived_$(dim)_$(field)"
                        value = get(confidence_diagnostics, src_key, nothing)
                        if value isa Number && haskey(tier_telemetry, dest_key)
                            push!(tier_telemetry[dest_key], Float64(value))
                        end
                    end
                end
                for (src_key, value) in confidence_diagnostics
                    src_key isa AbstractString || continue
                    startswith(src_key, "perception_component_") || continue
                    value isa Number || continue
                    if !haskey(tier_telemetry, src_key)
                        tier_telemetry[src_key] = Float64[]
                    end
                    push!(tier_telemetry[src_key], Float64(value))
                end
            end
        end

        # Track capital deployed by action type
        if act_type == "invest"
            amount = Float64(get(action, "amount", 0.0))
            capital_deployed["invest"] += amount
            opp_id = get(action, "opportunity_id", nothing)
            if !isnothing(opp_id)
                push!(opportunity_ids, string(opp_id))
            end
            # Track decision confidence for invest actions
            perception = get(action, "perception", Dict{String,Any}())
            conf = Float64(get(perception, "decision_confidence", 0.5))
            push!(invest_confidences, conf)
        elseif act_type == "innovate"
            rd_spend = Float64(get(action, "rd_spend", 0.0))
            capital_deployed["innovate"] += rd_spend
            if get(action, "success", false)
                ret = Float64(get(action, "innovation_return", 0.0))
                capital_returned["innovate"] += ret
            else
                rec = Float64(get(action, "recovery", 0.0))
                capital_returned["innovate"] += rec
            end
        elseif act_type == "explore"
            # Actions emit the spent amount under "explore_cost".
            cost = Float64(get(action, "explore_cost", 0.0))
            capital_deployed["explore"] += cost
            # Exploration doesn't have direct capital return
        end
    end

    # Add matured investment returns to capital_returned["invest"]
    for outcome in matured_outcomes
        ret = Float64(get(outcome, "capital_returned", 0.0))
        capital_returned["invest"] += ret
    end

    # Pivot recoveries are realized invest-channel returns: a pivot resolves an
    # investment at the haircut value, so they enter capital_returned["invest"]
    # alongside maturity payouts. The pivot_* telemetry columns below still
    # report the pivot channel separately.
    capital_returned["invest"] += pivot_recovered_total

    # ROIC by action type. NB this is a *round-level* PnL ratio: the numerator
    # is THIS round's matured returns (from investments deployed several rounds
    # ago), while the denominator is THIS round's new deploys — the two windows
    # do not match. When deployed=0 but returned>0 the ratio is undefined, so
    # it is emitted as NaN (not 0.0) to distinguish "no economic activity" from
    # "returns happened but no new deploys to ratio against." Consumers needing
    # a true cumulative ROIC should accumulate total_capital_deployed_* /
    # total_capital_returned_* across rounds; net_capital_flow_invest gives the
    # current round's signed delta.
    mean_roic_invest = capital_deployed["invest"] > 0 ?
        (capital_returned["invest"] - capital_deployed["invest"]) / capital_deployed["invest"] :
        (capital_returned["invest"] > 0 ? NaN : 0.0)
    mean_roic_innovate = capital_deployed["innovate"] > 0 ?
        (capital_returned["innovate"] - capital_deployed["innovate"]) / capital_deployed["innovate"] :
        (capital_returned["innovate"] > 0 ? NaN : 0.0)
    mean_roic_explore = 0.0  # Explore doesn't have direct return

    # Net capital flow by action type
    net_capital_flow_invest = capital_returned["invest"] - capital_deployed["invest"]
    net_capital_flow_innovate = capital_returned["innovate"] - capital_deployed["innovate"]

    # Calculate HHI (Herfindahl-Hirschman Index) for investment concentration
    overall_hhi = 0.0
    if !isempty(opportunity_ids)
        opp_counts = Dict{String,Int}()
        for opp_id in opportunity_ids
            opp_counts[opp_id] = get(opp_counts, opp_id, 0) + 1
        end
        total_invests = length(opportunity_ids)
        overall_hhi = sum((count / total_invests)^2 for count in values(opp_counts))
    end

    # Empty per-tier cells (no agents of that tier this round) are NaN, not
    # 0.0: an HHI/mean of zero observations is undefined, and a 0.0 sentinel
    # would deflate per-run means. Suite/analysis consumers aggregate these
    # columns through finite-value filters (e.g. finite_mean), so NaN cells
    # drop out instead of biasing toward zero.
    function exposure_hhi(ids::Vector{String})::Float64
        isempty(ids) && return NaN
        counts = Dict{String,Int}()
        for id in ids
            counts[id] = get(counts, id, 0) + 1
        end
        total = length(ids)
        return sum((count / total)^2 for count in values(counts))
    end

    mean_action_probability_entropy = isempty(action_probability_entropies) ? 0.0 : mean(action_probability_entropies)
    mean_visible_opportunities = isempty(visible_opportunity_counts) ? 0.0 : mean(visible_opportunity_counts)
    utility_floor_rate = utility_value_count > 0 ? utility_floor_hits / utility_value_count : 0.0
    utility_ceiling_rate = utility_value_count > 0 ? utility_ceiling_hits / utility_value_count : 0.0
    roi_clamp_hit_rate = experience_stats_count > 0 ? roi_clamp_hits / experience_stats_count : 0.0
    mean_raw_mean_roi = isempty(raw_mean_roi_values) ? 0.0 : mean(raw_mean_roi_values)
    mean_return_evidence = isempty(return_evidence_values) ? 0.0 : mean(return_evidence_values)
    confidence_saturation_rate = confidence_diagnostics_count > 0 ? confidence_saturation_hits / confidence_diagnostics_count : 0.0
    visible_hhi_none = exposure_hhi(visible_exposures_by_tier["none"])
    visible_hhi_basic = exposure_hhi(visible_exposures_by_tier["basic"])
    visible_hhi_advanced = exposure_hhi(visible_exposures_by_tier["advanced"])
    visible_hhi_premium = exposure_hhi(visible_exposures_by_tier["premium"])
    # NaN (not 0.0) for empty per-tier cells — see exposure_hhi comment above.
    mean_visible_by_tier = Dict(
        tier => isempty(vals) ? NaN : mean(vals)
        for (tier, vals) in visible_opportunity_counts_by_tier
    )
    mean_info_quality_by_tier = Dict(
        tier => isempty(vals) ? NaN : mean(vals)
        for (tier, vals) in info_quality_values_by_tier
    )
    mean_perception_telemetry_by_tier = Dict{String,Dict{String,Float64}}()
    for (tier, telemetry) in perception_telemetry_by_tier
        mean_perception_telemetry_by_tier[tier] = Dict(
            key => isempty(vals) ? NaN : mean(vals)
            for (key, vals) in telemetry
        )
    end

    # Action shares
    total_actions = sum(values(action_counts))
    action_shares = Dict(
        "invest" => total_actions > 0 ? get(action_counts, "invest", 0) / total_actions : 0.0,
        "innovate" => total_actions > 0 ? get(action_counts, "innovate", 0) / total_actions : 0.0,
        "explore" => total_actions > 0 ? get(action_counts, "explore", 0) / total_actions : 0.0,
        "maintain" => total_actions > 0 ? get(action_counts, "maintain", 0) / total_actions : 0.0
    )

    # AI tier shares
    total_ai = sum(values(ai_usage))
    ai_shares = Dict(
        "none" => total_ai > 0 ? ai_usage["none"] / total_ai : 0.0,
        "basic" => total_ai > 0 ? ai_usage["basic"] / total_ai : 0.0,
        "advanced" => total_ai > 0 ? ai_usage["advanced"] / total_ai : 0.0,
        "premium" => total_ai > 0 ? ai_usage["premium"] / total_ai : 0.0
    )

    # Innovation stats
    innovation_attempts = get(action_counts, "innovate", 0)
    innovation_successes = count(a -> get(a, "action", "") == "innovate" && get(a, "success", false), agent_actions)
    innovation_success_rate = innovation_attempts > 0 ? innovation_successes / innovation_attempts : 0.0

    # Mean confidence for invest actions
    mean_confidence_invest = isempty(invest_confidences) ? 0.0 : mean(invest_confidences)

    # Matured investment stats
    n_matured = length(matured_outcomes)
    n_success = count(o -> get(o, "success", false), matured_outcomes)
    n_failure = n_matured - n_success
    function matured_mean(key::String)::Float64
        values = Float64[]
        for outcome in matured_outcomes
            value = get(outcome, key, nothing)
            value isa Number || continue
            numeric = Float64(value)
            isfinite(numeric) && push!(values, numeric)
        end
        return isempty(values) ? NaN : mean(values)
    end
    function round_flow_count(event_type::String; source::Union{String,Nothing}=nothing)::Int
        return count(sim.market.opportunity_flow_events) do event
            event.round == round && event.event_type == event_type &&
                (isnothing(source) || event.source == source)
        end
    end

    # Uncertainty levels (formula-based, environment level).
    actor_ignorance = Float64(get(get(uncertainty_state, "actor_ignorance", Dict()), "level", 0.0))
    practical_indet = Float64(get(get(uncertainty_state, "practical_indeterminism", Dict()), "level", 0.0))
    agentic_novelty = Float64(get(get(uncertainty_state, "agentic_novelty", Dict()), "level", 0.0))
    competitive_rec = Float64(get(get(uncertainty_state, "competitive_recursion", Dict()), "level", 0.0))
    practical_state = get(uncertainty_state, "practical_indeterminism", Dict())
    competitive_state = get(uncertainty_state, "competitive_recursion", Dict())
    market_crowding_pressure = Float64(get(practical_state, "crowding_pressure", 0.0))
    market_opportunity_overlap = Float64(get(practical_state, "opportunity_overlap", 0.0))
    market_investment_concentration = Float64(get(practical_state, "investment_concentration", 0.0))
    # Opportunity-level competition/saturation exposure (investment-weighted mean
    # over opportunities holding capital). Crowding is modeled at the
    # opportunity level here.
    market_opportunity_competition = Float64(get(practical_state, "opportunity_competition", 0.0))
    market_ai_herding_intensity = Float64(get(competitive_state, "ai_herding_intensity",
        get(practical_state, "ai_herding_intensity", 0.0)))
    market_ai_action_correlation = Float64(get(competitive_state, "ai_action_correlation", 0.0))
    market_combo_reuse_pressure = Float64(get(competitive_state, "combo_reuse_pressure", 0.0))
    knightian_component_telemetry = _flatten_knightian_component_telemetry(uncertainty_state)

    # EMERGENT uncertainty (agent-level, computed from actual outcomes)
    # These metrics emerge from what actually happens to agents, not from formulas
    all_emergent_by_tier = aggregate_emergent_uncertainty_by_tier(
        sim.agents;
        include_dead=true,
    )
    survivor_emergent_by_tier = aggregate_emergent_uncertainty_by_tier(
        sim.agents;
        include_dead=false,
    )

    # Observation-weighted population aggregate across tiers. In fixed
    # single-tier runs only one tier has observations; in emergent/mixed-
    # distribution runs this yields a population-level series. Per-tier emergent
    # metrics for mixed runs are recomputed by the analysis scripts directly
    # from agents.
    tier_emergent = _population_emergent_aggregate(all_emergent_by_tier)
    tier_survivor_emergent = _population_emergent_aggregate(survivor_emergent_by_tier)

    emergent_actor_ignorance = Float64(get(tier_emergent, "actor_ignorance", 0.5))
    emergent_practical_indet = Float64(get(tier_emergent, "practical_indeterminism", 0.5))
    emergent_agentic_novelty = Float64(get(tier_emergent, "agentic_novelty", 0.5))
    emergent_competitive_rec = Float64(get(tier_emergent, "competitive_recursion", 0.0))
    survivor_emergent_actor_ignorance = Float64(get(tier_survivor_emergent, "actor_ignorance", emergent_actor_ignorance))
    survivor_emergent_practical_indet = Float64(get(tier_survivor_emergent, "practical_indeterminism", emergent_practical_indet))
    survivor_emergent_agentic_novelty = Float64(get(tier_survivor_emergent, "agentic_novelty", emergent_agentic_novelty))
    survivor_emergent_competitive_rec = Float64(get(tier_survivor_emergent, "competitive_recursion", emergent_competitive_rec))

    # --- Uncertainty Transformation Metrics ---
    # Store baseline on first round (or first round with uncertainty data)
    if isempty(sim.baseline_uncertainty_levels)
        sim.baseline_uncertainty_levels["actor_ignorance"] = actor_ignorance
        sim.baseline_uncertainty_levels["practical_indeterminism"] = practical_indet
        sim.baseline_uncertainty_levels["agentic_novelty"] = agentic_novelty
        sim.baseline_uncertainty_levels["competitive_recursion"] = competitive_rec
    end

    # Get previous levels (default to current if first round)
    prev_actor = get(sim.previous_uncertainty_levels, "actor_ignorance", actor_ignorance)
    prev_practical = get(sim.previous_uncertainty_levels, "practical_indeterminism", practical_indet)
    prev_agentic = get(sim.previous_uncertainty_levels, "agentic_novelty", agentic_novelty)
    prev_competitive = get(sim.previous_uncertainty_levels, "competitive_recursion", competitive_rec)

    # Get baseline levels
    base_actor = get(sim.baseline_uncertainty_levels, "actor_ignorance", actor_ignorance)
    base_practical = get(sim.baseline_uncertainty_levels, "practical_indeterminism", practical_indet)
    base_agentic = get(sim.baseline_uncertainty_levels, "agentic_novelty", agentic_novelty)
    base_competitive = get(sim.baseline_uncertainty_levels, "competitive_recursion", competitive_rec)

    # Compute delta (round-over-round change)
    delta_actor = actor_ignorance - prev_actor
    delta_practical = practical_indet - prev_practical
    delta_agentic = agentic_novelty - prev_agentic
    delta_competitive = competitive_rec - prev_competitive

    # Compute cumulative delta (change from baseline)
    cumulative_delta_actor = actor_ignorance - base_actor
    cumulative_delta_practical = practical_indet - base_practical
    cumulative_delta_agentic = agentic_novelty - base_agentic
    cumulative_delta_competitive = competitive_rec - base_competitive

    # Compute portfolio composition (shares)
    uncertainty_total = actor_ignorance + practical_indet + agentic_novelty + competitive_rec
    total_safe = max(uncertainty_total, 0.001)  # Avoid division by zero
    share_actor = actor_ignorance / total_safe
    share_practical = practical_indet / total_safe
    share_agentic = agentic_novelty / total_safe
    share_competitive = competitive_rec / total_safe

    # Compute HHI (Herfindahl-Hirschman Index) - concentration measure
    uncertainty_hhi = share_actor^2 + share_practical^2 + share_agentic^2 + share_competitive^2

    # Compute entropy - diversity measure (avoid log(0))
    eps = 1e-10
    uncertainty_entropy = -(
        share_actor * log(share_actor + eps) +
        share_practical * log(share_practical + eps) +
        share_agentic * log(share_agentic + eps) +
        share_competitive * log(share_competitive + eps)
    )

    # Update previous levels for next round
    sim.previous_uncertainty_levels["actor_ignorance"] = actor_ignorance
    sim.previous_uncertainty_levels["practical_indeterminism"] = practical_indet
    sim.previous_uncertainty_levels["agentic_novelty"] = agentic_novelty
    sim.previous_uncertainty_levels["competitive_recursion"] = competitive_rec

    # Mean AI trust
    trust_values = [Float64(get(a.traits, "ai_trust", 0.5)) for a in alive_agents]
    mean_trust = isempty(trust_values) ? 0.5 : mean(trust_values)
    std_trust = length(trust_values) < 2 ? 0.0 : std(trust_values)

    return Dict{String,Any}(
        "round" => round,
        "n_alive" => n_alive,
        "n_total" => n_total,
        "survival_rate" => n_total > 0 ? n_alive / n_total : 0.0,
        "mean_capital" => mean_capital,
        "median_capital" => median_capital,
        "std_capital" => std_capital,
        "total_capital" => sum(capitals),
        # Action counts
        "invest_count" => get(action_counts, "invest", 0),
        "innovate_count" => get(action_counts, "innovate", 0),
        "explore_count" => get(action_counts, "explore", 0),
        "maintain_count" => get(action_counts, "maintain", 0),
        # Action shares
        "action_share_invest" => action_shares["invest"],
        "action_share_innovate" => action_shares["innovate"],
        "action_share_explore" => action_shares["explore"],
        "action_share_maintain" => action_shares["maintain"],
        # AI tier counts and shares
        "ai_none_count" => ai_usage["none"],
        "ai_basic_count" => ai_usage["basic"],
        "ai_advanced_count" => ai_usage["advanced"],
        "ai_premium_count" => ai_usage["premium"],
        "ai_share_none" => ai_shares["none"],
        "ai_share_basic" => ai_shares["basic"],
        "ai_share_advanced" => ai_shares["advanced"],
        "ai_share_premium" => ai_shares["premium"],
        # Capital deployed/returned by action type
        "total_capital_deployed" => sum(values(capital_deployed)),
        "total_capital_returned" => sum(values(capital_returned)),
        "total_capital_deployed_invest" => capital_deployed["invest"],
        "total_capital_deployed_innovate" => capital_deployed["innovate"],
        "total_capital_deployed_explore" => capital_deployed["explore"],
        "total_ai_analysis_cost" => ai_analysis_cost_total,
        "total_ai_decision_analysis_cost" => ai_decision_analysis_cost_total,
        "total_ai_action_analysis_cost" => ai_action_analysis_cost_total,
        "total_ai_analysis_cost_none" => ai_analysis_cost_by_tier["none"],
        "total_ai_analysis_cost_basic" => ai_analysis_cost_by_tier["basic"],
        "total_ai_analysis_cost_advanced" => ai_analysis_cost_by_tier["advanced"],
        "total_ai_analysis_cost_premium" => ai_analysis_cost_by_tier["premium"],
        "total_capital_returned_invest" => capital_returned["invest"],
        "total_capital_returned_innovate" => capital_returned["innovate"],
        "total_capital_returned_explore" => capital_returned["explore"],
        "net_capital_flow_invest" => net_capital_flow_invest,
        "net_capital_flow_innovate" => net_capital_flow_innovate,
        # ROIC by action type
        "mean_roic_invest" => mean_roic_invest,
        "mean_roic_innovate" => mean_roic_innovate,
        "mean_roic_explore" => mean_roic_explore,
        # HHI and sector metrics
        "overall_hhi" => overall_hhi,
        "mean_action_probability_entropy" => mean_action_probability_entropy,
        "mean_visible_opportunities" => mean_visible_opportunities,
        "utility_floor_rate" => utility_floor_rate,
        "utility_ceiling_rate" => utility_ceiling_rate,
        "roi_clamp_hit_rate" => roi_clamp_hit_rate,
        "mean_raw_mean_roi" => mean_raw_mean_roi,
        "mean_return_evidence" => mean_return_evidence,
        "confidence_saturation_rate" => confidence_saturation_rate,
        "visible_hhi_none" => visible_hhi_none,
        "visible_hhi_basic" => visible_hhi_basic,
        "visible_hhi_advanced" => visible_hhi_advanced,
        "visible_hhi_premium" => visible_hhi_premium,
        "mean_visible_opportunities_none" => mean_visible_by_tier["none"],
        "mean_visible_opportunities_basic" => mean_visible_by_tier["basic"],
        "mean_visible_opportunities_advanced" => mean_visible_by_tier["advanced"],
        "mean_visible_opportunities_premium" => mean_visible_by_tier["premium"],
        "mean_info_quality_used_none" => mean_info_quality_by_tier["none"],
        "mean_info_quality_used_basic" => mean_info_quality_by_tier["basic"],
        "mean_info_quality_used_advanced" => mean_info_quality_by_tier["advanced"],
        "mean_info_quality_used_premium" => mean_info_quality_by_tier["premium"],
        [
            "mean_$(key)_$(tier)" => get(mean_perception_telemetry_by_tier[tier], key, 0.0)
            for tier in ("none", "basic", "advanced", "premium")
            for key in keys(mean_perception_telemetry_by_tier[tier])
        ]...,
        # Innovation metrics
        "innovation_attempts" => innovation_attempts,
        "innovation_successes" => innovation_successes,
        "innovation_success_rate" => innovation_success_rate,
        # A1 open-action pivot telemetry. Counts/sums are 0 when no pivots
        # occurred (a count of zero events is data); the recovery RATE over
        # zero pivots is undefined and emitted as NaN per the NaN-for-empty
        # convention. Suite consumers aggregate
        # through finite-value filters, so empty cells drop out instead of
        # deflating toward zero.
        "pivot_count" => pivot_count_total,
        "pivot_capital_recovered" => pivot_recovered_total,
        "pivot_capital_committed" => pivot_committed_total,
        "pivot_recovery_rate" => pivot_committed_total > 0.0 ?
            pivot_recovered_total / pivot_committed_total : NaN,
        # Confidence metrics
        "mean_confidence_invest" => mean_confidence_invest,
        "mean_ai_trust" => mean_trust,
        "ai_trust_std" => std_trust,
        # Matured investment stats
        "n_matured" => n_matured,
        "n_success" => n_success,
        "n_failure" => n_failure,
        "success_rate" => n_matured > 0 ? n_success / n_matured : 0.0,
        # Proximate causal chain, measured on investments maturing this round.
        # NaN means no maturity event occurred; consumers must not convert that
        # into a substantive zero.
        "mean_capacity_saturation_at_entry" =>
            matured_mean("capacity_saturation_at_entry"),
        "mean_capacity_saturation_at_maturity" =>
            matured_mean("capacity_saturation_at_maturity"),
        "mean_effective_capacity_saturation_at_maturity" =>
            matured_mean("effective_capacity_saturation_at_maturity"),
        "mean_capacity_saturation_change" =>
            matured_mean("capacity_saturation_change"),
        "mean_post_commitment_rival_capital" =>
            matured_mean("post_commitment_rival_capital"),
        "mean_post_commitment_rival_capacity_ratio" =>
            matured_mean("post_commitment_rival_capacity_ratio"),
        "mean_crowding_return_multiplier" =>
            matured_mean("crowding_return_multiplier"),
        "mean_realized_latent_capture" =>
            matured_mean("realized_latent_capture"),
        "mean_performative_effective_capacity" =>
            matured_mean("performative_effective_capacity"),
        "mean_performative_lagged_commitment" =>
            matured_mean("performative_lagged_commitment"),
        "mean_performative_commitment_share" =>
            matured_mean("performative_commitment_share"),
        "mean_performative_uplift_fraction" =>
            matured_mean("performative_uplift_fraction"),
        # Open-futures opportunity flows in this round. Initial stock is round
        # zero and therefore intentionally absent from these flow columns.
        "exploration_opportunities_created" =>
            round_flow_count("created"; source="exploration"),
        "innovation_opportunities_spawned" =>
            round_flow_count("created"; source="innovation"),
        "background_opportunities_created" =>
            round_flow_count("created"; source="background"),
        "opportunities_publicized" => round_flow_count("publicized"),
        "opportunities_culled" => round_flow_count("culled"),
        # Uncertainty levels (formula-based, kept for backwards compatibility)
        "actor_ignorance" => actor_ignorance,
        "practical_indeterminism" => practical_indet,
        "agentic_novelty" => agentic_novelty,
        "competitive_recursion" => competitive_rec,
        "market_crowding_pressure" => market_crowding_pressure,
        "market_opportunity_overlap" => market_opportunity_overlap,
        "market_investment_concentration" => market_investment_concentration,
        "market_opportunity_competition" => market_opportunity_competition,
        "market_ai_herding_intensity" => market_ai_herding_intensity,
        "market_ai_action_correlation" => market_ai_action_correlation,
        "market_combo_reuse_pressure" => market_combo_reuse_pressure,
        knightian_component_telemetry...,
        # EMERGENT uncertainty (agent-level, from actual outcomes)
        "emergent_actor_ignorance" => emergent_actor_ignorance,
        "emergent_practical_indeterminism" => emergent_practical_indet,
        "emergent_agentic_novelty" => emergent_agentic_novelty,
        "emergent_competitive_recursion" => emergent_competitive_rec,
        "all_agent_emergent_actor_ignorance" => emergent_actor_ignorance,
        "all_agent_emergent_practical_indeterminism" => emergent_practical_indet,
        "all_agent_emergent_agentic_novelty" => emergent_agentic_novelty,
        "all_agent_emergent_competitive_recursion" => emergent_competitive_rec,
        "survivor_emergent_actor_ignorance" => survivor_emergent_actor_ignorance,
        "survivor_emergent_practical_indeterminism" => survivor_emergent_practical_indet,
        "survivor_emergent_agentic_novelty" => survivor_emergent_agentic_novelty,
        "survivor_emergent_competitive_recursion" => survivor_emergent_competitive_rec,
        "all_agent_emergent_actor_ignorance_observations" =>
            Float64(get(tier_emergent, "actor_ignorance_observations", 0.0)),
        "all_agent_emergent_practical_indeterminism_observations" =>
            Float64(get(tier_emergent, "practical_indeterminism_observations", 0.0)),
        "all_agent_emergent_agentic_novelty_observations" =>
            Float64(get(tier_emergent, "agentic_novelty_observations", 0.0)),
        "all_agent_emergent_competitive_recursion_observations" =>
            Float64(get(tier_emergent, "competitive_recursion_observations", 0.0)),
        "survivor_emergent_actor_ignorance_observations" =>
            Float64(get(tier_survivor_emergent, "actor_ignorance_observations", 0.0)),
        "survivor_emergent_practical_indeterminism_observations" =>
            Float64(get(tier_survivor_emergent, "practical_indeterminism_observations", 0.0)),
        "survivor_emergent_agentic_novelty_observations" =>
            Float64(get(tier_survivor_emergent, "agentic_novelty_observations", 0.0)),
        "survivor_emergent_competitive_recursion_observations" =>
            Float64(get(tier_survivor_emergent, "competitive_recursion_observations", 0.0)),
        # Uncertainty transformation metrics
        "delta_actor_ignorance" => delta_actor,
        "delta_practical_indeterminism" => delta_practical,
        "delta_agentic_novelty" => delta_agentic,
        "delta_competitive_recursion" => delta_competitive,
        "cumulative_delta_actor" => cumulative_delta_actor,
        "cumulative_delta_practical" => cumulative_delta_practical,
        "cumulative_delta_agentic" => cumulative_delta_agentic,
        "cumulative_delta_competitive" => cumulative_delta_competitive,
        "uncertainty_total" => uncertainty_total,
        "share_actor_ignorance" => share_actor,
        "share_practical_indeterminism" => share_practical,
        "share_agentic_novelty" => share_agentic,
        "share_competitive_recursion" => share_competitive,
        "uncertainty_hhi" => uncertainty_hhi,
        "uncertainty_entropy" => uncertainty_entropy,
        # Agent counts
        "alive_agents" => n_alive,
        "dead_agents" => n_total - n_alive
    )
end

"""
Convert simulation history to DataFrame.
"""
function history_to_dataframe(sim::EmergentSimulation)::DataFrame
    if isempty(sim.history)
        return DataFrame()
    end

    # Union of column names across ALL rounds (sorted for stable column
    # order). Taking only round 1's keys silently dropped any telemetry key
    # that first appears later, e.g. per-tier perception components for tiers
    # with no agents in round 1.
    col_set = Set{String}()
    for h in sim.history
        union!(col_set, keys(h))
    end
    cols = sort!(collect(col_set))

    # Create DataFrame
    df = DataFrame()
    for col in cols
        df[!, Symbol(col)] = [get(h, col, missing) for h in sim.history]
    end

    return df
end

"""
Get final agent data as DataFrame.
"""
function agents_to_dataframe(sim::EmergentSimulation)::DataFrame
    data = [snapshot(agent, sim.current_round) for agent in sim.agents]

    if isempty(data)
        return DataFrame()
    end

    cols = collect(keys(data[1]))
    df = DataFrame()
    for col in cols
        df[!, Symbol(col)] = [get(d, col, missing) for d in data]
    end

    return df
end

"""
Get summary statistics for the simulation.
"""
function summary_stats(sim::EmergentSimulation)::Dict{String,Any}
    alive_agents = [a for a in sim.agents if a.alive]

    # Final survival rate
    survival_rate = length(alive_agents) / length(sim.agents)

    # Capital statistics
    final_capitals = [get_capital(a) for a in alive_agents]
    mean_final_capital = isempty(final_capitals) ? 0.0 : mean(final_capitals)

    # AI tier distribution at end
    ai_distribution = Dict{String,Int}("none" => 0, "basic" => 0, "advanced" => 0, "premium" => 0)
    for agent in sim.agents
        tier = get_ai_level(agent)
        if haskey(ai_distribution, tier)
            ai_distribution[tier] += 1
        end
    end

    # Success/failure totals
    total_successes = sum(a.success_count for a in sim.agents)
    total_failures = sum(a.failure_count for a in sim.agents)
    total_innovations = sum(a.innovation_count for a in sim.agents)

    confidence_outcome = confidence_outcome_stats(sim.agents)
    venture_events = [
        event for agent in sim.agents for event in agent.venture_ledger
    ]
    function venture_mean(index::Int)
        values = Float64[]
        for event in venture_events
            index <= length(event) || continue
            value = Float64(event[index])
            isfinite(value) && push!(values, value)
        end
        return isempty(values) ? missing : mean(values)
    end
    function venture_ratio_mean(numerator_index::Int, denominator_index::Int)
        values = Float64[]
        for event in venture_events
            max(numerator_index, denominator_index) <= length(event) || continue
            numerator = Float64(event[numerator_index])
            denominator = Float64(event[denominator_index])
            if isfinite(numerator) && isfinite(denominator) && denominator > 0.0
                push!(values, numerator / denominator)
            end
        end
        return isempty(values) ? missing : mean(values)
    end
    flow_counts = opportunity_flow_counts(sim.market)

    # Uncertainty averages from history
    if !isempty(sim.history)
        mean_actor_ignorance = mean(get(h, "actor_ignorance", 0.0) for h in sim.history)
        mean_practical_indet = mean(get(h, "practical_indeterminism", 0.0) for h in sim.history)
        mean_agentic_novelty = mean(get(h, "agentic_novelty", 0.0) for h in sim.history)
        mean_competitive_rec = mean(get(h, "competitive_recursion", 0.0) for h in sim.history)
    else
        mean_actor_ignorance = 0.0
        mean_practical_indet = 0.0
        mean_agentic_novelty = 0.0
        mean_competitive_rec = 0.0
    end

    stats = Dict{String,Any}(
        "run_id" => sim.run_id,
        "n_agents" => length(sim.agents),
        "n_rounds" => sim.config.N_ROUNDS,
        "final_survival_rate" => survival_rate,
        "n_survivors" => length(alive_agents),
        "mean_final_capital" => mean_final_capital,
        "total_successes" => total_successes,
        "total_failures" => total_failures,
        "total_innovations" => total_innovations,
        "matured_investment_count" => length(venture_events),
        "mean_capacity_saturation_at_entry" => venture_mean(10),
        "mean_capacity_saturation_at_maturity" => venture_mean(11),
        "mean_capacity_saturation_change" => venture_mean(13),
        "mean_post_commitment_rival_capital" => venture_mean(14),
        "mean_post_commitment_rival_capacity_ratio" => venture_ratio_mean(14, 8),
        "mean_crowding_return_multiplier" => venture_mean(15),
        "mean_realized_latent_capture" => venture_mean(16),
        "mean_performative_effective_capacity" => venture_mean(17),
        "mean_performative_lagged_commitment" => venture_mean(18),
        "mean_performative_commitment_share" => venture_mean(19),
        "mean_performative_uplift_fraction" => venture_mean(20),
        "ai_none_count" => ai_distribution["none"],
        "ai_basic_count" => ai_distribution["basic"],
        "ai_advanced_count" => ai_distribution["advanced"],
        "ai_premium_count" => ai_distribution["premium"],
        "mean_actor_ignorance" => mean_actor_ignorance,
        "mean_practical_indeterminism" => mean_practical_indet,
        "mean_agentic_novelty" => mean_agentic_novelty,
        "mean_competitive_recursion" => mean_competitive_rec,
        "confidence_outcome_weighted_gap_mean" => confidence_outcome["confidence_outcome_weighted_gap_mean"],
        "confidence_outcome_weighted_abs_gap_mean" => confidence_outcome["confidence_outcome_weighted_abs_gap_mean"],
        "confidence_outcome_raw_gap_mean" => confidence_outcome["confidence_outcome_raw_gap_mean"],
        "confidence_outcome_abs_gap_mean" => confidence_outcome["confidence_outcome_abs_gap_mean"],
        "confidence_outcome_positive_gap_rate" => confidence_outcome["confidence_outcome_positive_gap_rate"],
        "confidence_outcome_negative_gap_rate" => confidence_outcome["confidence_outcome_negative_gap_rate"],
        "confidence_outcome_agent_coverage" => confidence_outcome["confidence_outcome_agent_coverage"],
        "confidence_outcome_observations" => confidence_outcome["confidence_outcome_observations"],
        "confidence_outcome_realized_multiple_mean" => confidence_outcome["confidence_outcome_realized_multiple_mean"],
        "elapsed_seconds" => (now() - sim.start_time).value / 1000.0
    )
    for (key, value) in flow_counts
        stats[key] = value
    end
    stats["exploration_niche_events_total"] = Float64(sum(
        a.uncertainty_metrics.niches_discovered for a in sim.agents;
        init=0,
    ))
    return stats
end

"""
Save simulation results to disk.
"""
function save_results!(sim::EmergentSimulation)
    mkpath(sim.output_dir)

    # Save history
    history_df = history_to_dataframe(sim)
    if nrow(history_df) > 0
        save_dataframe_csv(history_df, joinpath(sim.output_dir, "history.csv"))
        save_dataframe_arrow(history_df, joinpath(sim.output_dir, "history.arrow"))
    end

    # Save agent data
    agents_df = agents_to_dataframe(sim)
    if nrow(agents_df) > 0
        save_dataframe_csv(agents_df, joinpath(sim.output_dir, "final_agents.csv"))
    end

    # Save config
    save_config_snapshot(sim.config, joinpath(sim.output_dir, "config_snapshot.json"))

    # Save summary
    stats = summary_stats(sim)
    open(joinpath(sim.output_dir, "summary.json"), "w") do io
        JSON3.write(io, stats)
    end

    println("[$(sim.run_id)] Results saved to $(sim.output_dir)")
end

# ============================================================================
# BATCH SIMULATION UTILITIES
# ============================================================================

"""
Run multiple simulations with different configurations.
"""
function run_batch(;
    base_config::EmergentConfig = EmergentConfig(),
    n_runs::Int = 10,
    output_base::String = "results",
    fixed_ai_levels::Vector{String} = String[],
    parallel::Bool = false
)::Vector{EmergentSimulation}
    results = EmergentSimulation[]

    if isempty(fixed_ai_levels)
        # Run with adaptive AI
        for run_idx in 1:n_runs
            config = deepcopy(base_config)
            config.RANDOM_SEED = base_config.RANDOM_SEED + run_idx

            run_id = "run_$(run_idx)"
            output_dir = joinpath(output_base, run_id)

            sim = EmergentSimulation(
                config=config,
                output_dir=output_dir,
                run_id=run_id,
                seed=config.RANDOM_SEED
            )

            run!(sim)
            save_results!(sim)
            push!(results, sim)
        end
    else
        # Run fixed AI tier sweep
        for (tier_idx, ai_level) in enumerate(fixed_ai_levels)
            for run_idx in 1:n_runs
                config = deepcopy(base_config)
                config.RANDOM_SEED = base_config.RANDOM_SEED + (tier_idx - 1) * n_runs + run_idx

                run_id = "Fixed_AI_Level_$(ai_level)_run_$(run_idx)"
                output_dir = joinpath(output_base, run_id)

                sim = EmergentSimulation(
                    config=config,
                    output_dir=output_dir,
                    run_id=run_id,
                    seed=config.RANDOM_SEED
                )

                # Set fixed AI level for all agents
                initialize_agents!(sim; fixed_ai_level=ai_level)

                run!(sim)
                save_results!(sim)
                push!(results, sim)
            end
        end
    end

    return results
end

"""
Aggregate results from multiple simulations.
"""
function aggregate_results(simulations::Vector{EmergentSimulation})::DataFrame
    all_stats = Dict{String,Any}[]

    for sim in simulations
        stats = summary_stats(sim)
        push!(all_stats, stats)
    end

    if isempty(all_stats)
        return DataFrame()
    end

    cols = collect(keys(all_stats[1]))
    df = DataFrame()
    for col in cols
        df[!, Symbol(col)] = [get(s, col, missing) for s in all_stats]
    end

    return df
end
