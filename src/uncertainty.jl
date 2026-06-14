"""
Knightian Uncertainty Environment for GlimpseABM.jl

This module implements the four-dimensional Knightian uncertainty framework
from Townsend et al. (2025, AMR).
"""

using Random
using Statistics

const _UNCERTAINTY_DIMENSIONS = (
    "actor_ignorance",
    "practical_indeterminism",
    "agentic_novelty",
    "competitive_recursion",
)

const _DEFAULT_RECURSION_WEIGHTS = Dict{String,Float64}(
    "crowd_weight" => 0.35,
    "volatility_weight" => 0.30,
    "ai_herd_weight" => 0.40,
    "reuse_weight" => 0.20,
)

_perception_weights(config::EmergentConfig)::UncertaintyPerceptionWeights =
    hasfield(typeof(config), :UNCERTAINTY_PERCEPTION_WEIGHTS) ?
    config.UNCERTAINTY_PERCEPTION_WEIGHTS :
    UncertaintyPerceptionWeights()

_producer_weights(config::EmergentConfig)::KnightianProducerWeights =
    hasfield(typeof(config), :KNIGHTIAN_PRODUCER_WEIGHTS) ?
    config.KNIGHTIAN_PRODUCER_WEIGHTS :
    KnightianProducerWeights()

function _perception_weight(config::EmergentConfig, key::String)::Float64
    weights = _perception_weights(config)
    return Float64(getfield(weights, Symbol(key)))
end

function _legacy_recursion_weight(config::EmergentConfig, key::String, typed_value::Float64)::Float64
    legacy = hasfield(typeof(config), :RECURSION_WEIGHTS) ? config.RECURSION_WEIGHTS : _DEFAULT_RECURSION_WEIGHTS
    default = _DEFAULT_RECURSION_WEIGHTS[key]
    value = Float64(get(legacy, key, default))
    return isapprox(value, default; atol=1e-12, rtol=0.0) ? typed_value : value
end

function _dimension_state_level(
    env,
    env_state::Dict{String,Any},
    dimension::String,
    fallback::Float64,
)::Float64
    state = get(env_state, dimension, nothing)
    if !(state isa Dict)
        state = if dimension == "actor_ignorance"
            env.actor_ignorance_state
        elseif dimension == "practical_indeterminism"
            env.practical_indeterminism_state
        elseif dimension == "agentic_novelty"
            env.agentic_novelty_state
        else
            env.competitive_recursion_state
        end
    end
    level = Float64(get(state, "level", fallback))
    return isfinite(level) ? clamp(level, 0.0, 1.0) : clamp(fallback, 0.0, 1.0)
end

_bounded01(value::Real)::Float64 = clamp(Float64(value), 0.0, 1.0)

function _bounded01_with_hit(value::Real)::Tuple{Float64,Float64}
    raw = Float64(value)
    bounded = _bounded01(raw)
    hit = isfinite(raw) && !isapprox(raw, bounded; atol=1e-12, rtol=0.0) ? 1.0 : 0.0
    return bounded, hit
end

function _prior_adjusted_raw(prior::Real, local_raw::Real, prior_weight::Real)::Float64
    w = clamp(Float64(prior_weight), 0.0, 1.0)
    return w * Float64(prior) + (1.0 - w) * Float64(local_raw)
end

function _record_perception_dimension_diagnostics!(
    diagnostics::Dict{String,Float64},
    dimension::String;
    prior::Real,
    local_raw::Real,
    raw_score::Real,
    normalized_score::Real,
    bounded_level::Real,
    bound_hit::Real,
)
    prefix = "perception_" * dimension
    diagnostics[prefix * "_prior"] = Float64(prior)
    diagnostics[prefix * "_local_raw"] = Float64(local_raw)
    diagnostics[prefix * "_raw_score"] = Float64(raw_score)
    diagnostics[prefix * "_normalized_score"] = Float64(normalized_score)
    diagnostics[prefix * "_bounded_level"] = Float64(bounded_level)
    diagnostics[prefix * "_bound_hit"] = Float64(bound_hit)
    diagnostics[prefix * "_gap_vs_prior"] = Float64(bounded_level) - Float64(prior)
    # Backward-compatible aliases for any existing ad hoc diagnostics readers.
    diagnostics[prefix * "_raw"] = Float64(raw_score)
    diagnostics[prefix * "_bounded"] = Float64(bounded_level)
end

function _record_perception_component_diagnostics!(
    diagnostics::Dict{String,Float64},
    dimension::String,
    components::Dict{String,Float64},
)
    prefix = "perception_component_" * dimension * "_"
    for (key, value) in components
        diagnostics[prefix * key] = Float64(value)
    end
end

# ============================================================================
# KNIGHTIAN UNCERTAINTY ENVIRONMENT
# ============================================================================

"""
Environment tracking and measuring the four dimensions of Knightian uncertainty.

The four dimensions are:
1. Actor Ignorance - Information gaps about current states
2. Practical Indeterminism - Unpredictable execution outcomes
3. Agentic Novelty - Genuinely new possibilities from creative action
4. Competitive Recursion - Strategic interdependence effects
"""
mutable struct KnightianUncertaintyEnvironment
    config::EmergentConfig

    # Optional knowledge base reference
    knowledge_base::Union{Nothing,Any}

    # Actor Ignorance State
    actor_ignorance_state::Dict{String,Any}

    # Practical Indeterminism State
    practical_indeterminism_state::Dict{String,Any}

    # Agentic Novelty State
    agentic_novelty_state::Dict{String,Any}

    # Competitive Recursion State
    competitive_recursion_state::Dict{String,Any}

    # Uncertainty evolution history
    uncertainty_evolution::Vector{Any}

    # AI-specific signals
    ai_uncertainty_signals::Dict{String,Any}

    # Innovation tracking (circular buffer)
    innovation_success_tracker::Vector{Dict{String,Any}}

    # Exploration outcomes tracking
    exploration_outcomes::Vector{Dict{String,Any}}

    # Market regime history
    market_regime_history::Vector{String}

    # Opportunity discovery rate
    opportunity_discovery_rate::Float64

    # Global short-term buffers for volatility calculation
    _global_short_term_buffers::Dict{String,Vector{Float64}}

    # Per-agent short-term buffers
    _agent_short_term_buffers::Dict{Int,Dict{String,Vector{Float64}}}

    # Short-term window size
    short_term_window::Int

    # Short-term decay factor
    short_term_decay_factor::Float64

    # Volatility state
    _volatility_state::Dict{String,Any}

    # AI signal history limit
    _ai_signal_history::Int

    # Action history for tracking behavioral patterns
    _action_history::Vector{Dict{String,Any}}

    # Novelty diagnostics
    _novelty_diagnostics::Dict{String,Any}

    # Last environment state for caching
    _last_environment_state::Dict{String,Any}

    # Last alive agents count
    _last_alive_agents::Int

    rng::Random.AbstractRNG
end

function KnightianUncertaintyEnvironment(
    config::EmergentConfig;
    knowledge_base::Union{Nothing,Any} = nothing,
    rng::Random.AbstractRNG = Random.default_rng()
)
    history_window = max(5, Int(getfield_default(config, :NOVELTY_HISTORY_WINDOW, 15)))
    short_term_window = Int(getfield_default(config, :UNCERTAINTY_SHORT_WINDOW, 18))
    short_term_decay = Float64(getfield_default(config, :UNCERTAINTY_SHORT_DECAY, 0.0))
    ai_signal_history = max(10, Int(getfield_default(config, :AI_SIGNAL_HISTORY, 420)))

    return KnightianUncertaintyEnvironment(
        config,
        knowledge_base,
        Dict{String,Any}(
            "unknown_opportunities" => Dict{String,Float64}(),
            "knowledge_gaps" => Dict{String,Float64}(),
            "emergence_potential" => 0.0,
            "level" => 0.0
        ),
        Dict{String,Any}(
            "path_volatility" => 0.0,
            "timing_criticality" => Dict{String,Float64}(),
            "regime_instability" => 0.0,
            "level" => 0.0,
            "crowding_pressure" => 0.0,
            "ai_pressure" => 0.0,
            "ai_herding_intensity" => 0.0,
            "timing_variability" => 0.0
        ),
        Dict{String,Any}(
            "creative_momentum" => 0.0,
            "disruption_potential" => Dict{String,Float64}(),
            "novelty_level" => 0.0,
            "level" => 0.0,
            "component_scarcity" => 0.5,
            "scarcity_signal" => 0.5,
            "reuse_pressure" => 0.0,
            "new_possibilities" => 0,
            "new_possibility_rate" => 0.0,
            "innovation_births" => 0,
            "tier_drivers" => Dict{String,Any}(),
            "drivers" => Dict{String,Any}()
        ),
        Dict{String,Any}(
            "strategic_opacity" => 0.0,
            "herding_pressure" => Dict{String,Float64}(),
            "game_complexity" => 0.0,
            "level" => 0.0
        ),
        Any[],  # uncertainty_evolution (stores tuples)
        Dict{String,Any}(
            "hallucination_events" => Dict{String,Any}[],
            "ai_signal_exposures" => Dict{String,Any}[],
            "observable_disconfirmation_events" => Dict{String,Any}[],
            "confidence_miscalibration" => Float64[],
            "ai_herding_patterns" => Dict{String,Float64}()
        ),
        Dict{String,Any}[],  # innovation_success_tracker
        Dict{String,Any}[],  # exploration_outcomes
        String[],  # market_regime_history
        0.5,  # opportunity_discovery_rate
        Dict{String,Vector{Float64}}(
            "actor_ignorance" => Float64[],
            "practical_indeterminism" => Float64[],
            "agentic_novelty" => Float64[],
            "competitive_recursion" => Float64[]
        ),
        Dict{Int,Dict{String,Vector{Float64}}}(),  # _agent_short_term_buffers
        short_term_window,
        short_term_decay,
        Dict{String,Any}(
            "action_prev" => zeros(4),
            "ai_prev" => zeros(4),
            "market_prev" => 0.0,
            "volatility_metric" => 0.0,
            "history" => Float64[],
            "last_action_shares" => zeros(4),
            "last_ai_shares" => zeros(4)
        ),
        ai_signal_history,
        Dict{String,Any}[],  # _action_history
        Dict{String,Any}(),  # _novelty_diagnostics
        Dict{String,Any}(),  # _last_environment_state
        Int(getfield_default(config, :N_AGENTS, 100)),  # _last_alive_agents
        rng
    )
end

"""
Get short-term buffer for a metric (global or per-agent).
"""
function _short_buffer(env::KnightianUncertaintyEnvironment, metric::String, agent_id::Union{Int,Nothing})::Vector{Float64}
    if isnothing(agent_id)
        return env._global_short_term_buffers[metric]
    end
    if !haskey(env._agent_short_term_buffers, agent_id)
        env._agent_short_term_buffers[agent_id] = Dict{String,Vector{Float64}}(
            "actor_ignorance" => Float64[],
            "practical_indeterminism" => Float64[],
            "agentic_novelty" => Float64[],
            "competitive_recursion" => Float64[]
        )
    end
    return env._agent_short_term_buffers[agent_id][metric]
end

"""
Update short-term buffer for a metric.
"""
function _update_short_term!(env::KnightianUncertaintyEnvironment, metric::String, value::Float64; agent_id::Union{Int,Nothing} = nothing)
    if isnothing(value) || !isfinite(value)
        return
    end
    buffer = _short_buffer(env, metric, agent_id)
    push!(buffer, Float64(value))
    # Keep buffer size limited
    max_size = env.short_term_window
    while length(buffer) > max_size
        popfirst!(buffer)
    end
end

"""
Normalize metric value (floor at zero).
"""
function _normalize_metric(env::KnightianUncertaintyEnvironment, metric::String, value::Float64)::Float64
    if !isfinite(value)
        return 0.0
    end
    return max(0.0, value)
end

"""
Get short-term average for a metric.
"""
function _get_short_term_average(env::KnightianUncertaintyEnvironment, metric::String; fallback::Float64 = 0.0, agent_id::Union{Int,Nothing} = nothing)::Float64
    # First try agent-specific buffer
    buffer = nothing
    if !isnothing(agent_id) && haskey(env._agent_short_term_buffers, agent_id)
        buffer = get(env._agent_short_term_buffers[agent_id], metric, nothing)
    end
    # Fall back to global buffer
    if isnothing(buffer)
        buffer = get(env._global_short_term_buffers, metric, Float64[])
    end
    if !isempty(buffer)
        avg = mean(buffer)
    else
        avg = isfinite(fallback) ? fallback : 0.0
    end
    return _normalize_metric(env, metric, avg)
end

"""
Update volatility state based on action and AI share changes.
"""
function _update_volatility_state!(env::KnightianUncertaintyEnvironment, action_shares::Vector{Float64}, ai_shares::Vector{Float64}, market)::Float64
    cfg = env.config

    action_prev = get(env._volatility_state, "action_prev", nothing)
    if isnothing(action_prev) || length(action_prev) != length(action_shares)
        action_prev = zeros(length(action_shares))
    end
    ai_prev = get(env._volatility_state, "ai_prev", nothing)
    if isnothing(ai_prev) || length(ai_prev) != length(ai_shares)
        ai_prev = zeros(length(ai_shares))
    end

    action_delta = mean(abs.(action_shares .- action_prev))
    ai_delta = mean(abs.(ai_shares .- ai_prev))

    market_momentum = hasfield(typeof(market), :market_momentum) ? Float64(market.market_momentum) : 0.0
    market_volatility = hasfield(typeof(market), :volatility) ? Float64(market.volatility) : 0.0
    market_signal = abs(market_momentum) + abs(market_volatility)
    market_prev = Float64(get(env._volatility_state, "market_prev", 0.0))
    market_delta = abs(market_signal - market_prev)

    action_weight = Float64(getfield_default(cfg, :UNCERTAINTY_ACTION_VARIANCE_WEIGHT, 0.14))
    ai_weight = Float64(getfield_default(cfg, :UNCERTAINTY_AI_SWITCH_WEIGHT, 0.09))
    market_weight = Float64(getfield_default(cfg, :UNCERTAINTY_MARKET_RETURN_WEIGHT, 0.14))

    raw = action_weight * action_delta + ai_weight * ai_delta + market_weight * market_delta
    decay = Float64(getfield_default(cfg, :UNCERTAINTY_VOLATILITY_DECAY, 0.87))
    # The EWMA state must be kept on the unscaled (raw) axis; feeding the
    # scaled output back in as the prior would compound the scaling each round,
    # collapsing the effective persistence to decay*scaling and attenuating the
    # steady state by ~1/(1 + decay*(1 - scaling)/(1 - decay)).
    prev_smoothed = Float64(get(env._volatility_state, "volatility_smoothed", 0.0))
    smoothed = prev_smoothed * decay + (1.0 - decay) * raw
    scaling = Float64(getfield_default(cfg, :UNCERTAINTY_VOLATILITY_SCALING, 0.45))
    volatility = smoothed * scaling

    env._volatility_state["action_prev"] = copy(action_shares)
    env._volatility_state["ai_prev"] = copy(ai_shares)
    env._volatility_state["market_prev"] = market_signal
    env._volatility_state["volatility_smoothed"] = smoothed
    env._volatility_state["volatility_metric"] = volatility

    return volatility
end

function _ai_signal_exposure_records(env::KnightianUncertaintyEnvironment)
    records = get(env.ai_uncertainty_signals, "ai_signal_exposures", nothing)
    if records isa Vector{Dict{String,Any}}
        return records
    end

    records = Dict{String,Any}[]
    env.ai_uncertainty_signals["ai_signal_exposures"] = records
    return records
end

function _observable_ai_disconfirmation_records(env::KnightianUncertaintyEnvironment)
    records = get(env.ai_uncertainty_signals, "observable_disconfirmation_events", nothing)
    if records isa Vector{Dict{String,Any}}
        return records
    end

    records = Dict{String,Any}[]
    env.ai_uncertainty_signals["observable_disconfirmation_events"] = records
    return records
end

function _rolling_ai_signal_stats(env::KnightianUncertaintyEnvironment)
    exposures = _observable_ai_disconfirmation_records(env)
    window_size = min(100, env._ai_signal_history)

    recent = if isempty(exposures)
        Dict{String,Any}[]
    else
        exposures[max(1, length(exposures) - window_size + 1):end]
    end

    exposure_count = length(recent)
    disconfirmation_count = count(record -> Bool(get(record, "suspected_ai_failure", false)), recent)
    # Overall rate across ALL matured-investment exposures (AI and human alike).
    # Forecast disconfirmation is a property of venture-return unpredictability,
    # not of AI presence, so the headline rate covers everyone.
    disconfirmation_rate = exposure_count > 0 ? clamp(disconfirmation_count / exposure_count, 0.0, 1.0) : 0.0

    # Population split: AI-assisted exposures vs the human (non-AI) baseline.
    # Records written before the symmetrization carry no "ai_assisted" flag;
    # they were AI-only by construction, so fall back to the behavior tier.
    _record_ai_assisted(record) = Bool(get(record, "ai_assisted",
        String(get(record, "ai_level", "none")) != "none"))
    ai_exposure_count = 0
    ai_disconfirmation_count = 0
    baseline_exposure_count = 0
    baseline_disconfirmation_count = 0
    for record in recent
        failed = Bool(get(record, "suspected_ai_failure", false))
        if _record_ai_assisted(record)
            ai_exposure_count += 1
            ai_disconfirmation_count += failed ? 1 : 0
        else
            baseline_exposure_count += 1
            baseline_disconfirmation_count += failed ? 1 : 0
        end
    end
    ai_rate = ai_exposure_count > 0 ?
        clamp(ai_disconfirmation_count / ai_exposure_count, 0.0, 1.0) : 0.0
    baseline_rate = baseline_exposure_count > 0 ?
        clamp(baseline_disconfirmation_count / baseline_exposure_count, 0.0, 1.0) : 0.0
    # Uncertainty attributable to AI is its EXCESS forecast error over the
    # human baseline. When AI forecasts are no worse than human ones this is
    # ~0 instead of a hard-wired AI-presence offset (a known asymmetry: uncertainty re-entering only for AI users through a
    # channel with no human analogue).
    excess_ai_rate = max(0.0, ai_rate - baseline_rate)

    severity_values = Float64[]
    miscalibration_values = Float64[]
    # (design note): miscalibration is split by population the
    # same way the disconfirmation rate is. Confident-but-wrong forecasts are
    # a property of Pareto-tailed venture returns for everyone; only the
    # EXCESS of the AI population's miscalibration over the human baseline is
    # AI-attributable.
    ai_miscalibration_values = Float64[]
    baseline_miscalibration_values = Float64[]
    for record in recent
        severity = get(record, "severity", nothing)
        if !isnothing(severity)
            try
                value = Float64(severity)
                if isfinite(value)
                    push!(severity_values, clamp(value, 0.0, 1.0))
                end
            catch
                # Ignore malformed diagnostic payloads.
            end
        end

        miscalibration = get(record, "observable_miscalibration", nothing)
        if isnothing(miscalibration)
            continue
        end
        try
            value = Float64(miscalibration)
            if isfinite(value)
                push!(miscalibration_values, value)
                if _record_ai_assisted(record)
                    push!(ai_miscalibration_values, value)
                else
                    push!(baseline_miscalibration_values, value)
                end
            end
        catch
            # Ignore malformed diagnostic payloads.
        end
    end

    _finite_mean_or_zero(values::Vector{Float64}) = begin
        m = !isempty(values) ? safe_mean(values) : 0.0
        isfinite(m) ? m : 0.0
    end

    # Pooled value retained for symmetric/diagnostic consumers.
    confidence_miscalibration = _finite_mean_or_zero(miscalibration_values)
    ai_miscalibration = _finite_mean_or_zero(ai_miscalibration_values)
    baseline_miscalibration = _finite_mean_or_zero(baseline_miscalibration_values)
    # AI-attributable miscalibration: excess over the human baseline (~0 when
    # AI forecasts are calibrated no worse than human ones). AI-specific
    # consumers (the recursion-perception component) read THIS.
    excess_miscalibration = max(0.0, ai_miscalibration - baseline_miscalibration)
    disconfirmation_severity = _finite_mean_or_zero(severity_values)

    return (
        exposure_count=exposure_count,
        disconfirmation_count=disconfirmation_count,
        # disconfirmation_rate keeps its name but is now the OVERALL rate
        # across both populations (the field overall_rate aliases it).
        disconfirmation_rate=disconfirmation_rate,
        disconfirmation_severity=disconfirmation_severity,
        confidence_miscalibration=confidence_miscalibration,
        ai_miscalibration=ai_miscalibration,
        baseline_miscalibration=baseline_miscalibration,
        excess_miscalibration=excess_miscalibration,
        overall_rate=disconfirmation_rate,
        baseline_rate=baseline_rate,
        ai_rate=ai_rate,
        excess_ai_rate=excess_ai_rate,
        ai_exposure_count=ai_exposure_count,
        baseline_exposure_count=baseline_exposure_count,
    )
end

"""
(design note): once-per-round cache for
`_rolling_ai_signal_stats`. The rolling stats are a pure function of the
recorded exposure window, but the uncached function was called once per agent
from perceive_uncertainty (plus once from measure_uncertainty_state!), an
O(N_agents × window) rescan per round. The cache lives on
env.ai_uncertainty_signals and is keyed by (round, record count): the record
count guard keeps the value EXACT when new exposures land mid-round (pivot
exposures are recorded during the decision phase — F2b), instead of freezing
the first agent's view for the rest of the round.

Thread-safety: a simulation is single-threaded internally (no @threads/@spawn
anywhere in src/); the only threaded harness is the robustness suite script,
which threads across RUNS (Threads.@threads over run_idx,
run_reviewer_robustness_suite.jl) with each run owning its own
KnightianUncertaintyEnvironment — so this per-env cache is never shared
across threads (same isolation argument as the strategic-opacity
once-per-round guard).
"""
function _rolling_ai_signal_stats_cached(env::KnightianUncertaintyEnvironment, round_num::Int)
    signals = env.ai_uncertainty_signals
    n_records = length(_observable_ai_disconfirmation_records(env))
    cached_round = get(signals, "_ai_signal_stats_cache_round", -1)
    cached_count = get(signals, "_ai_signal_stats_cache_records", -1)
    if cached_round == round_num && cached_count == n_records
        cached = get(signals, "_ai_signal_stats_cache", nothing)
        isnothing(cached) || return cached
    end
    stats = _rolling_ai_signal_stats(env)
    signals["_ai_signal_stats_cache"] = stats
    signals["_ai_signal_stats_cache_round"] = round_num
    signals["_ai_signal_stats_cache_records"] = n_records
    return stats
end

function _rolling_hidden_ai_signal_stats(env::KnightianUncertaintyEnvironment)
    exposures = _ai_signal_exposure_records(env)
    window_size = min(100, env._ai_signal_history)

    recent = if isempty(exposures)
        Dict{String,Any}[]
    else
        exposures[max(1, length(exposures) - window_size + 1):end]
    end

    exposure_count = length(recent)
    hallucination_count = count(record -> Bool(get(record, "hallucinated", false)), recent)
    hallucination_rate = exposure_count > 0 ? clamp(hallucination_count / exposure_count, 0.0, 1.0) : 0.0

    miscalibration_values = Float64[]
    for record in recent
        miscalibration = get(record, "hidden_miscalibration", nothing)
        isnothing(miscalibration) && continue
        try
            value = Float64(miscalibration)
            if isfinite(value)
                push!(miscalibration_values, value)
            end
        catch
            # Ignore malformed diagnostic payloads.
        end
    end

    confidence_miscalibration = !isempty(miscalibration_values) ? safe_mean(miscalibration_values) : 0.0
    if !isfinite(confidence_miscalibration)
        confidence_miscalibration = 0.0
    end

    return (
        exposure_count=exposure_count,
        hallucination_count=hallucination_count,
        hallucination_rate=hallucination_rate,
        confidence_miscalibration=confidence_miscalibration,
    )
end

function record_observable_ai_disconfirmation!(
    env::KnightianUncertaintyEnvironment,
    round_num::Int,
    agent_id::Union{Int,Nothing},
    ai_level::String,
    domain::String;
    suspected_ai_failure::Bool,
    disconfirmation_score::Float64 = 0.0,
    severity::Float64 = 0.0,
    return_error::Float64 = 0.0,
    ai_confidence::Float64 = 0.5,
    accuracy_score::Float64 = 0.5,
    success::Bool = false,
    # Forecast-disconfirmation events are now recorded for EVERY matured
    # investment, not only AI-assisted ones (symmetric treatment of Knightian
    # uncertainty: Pareto-tailed venture returns disconfirm human forecasts
    # too). The flag splits the rolling stats into an AI population and a
    # human baseline so AI-specific channels can read EXCESS error only.
    # Default mirrors the historical behavior for callers that predate the
    # flag: a non-"none" behavior tier implies AI assistance.
    ai_assisted::Bool = ai_level != "none",
)
    records = _observable_ai_disconfirmation_records(env)
    confidence = isfinite(ai_confidence) ? clamp(ai_confidence, 0.0, 1.0) : 0.5
    accuracy = isfinite(accuracy_score) ? clamp(accuracy_score, 0.0, 1.0) : 0.5
    observable_miscalibration = max(0.0, confidence - accuracy)

    push!(records, Dict{String,Any}(
        "round" => round_num,
        "agent_id" => agent_id,
        "ai_level" => ai_level,
        "ai_assisted" => ai_assisted,
        "domain" => domain,
        "suspected_ai_failure" => suspected_ai_failure,
        "disconfirmation_score" => clamp(disconfirmation_score, 0.0, 1.0),
        "severity" => clamp(severity, 0.0, 1.0),
        "return_error" => max(0.0, return_error),
        "observable_miscalibration" => observable_miscalibration,
        "success" => success,
    ))

    while length(records) > env._ai_signal_history
        popfirst!(records)
    end
end

"""
Record AI signals from agent actions.
"""
function record_ai_signals!(
    env::KnightianUncertaintyEnvironment,
    round_num::Int,
    agent_actions::Vector{Dict{String,Any}}
)
    if isempty(agent_actions)
        return
    end

    hallucinations = env.ai_uncertainty_signals["hallucination_events"]
    exposures = _ai_signal_exposure_records(env)
    confidence_vals = env.ai_uncertainty_signals["confidence_miscalibration"]
    herding_patterns = env.ai_uncertainty_signals["ai_herding_patterns"]

    herding_counts = Dict{String,Int}()

    for action in agent_actions
        ai_level = action_behavior_ai_level(action)
        if ai_level == "none"
            continue
        end
        agent_id = get(action, "agent_id", nothing)
        domain = get(action, "ai_analysis_domain", nothing)
        has_signal_payload =
            haskey(action, "ai_contains_hallucination") ||
            haskey(action, "ai_confidence") ||
            haskey(action, "ai_actual_accuracy")
        contains_hallucination = Bool(get(action, "ai_contains_hallucination", false))

        # Track hidden hallucinations for diagnostics only. Behavioral uncertainty
        # pressure comes from observable post-maturity disconfirmation events,
        # not privileged access to this latent flag.
        if contains_hallucination
            push!(hallucinations, Dict(
                "round" => round_num,
                "agent_id" => agent_id,
                "ai_level" => ai_level,
                "domain" => domain
            ))
        end

        # Track confidence miscalibration
        confidence = get(action, "ai_confidence", nothing)
        accuracy = get(action, "ai_actual_accuracy", nothing)
        hidden_miscalibration = nothing
        try
            if !isnothing(confidence) && !isnothing(accuracy)
                miscal = Float64(confidence) - Float64(accuracy)
                if isfinite(miscal)
                    push!(confidence_vals, miscal)
                    hidden_miscalibration = miscal
                end
            end
        catch
            # Ignore type conversion errors
        end

        if has_signal_payload
            exposure_record = Dict{String,Any}(
                "round" => round_num,
                "agent_id" => agent_id,
                "ai_level" => ai_level,
                "domain" => domain,
                "hallucinated" => contains_hallucination,
            )
            if !isnothing(confidence)
                exposure_record["confidence"] = confidence
            end
            if !isnothing(hidden_miscalibration)
                exposure_record["hidden_miscalibration"] = hidden_miscalibration
            end
            push!(exposures, exposure_record)
        end

        # Track herding
        if get(action, "action", "") == "invest"
            details = get(action, "chosen_opportunity_details", Dict())
            opp_id = get(details, "id", get(action, "opportunity_id", nothing))
            if !isnothing(opp_id)
                herding_counts[string(opp_id)] = get(herding_counts, string(opp_id), 0) + 1
            end
        end
    end

    # Update herding patterns with decay. Decay is applied exactly once, in the
    # first loop over all keys; the second loop then adds this round's new
    # counts to the already-decayed priors (applying decay again in the second
    # loop would double-decay any opp receiving new counts).
    decay = Float64(getfield_default(env.config, :AI_HERDING_DECAY, 0.5))
    for key in collect(keys(herding_patterns))
        herding_patterns[key] *= decay
        # Prune entries that have decayed to numerical irrelevance so the
        # pattern dict tracks recent behavior rather than run-lifetime history.
        if herding_patterns[key] < 1e-4
            delete!(herding_patterns, key)
        end
    end
    for (opp_id, count) in herding_counts
        prior = get(herding_patterns, opp_id, 0.0)  # already decayed above
        herding_patterns[opp_id] = prior + count
    end

    # Maintain matching decayed activity stocks so herding intensity can be
    # normalized as a share of recent activity rather than a cumulative stock
    # divided by a per-round flow (which pins at the clamp within a few rounds).
    ai_invest_new = isempty(herding_counts) ? 0 : sum(values(herding_counts))
    signals = env.ai_uncertainty_signals
    signals["ai_invest_stock"] =
        Float64(get(signals, "ai_invest_stock", 0.0)) * decay + ai_invest_new
    signals["action_stock"] =
        Float64(get(signals, "action_stock", 0.0)) * decay + length(agent_actions)

    # Limit history size
    max_history = env._ai_signal_history
    while length(hallucinations) > max_history
        popfirst!(hallucinations)
    end
    while length(confidence_vals) > max_history
        popfirst!(confidence_vals)
    end
    while length(exposures) > max_history
        popfirst!(exposures)
    end

    # Hidden hallucination/actual-accuracy telemetry is diagnostic only. Do not
    # update behavioral short-term buffers here; public uncertainty responds to
    # observable forecast disconfirmation recorded after matured outcomes
    # (every matured investment, AI-assisted or not — see
    # record_observable_ai_disconfirmation!).
end

"""
Summarize agent actions for uncertainty calculation.

Aggregates per-action and per-tier counts, investment concentration (HHI and
opportunity overlap), herding, combination/niche rates, and an AI action
correlation metric used by the competitive-recursion dimension.
"""
function _summarize_actions(env::KnightianUncertaintyEnvironment, agent_actions::Vector{Dict{String,Any}}, market)::Dict{String,Any}
    summary = Dict{String,Any}(
        "new_combos" => 0,
        "innovate" => 0,
        "new_niches" => 0,
        "explore" => 0,
        "derivative_adoption" => 0,
        "invest" => 0,
        "combo_hhi" => 0.0,
        "sector_hhi" => 0.0,
        "invest_hhi" => 0.0,
        "opportunity_overlap" => 0.0,
        "herding_counts" => Dict{String,Int}(),
        "invest_by_ai" => Dict{String,Int}(),
        "tier_stats" => Dict{String,Any}(),
        "ai_action_correlation" => 0.0  # AI-induced action correlation
    )

    invest_counts = Dict{String,Int}()
    invest_by_ai = Dict{String,Int}()
    tiers = ["none", "basic", "advanced", "premium"]

    # Track actions and opportunities by tier for the correlation calculation.
    actions_by_tier = Dict{String,Vector{String}}(tier => String[] for tier in tiers)
    opportunities_by_tier = Dict{String,Dict{String,Int}}(tier => Dict{String,Int}() for tier in tiers)

    tier_template() = Dict{String,Int}(
        "total_actions" => 0,
        "innovate" => 0,
        "new_combos" => 0,
        "reuse_hits" => 0,
        "explore" => 0,
        "new_niches" => 0,
        "invest" => 0,
        "derivative_adoption" => 0
    )

    tier_stats = Dict{String,Dict{String,Int}}(tier => tier_template() for tier in tiers)

    for action in agent_actions
        act_type = get(action, "action", nothing)
        tier = action_behavior_ai_level(action)

        if !haskey(tier_stats, tier)
            tier_stats[tier] = tier_template()
        end
        tier_stats[tier]["total_actions"] += 1

        # Track actions for the correlation calculation.
        if !isnothing(act_type)
            push!(actions_by_tier[tier], act_type)
        end

        if act_type == "innovate"
            summary["innovate"] += 1
            if get(action, "is_new_combination", false)
                summary["new_combos"] += 1
                tier_stats[tier]["new_combos"] += 1
            elseif !isnothing(get(action, "combination_signature", nothing))
                tier_stats[tier]["reuse_hits"] += 1
            end
            tier_stats[tier]["innovate"] += 1
        elseif act_type == "explore"
            summary["explore"] += 1
            if get(action, "created_niche", false) || get(action, "discovered_niche", false)
                summary["new_niches"] += 1
                tier_stats[tier]["new_niches"] += 1
            end
            tier_stats[tier]["explore"] += 1
        elseif act_type == "invest"
            summary["invest"] += 1
            tier_stats[tier]["invest"] += 1
            if get(action, "invested_derivative", false)
                summary["derivative_adoption"] += 1
                tier_stats[tier]["derivative_adoption"] += 1
            end
            details = get(action, "chosen_opportunity_details", Dict())
            opp_id = get(details, "id", get(action, "opportunity_id", nothing))
            if !isnothing(opp_id)
                opp_id_str = string(opp_id)
                invest_counts[opp_id_str] = get(invest_counts, opp_id_str, 0) + 1
                # Track opportunity choices by tier.
                opportunities_by_tier[tier][opp_id_str] = get(opportunities_by_tier[tier], opp_id_str, 0) + 1
            end
            invest_by_ai[tier] = get(invest_by_ai, tier, 0) + 1
        end
    end

    # Compute investment HHI
    invest_total = summary["invest"]
    if invest_total > 0 && !isempty(invest_counts)
        invest_hhi = sum((count / invest_total)^2 for count in values(invest_counts))
        summary["invest_hhi"] = invest_hhi
        if invest_total > 1
            min_hhi = 1.0 / invest_total
            summary["opportunity_overlap"] = clamp((invest_hhi - min_hhi) / max(1e-9, 1.0 - min_hhi), 0.0, 1.0)
        end
    end
    summary["herding_counts"] = invest_counts
    summary["invest_by_ai"] = invest_by_ai

    # Compute tier-specific rates
    tier_combo_rate = Dict{String,Float64}()
    tier_reuse_pressure = Dict{String,Float64}()
    tier_new_poss_rate = Dict{String,Float64}()
    tier_derivative_adoption_rate = Dict{String,Float64}()

    for (tier, stats) in tier_stats
        innov = max(1, stats["innovate"])
        invest_count = max(1, stats["invest"])
        action_total = max(1, stats["total_actions"])
        tier_combo_rate[tier] = clamp(stats["new_combos"] / innov, 0.0, 1.0)
        tier_reuse_pressure[tier] = clamp(stats["reuse_hits"] / innov, 0.0, 1.0)
        tier_new_poss_rate[tier] = clamp((stats["new_combos"] + stats["new_niches"]) / action_total, 0.0, 1.0)
        tier_derivative_adoption_rate[tier] = clamp(stats["derivative_adoption"] / invest_count, 0.0, 1.0)
    end

    summary["tier_stats"] = tier_stats
    summary["tier_combo_rate"] = tier_combo_rate
    summary["tier_reuse_pressure"] = tier_reuse_pressure
    summary["tier_new_possibility_rate"] = tier_new_poss_rate
    summary["tier_derivative_adoption_rate"] = tier_derivative_adoption_rate
    # Compatibility alias for downstream tables that still read adoption_rate.
    summary["tier_adoption_rate"] = tier_derivative_adoption_rate

    # Get combination diversity metrics from market if available
    if !isnothing(market)
        combo_hhi, sector_hhi = get_combination_diversity_metrics(market)
        summary["combo_hhi"] = combo_hhi
        summary["sector_hhi"] = sector_hhi
    end

    # Calculate AI action correlation for competitive recursion: when AI-using
    # agents converge on similar actions/opportunities, recursion increases.
    ai_tiers = ["basic", "advanced", "premium"]
    ai_action_counts = sum(length(actions_by_tier[tier]) for tier in ai_tiers)

    if ai_action_counts >= 2
        # Calculate correlation based on opportunity clustering among AI agents
        ai_opportunity_hhi = 0.0
        total_ai_investments = sum(sum(values(opportunities_by_tier[tier])) for tier in ai_tiers)

        if total_ai_investments > 0
            # HHI of opportunities among AI agents (higher = more correlated)
            all_ai_opps = Dict{String,Int}()
            for tier in ai_tiers
                for (opp_id, count) in opportunities_by_tier[tier]
                    all_ai_opps[opp_id] = get(all_ai_opps, opp_id, 0) + count
                end
            end

            ai_opportunity_hhi = sum((count / total_ai_investments)^2 for count in values(all_ai_opps))
        end

        # Correlation ranges from baseline 0.3 (no AI) to 0.6+ (high AI adoption with clustering)
        # Scale by AI adoption rate and opportunity clustering
        ai_adoption_rate = ai_action_counts / max(1, length(agent_actions))
        summary["ai_action_correlation"] = clamp(
            0.30 + 0.30 * ai_adoption_rate * ai_opportunity_hhi,
            0.30,
            0.70
        )
    else
        summary["ai_action_correlation"] = 0.30  # Baseline correlation without AI
    end

    return summary
end

"""
Compute component scarcity from knowledge base.
"""
function _compute_component_scarcity(env::KnightianUncertaintyEnvironment)::Float64
    if !isnothing(env.knowledge_base)
        raw_scarcity = Float64(get_component_scarcity_metric(env.knowledge_base))
    else
        raw_scarcity = Float64(get(env.agentic_novelty_state, "scarcity_signal", 0.5))
    end
    prev = get(env.agentic_novelty_state, "scarcity_signal", nothing)
    if isnothing(prev)
        return raw_scarcity
    end
    blend = clamp(0.65 * prev + 0.35 * raw_scarcity, 0.0, 1.0)
    return blend
end

"""
Measure environment-level uncertainty across the four Knightian dimensions
(actor ignorance, practical indeterminism, agentic novelty, competitive
recursion) from this round's agent actions, innovations, and market state.
"""
function measure_uncertainty_state!(
    env::KnightianUncertaintyEnvironment,
    market::MarketEnvironment,
    agent_actions::Vector{Dict{String,Any}},
    innovations::Vector{Innovation},
    round_num::Int
)::Dict{String,Dict{String,Any}}
    state = Dict{String,Dict{String,Any}}()
    ai_effects = Dict{String,Float64}()
    total_actions = length(agent_actions)
    producer_weights = _producer_weights(env.config)
    actor_w = producer_weights.actor_ignorance
    practical_w = producer_weights.practical_indeterminism
    novelty_w = producer_weights.agentic_novelty
    recursion_w_base = producer_weights.competitive_recursion
    recursion_w = CompetitiveRecursionProducerWeights(
        crowd_weight=_legacy_recursion_weight(env.config, "crowd_weight", recursion_w_base.crowd_weight),
        volatility_weight=_legacy_recursion_weight(env.config, "volatility_weight", recursion_w_base.volatility_weight),
        ai_herd_weight=_legacy_recursion_weight(env.config, "ai_herd_weight", recursion_w_base.ai_herd_weight),
        reuse_weight=_legacy_recursion_weight(env.config, "reuse_weight", recursion_w_base.reuse_weight),
        invest_hhi_weight=recursion_w_base.invest_hhi_weight,
        knowledge_overlap_weight=recursion_w_base.knowledge_overlap_weight,
        ai_action_correlation_excess_weight=recursion_w_base.ai_action_correlation_excess_weight,
        population_scale_base=recursion_w_base.population_scale_base,
        population_scale_alive_weight=recursion_w_base.population_scale_alive_weight,
        no_ai_herding_weight=recursion_w_base.no_ai_herding_weight,
    )

    # Count actions by type
    action_counter = Dict{String,Int}()
    for action in agent_actions
        act_type = get(action, "action", "maintain")
        action_counter[act_type] = get(action_counter, act_type, 0) + 1
    end

    if total_actions > 0
        action_shares = Float64[
            get(action_counter, "invest", 0) / total_actions,
            get(action_counter, "innovate", 0) / total_actions,
            get(action_counter, "explore", 0) / total_actions,
            get(action_counter, "maintain", 0) / total_actions
        ]
    else
        action_shares = Float64[0.0, 0.0, 0.0, 1.0]
    end

    # Count AI tier usage
    ai_counter = Dict{String,Int}()
    for action in agent_actions
        tier = action_behavior_ai_level(action)
        ai_counter[tier] = get(ai_counter, tier, 0) + 1
    end

    if total_actions > 0
        ai_shares = Float64[
            get(ai_counter, "none", 0) / total_actions,
            get(ai_counter, "basic", 0) / total_actions,
            get(ai_counter, "advanced", 0) / total_actions,
            get(ai_counter, "premium", 0) / total_actions
        ]
    else
        ai_shares = Float64[1.0, 0.0, 0.0, 0.0]
    end

    ai_share_none = ai_shares[1]

    # Summarize actions (tier-specific tracking)
    action_summary = _summarize_actions(env, agent_actions, market)

    # Update tier invest shares on market
    tier_invest_share = Dict{String,Float64}()
    invest_total = get(action_summary, "invest", 0)
    if invest_total > 0
        for (tier, count) in get(action_summary, "invest_by_ai", Dict())
            tier_invest_share[tier] = clamp(count / invest_total, 0.0, 1.0)
        end
    end
    if hasfield(typeof(market), :_tier_invest_share) && !isempty(tier_invest_share)
        market._tier_invest_share = tier_invest_share
    end
    if isempty(tier_invest_share)
        tier_invest_share = Dict{String,Float64}(
            "none" => ai_shares[1],
            "basic" => ai_shares[2],
            "advanced" => ai_shares[3],
            "premium" => ai_shares[4]
        )
    end

    # Record action history
    push!(env._action_history, action_summary)
    history_window = max(5, Int(getfield_default(env.config, :NOVELTY_HISTORY_WINDOW, 15)))
    while length(env._action_history) > history_window
        popfirst!(env._action_history)
    end

    # AI herding intensity = participation x excess concentration.
    # Participation is the decayed share of recent actions that were AI-assisted
    # investments; concentration is the HHI of the decayed per-opportunity
    # AI-invest distribution in excess of the HHI uniform random allocation
    # would produce at the same volume (E[HHI_uniform] = 1/K + (1 - 1/K)/n).
    # Dispersed AI investing therefore scores ~0 no matter how much AI activity
    # exists; only convergence on few opportunities registers as herding.
    ai_herding_patterns = get(env.ai_uncertainty_signals, "ai_herding_patterns", Dict())
    ai_invest_stock = Float64(get(env.ai_uncertainty_signals, "ai_invest_stock", 0.0))
    action_stock = Float64(get(env.ai_uncertainty_signals, "action_stock", 0.0))
    n_opps_active = (!isnothing(market) && hasfield(typeof(market), :opportunities)) ?
        length(market.opportunities) : 0
    if !isempty(ai_herding_patterns) && ai_invest_stock > 1e-9 && n_opps_active >= 2
        pattern_total = sum(values(ai_herding_patterns))
        herding_hhi = sum((v / pattern_total)^2 for v in values(ai_herding_patterns))
        k_opps = Float64(n_opps_active)
        expected_uniform_hhi = 1.0 / k_opps +
            (1.0 - 1.0 / k_opps) / max(1.0, ai_invest_stock)
        excess_concentration = clamp(
            (herding_hhi - expected_uniform_hhi) / max(1e-9, 1.0 - expected_uniform_hhi),
            0.0, 1.0)
        ai_participation = clamp(ai_invest_stock / max(1.0, action_stock), 0.0, 1.0)
        ai_herding_intensity = clamp(ai_participation * excess_concentration, 0.0, 1.0)
    else
        ai_herding_intensity = 0.0
    end

    # Count new possibilities
    new_possibilities = 0
    for action in agent_actions
        if get(action, "discovered_niche", false) || get(action, "created_opportunity", false) || !isnothing(get(action, "new_opportunity_id", nothing))
            new_possibilities += 1
        end
    end
    innovation_births = count(inn -> !isnothing(inn), innovations)
    new_possibility_rate = new_possibilities / max(1, total_actions)

    # Update volatility state
    volatility = _update_volatility_state!(env, action_shares, ai_shares, market)
    env._volatility_state["last_action_shares"] = copy(action_shares)
    env._volatility_state["last_ai_shares"] = copy(ai_shares)

    share_invest, share_innovate, share_explore, share_maintain = action_shares

    # ========================================================================
    # ACTOR IGNORANCE
    # ========================================================================
    # Gather knowledge gaps from agent perceptions
    knowledge_gaps = Dict{String,Float64}()
    for action in agent_actions
        perception = get(action, "perception_at_decision", Dict())
        ignorance = get(perception, "actor_ignorance", Dict())
        gap_map = get(ignorance, "knowledge_gaps", Dict())
        if !isempty(gap_map)
            merge!(knowledge_gaps, gap_map)
        end
    end

    # Calculate knowledge-based metrics. The gap term is the share of the
    # collectively knowable (distinct pieces in the knowledge base) that the
    # average agent does NOT hold; it falls as diffusion spreads knowledge and
    # rises again when innovation expands the frontier faster than agents learn.
    avg_knowledge, total_knowledge = if !isnothing(env.knowledge_base)
        (Float64(get_average_agent_knowledge(env.knowledge_base)),
         Float64(get_total_knowledge_count(env.knowledge_base)))
    else
        (0.0, 0.0)
    end
    total_opportunities = length(market.opportunities)
    knowledge_norm = avg_knowledge / max(1.0, total_knowledge)

    gap_values = collect(values(knowledge_gaps))
    gap_pressure = !isempty(gap_values) ? clamp(safe_mean(gap_values), 0.0, 1.0) : 0.0
    gap_coverage = total_opportunities > 0 ? length(knowledge_gaps) / total_opportunities : 0.0
    # F5: round-keyed cached read (measure_uncertainty_state! is the round
    # producer; perceive_uncertainty reuses the same cache per agent).
    ai_signal_stats = _rolling_ai_signal_stats_cached(env, round_num)
    hidden_ai_signal_stats = _rolling_hidden_ai_signal_stats(env)
    # Actor ignorance reads the OVERALL forecast-disconfirmation rate: every
    # agent faces irreducible venture-return unpredictability, AI-assisted or
    # not. Only the EXCESS of AI error over the human baseline is treated as
    # AI-attributable (see the recursion-perception consumers).
    disconfirmation_rate = ai_signal_stats.disconfirmation_rate
    knowledge_gap_term = 1.0 - clamp(knowledge_norm, 0.0, 1.0)

    # ACTOR IGNORANCE: Linear additive formula (consistent with other dimensions)
    # Base level + weighted drivers that can accumulate or decline
    # Theoretical rationale: Ignorance can spike during paradigm shifts or collapse
    # with learning breakthroughs - sigmoid would artificially constrain these dynamics

    actor_components = Dict{String,Float64}(
        "base" => actor_w.base_level,
        "knowledge_gap" => actor_w.knowledge_gap_weight * knowledge_gap_term,
        "gap_pressure" => actor_w.gap_pressure_weight * gap_pressure,
        "gap_coverage" => actor_w.gap_coverage_weight * gap_coverage,
        # Renamed from "ai_disconfirmation": the channel is symmetric now
        # (every matured investment, AI or not, contributes an exposure), so
        # the component reflects forecast unpredictability faced by everyone.
        # The weight field keeps its historical name for config compatibility.
        "forecast_disconfirmation" => actor_w.ai_disconfirmation_weight * disconfirmation_rate,
        "exploration_reduction" => -actor_w.exploration_reduction_weight * share_explore,
        "maintain_increase" => actor_w.maintain_increase_weight * share_maintain,
    )
    actor_raw = sum(values(actor_components))
    actor_level = clamp(actor_raw, 0.0, 1.0)

    # Baseline (no AI) actor ignorance. The no-AI counterfactual is the human
    # baseline disconfirmation rate, not zero: forecast disconfirmation exists
    # without AI, so only the AI-attributable excess enters the AI delta.
    actor_components_no_ai = copy(actor_components)
    actor_components_no_ai["forecast_disconfirmation"] =
        actor_w.ai_disconfirmation_weight * ai_signal_stats.baseline_rate
    actor_level_no_ai = clamp(sum(values(actor_components_no_ai)), 0.0, 1.0)
    ai_effects["ai_ignorance_delta"] = actor_level - actor_level_no_ai

    _update_short_term!(env, "actor_ignorance", actor_level)
    env.actor_ignorance_state["level"] = actor_level
    # "ai_disconfirmation_rate" is retained as a legacy alias; since the
    # symmetrization it holds the OVERALL forecast-disconfirmation rate. The
    # population-split rates are exported under explicit new names below.
    env.actor_ignorance_state["ai_disconfirmation_rate"] = disconfirmation_rate
    env.actor_ignorance_state["hallucination_rate"] = disconfirmation_rate  # legacy alias
    env.actor_ignorance_state["forecast_disconfirmation_rate"] = disconfirmation_rate
    env.actor_ignorance_state["baseline_disconfirmation_rate"] = ai_signal_stats.baseline_rate
    env.actor_ignorance_state["ai_population_disconfirmation_rate"] = ai_signal_stats.ai_rate
    env.actor_ignorance_state["excess_ai_disconfirmation_rate"] = ai_signal_stats.excess_ai_rate
    env.actor_ignorance_state["ai_signal_exposures"] = ai_signal_stats.exposure_count
    env.actor_ignorance_state["ai_disconfirmation_count"] = ai_signal_stats.disconfirmation_count
    env.actor_ignorance_state["ai_hallucination_count"] = ai_signal_stats.disconfirmation_count  # legacy alias
    env.actor_ignorance_state["hidden_ai_signal_exposures"] = hidden_ai_signal_stats.exposure_count
    env.actor_ignorance_state["hidden_ai_hallucination_count"] = hidden_ai_signal_stats.hallucination_count
    env.actor_ignorance_state["hidden_hallucination_rate"] = hidden_ai_signal_stats.hallucination_rate
    env.actor_ignorance_state["raw_score"] = actor_raw
    env.actor_ignorance_state["components"] = actor_components

    state["actor_ignorance"] = Dict{String,Any}(
        "level" => actor_level,
        "knowledge_gaps" => knowledge_gaps,
        "gap_pressure" => gap_pressure,
        "volatility" => volatility,
        "ai_signal_exposures" => ai_signal_stats.exposure_count,
        "ai_disconfirmation_count" => ai_signal_stats.disconfirmation_count,
        "ai_disconfirmation_rate" => disconfirmation_rate,                  # legacy alias: now OVERALL rate
        "ai_disconfirmation_severity" => ai_signal_stats.disconfirmation_severity,
        "forecast_disconfirmation_rate" => disconfirmation_rate,
        "baseline_disconfirmation_rate" => ai_signal_stats.baseline_rate,
        "ai_population_disconfirmation_rate" => ai_signal_stats.ai_rate,
        "excess_ai_disconfirmation_rate" => ai_signal_stats.excess_ai_rate,
        "ai_hallucination_count" => ai_signal_stats.disconfirmation_count,  # legacy alias
        "hallucination_rate" => disconfirmation_rate,                       # legacy alias
        "hidden_ai_signal_exposures" => hidden_ai_signal_stats.exposure_count,
        "hidden_ai_hallucination_count" => hidden_ai_signal_stats.hallucination_count,
        "hidden_hallucination_rate" => hidden_ai_signal_stats.hallucination_rate,
        "ai_delta" => get(ai_effects, "ai_ignorance_delta", 0.0),
        "raw_score" => actor_raw,
        "components" => actor_components,
    )

    # ========================================================================
    # PRACTICAL INDETERMINISM
    # ========================================================================
    timing_levels = Float64[]
    timing_pressure = Dict{String,Float64}()
    if !isempty(market.opportunities)
        for opp in market.opportunities
            acceleration = Float64(opp.market_share) * Float64(opp.competition)
            timing_pressure[opp.id] = acceleration
            push!(timing_levels, acceleration)
        end
    end
    base_practical = !isempty(timing_levels) ? safe_mean(timing_levels) : 0.3
    # Bessel-corrected std of a single element is NaN in Julia (NumPy ddof=0
    # returns 0.0); guard so a one-opportunity market can't poison the round.
    timing_variability = length(timing_levels) > 1 ? std(timing_levels) : 0.0

    market_volatility = Float64(market.volatility)
    regime = hasfield(typeof(market), :market_regime) ? market.market_regime : "normal"
    regime_uncertainty = get(Dict(
        "boom" => 0.35, "growth" => 0.22, "normal" => 0.18,
        "volatile" => 0.32, "recession" => 0.45, "crisis" => 0.65
    ), regime, 0.2)
    path_component = clamp(timing_variability * 1.5, 0.0, 1.0)

    # Behavioral crowding metrics. Crowding comes from actual investment
    # concentration, opportunity overlap, and clearing load — not from an
    # action-share HHI, which would label a market of agents all maintaining
    # as "crowded."
    market_crowding_metrics = hasfield(typeof(market), :crowding_metrics) ? market.crowding_metrics : Dict()
    action_concentration = Float64(get(market_crowding_metrics, "crowding_index", 0.0))
    investment_concentration = Float64(get(action_summary, "invest_hhi", 0.0))
    opportunity_overlap = Float64(get(action_summary, "opportunity_overlap", 0.0))
    # Opportunity-level competition/saturation exposure: investment-weighted
    # mean, over opportunities holding capital, of max(competition_trace,
    # saturation) — the same competition and capacity-saturation the return
    # penalty uses. Crowding in this model is an opportunity-level phenomenon,
    # not a sector-level one.
    opp_crowd_weighted_sum = 0.0
    opp_crowd_weight = 0.0
    if !isempty(market.opportunities)
        for opp in market.opportunities
            comp = hasfield(typeof(opp), :competition) ? max(0.0, Float64(opp.competition)) : 0.0
            comp_trace = 1.0 - exp(-comp / 1.5)
            sat = (hasfield(typeof(opp), :capacity) && opp.capacity > 0.0 &&
                   hasfield(typeof(opp), :total_invested)) ?
                clamp(Float64(opp.total_invested) / Float64(opp.capacity), 0.0, 1.0) : 0.0
            crowd = max(comp_trace, sat)
            w = hasfield(typeof(opp), :total_invested) ? max(0.0, Float64(opp.total_invested)) : 0.0
            opp_crowd_weighted_sum += w * crowd
            opp_crowd_weight += w
        end
    end
    opportunity_competition = opp_crowd_weight > 0.0 ?
        clamp(opp_crowd_weighted_sum / opp_crowd_weight, 0.0, 1.0) : 0.0
    investment_volume = clamp(share_invest, 0.0, 1.0)
    crowding_components = Dict{String,Float64}(
        "investment_overlap" => practical_w.crowding_investment_overlap_weight * investment_volume * opportunity_overlap,
        "opportunity_competition" => practical_w.crowding_opportunity_competition_weight * opportunity_competition,
        "investment_concentration" => practical_w.crowding_investment_concentration_weight * investment_volume * investment_concentration,
    )
    crowding_pressure = clamp(sum(values(crowding_components)), 0.0, 1.0)
    ai_usage_pressure = clamp(1.0 - ai_share_none, 0.0, 1.0)

    crowd_weight = Float64(getfield_default(env.config, :UNCERTAINTY_CROWDING_WEIGHT, 0.18))
    competitive_weight = Float64(getfield_default(env.config, :UNCERTAINTY_COMPETITIVE_WEIGHT, 0.12))
    herding_weight = Float64(getfield_default(env.config, :UNCERTAINTY_AI_HERDING_WEIGHT, 0.10))

    practical_components = Dict{String,Float64}(
        "base" => practical_w.base_level,
        "market_volatility" => practical_w.market_volatility_weight * market_volatility,
        "regime_uncertainty" => practical_w.regime_uncertainty_weight * regime_uncertainty,
        "timing_pressure" => practical_w.timing_pressure_weight * clamp(base_practical, 0.0, 1.0),
        "path_volatility" => practical_w.path_volatility_weight * path_component,
        "invest_share" => practical_w.invest_share_weight * share_invest,
        "crowding_pressure" => crowd_weight * crowding_pressure,
        "competitive_overlap" => competitive_weight * opportunity_overlap,
        "ai_herding" => herding_weight * ai_herding_intensity,
    )
    practical_raw = sum(values(practical_components))
    practical_level = clamp(practical_raw, 0.0, 1.0)

    practical_components_no_ai = copy(practical_components)
    practical_components_no_ai["ai_herding"] = 0.0
    practical_level_no_ai = clamp(sum(values(practical_components_no_ai)), 0.0, 1.0)
    ai_effects["ai_indeterminism_delta"] = practical_level - practical_level_no_ai

    _update_short_term!(env, "practical_indeterminism", practical_level)
    env.practical_indeterminism_state["path_volatility"] = path_component
    env.practical_indeterminism_state["timing_variability"] = timing_variability
    env.practical_indeterminism_state["regime_instability"] = regime_uncertainty
    env.practical_indeterminism_state["crowding_pressure"] = crowding_pressure
    env.practical_indeterminism_state["ai_pressure"] = ai_usage_pressure
    env.practical_indeterminism_state["ai_usage_share"] = ai_usage_pressure
    env.practical_indeterminism_state["action_concentration"] = action_concentration
    env.practical_indeterminism_state["investment_volume"] = investment_volume
    env.practical_indeterminism_state["investment_concentration"] = investment_concentration
    env.practical_indeterminism_state["opportunity_overlap"] = opportunity_overlap
    env.practical_indeterminism_state["opportunity_competition"] = opportunity_competition
    env.practical_indeterminism_state["ai_herding_intensity"] = ai_herding_intensity
    env.practical_indeterminism_state["level"] = practical_level
    env.practical_indeterminism_state["raw_score"] = practical_raw
    env.practical_indeterminism_state["components"] = practical_components
    env.practical_indeterminism_state["crowding_components"] = crowding_components

    state["practical_indeterminism"] = Dict{String,Any}(
        "level" => practical_level,
        "timing_pressure" => timing_pressure,
        "volatility" => volatility,
        "crowding_pressure" => crowding_pressure,
        "ai_usage_share" => ai_usage_pressure,
        "action_concentration" => action_concentration,
        "investment_volume" => investment_volume,
        "investment_concentration" => investment_concentration,
        "opportunity_overlap" => opportunity_overlap,
        "opportunity_competition" => opportunity_competition,
        "ai_herding_intensity" => ai_herding_intensity,
        "ai_delta" => get(ai_effects, "ai_indeterminism_delta", 0.0),
        "raw_score" => practical_raw,
        "components" => practical_components,
        "crowding_components" => crowding_components,
    )

    # ========================================================================
    # AGENTIC NOVELTY
    # ========================================================================
    history = env._action_history
    window_size = min(5, length(history))
    recent_history = window_size > 0 ? history[end-window_size+1:end] : []

    total_innov = sum(get(item, "innovate", 0) for item in recent_history)
    total_explore = sum(get(item, "explore", 0) for item in recent_history)
    total_invest = sum(get(item, "invest", 0) for item in recent_history)

    combo_rate = total_innov > 0 ? sum(get(item, "new_combos", 0) for item in recent_history) / total_innov : 0.0
    niche_rate = total_explore > 0 ? sum(get(item, "new_niches", 0) for item in recent_history) / total_explore : 0.0
    derivative_adoption_rate = total_invest > 0 ? sum(get(item, "derivative_adoption", 0) for item in recent_history) / total_invest : 0.0

    combo_hhi_avg = safe_mean([get(item, "combo_hhi", 0.0) for item in recent_history])
    sector_hhi_avg = safe_mean([get(item, "sector_hhi", 0.0) for item in recent_history])
    diversity_term = clamp(1.0 - combo_hhi_avg, 0.0, 1.0)
    sector_diversity = clamp(1.0 - sector_hhi_avg, 0.0, 1.0)

    scarcity_signal = _compute_component_scarcity(env)
    innovation_intensity = innovation_births / max(1, total_actions)

    # Disruption state with decay
    disruption_state = get(env.agentic_novelty_state, "disruption_potential", Dict{String,Float64}())
    if isempty(disruption_state) && hasfield(typeof(env.config), :SECTORS)
        for sector in env.config.SECTORS
            disruption_state[sector] = 0.3
        end
    end

    disruption_decay = Float64(getfield_default(env.config, :DISRUPTION_STATE_DECAY, 0.92))
    for sector in collect(keys(disruption_state))
        disruption_state[sector] = clamp(disruption_state[sector] * disruption_decay, 0.05, 0.95)
    end

    # Update disruption from innovations
    for innovation in innovations
        if isnothing(innovation)
            continue
        end
        sector = innovation.sector
        if isnothing(sector)
            continue
        end
        success_flag = innovation.success ? 1.0 : 0.0
        novelty_score = Float64(isnothing(innovation.novelty) ? 0.0 : innovation.novelty)
        impact_signal = Float64(hasfield(typeof(innovation), :market_impact) ? (isnothing(innovation.market_impact) ? 0.0 : innovation.market_impact) : 0.0)
        bump = novelty_w.disruption_base_bump +
            novelty_w.disruption_novelty_weight * novelty_score +
            novelty_w.disruption_impact_weight * max(impact_signal, 0.0) +
            novelty_w.disruption_success_weight * success_flag
        disruption_state[sector] = clamp(get(disruption_state, sector, 0.3) + bump, 0.05, 0.95)
    end

    # Update from explore actions
    for action in agent_actions
        if get(action, "action", "") != "explore"
            continue
        end
        sector = get(action, "base_sector", get(action, "domain", nothing))
        if isnothing(sector)
            continue
        end
        bump = (get(action, "created_niche", false) || get(action, "discovered_niche", false)) ?
            novelty_w.explore_niche_bump :
            novelty_w.explore_general_bump
        disruption_state[sector] = clamp(get(disruption_state, sector, 0.3) + bump, 0.05, 0.95)
    end

    disruption_avg = !isempty(disruption_state) ? clamp(fast_mean(values(disruption_state)), 0.0, 1.0) : 0.3
    innovation_momentum = clamp(innovation_intensity + new_possibility_rate, 0.0, 1.0)
    reuse_pressure = novelty_w.reuse_combo_hhi_weight * combo_hhi_avg +
        novelty_w.reuse_sector_hhi_weight * sector_hhi_avg

    # AGENTIC NOVELTY: Linear additive formula (consistent with other dimensions)
    # Theoretical rationale: Novelty can collapse (combinatorial exhaustion) or
    # explode (paradigm shifts) - sigmoid would artificially prevent these valid extremes

    novelty_diagnostics = Dict{String,Any}(
        "combo_rate" => combo_rate,
        "niche_rate" => niche_rate,
        "adoption_rate" => derivative_adoption_rate,
        "derivative_adoption_rate" => derivative_adoption_rate,
        "diversity_term" => diversity_term,
        "sector_diversity" => sector_diversity,
        "scarcity" => scarcity_signal,
        "innovation_intensity" => innovation_intensity,
        "new_possibility_rate" => new_possibility_rate,
        "reuse_pressure" => reuse_pressure,
        "disruption_avg" => disruption_avg
    )

    # Optional direct AI novelty component. Baseline defaults to zero so
    # novelty emerges from observed combinations, niches, innovations,
    # diversity, and reuse. Robustness cells can opt into a direct channel.
    ai_novelty_uplift = Float64(getfield_default(env.config, :AI_NOVELTY_UPLIFT, 0.0))
    ai_direct_novelty_uplift = ai_novelty_uplift * ai_usage_pressure
    novelty_diagnostics["ai_usage_share"] = ai_usage_pressure
    novelty_diagnostics["ai_direct_novelty_uplift"] = ai_direct_novelty_uplift

    # Linear additive: drivers that increase novelty potential minus drags
    # Positive: new combinations, new possibilities, niches, innovation, disruption, diversity
    # Negative: derivative adoption (known patterns), reuse pressure (combinatorial exhaustion)
    novelty_components = Dict{String,Float64}(
        "base" => novelty_w.base_level,
        "combo_rate" => novelty_w.combo_rate_weight * combo_rate,
        "new_possibility_rate" => novelty_w.new_possibility_rate_weight * new_possibility_rate,
        "niche_rate" => novelty_w.niche_rate_weight * niche_rate,
        "innovation_intensity" => novelty_w.innovation_intensity_weight * innovation_intensity,
        "disruption" => novelty_w.disruption_weight * disruption_avg,
        "sector_diversity" => novelty_w.sector_diversity_weight * sector_diversity,
        "derivative_adoption_drag" => -novelty_w.derivative_adoption_drag * derivative_adoption_rate,
        "reuse_pressure_drag" => -novelty_w.reuse_pressure_drag * reuse_pressure,
        "ai_direct_novelty_uplift" => ai_direct_novelty_uplift,
    )
    agentic_raw = sum(values(novelty_components))
    agentic_level = clamp(agentic_raw, 0.0, 1.0)

    novelty_components_no_ai = copy(novelty_components)
    novelty_components_no_ai["ai_direct_novelty_uplift"] = 0.0
    agentic_level_no_ai = clamp(sum(values(novelty_components_no_ai)), 0.0, 1.0)

    _update_short_term!(env, "agentic_novelty", agentic_level)

    # Update agentic novelty state
    env.agentic_novelty_state["creative_momentum"] = agentic_level
    env.agentic_novelty_state["new_possibilities"] = new_possibilities
    env.agentic_novelty_state["new_possibility_rate"] = new_possibility_rate
    env.agentic_novelty_state["innovation_births"] = innovation_births
    env.agentic_novelty_state["scarcity_signal"] = scarcity_signal
    env.agentic_novelty_state["recent_scarcity_signal"] = scarcity_signal
    env.agentic_novelty_state["novelty_level"] = agentic_level
    env.agentic_novelty_state["drivers"] = novelty_diagnostics
    env.agentic_novelty_state["disruption_potential"] = disruption_state
    env.agentic_novelty_state["level"] = agentic_level
    env.agentic_novelty_state["ai_usage_share"] = ai_usage_pressure
    env.agentic_novelty_state["ai_direct_novelty_uplift"] = ai_direct_novelty_uplift
    env.agentic_novelty_state["raw_score"] = agentic_raw
    env.agentic_novelty_state["components"] = novelty_components
    # Emit the keys realized_return reads: `novelty_potential` and
    # `component_scarcity`. Writing only the internal keys ("level" /
    # "scarcity_signal") would trip realized_return's silent-zero fallbacks,
    # defaulting novelty_signal to 0.5 and freezing scarcity at each opp's
    # baked-in component_scarcity.
    env.agentic_novelty_state["novelty_potential"] = agentic_level
    env.agentic_novelty_state["component_scarcity"] = scarcity_signal
    env._novelty_diagnostics = novelty_diagnostics

    # Get tier-specific rates from action summary
    tier_combo_rate = get(action_summary, "tier_combo_rate", Dict{String,Float64}())
    tier_reuse_pressure = get(action_summary, "tier_reuse_pressure", Dict{String,Float64}())
    tier_new_poss_rate = get(action_summary, "tier_new_possibility_rate", Dict{String,Float64}())
    tier_derivative_adoption_rate = get(
        action_summary,
        "tier_derivative_adoption_rate",
        get(action_summary, "tier_adoption_rate", Dict{String,Float64}()),
    )

    tier_drivers = Dict{String,Any}(
        "combo_rate" => tier_combo_rate,
        "reuse_pressure" => tier_reuse_pressure,
        "new_possibility_rate" => tier_new_poss_rate,
        "adoption_rate" => tier_derivative_adoption_rate,
        "derivative_adoption_rate" => tier_derivative_adoption_rate
    )
    env.agentic_novelty_state["tier_drivers"] = tier_drivers

    state["agentic_novelty"] = Dict{String,Any}(
        "level" => agentic_level,
        "novelty_potential" => agentic_level,
        "new_possibilities" => new_possibilities,
        "new_possibility_rate" => new_possibility_rate,
        "innovation_births" => innovation_births,
        "volatility" => volatility,
        "scarcity_signal" => scarcity_signal,
        "component_scarcity" => scarcity_signal,
        "disruption_potential" => disruption_state,
        "combo_rate" => combo_rate,
        "reuse_pressure" => reuse_pressure,
        "adoption_rate" => derivative_adoption_rate,
        "derivative_adoption_rate" => derivative_adoption_rate,
        "drivers" => novelty_diagnostics,
        "ai_usage_share" => ai_usage_pressure,
        "ai_direct_novelty_uplift" => ai_direct_novelty_uplift,
        "ai_delta" => agentic_level - agentic_level_no_ai,
        "raw_score" => agentic_raw,
        "components" => novelty_components,
        "tier_combo_rate" => tier_combo_rate,
        "tier_reuse_pressure" => tier_reuse_pressure,
        "tier_new_possibility_rate" => tier_new_poss_rate,
        "tier_adoption_rate" => tier_derivative_adoption_rate,
        "tier_derivative_adoption_rate" => tier_derivative_adoption_rate
    )

    # ========================================================================
    # COMPETITIVE RECURSION
    # ========================================================================
    invest_hhi = get(action_summary, "invest_hhi", 0.0)
    premium_share = ai_shares[4]
    tier_reuse_map = get(action_summary, "tier_reuse_pressure", Dict())
    tier_stats_map = get(action_summary, "tier_stats", Dict())
    knowledge_overlap = Float64(get(action_summary, "sector_hhi", 0.0))

    combo_reuse_numer = 0.0
    combo_reuse_denom = 0.0
    ai_combo_reuse_numer = 0.0
    ai_combo_reuse_denom = 0.0
    for tier in ("none", "basic", "advanced", "premium")
        stats = get(tier_stats_map, tier, Dict())
        innovate_count = Float64(get(stats, "innovate", 0))
        if innovate_count <= 0
            continue
        end
        tier_reuse = Float64(get(tier_reuse_map, tier, 0.0))
        combo_reuse_numer += tier_reuse * innovate_count
        combo_reuse_denom += innovate_count
        if tier != "none"
            ai_combo_reuse_numer += tier_reuse * innovate_count
            ai_combo_reuse_denom += innovate_count
        end
    end
    combo_reuse_pressure = combo_reuse_denom > 0 ? clamp(combo_reuse_numer / combo_reuse_denom, 0.0, 1.0) : 0.0
    ai_combo_reuse_pressure = ai_combo_reuse_denom > 0 ? clamp(ai_combo_reuse_numer / ai_combo_reuse_denom, 0.0, 1.0) : 0.0

    agent_count = max(1, Int(getfield_default(env.config, :N_AGENTS, 1)))
    alive_agents = env._last_alive_agents
    population_factor = clamp(alive_agents / agent_count, 0.15, 1.0)

    # Track AI action correlation and let only excess behavioral convergence
    # above the no-clustering baseline contribute to recursion.
    ai_correlation = Float64(get(action_summary, "ai_action_correlation", 0.30))
    ai_correlation_excess = max(0.0, ai_correlation - 0.30)

    # Crowding component based on behavioral convergence, not tier labels.
    # Frontier AI share remains diagnostic only; recursion emerges from concentrated
    # investment, overlap, action correlation, herding, and repeated strategies.
    behavioral_crowding_components = Dict{String,Float64}(
        "invest_hhi" => recursion_w.invest_hhi_weight * invest_hhi,
        "crowding_pressure" => recursion_w.crowd_weight * crowding_pressure,
        "knowledge_overlap" => recursion_w.knowledge_overlap_weight * knowledge_overlap,
        "combo_reuse" => recursion_w.reuse_weight * combo_reuse_pressure,
    )
    behavioral_crowding_component = sum(values(behavioral_crowding_components))
    ai_convergence_component = recursion_w.ai_action_correlation_excess_weight * ai_correlation_excess
    crowding_component = (
        behavioral_crowding_component +
        ai_convergence_component
    )

    scale = recursion_w.population_scale_base + recursion_w.population_scale_alive_weight * population_factor
    recursion_components = Dict{String,Float64}(
        "behavioral_crowding_scaled" => behavioral_crowding_component * scale,
        "ai_action_correlation_excess" => ai_convergence_component * scale,
        "volatility" => recursion_w.volatility_weight * volatility,
        "ai_herding" => recursion_w.ai_herd_weight * ai_herding_intensity,
    )
    recursion_raw = sum(values(recursion_components))
    recursion_level = clamp(recursion_raw, 0.0, 1.0)

    recursion_components_no_ai = copy(recursion_components)
    recursion_components_no_ai["ai_action_correlation_excess"] = 0.0
    recursion_components_no_ai["ai_herding"] = recursion_w.no_ai_herding_weight * ai_herding_intensity
    recursion_level_no_ai = clamp(
        sum(values(recursion_components_no_ai)),
        0.0,
        1.0,
    )
    ai_effects["ai_recursion_delta"] = recursion_level - recursion_level_no_ai

    _update_short_term!(env, "competitive_recursion", recursion_level)

    herding_pressure_map = Dict{String,Float64}()
    herding_counts = get(action_summary, "herding_counts", Dict())
    invest_count = max(1, get(action_summary, "invest", 1))
    for (opp_id, count) in herding_counts
        herding_pressure_map[opp_id] = count / invest_count
    end

    env.competitive_recursion_state["level"] = recursion_level
    env.competitive_recursion_state["herding_pressure"] = herding_pressure_map
    env.competitive_recursion_state["raw_score"] = recursion_raw
    env.competitive_recursion_state["components"] = recursion_components
    env.competitive_recursion_state["behavioral_crowding_components"] = behavioral_crowding_components

    state["competitive_recursion"] = Dict{String,Any}(
        "level" => recursion_level,
        "herding_pressure" => herding_pressure_map,
        "volatility" => volatility,
        "ai_premium_share" => premium_share,
        "ai_herding_intensity" => ai_herding_intensity,
        "ai_action_correlation" => ai_correlation,
        "ai_action_correlation_excess" => ai_correlation_excess,
        "combo_reuse_pressure" => combo_reuse_pressure,
        "ai_combo_reuse_pressure" => ai_combo_reuse_pressure,
        "ai_delta" => get(ai_effects, "ai_recursion_delta", 0.0),
        "raw_score" => recursion_raw,
        "components" => recursion_components,
        "behavioral_crowding_components" => behavioral_crowding_components,
    )

    # Record evolution as tuple (round, state)
    push!(env.uncertainty_evolution, (round_num, state))
    env._last_environment_state = state

    return state
end

"""
Get the current uncertainty state as a simple dictionary.
"""
function get_uncertainty_state(env::KnightianUncertaintyEnvironment)::Dict{String,Any}
    return Dict{String,Any}(
        "actor_ignorance" => env.actor_ignorance_state,
        "practical_indeterminism" => env.practical_indeterminism_state,
        "agentic_novelty" => env.agentic_novelty_state,
        "competitive_recursion" => env.competitive_recursion_state
    )
end

"""
Calculate composite uncertainty score.
"""
function get_composite_uncertainty(env::KnightianUncertaintyEnvironment)::Float64
    levels = [
        Float64(get(env.actor_ignorance_state, "level", 0.0)),
        Float64(get(env.practical_indeterminism_state, "level", 0.0)),
        Float64(get(env.agentic_novelty_state, "level", 0.0)),
        Float64(get(env.competitive_recursion_state, "level", 0.0))
    ]
    return mean(levels)
end

const _RETURN_MULTIPLE_FLOOR = 0.01
const _RETURN_EVIDENCE_PRIOR_WEIGHT = 4.0
const _RETURN_TAIL_THRESHOLD_MULTIPLE = 3.0
const _RETURN_TAIL_WEIGHT = 0.25
const _RETURN_TAIL_EXPONENT = 1.15

function _decision_confidence_diagnostics_from_components(;
    total_uncertainty::Float64,
    competence_trait::Float64,
    recent_success_rate::Float64,
    ai_success_rate::Float64,
    avg_ai_trust::Float64,
    # When false, the AI-experience multiplier and AI-trust/hallucination
    # confidence terms are neutralized: an agent not using AI should derive
    # neither confidence nor doubt from beliefs about a tool it is not using
    # (mirrors the counts_as_ai_use gate on ai_interpretation_reliability).
    ai_used::Bool = true,
    mean_roi::Float64 = 0.0,
    return_evidence::Union{Float64,Nothing} = nothing,
    risk_tolerance::Float64,
    hallucination_rate::Float64,
    # Inert by design: confidence depends on information quality only through
    # total_uncertainty, never directly. Kept as a parameter so the invariant
    # (output independent of info_quality) stays regression-tested.
    info_quality::Float64 = 0.0,
)::Dict{String,Float64}
    raw_confidence = competence_trait * exp(-total_uncertainty * 0.5)
    raw_confidence = isfinite(raw_confidence) ? raw_confidence : 0.5

    experience_multiplier = clamp(0.4 + 1.4 * recent_success_rate * competence_trait, 0.2, 1.8)
    ai_effective_trust = clamp(avg_ai_trust, 0.0, 1.0)
    ai_experience_multiplier = ai_used ?
        clamp(0.4 + 1.4 * ai_success_rate * ai_effective_trust, 0.2, 1.8) : 1.0
    evidence = isnothing(return_evidence) ? mean_roi : return_evidence
    evidence = isfinite(evidence) ? evidence : 0.0

    # Return learning is multiplicative: realized multiples enter through
    # log-return evidence upstream, so a 50x outcome teaches more than a 5x
    # outcome without making confidence linear in raw ROI.
    return_sensitivity = 0.35 + 0.25 * clamp(risk_tolerance, 0.0, 1.0)
    return_multiplier = safe_exp(return_sensitivity * evidence)

    raw_pre_clamp = raw_confidence * experience_multiplier * ai_experience_multiplier * return_multiplier
    raw_confidence = clamp(raw_pre_clamp, 0.01, 2.0)
    raw_clamp_hit = raw_confidence != raw_pre_clamp

    decision_confidence = stable_sigmoid(3.0 * (raw_confidence - 0.5))
    decision_confidence = isfinite(decision_confidence) ? decision_confidence : 0.5

    confidence_multiplier = ai_used ?
        (1 - 0.4 * hallucination_rate + 0.3 * avg_ai_trust) : 1.0
    confidence_multiplier = clamp(confidence_multiplier, 0.5, 1.3)
    final_pre_clamp = decision_confidence * confidence_multiplier
    final_confidence = clamp(final_pre_clamp, 0.02, 0.98)
    final_clamp_hit = final_confidence != final_pre_clamp

    return Dict{String,Float64}(
        "decision_confidence" => final_confidence,
        "return_evidence" => evidence,
        "return_sensitivity" => return_sensitivity,
        "return_multiplier" => isfinite(return_multiplier) ? return_multiplier : 1.0e12,
        "pre_clamp_raw_confidence" => raw_pre_clamp,
        "raw_confidence" => raw_confidence,
        "raw_confidence_clamp_hit" => raw_clamp_hit ? 1.0 : 0.0,
        "final_confidence_pre_clamp" => final_pre_clamp,
        "final_confidence_clamp_hit" => final_clamp_hit ? 1.0 : 0.0,
        "confidence_saturation_hit" => (raw_clamp_hit || final_clamp_hit) ? 1.0 : 0.0,
    )
end

function _decision_confidence_from_components(; kwargs...)::Float64
    return _decision_confidence_diagnostics_from_components(; kwargs...)["decision_confidence"]
end

function _experience_return_multiple(outcome::Dict{String,Any})::Union{Float64,Nothing}
    if haskey(outcome, "cash_multiple")
        return Float64(outcome["cash_multiple"])
    elseif haskey(outcome, "return_multiple")
        return Float64(outcome["return_multiple"])
    else
        investment = get(outcome, "investment", Dict())
        invested = Float64(get(outcome, "investment_amount", get(outcome, "amount", get(investment, "amount", 0.0))))
        returned = Float64(get(outcome, "capital_returned", 0.0))
        if invested > 0
            return returned / invested
        end
    end
    return nothing
end

function _recent_outcome_experience_stats(
    recent_outcomes::Vector{Dict{String,Any}}
)::Dict{String,Any}
    recent_success_rate = 0.5
    ai_success_rate = 0.5
    raw_mean_roi = 0.0
    raw_mean_log_multiple = 0.0
    tail_return_evidence = 0.0
    return_evidence = 0.0
    return_evidence_shrinkage = 0.0
    n_outcomes = 0
    n_roi_outcomes = 0

    if !isempty(recent_outcomes)
        # "pivot" records (open-action extension A1) only exist when
        # ENABLE_PIVOT is on, so including them leaves all default runs
        # bit-identical while letting experience-based learning see the
        # realized recovery multiple (recovered/committed < 1) honestly.
        observable_outcomes = filter(o -> get(o, "action", "") in ("invest", "innovate", "explore", "pivot"), recent_outcomes)
        n_outcomes = length(observable_outcomes)
        if !isempty(observable_outcomes)
            # (design note): pivots are EXCLUDED from the
            # Boolean success channels — a censored salvage at a
            # policy-determined haircut is neither a success nor a
            # maturity-grade failure, and counting every pivot as a failure
            # would double-punish the abandonment option (the cash multiple
            # below already carries the loss into return_evidence honestly).
            boolean_outcomes = filter(o -> get(o, "action", "") != "pivot", observable_outcomes)
            if !isempty(boolean_outcomes)
                recent_success_rate = count(o -> get(o, "success", false), boolean_outcomes) / length(boolean_outcomes)
            end
            ai_outcomes = filter(o -> get(o, "ai_used", false), boolean_outcomes)
            if !isempty(ai_outcomes)
                ai_success_rate = count(o -> get(o, "success", false), ai_outcomes) / length(ai_outcomes)
            else
                ai_success_rate = recent_success_rate
            end

            multiples = Float64[]
            for outcome in observable_outcomes
                multiple = _experience_return_multiple(outcome)
                if !isnothing(multiple) && isfinite(multiple)
                    push!(multiples, multiple)
                end
            end

            n_roi_outcomes = length(multiples)
            if !isempty(multiples)
                raw_mean_roi = mean(m - 1.0 for m in multiples)
                log_values = [log(max(m, _RETURN_MULTIPLE_FLOOR)) for m in multiples]
                raw_mean_log_multiple = mean(log_values)
                tail_threshold = log(_RETURN_TAIL_THRESHOLD_MULTIPLE)
                tail_values = [
                    sign(lv) * max(abs(lv) - tail_threshold, 0.0)^_RETURN_TAIL_EXPONENT
                    for lv in log_values
                ]
                tail_return_evidence = mean(tail_values)
                return_evidence_shrinkage = n_roi_outcomes / (n_roi_outcomes + _RETURN_EVIDENCE_PRIOR_WEIGHT)
                return_evidence = return_evidence_shrinkage * (
                    raw_mean_log_multiple + _RETURN_TAIL_WEIGHT * tail_return_evidence
                )
            end
        end
    end

    return Dict{String,Any}(
        "recent_success_rate" => recent_success_rate,
        "ai_success_rate" => ai_success_rate,
        "raw_mean_roi" => raw_mean_roi,
        "raw_mean_log_multiple" => raw_mean_log_multiple,
        "tail_return_evidence" => tail_return_evidence,
        "return_evidence_shrinkage" => return_evidence_shrinkage,
        "return_evidence" => return_evidence,
        "mean_roi" => raw_mean_roi,
        "roi_clamp_hit" => false,
        "n_outcomes" => n_outcomes,
        "n_roi_outcomes" => n_roi_outcomes,
    )
end

function _observable_opp_float(opp, field::Symbol, default::Float64)::Float64
    hasfield(typeof(opp), field) || return default
    value = getfield(opp, field)
    isnothing(value) && return default
    try
        parsed = Float64(value)
        return isfinite(parsed) ? parsed : default
    catch
        return default
    end
end

function _observable_opp_string(opp, field::Symbol, default::String)::String
    hasfield(typeof(opp), field) || return default
    value = getfield(opp, field)
    return isnothing(value) ? default : string(value)
end

function _observable_opportunity_signal(
    opp,
    weights::UncertaintyPerceptionWeights = UncertaintyPerceptionWeights(),
)::Float64
    complexity = clamp(_observable_opp_float(opp, :complexity, 1.0) / 2.0, 0.0, 1.0)
    novelty = clamp(_observable_opp_float(opp, :novelty_score, 0.5), 0.0, 1.0)
    combo_unc = clamp(_observable_opp_float(opp, :combination_uncertainty, 0.5), 0.0, 1.0)
    threshold = clamp(_observable_opp_float(opp, :required_discovery_threshold, 0.0), 0.0, 1.0)

    intrinsic_legibility = (
        weights.opportunity_legibility_complexity_weight * (1.0 - complexity) +
        weights.opportunity_legibility_novelty_weight * (1.0 - novelty) +
        weights.opportunity_legibility_combo_uncertainty_weight * (1.0 - combo_unc) +
        weights.opportunity_legibility_discovery_threshold_weight * (1.0 - threshold)
    )

    capital_requirements = _observable_opp_float(opp, :capital_requirements, 10_000.0)
    capacity = max(_observable_opp_float(opp, :capacity, capital_requirements * 10.0), capital_requirements * 10.0, 1.0)
    invested_trace = 1.0 - exp(-max(0.0, _observable_opp_float(opp, :total_invested, 0.0)) / capacity)
    competition_trace = 1.0 - exp(-max(0.0, _observable_opp_float(opp, :competition, 0.0)) / 1.5)
    impact_trace = 1.0 - exp(-max(0.0, _observable_opp_float(opp, :market_impact, 0.0)) / 3.0)
    market_trace = (
        weights.opportunity_market_invested_weight * invested_trace +
        weights.opportunity_market_competition_weight * competition_trace +
        weights.opportunity_market_impact_weight * impact_trace
    )

    # Constant lifecycle term pinned at the "emerging" value (0.30): the staged
    # opportunity lifecycle was excised 2026-06-09 as unreachable (review
    # decision #4) — every opportunity stayed "emerging" in practice.
    lifecycle_signal = 0.30

    signal = (
        weights.opportunity_signal_intrinsic_weight * intrinsic_legibility +
        weights.opportunity_signal_market_weight * market_trace +
        weights.opportunity_signal_lifecycle_weight * lifecycle_signal
    )
    if hasfield(typeof(opp), :truly_unknown) && Bool(getfield(opp, :truly_unknown))
        signal *= weights.truly_unknown_signal_multiplier
    end
    if threshold > 0.0
        signal *= (1.0 - weights.discovery_threshold_signal_discount * threshold)
    end
    return clamp(signal, 0.0, 1.0)
end

function _observable_information_context(
    visible_opportunities::Vector,
    agent_knowledge,
    sector_knowledge::Dict{String,Float64},
    total_sectors::Int,
    weights::UncertaintyPerceptionWeights = UncertaintyPerceptionWeights(),
)
    visible_count = length(visible_opportunities)
    if visible_count == 0
        return (
            visible_count=0,
            visible_coverage=0.0,
            visible_sector_span=0.0,
            avg_sector_familiarity=0.0,
            avg_component_familiarity=0.0,
            avg_signal_strength=0.0,
            opportunity_signals=Float64[],
            data_rich_rate=0.0,
            info_quality=0.0,
            info_breadth=0.0,
            evidence_strength=0.0,
        )
    end

    sector_set = Set{String}()
    sector_familiarities = Float64[]
    component_familiarities = Float64[]
    signal_strengths = Float64[]

    for opp in visible_opportunities
        sector = _observable_opp_string(opp, :sector, "unknown")
        if sector != "unknown"
            push!(sector_set, sector)
        end

        sector_familiarity_value = sector_familiarity(
            sector_knowledge,
            sector;
            opportunity=opp,
            default=0.0,
        )
        push!(sector_familiarities, sector_familiarity_value)

        component_familiarity = sector_familiarity_value
        if hasfield(typeof(opp), :knowledge_components) && !isnothing(getfield(opp, :knowledge_components))
            component_ids = Set(getfield(opp, :knowledge_components))
            if !isempty(component_ids) && !isempty(agent_knowledge)
                familiar = length(intersect(agent_knowledge, component_ids))
                component_familiarity = familiar / max(1, length(component_ids))
            end
        end
        push!(component_familiarities, clamp(component_familiarity, 0.0, 1.0))
        push!(signal_strengths, _observable_opportunity_signal(opp, weights))
    end

    denom_sectors = max(1, total_sectors)
    visible_sector_span = clamp(length(sector_set) / denom_sectors, 0.0, 1.0)
    coverage_scale = max(4.0, 4.0 * denom_sectors)
    visible_coverage = 1.0 - exp(-visible_count / coverage_scale)
    avg_sector_familiarity = safe_mean(sector_familiarities)
    avg_component_familiarity = safe_mean(component_familiarities)
    avg_signal_strength = safe_mean(signal_strengths)
    data_rich_rate = count(s -> s >= 0.45, signal_strengths) / visible_count

    # These are observable sufficiency signals, not raw tier capability labels.
    observable_quality = clamp(
        weights.observable_quality_sector_familiarity_weight * avg_sector_familiarity +
        weights.observable_quality_component_familiarity_weight * avg_component_familiarity +
        weights.observable_quality_signal_strength_weight * avg_signal_strength +
        weights.observable_quality_visible_coverage_weight * visible_coverage +
        weights.observable_quality_data_rich_weight * data_rich_rate,
        0.0, 1.0,
    )
    observable_breadth = clamp(
        weights.observable_breadth_sector_span_weight * visible_sector_span +
        weights.observable_breadth_visible_coverage_weight * visible_coverage +
        weights.observable_breadth_data_rich_weight * data_rich_rate,
        0.0, 1.0,
    )
    evidence_strength = clamp(
        weights.evidence_quality_weight * observable_quality +
        weights.evidence_breadth_weight * observable_breadth,
        0.0, 1.0,
    )

    return (
        visible_count=visible_count,
        visible_coverage=visible_coverage,
        visible_sector_span=visible_sector_span,
        avg_sector_familiarity=avg_sector_familiarity,
        avg_component_familiarity=avg_component_familiarity,
        avg_signal_strength=avg_signal_strength,
        opportunity_signals=signal_strengths,
        data_rich_rate=data_rich_rate,
        info_quality=observable_quality,
        info_breadth=observable_breadth,
        evidence_strength=evidence_strength,
    )
end

# ============================================================================
# AGENT PERCEPTION MODEL
# ============================================================================

"""
Compute an individual agent's subjective perception of uncertainty.

While measure_uncertainty_state() computes objective environment-level
uncertainty, this method models how individual agents perceive uncertainty
based on their traits, knowledge, AI augmentation level, and experience.

The gap between objective and perceived uncertainty is central to the
paradox of future knowledge (Townsend et al., 2025).

Parameters
----------
env : KnightianUncertaintyEnvironment
    The uncertainty environment
agent_traits : Dict
    Agent's psychological traits
visible_opportunities : Vector
    Opportunities currently visible to the agent
market_conditions : Dict
    Current market state
ai_level : String
    Agent's current AI augmentation tier
agent_knowledge : Set{String}
    Knowledge IDs known to the agent
sector_knowledge : Dict{String,Float64}
    Agent's sector knowledge levels
action_history : Vector{String}
    Agent's recent action history for behavioral pattern analysis
ai_learning_profile : AILearningProfile
    Agent's learned understanding of AI capabilities
agent_id : Int
    Agent identifier for agent-specific short-term buffering
agent_resources : Any
    Agent's current resource state
cached_metrics : Dict
    Pre-computed metrics to avoid redundant calculation
recent_outcomes : Vector{Dict}
    Recent investment/innovation outcomes

Returns
-------
Dict
    Agent's subjective uncertainty perception across all dimensions
"""
function perceive_uncertainty(
    env::KnightianUncertaintyEnvironment,
    agent_traits::Dict{String,Float64},
    visible_opportunities::Vector,
    market_conditions::MarketConditions;
    ai_level::String = "none",
    agent_knowledge::Set{String} = Set{String}(),
    sector_knowledge::Dict{String,Float64} = Dict{String,Float64}(),
    action_history::Vector{String} = String[],
    ai_learning_profile::Union{AILearningProfile,Nothing} = nothing,
    agent_id::Union{Int,Nothing} = nothing,
    agent_resources::Any = nothing,
    cached_metrics::Union{Dict{String,Any},Nothing} = nothing,
    recent_outcomes::Vector{Dict{String,Any}} = Dict{String,Any}[]
)::Perception
    env_state = env._last_environment_state
    current_round = market_conditions.round
    volatility = Float64(get(env._volatility_state, "volatility_metric", 0.0))
    perception_diagnostics = Dict{String,Float64}()
    pw = _perception_weights(env.config)

    # Get global action shares from volatility state
    global_action_shares = get(env._volatility_state, "last_action_shares", zeros(4))
    if isnothing(global_action_shares) || length(global_action_shares) != 4
        global_share_invest = global_share_innovate = global_share_explore = global_share_maintain = 0.0
    else
        global_share_invest, global_share_innovate, global_share_explore, global_share_maintain = Float64.(global_action_shares)
    end

    # Compute agent-specific action shares from history
    agent_counts = Dict{String,Int}()
    for action in action_history
        agent_counts[action] = get(agent_counts, action, 0) + 1
    end
    agent_total = length(action_history)

    if agent_total > 0
        agent_share_invest = get(agent_counts, "invest", 0) / agent_total
        agent_share_innovate = get(agent_counts, "innovate", 0) / agent_total
        agent_share_explore = get(agent_counts, "explore", 0) / agent_total
        agent_share_maintain = get(agent_counts, "maintain", 0) / agent_total
    else
        agent_share_invest = global_share_invest
        agent_share_innovate = global_share_innovate
        agent_share_explore = global_share_explore
        agent_share_maintain = global_share_maintain
    end

    # Blend agent-specific and global action shares. The weight is named in
    # config so it is visible as calibration rather than an inline constant.
    blend = clamp(pw.action_history_blend, 0.0, 1.0)
    share_invest = blend * agent_share_invest + (1.0 - blend) * global_share_invest
    share_innovate = blend * agent_share_innovate + (1.0 - blend) * global_share_innovate
    share_explore = blend * agent_share_explore + (1.0 - blend) * global_share_explore
    share_maintain = blend * agent_share_maintain + (1.0 - blend) * global_share_maintain

    # Get global AI shares
    last_ai_shares = get(env._volatility_state, "last_ai_shares", zeros(4))
    if isnothing(last_ai_shares) || length(last_ai_shares) != 4
        ai_shares = Float64[1.0, 0.0, 0.0, 0.0]
    else
        ai_shares = Float64.(last_ai_shares)
    end

    # Extract agent traits
    competence_trait = Float64(get(agent_traits, "competence", 0.5))
    ai_trust_trait = Float64(get(agent_traits, "ai_trust", 0.5))
    exploration_trait = Float64(get(agent_traits, "exploration_tendency", 0.5))
    innovation_trait = Float64(get(agent_traits, "innovativeness", 0.5))
    uncertainty_tolerance = Float64(get(agent_traits, "uncertainty_tolerance", 0.5))
    analytical_ability = Float64(get(agent_traits, "analytical_ability", 0.5))
    risk_tolerance = Float64(get(agent_traits, "risk_tolerance", 0.5))

    total_sectors = hasfield(typeof(env.config), :SECTORS) ? length(env.config.SECTORS) : 0
    info_context = _observable_information_context(
        visible_opportunities,
        agent_knowledge,
        sector_knowledge,
        total_sectors,
        pw,
    )
    observable_info_quality = Float64(info_context.info_quality)
    observable_info_breadth = Float64(info_context.info_breadth)
    observable_evidence_strength = Float64(info_context.evidence_strength)

    # AI trust calibration
    base_ai_trust = ai_trust_trait
    avg_ai_trust = base_ai_trust
    # Historical field name: hallucination_experiences now stores observable
    # suspected AI disconfirmation events rather than hidden hallucination truth.
    hallucination_rate = 0.0
    if !isnothing(ai_learning_profile)
        trusts = collect(values(ai_learning_profile.domain_trust))
        if !isempty(trusts)
            avg_ai_trust = clamp(mean(trusts), 0.0, 1.0)
        end
        total_hall = sum(values(ai_learning_profile.hallucination_experiences))
        total_usage = max(1, sum(values(ai_learning_profile.usage_count)))
        hallucination_rate = clamp(total_hall / total_usage, 0.0, 1.0)
    end
    avg_ai_trust = clamp(
        pw.ai_trust_trait_weight * base_ai_trust +
        pw.ai_trust_learned_weight * avg_ai_trust,
        0.0, 1.0,
    )
    ai_interpretation_reliability = counts_as_ai_use(env.config, ai_level) ?
        avg_ai_trust * (1.0 - hallucination_rate) : 0.0

    # ========================================================================
    # ACTOR IGNORANCE PERCEPTION
    # ========================================================================

    # Personal knowledge assessment
    personal_knowledge = 0.0
    knowledge_span = 0.0
    if !isempty(sector_knowledge)
        knowledge_vals = collect(values(sector_knowledge))
        personal_knowledge = clamp(mean(knowledge_vals), 0.0, 1.0)
        knowledge_span = length(sector_knowledge) / max(1, length(env.config.SECTORS))
    end

    # Knowledge deficit
    avg_knowledge_level = clamp(
        pw.avg_knowledge_personal_weight * personal_knowledge +
        pw.avg_knowledge_component_weight * Float64(info_context.avg_component_familiarity),
        0.0, 1.0,
    )
    knowledge_deficit = max(0.0, 1.0 - avg_knowledge_level)

    # Discovery momentum
    discovery_momentum = pw.discovery_momentum_base +
        pw.discovery_momentum_explore_weight * share_explore +
        pw.discovery_momentum_coverage_weight * Float64(info_context.visible_coverage)

    # Base ignorance calculation
    discovery_shortfall = max(
        0.0,
        1.0 - min(
            1.0,
            discovery_momentum +
            pw.discovery_shortfall_explore_weight * share_explore +
            pw.discovery_shortfall_sector_span_weight * Float64(info_context.visible_sector_span),
        ),
    )
    base_ignorance = (
        pw.actor_info_quality_gap_weight * (1.0 - observable_info_quality) +
        pw.actor_info_breadth_gap_weight * (1.0 - observable_info_breadth) +
        pw.actor_knowledge_deficit_weight * knowledge_deficit +
        pw.actor_discovery_shortfall_weight * discovery_shortfall +
        pw.actor_signal_gap_weight * (1.0 - Float64(info_context.avg_signal_strength))
    )

    # Ignorance reduction comes from observable evidence quality and learned
    # reliability, not raw tier capability labels.
    ignorance_reduction =
        pw.actor_info_quality_reduction * observable_info_quality +
        pw.actor_info_breadth_reduction * observable_info_breadth +
        pw.actor_evidence_reliability_reduction * observable_evidence_strength * ai_interpretation_reliability

    # Calculate knowledge gaps for visible opportunities
    knowledge_gaps = Dict{String,Float64}()
    analytical_modifier = 1 - analytical_ability * pw.analytical_gap_modifier_weight

    for (opp_idx, opp) in enumerate(visible_opportunities)
        sector = hasfield(typeof(opp), :sector) ? opp.sector : "unknown"
        sector_familiarity_value = sector_familiarity(
            sector_knowledge,
            sector;
            opportunity=opp,
            default=0.0,
        )

        # Component familiarity
        component_familiarity = sector_familiarity_value
        if hasfield(typeof(opp), :knowledge_components) && !isnothing(opp.knowledge_components)
            component_ids = Set(opp.knowledge_components)
            if !isempty(component_ids) && !isempty(agent_knowledge)
                familiar = length(intersect(agent_knowledge, component_ids))
                component_familiarity = familiar / max(1, length(component_ids))
            end
        end

        base_gap = 1.0 - max(sector_familiarity_value, component_familiarity)
        opportunity_signal = Float64(info_context.opportunity_signals[opp_idx])
        gap = clamp(
            base_gap * analytical_modifier * (1.0 - pw.opportunity_signal_gap_discount * opportunity_signal),
            0.0,
            1.0,
        )

        opp_id = hasfield(typeof(opp), :id) ? opp.id : "opp_unknown"
        if gap > 0.02
            knowledge_gaps[opp_id] = gap
        end
    end

    gap_pressure = isempty(knowledge_gaps) ? 0.0 : clamp(fast_mean(values(knowledge_gaps)), 0.0, 1.0)

    # Final ignorance level starts from the measured environment state and then
    # adjusts locally for the agent's visible evidence, knowledge, exploration,
    # and observable AI disconfirmation history.
    actor_perception_components = Dict{String,Float64}(
        "base_ignorance" => base_ignorance,
        "observable_reduction" => -ignorance_reduction,
        "exploration_reduction" => -pw.actor_explore_reduction * share_explore,
        "volatility_reduction" => -pw.actor_volatility_reduction * volatility,
        "maintain_increase" => pw.actor_maintain_increase * share_maintain,
        "personal_knowledge_reduction" => -pw.actor_personal_knowledge_reduction * personal_knowledge,
        "knowledge_span_reduction" => -pw.actor_knowledge_span_reduction * knowledge_span,
        "visible_coverage_reduction" => -pw.actor_visible_coverage_reduction * Float64(info_context.visible_coverage),
        "disconfirmation_increase" => pw.actor_disconfirmation_increase * hallucination_rate,
        "incompetence_increase" => pw.actor_incompetence_increase * (1.0 - competence_trait),
    )
    actor_local_raw = sum(values(actor_perception_components))
    actor_prior = _dimension_state_level(env, env_state, "actor_ignorance", actor_local_raw)
    actor_raw = _prior_adjusted_raw(
        actor_prior,
        actor_local_raw,
        pw.actor_prior_weight,
    )

    # Short-term buffer integration
    short_actor = _get_short_term_average(env, "actor_ignorance"; fallback=actor_raw, agent_id=agent_id)
    short_weight = clamp(
        pw.actor_short_base -
        pw.actor_short_evidence_reduction * observable_evidence_strength +
        pw.actor_short_discovery_shortfall_increase * discovery_shortfall,
        pw.actor_short_min,
        pw.actor_short_max,
    )
    actor_normalized_score = (1.0 - short_weight) * actor_raw + short_weight * short_actor
    ignorance_level, actor_bound_hit = _bounded01_with_hit(actor_normalized_score)
    _record_perception_dimension_diagnostics!(
        perception_diagnostics,
        "actor_ignorance";
        prior=actor_prior,
        local_raw=actor_local_raw,
        raw_score=actor_raw,
        normalized_score=actor_normalized_score,
        bounded_level=ignorance_level,
        bound_hit=actor_bound_hit,
    )
    _record_perception_component_diagnostics!(
        perception_diagnostics,
        "actor_ignorance",
        actor_perception_components,
    )

    # Update short-term buffer
    _update_short_term!(env, "actor_ignorance", ignorance_level; agent_id=agent_id)

    # Knowledge signal: info_quality/info_breadth are observable sufficiency
    # summaries from the visible opportunity set, not raw AI-tier parameters.
    # knowledge_gaps is locally constructed, so no copy is needed.
    knowledge_signal_struct = KnowledgeSignal(
        max(0.0, 1.0 - ignorance_level),  # confidence
        ignorance_level,
        knowledge_gaps,
        gap_pressure,
        observable_info_quality,
        observable_info_breadth,
        personal_knowledge,
        knowledge_span,
        discovery_momentum,
        volatility,
    )

    actor_ignorance_struct = ActorIgnorance(
        ignorance_level,
        ignorance_level,                  # level (alias)
        knowledge_gaps,
        observable_info_quality,          # info_sufficiency
        discovery_momentum,
        volatility,
        gap_pressure,
    )

    # ========================================================================
    # PRACTICAL INDETERMINISM PERCEPTION
    # ========================================================================

    market_volatility = Float64(market_conditions.volatility)
    regime = market_conditions.regime
    regime_uncertainty_map = Dict(
        "boom" => 0.3, "growth" => 0.2, "normal" => 0.1,
        "recession" => 0.4, "crisis" => 0.7
    )
    regime_uncertainty = get(regime_uncertainty_map, regime, 0.2)
    regime_uncertainty = clamp(
        regime_uncertainty * (1.0 - pw.practical_regime_info_reduction * observable_info_quality * ai_interpretation_reliability) +
        hallucination_rate * pw.practical_regime_disconfirmation_increase,
        0.0, 1.0,
    )

    # Path complexity
    visible_sectors = Set(hasfield(typeof(opp), :sector) ? opp.sector : "unknown" for opp in visible_opportunities)
    known_sectors = count(v -> v >= pw.known_sector_threshold, values(sector_knowledge))
    path_complexity = max(0.0, length(visible_sectors) - known_sectors) / max(1, length(env.config.SECTORS))

    # Timing pressure
    timing_pressure = Dict{String,Float64}()
    for opp in visible_opportunities
        competition = hasfield(typeof(opp), :competition) ? opp.competition : 0.0
        adoption = hasfield(typeof(opp), :market_share) ? opp.market_share : 0.0

        # The former per-stage lifecycle multiplier was excised 2026-06-09
        # (review decision #4): every opportunity stayed "emerging", whose
        # multiplier was 1.0, so it never altered urgency.
        urgency = competition * pw.timing_competition_weight + adoption * pw.timing_adoption_weight

        opp_id = hasfield(typeof(opp), :id) ? opp.id : "opp_unknown"
        timing_pressure[opp_id] = urgency
    end
    avg_timing_pressure = isempty(timing_pressure) ? 0.0 : mean(values(timing_pressure))

    # Crowding and AI diagnostics from environment state. AI usage share is
    # carried for reporting; perceived path risk uses behavioral crowding,
    # opportunity overlap, clearing pressure, and herding.
    crowding_pressure = Float64(get(env.practical_indeterminism_state, "crowding_pressure", 0.0))
    ai_pressure_level = Float64(get(env.practical_indeterminism_state, "ai_pressure", 0.0))
    opportunity_overlap = Float64(get(env.practical_indeterminism_state, "opportunity_overlap", 0.0))
    opportunity_competition = Float64(get(env.practical_indeterminism_state, "opportunity_competition", 0.0))
    ai_herding_intensity = Float64(get(env.practical_indeterminism_state, "ai_herding_intensity", 0.0))

    # Agent-specific path risk
    agent_path_risk = (
        pw.path_risk_base +
        pw.path_risk_gap_pressure_weight * gap_pressure +
        pw.path_risk_invest_share_weight * share_invest +
        pw.path_risk_innovate_over_explore_weight * max(0.0, share_innovate - share_explore)
    )
    agent_volatility_factor = clamp(abs(share_invest - share_maintain) + abs(share_innovate - share_explore), 0.0, 1.0)

    agent_uncertainty = (
        pw.agent_uncertainty_incompetence_weight * (1.0 - competence_trait) +
        pw.agent_uncertainty_knowledge_shortfall_weight * (pw.agent_uncertainty_knowledge_reference - personal_knowledge) +
        pw.agent_uncertainty_disconfirmation_weight * hallucination_rate +
        pw.agent_uncertainty_discovery_shortfall_weight * discovery_shortfall
    )

    system_term = (
        pw.system_market_volatility_weight * market_volatility +
        pw.system_regime_uncertainty_weight * regime_uncertainty +
        pw.system_timing_pressure_weight * avg_timing_pressure +
        pw.system_path_complexity_weight * path_complexity +
        pw.system_volatility_weight * volatility
    )

    crowd_term = (
        pw.crowding_pressure_weight * max(0.0, crowding_pressure) +
        pw.opportunity_overlap_weight * max(0.0, opportunity_overlap) +
        pw.opportunity_competition_weight * max(0.0, opportunity_competition) +
        pw.ai_herding_pressure_weight * ai_herding_intensity
    )

    practical_perception_components = Dict{String,Float64}(
        "agent_path_risk" => agent_path_risk,
        "agent_volatility" => pw.practical_agent_volatility_weight * agent_volatility_factor,
        "agent_uncertainty" => agent_uncertainty,
        "system_term" => system_term,
        "crowd_term" => crowd_term,
    )
    practical_local_raw = sum(values(practical_perception_components))
    practical_prior = _dimension_state_level(
        env,
        env_state,
        "practical_indeterminism",
        practical_local_raw,
    )
    practical_raw = _prior_adjusted_raw(
        practical_prior,
        practical_local_raw,
        pw.practical_prior_weight,
    )

    # Short-term buffer integration
    short_practical = _get_short_term_average(env, "practical_indeterminism"; fallback=practical_raw, agent_id=agent_id)
    short_weight_practical = clamp(
        pw.practical_short_base +
        pw.practical_short_market_volatility_weight * market_volatility +
        pw.practical_short_agent_volatility_weight * agent_volatility_factor -
        pw.practical_short_evidence_reduction * observable_evidence_strength,
        pw.practical_short_min,
        pw.practical_short_max,
    )
    practical_normalized_score = (
        (1.0 - short_weight_practical) * practical_raw +
        short_weight_practical * short_practical
    )
    indeterminism_level, practical_bound_hit = _bounded01_with_hit(practical_normalized_score)
    _record_perception_dimension_diagnostics!(
        perception_diagnostics,
        "practical_indeterminism";
        prior=practical_prior,
        local_raw=practical_local_raw,
        raw_score=practical_raw,
        normalized_score=practical_normalized_score,
        bounded_level=indeterminism_level,
        bound_hit=practical_bound_hit,
    )
    _record_perception_component_diagnostics!(
        perception_diagnostics,
        "practical_indeterminism",
        practical_perception_components,
    )

    # Update short-term buffer
    _update_short_term!(env, "practical_indeterminism", indeterminism_level; agent_id=agent_id)

    # Execution risk signal. timing_pressure is constructed locally above and
    # deliberately shared by reference between execution_risk and
    # practical_indeterminism (both point to the same Dict object).
    execution_risk_struct = ExecutionRisk(
        indeterminism_level,              # risk_level
        timing_pressure,
        regime_uncertainty,               # market_regime_risk
        volatility,
        crowding_pressure,
        ai_pressure_level,                # ai_pressure
        ai_herding_intensity,
    )

    practical_indeterminism_struct = PracticalIndeterminism(
        indeterminism_level,
        indeterminism_level,              # level (alias)
        timing_pressure,                  # timing_criticality (alias)
        timing_pressure,
        regime_uncertainty,               # market_regime_risk
        regime_uncertainty,               # regime_stability (alias)
        volatility,
        crowding_pressure,
        ai_pressure_level,
        ai_herding_intensity,
    )

    # ========================================================================
    # AGENTIC NOVELTY PERCEPTION
    # ========================================================================

    novelty_state = env.agentic_novelty_state
    recent_new_possibility_rate = Float64(get(novelty_state, "new_possibility_rate", 0.0))
    recent_innovation_births = Int(get(novelty_state, "innovation_births", 0))
    recent_new_possibilities = Int(get(novelty_state, "new_possibilities", 0))
    scarcity_signal = Float64(get(novelty_state, "recent_scarcity_signal", get(novelty_state, "scarcity_signal", 0.5)))

    novelty_level_global = Float64(get(novelty_state, "novelty_level", 0.5))
    driver_snapshot = get(novelty_state, "drivers", Dict())

    # Get base rates from driver snapshot
    base_combo_rate = Float64(get(driver_snapshot, "combo_rate", 0.0))
    base_reuse_pressure = Float64(get(driver_snapshot, "reuse_pressure", 0.0))
    base_new_poss_rate = Float64(get(driver_snapshot, "new_possibility_rate", recent_new_possibility_rate))
    base_derivative_adoption_rate = Float64(get(
        driver_snapshot,
        "derivative_adoption_rate",
        get(driver_snapshot, "adoption_rate", 0.0),
    ))

    # Tier-specific rates from tier_drivers
    tier_driver_map = get(novelty_state, "tier_drivers", Dict())
    tier_combo_rate = Float64(get(get(tier_driver_map, "combo_rate", Dict()), ai_level, base_combo_rate))
    tier_reuse_pressure = Float64(get(get(tier_driver_map, "reuse_pressure", Dict()), ai_level, base_reuse_pressure))
    tier_new_poss_rate = Float64(get(get(tier_driver_map, "new_possibility_rate", Dict()), ai_level, base_new_poss_rate))
    tier_derivative_map = get(
        tier_driver_map,
        "derivative_adoption_rate",
        get(tier_driver_map, "adoption_rate", Dict()),
    )
    tier_derivative_adoption_rate = Float64(get(tier_derivative_map, ai_level, base_derivative_adoption_rate))

    # Tier-specific novelty: new combinations/niches raise novelty; derivative
    # adoption means following known patterns and therefore lowers novelty.
    novelty_adjustment = (
        pw.novelty_combo_rate_weight * (tier_combo_rate - base_combo_rate) -
        pw.novelty_reuse_pressure_drag * (tier_reuse_pressure - base_reuse_pressure) +
        pw.novelty_new_possibility_rate_weight * (tier_new_poss_rate - base_new_poss_rate) -
        pw.novelty_derivative_adoption_drag * (tier_derivative_adoption_rate - base_derivative_adoption_rate)
    )

    novelty_prior = _dimension_state_level(
        env,
        env_state,
        "agentic_novelty",
        novelty_level_global,
    )
    novelty_perception_components = Dict{String,Float64}(
        "global_novelty" => novelty_level_global,
        "tier_driver_adjustment" => novelty_adjustment,
        "innovation_trait" => pw.novelty_innovation_trait_weight * (innovation_trait - 0.5),
        "exploration_trait" => pw.novelty_exploration_trait_weight * (exploration_trait - 0.5),
        "personal_knowledge" => pw.novelty_personal_knowledge_weight * personal_knowledge,
    )
    novelty_local_raw = sum(values(novelty_perception_components))
    novelty_raw = _prior_adjusted_raw(
        novelty_prior,
        novelty_local_raw,
        pw.novelty_prior_weight,
    )

    # Short-term buffer integration
    short_novelty = _get_short_term_average(env, "agentic_novelty"; fallback=novelty_raw, agent_id=agent_id)
    short_weight_novelty = clamp(
        pw.novelty_short_base +
        pw.novelty_short_volatility_weight * volatility +
        pw.novelty_short_scarcity_weight * scarcity_signal -
        pw.novelty_short_info_breadth_reduction * observable_info_breadth,
        pw.novelty_short_min,
        pw.novelty_short_max,
    )
    novelty_normalized_score = (
        (1.0 - short_weight_novelty) * novelty_raw +
        short_weight_novelty * short_novelty
    )
    novelty_level_agent, novelty_bound_hit = _bounded01_with_hit(novelty_normalized_score)
    _record_perception_dimension_diagnostics!(
        perception_diagnostics,
        "agentic_novelty";
        prior=novelty_prior,
        local_raw=novelty_local_raw,
        raw_score=novelty_raw,
        normalized_score=novelty_normalized_score,
        bounded_level=novelty_level_agent,
        bound_hit=novelty_bound_hit,
    )
    _record_perception_component_diagnostics!(
        perception_diagnostics,
        "agentic_novelty",
        novelty_perception_components,
    )

    # Update short-term buffer
    _update_short_term!(env, "agentic_novelty", novelty_level_agent; agent_id=agent_id)

    creative_momentum = novelty_level_agent

    # Get disruption potential from state
    disruption_potential = Dict{String,Float64}(get(novelty_state, "disruption_potential", Dict()))
    if isempty(disruption_potential) && hasfield(typeof(env.config), :SECTORS)
        for sector in env.config.SECTORS
            disruption_potential[sector] = 0.3
        end
    end

    # Innovation signal. A struct with all fields typed Float64/Int; downstream
    # consumers read it via .field access or via the dict-shim defined
    # alongside the struct in models.jl.
    innovation_signal_struct = InnovationSignal(
        novelty_level_agent,              # novelty_potential
        creative_momentum,                # creative_confidence
        scarcity_signal,                  # component_scarcity
        recent_new_possibility_rate,
        recent_innovation_births,
        recent_new_possibilities,
        tier_combo_rate,                  # combo_rate
        tier_reuse_pressure,              # reuse_pressure
        tier_derivative_adoption_rate,    # derivative adoption rate
    )

    # driver_snapshot is read out of env.agentic_novelty_state and may be
    # mutated by later rounds. Copy it to a typed dict so the Perception
    # snapshot stays stable for the lifetime of this decision; a shared mutable
    # ref would silently drift as the env updates.
    drivers_snapshot_copy = Dict{String,Any}(
        String(k) => v for (k, v) in (driver_snapshot isa Dict ? driver_snapshot : Dict())
    )

    agentic_novelty_struct = AgenticNovelty(
        novelty_level_agent,              # novelty_potential
        novelty_level_agent,              # level (alias)
        creative_momentum,                # creative_confidence
        copy(disruption_potential),       # snapshot-isolated
        volatility,
        recent_new_possibility_rate,
        recent_innovation_births,
        recent_new_possibilities,
        scarcity_signal,                  # component_scarcity
        drivers_snapshot_copy,
        tier_combo_rate,                  # combo_rate
        tier_reuse_pressure,              # reuse_pressure
        tier_derivative_adoption_rate,    # derivative adoption rate
        tier_combo_rate,
        tier_reuse_pressure,
        tier_new_poss_rate,               # tier_new_possibility_rate
        tier_derivative_adoption_rate,
        base_combo_rate,                  # global_combo_rate
        base_reuse_pressure,              # global_reuse_pressure
        novelty_adjustment,               # tier_novelty_adjustment
    )

    # ========================================================================
    # COMPETITIVE RECURSION PERCEPTION
    # ========================================================================

    # Get herding patterns from AI uncertainty signals
    herding_patterns = get(env.ai_uncertainty_signals, "ai_herding_patterns", Dict())

    # Compute herding pressure from visible opportunities
    herding_pressure_map = Dict{String,Float64}()
    for opp in visible_opportunities
        demand = hasfield(typeof(opp), :competition) ? Float64(opp.competition) : 0.0
        opp_id = hasfield(typeof(opp), :id) ? opp.id : "opp_$(objectid(opp))"
        herding_pressure_map[opp_id] = clamp(demand, 0.0, 1.0)
    end

    pressure_vals = collect(values(herding_pressure_map))
    avg_pressure = !isempty(pressure_vals) ? fast_mean(pressure_vals) : 0.0
    if !isfinite(avg_pressure)
        avg_pressure = 0.0
    end

    # Use exposure-normalized observable AI disconfirmation for market-level
    # recursion. The agent-specific hallucination_rate above is based on the
    # agent's learned suspected disconfirmation history; hidden AI telemetry is
    # diagnostic only and does not enter this behavioral channel.
    #
    # This is an AI-SPECIFIC channel, so it reads the EXCESS of the AI
    # population's forecast-disconfirmation rate over the human baseline, not
    # the overall rate. Pareto-tailed venture returns make large forecast
    # errors ubiquitous for everyone; charging the full rate to AI presence
    # would hard-wire an AI offset with no human analogue (a modeling concern). When AI forecasts are no worse than human ones this pressure is ~0.
    # F5: cached per (round, record count) — perceive_uncertainty runs once
    # per agent, and the rolling stats are a pure function of the recorded
    # exposure window, so per-agent recomputation was pure waste.
    ai_signal_stats = _rolling_ai_signal_stats_cached(env, market_conditions.round)
    ai_disconfirmation_pressure = ai_signal_stats.excess_ai_rate
    # F4: this is an AI-SPECIFIC perception component, so it reads the EXCESS
    # of the AI population's confidence miscalibration over the human
    # baseline (the pooled confidence_miscalibration field remains available
    # for symmetric/diagnostic consumers).
    confidence_miscalibration = ai_signal_stats.excess_miscalibration

    ai_herding_intensity = Float64(get(env.practical_indeterminism_state, "ai_herding_intensity", 0.0))
    capital_crowding = Float64(get(market_conditions.crowding_metrics, "crowding_index", 0.0))
    competitive_state = get(env_state, "competitive_recursion", env.competitive_recursion_state)
    practical_state = get(env_state, "practical_indeterminism", env.practical_indeterminism_state)
    perceived_opportunity_overlap = Float64(get(practical_state, "opportunity_overlap", 0.0))
    perceived_combo_reuse_pressure = Float64(get(competitive_state, "combo_reuse_pressure", 0.0))
    perceived_ai_action_correlation = Float64(get(
        competitive_state,
        "ai_action_correlation",
        pw.recursion_action_correlation_baseline,
    ))
    perceived_ai_action_correlation_excess = max(
        0.0,
        perceived_ai_action_correlation - pw.recursion_action_correlation_baseline,
    )

    # Local recursion adjusts the measured market prior for what this agent can
    # see: visible crowding, opportunity overlap, reuse, action correlation,
    # AI herding, and observable disconfirmation.
    recursion_perception_components = Dict{String,Float64}(
        "base" => pw.recursion_local_base,
        "visible_pressure" => avg_pressure * pw.recursion_visible_pressure_weight,
        "confidence_miscalibration" => confidence_miscalibration * pw.recursion_confidence_miscalibration_weight,
        "disconfirmation" => ai_disconfirmation_pressure * pw.recursion_disconfirmation_weight,
        "volatility" => volatility * pw.recursion_volatility_weight,
        "invest_share" => share_invest * pw.recursion_invest_share_weight,
        "ai_herding" => ai_herding_intensity * pw.recursion_ai_herding_weight,
        "capital_crowding" => max(0.0, capital_crowding) * pw.recursion_capital_crowding_weight,
        "opportunity_overlap" => max(0.0, perceived_opportunity_overlap) * pw.recursion_opportunity_overlap_weight,
        "combo_reuse" => max(0.0, perceived_combo_reuse_pressure) * pw.recursion_combo_reuse_weight,
        "action_correlation_excess" => perceived_ai_action_correlation_excess * pw.recursion_action_correlation_excess_weight,
        "knowledge_span_excess" => pw.recursion_knowledge_span_excess_weight * max(0.0, knowledge_span - pw.recursion_knowledge_span_threshold),
        "exploration_trait_reduction" => -pw.recursion_exploration_trait_reduction * (exploration_trait - 0.5),
        "low_knowledge" => pw.recursion_low_knowledge_weight * max(0.0, pw.recursion_low_knowledge_reference - personal_knowledge),
    )
    recursion_local_raw = sum(values(recursion_perception_components))
    recursion_prior = _dimension_state_level(
        env,
        env_state,
        "competitive_recursion",
        recursion_local_raw,
    )
    recursion_raw = _prior_adjusted_raw(
        recursion_prior,
        recursion_local_raw,
        pw.recursion_prior_weight,
    )

    # Update strategic opacity at most once per round. perceive_uncertainty is
    # called once per agent, so an unguarded update would decay the shared
    # state by decay^N_alive per round (N- and order-dependent), violating the
    # parameter's per-round decay semantics. Diagnostic-only state: the first
    # perceiving agent's signals set this round's update.
    opacity_round = Int(get(env.competitive_recursion_state, "strategic_opacity_round", -1))
    if opacity_round != market_conditions.round
        strategic_opacity = clamp(
            Float64(get(env.competitive_recursion_state, "strategic_opacity", 0.0)) * pw.strategic_opacity_decay +
            ai_disconfirmation_pressure * pw.strategic_opacity_disconfirmation_weight +
            avg_pressure * pw.strategic_opacity_pressure_weight +
            volatility * pw.strategic_opacity_volatility_weight,
            0.0, 1.0
        )
        env.competitive_recursion_state["strategic_opacity"] = strategic_opacity
        env.competitive_recursion_state["strategic_opacity_round"] = market_conditions.round
    else
        strategic_opacity = Float64(get(env.competitive_recursion_state, "strategic_opacity", 0.0))
    end

    # Short-term buffer integration
    short_recursion = _get_short_term_average(env, "competitive_recursion"; fallback=recursion_raw, agent_id=agent_id)
    recursion_short_weight = clamp(
        pw.recursion_short_base +
        pw.recursion_short_volatility_weight * volatility +
        pw.recursion_short_herding_weight * ai_herding_intensity -
        pw.recursion_short_info_quality_reduction * observable_info_quality,
        pw.recursion_short_min,
        pw.recursion_short_max,
    )
    recursion_normalized_score = (
        (1.0 - recursion_short_weight) * recursion_raw +
        recursion_short_weight * short_recursion
    )
    recursion_level, recursion_bound_hit = _bounded01_with_hit(recursion_normalized_score)
    _record_perception_dimension_diagnostics!(
        perception_diagnostics,
        "competitive_recursion";
        prior=recursion_prior,
        local_raw=recursion_local_raw,
        raw_score=recursion_raw,
        normalized_score=recursion_normalized_score,
        bounded_level=recursion_level,
        bound_hit=recursion_bound_hit,
    )
    _record_perception_component_diagnostics!(
        perception_diagnostics,
        "competitive_recursion",
        recursion_perception_components,
    )

    # Update short-term buffer
    _update_short_term!(env, "competitive_recursion", recursion_level; agent_id=agent_id)

    ai_usage_share = 1.0 - ai_shares[1]  # 1 - none share

    # Competition signal.
    competition_signal_struct = CompetitionSignal(
        recursion_level,                  # pressure_level
        avg_pressure,                     # herding_awareness
        herding_pressure_map,
        ai_usage_share,
        capital_crowding,
        ai_herding_intensity,
    )

    # ai_herding_patterns: convert the (possibly Dict{Any,Any}) generic dict
    # to typed Dict{String,Float64} once, matching the previous inline coerce.
    ai_herding_patterns_typed = Dict{String,Float64}(
        String(k) => Float64(v) for (k, v) in herding_patterns
    )

    competitive_recursion_struct = CompetitiveRecursion(
        recursion_level,
        recursion_level,                  # level (alias)
        avg_pressure,                     # herding_awareness
        herding_pressure_map,
        ai_herding_patterns_typed,
        ai_herding_intensity,
        strategic_opacity,
        volatility,
        ai_usage_share,
    )

    # Snapshot-isolate crowding_metrics with copy(). Holding the live ref to
    # market.crowding_metrics would let later market mutations leak into cached
    # Perception objects.
    crowding_metrics_snapshot = copy(market_conditions.crowding_metrics)

    action_profile_struct = Dict{String,Float64}(
        "invest" => share_invest,
        "innovate" => share_innovate,
        "explore" => share_explore,
        "maintain" => share_maintain,
    )

    # ========================================================================
    # DECISION CONFIDENCE
    # ========================================================================

    # Overall uncertainty from all dimensions.
    #
    # DESIGN STATEMENT — agentic-novelty sign (decision resolved 2026-06-09,
    # the design notes): novelty enters DECISION
    # CONFIDENCE inverted, via (1 − novelty) under the *_novelty_resolved_
    # weight. The theory: agentic novelty the agent itself perceives represents
    # opportunity space the agent has actively opened — self-authored novelty
    # is experienced as agency, not as threat, so higher perceived novelty
    # REDUCES felt uncertainty at the decision moment even though the
    # MEASUREMENT layer correctly reports novelty as an uncertainty dimension
    # (the agent's confidence and the analyst's uncertainty accounting are
    # different constructs and may legitimately diverge). This must be stated
    # explicitly in the model description, where a reader seeing this
    # composition without the design statement will call it a sign error.
    # The field name (novelty_resolved_weight) encodes the intent.
    total_uncertainty = (
        ignorance_level * pw.total_uncertainty_actor_weight +
        indeterminism_level * pw.total_uncertainty_practical_weight +
        (1.0 - novelty_level_agent) * pw.total_uncertainty_novelty_resolved_weight +
        recursion_level * pw.total_uncertainty_recursion_weight
    )

    # Calculate experience-based multipliers from recent outcomes. Venture-tail
    # outcomes feed confidence through log-multiple return evidence with
    # sample-size shrinkage; raw ROI is retained as telemetry.
    experience_stats = _recent_outcome_experience_stats(recent_outcomes)
    recent_success_rate = Float64(experience_stats["recent_success_rate"])
    ai_success_rate = Float64(experience_stats["ai_success_rate"])
    return_evidence = Float64(experience_stats["return_evidence"])

    confidence_diagnostics = _decision_confidence_diagnostics_from_components(
        total_uncertainty=total_uncertainty,
        competence_trait=competence_trait,
        recent_success_rate=recent_success_rate,
        ai_success_rate=ai_success_rate,
        avg_ai_trust=avg_ai_trust,
        ai_used=counts_as_ai_use(env.config, ai_level),
        return_evidence=return_evidence,
        risk_tolerance=risk_tolerance,
        hallucination_rate=hallucination_rate,
    )
    decision_confidence = Float64(confidence_diagnostics["decision_confidence"])
    merge!(confidence_diagnostics, perception_diagnostics)

    return Perception(
        knowledge_signal_struct,
        actor_ignorance_struct,
        execution_risk_struct,
        practical_indeterminism_struct,
        innovation_signal_struct,
        agentic_novelty_struct,
        competition_signal_struct,
        competitive_recursion_struct,
        crowding_metrics_snapshot,
        volatility,
        action_profile_struct,
        decision_confidence,
        confidence_diagnostics,
    )
end
