"""
Agent resources and decision logic for GlimpseABM.jl

This module implements the entrepreneurial agent model, including resource
management, decision-making under uncertainty, and learning from outcomes.

Note: UncertaintyResponseProfile and PerformanceTracker are defined in models.jl
"""

using Random
using Statistics
using Distributions

# ============================================================================
# HOT-PATH CONSTANTS (hoisted to avoid per-call allocation in evaluators)
# ============================================================================
# Regime-multiplier lookup used inside evaluate_opportunity_basic, a hot path
# called millions of times per run. Hoisting it to a const avoids reallocating
# the Dict on every call (which otherwise generates gigabytes of garbage).
const _REGIME_MULTIPLIER = Dict(
    "boom" => 1.2, "growth" => 1.1, "normal" => 1.0,
    "volatile" => 0.9, "crisis" => 0.7,
)

# ============================================================================
# AGENT RESOURCES
# ============================================================================

"""
Represents an agent's multidimensional assets including cognitive constraints.
"""
mutable struct AgentResources
    capital::Float64
    knowledge::Dict{String,Float64}
    capabilities::Dict{String,Float64}
    experience_units::Int
    performance::PerformanceTracker
    knowledge_last_used::Dict{String,Int}
    private_opportunity_ids::Set{String}
end

function AgentResources(capital::Float64)
    return AgentResources(
        capital,
        Dict("tech" => 0.1, "retail" => 0.1, "service" => 0.1, "manufacturing" => 0.1),
        Dict(
            "market_timing" => 0.1,
            "opportunity_evaluation" => 0.1,
            "innovation" => 0.5,
            "uncertainty_management" => 0.1
        ),
        0,
        PerformanceTracker(capital),
        Dict{String,Int}(),
        Set{String}()
    )
end

# ============================================================================
# AGENT UNCERTAINTY METRICS (Emergent, Agent-Level)
# ============================================================================

"""
Tracks agent-level uncertainty metrics that emerge from realized outcomes
rather than from environment-level formulas.

The four Knightian dimensions are computed from what actually happens to agents:
- Actor Ignorance: How wrong were the agent's estimates? (estimation error)
- Practical Indeterminism: How variable were outcomes? (return variance)
- Agentic Novelty: How much new value did the agent create? (creative output)
- Competitive Recursion: How crowded were investments? (competition experienced)
"""
mutable struct AgentUncertaintyMetrics
    # Actor Ignorance - tracks estimation accuracy
    # |estimated_return - actual_return| / max(0.01, |actual_return|)
    estimation_errors::Vector{Float64}

    # Practical Indeterminism - tracks outcome variance
    # Realized return multiples from investments
    return_history::Vector{Float64}

    # Agentic Novelty - tracks creative output
    new_combinations_created::Int
    niches_discovered::Int
    derivative_adoptions::Int  # Following existing patterns vs creating
    total_actions::Int

    # Competitive Recursion - tracks crowding experienced
    # Competition levels on opportunities when invested
    competition_levels::Vector{Float64}

    # Round tracking for temporal analysis
    round_metrics::Vector{Dict{String,Float64}}

    # Knowledge Recombination Metrics - tracks innovation quality from knowledge system
    innovation_qualities::Vector{Float64}    # Quality score (0-1) of each innovation
    innovation_novelties::Vector{Float64}    # Novelty score (0-1) of each innovation
    innovation_scarcities::Vector{Float64}   # Scarcity bonus (0-1) of each innovation
end

function AgentUncertaintyMetrics()
    return AgentUncertaintyMetrics(
        Float64[],  # estimation_errors
        Float64[],  # return_history
        0,          # new_combinations_created
        0,          # niches_discovered
        0,          # derivative_adoptions
        0,          # total_actions
        Float64[],  # competition_levels
        Dict{String,Float64}[],  # round_metrics
        Float64[],  # innovation_qualities
        Float64[],  # innovation_novelties
        Float64[]   # innovation_scarcities
    )
end

"""
Compute the four uncertainty dimensions from tracked metrics.
Returns values in [0, 1] range for comparison.
"""
function _relative_error_ignorance_score(error::Real)::Float64
    e = max(0.0, Float64(error))
    return e / (e + 2.0)
end

function _competition_recursion_score(exposure::Real)::Float64
    c = max(0.0, Float64(exposure))
    return 1.0 - exp(-c / 1.5)
end

function compute_emergent_uncertainty(metrics::AgentUncertaintyMetrics)::Dict{String,Float64}
    # Actor Ignorance: smooth transform of relative estimation error.
    # The smooth saturating map e/(e+2) preserves ordering across venture-scale
    # misses without letting a single outlier saturate the bounded diagnostic,
    # which a hard clamp at 1.0 would do.
    actor_ignorance = if isempty(metrics.estimation_errors)
        0.5  # No data, neutral
    else
        mean(_relative_error_ignorance_score.(metrics.estimation_errors))
    end

    # Practical Indeterminism: coefficient of variation of returns
    practical_indeterminism = if length(metrics.return_history) < 2
        0.5  # Not enough data
    else
        μ = mean(metrics.return_history)
        σ = std(metrics.return_history)
        # CV normalized to [0, 1] - higher variance = higher indeterminism
        clamp(σ / max(0.1, abs(μ)), 0.0, 1.0)
    end

    # Agentic Novelty: ratio of creative to total actions
    agentic_novelty = if metrics.total_actions == 0
        0.5  # No data
    else
        creative_actions = metrics.new_combinations_created + metrics.niches_discovered
        following_actions = metrics.derivative_adoptions
        total_creative_relevant = creative_actions + following_actions
        if total_creative_relevant == 0
            0.5
        else
            clamp(creative_actions / total_creative_relevant, 0.0, 1.0)
        end
    end

    # Competitive Recursion: transformed crowding exposure experienced.
    competitive_recursion = if isempty(metrics.competition_levels)
        0.0  # No investments, no recursion
    else
        # Opportunity competition is calibrated so ~1 is normal pressure and
        # ~3 is severe crowding. The 1 - exp(-c/1.5) transform keeps this
        # channel sensitive across that range rather than compressing it to
        # near zero in ordinary runs.
        mean(_competition_recursion_score.(metrics.competition_levels))
    end

    return Dict{String,Float64}(
        "actor_ignorance" => actor_ignorance,
        "practical_indeterminism" => practical_indeterminism,
        "agentic_novelty" => agentic_novelty,
        "competitive_recursion" => competitive_recursion
    )
end

"""
Observation counts behind the emergent uncertainty diagnostics.

These counts are telemetry, not uncertainty levels. They let analysis scripts
distinguish a true low/high dimension value from a neutral fallback produced by
sparse outcome evidence.
"""
function emergent_uncertainty_observation_counts(metrics::AgentUncertaintyMetrics)::Dict{String,Float64}
    creative_relevant = metrics.new_combinations_created +
        metrics.niches_discovered +
        metrics.derivative_adoptions
    return Dict{String,Float64}(
        "actor_ignorance_observations" => Float64(length(metrics.estimation_errors)),
        "practical_indeterminism_observations" => Float64(length(metrics.return_history)),
        "agentic_novelty_observations" => Float64(metrics.total_actions),
        "agentic_novelty_relevant_observations" => Float64(creative_relevant),
        "competitive_recursion_observations" => Float64(length(metrics.competition_levels)),
    )
end

"""
Record an investment outcome for uncertainty tracking.
"""
function record_investment_outcome!(
    metrics::AgentUncertaintyMetrics,
    estimated_return::Float64,
    actual_return::Float64,
    competition_level::Float64
)
    # Track estimation error (Actor Ignorance)
    if isfinite(estimated_return) && isfinite(actual_return)
        error = abs(estimated_return - actual_return) / max(0.01, abs(actual_return))
        push!(metrics.estimation_errors, max(0.0, error))
    end

    # Track return (Practical Indeterminism)
    if isfinite(actual_return)
        push!(metrics.return_history, actual_return)
    end

    # Track competition (Competitive Recursion)
    if isfinite(competition_level)
        push!(metrics.competition_levels, max(0.0, competition_level))
    end

    # Keep history bounded (last 50 investments)
    max_history = 50
    if length(metrics.estimation_errors) > max_history
        metrics.estimation_errors = metrics.estimation_errors[end-max_history+1:end]
    end
    if length(metrics.return_history) > max_history
        metrics.return_history = metrics.return_history[end-max_history+1:end]
    end
    if length(metrics.competition_levels) > max_history
        metrics.competition_levels = metrics.competition_levels[end-max_history+1:end]
    end
end

"""
Record a creative action for novelty tracking.
"""
function record_creative_action!(
    metrics::AgentUncertaintyMetrics;
    new_combination::Bool = false,
    niche_discovered::Bool = false,
    derivative_adoption::Bool = false
)
    metrics.total_actions += 1
    if new_combination
        metrics.new_combinations_created += 1
    end
    if niche_discovered
        metrics.niches_discovered += 1
    end
    if derivative_adoption
        metrics.derivative_adoptions += 1
    end
end

"""
Get the emergent uncertainty dimensions for an agent.
Returns Dict with actor_ignorance, practical_indeterminism, agentic_novelty, competitive_recursion.
"""
function get_emergent_uncertainty(agent)::Dict{String,Float64}
    return compute_emergent_uncertainty(agent.uncertainty_metrics)
end

"""
Aggregate emergent uncertainty across multiple agents by AI tier.
Returns Dict[tier => Dict[dimension => value]].
Uses fixed_ai_level if set, otherwise current_ai_level.
"""
function aggregate_emergent_uncertainty_by_tier(
    agents::Vector;
    include_dead::Bool = false,
)::Dict{String,Dict{String,Float64}}
    tier_metrics = Dict{String,Vector{Dict{String,Float64}}}()
    tier_counts = Dict{String,Dict{String,Float64}}()
    tier_alive = Dict{String,Float64}()

    for agent in agents
        if !include_dead && !agent.alive
            continue
        end
        # Use fixed_ai_level if set, otherwise current_ai_level
        tier = if !isnothing(agent.fixed_ai_level)
            agent.fixed_ai_level
        else
            agent.current_ai_level
        end
        if !haskey(tier_metrics, tier)
            tier_metrics[tier] = Dict{String,Float64}[]
        end
        push!(tier_metrics[tier], get_emergent_uncertainty(agent))

        if !haskey(tier_counts, tier)
            tier_counts[tier] = Dict{String,Float64}()
            tier_alive[tier] = 0.0
        end
        agent.alive && (tier_alive[tier] += 1.0)
        obs_counts = emergent_uncertainty_observation_counts(agent.uncertainty_metrics)
        for (key, value) in obs_counts
            tier_counts[tier][key] = get(tier_counts[tier], key, 0.0) + value
        end
    end

    # Aggregate by taking means
    result = Dict{String,Dict{String,Float64}}()
    for (tier, metrics_list) in tier_metrics
        if isempty(metrics_list)
            continue
        end
        result[tier] = Dict{String,Float64}(
            "actor_ignorance" => mean(m["actor_ignorance"] for m in metrics_list),
            "practical_indeterminism" => mean(m["practical_indeterminism"] for m in metrics_list),
            "agentic_novelty" => mean(m["agentic_novelty"] for m in metrics_list),
            "competitive_recursion" => mean(m["competitive_recursion"] for m in metrics_list),
            "n_agents" => Float64(length(metrics_list)),
            "n_alive" => Float64(get(tier_alive, tier, 0.0)),
            "n_failed" => Float64(length(metrics_list)) - Float64(get(tier_alive, tier, 0.0)),
        )
        merge!(result[tier], get(tier_counts, tier, Dict{String,Float64}()))
    end

    return result
end

# ============================================================================
# EMERGENT AGENT
# ============================================================================

"""
Tracks confidence calibration against realized outcomes.

`weighted_gap` preserves the old rolling, AI-weighted diagnostic, but the
additional sums are raw observation-level telemetry so downstream analysis can
separate sign, action channel, AI usage, and realized-return distributions.
"""
mutable struct ConfidenceOutcomeDiagnostics
    weighted_gap::Float64
    obs_count::Int
    raw_gap_sum::Float64
    abs_gap_sum::Float64
    positive_gap_count::Int
    negative_gap_count::Int
    decision_confidence_sum::Float64
    outcome_score_sum::Float64
    realized_multiple_sum::Float64
    realized_multiple_sq_sum::Float64
    investment_gap_sum::Float64
    investment_obs_count::Int
    innovate_explore_gap_sum::Float64
    innovate_explore_obs_count::Int
    ai_gap_sum::Float64
    ai_obs_count::Int
    non_ai_gap_sum::Float64
    non_ai_obs_count::Int
    realized_multiples::Vector{Float64}
end

function ConfidenceOutcomeDiagnostics()
    return ConfidenceOutcomeDiagnostics(
        0.0,  # weighted_gap
        0,    # obs_count
        0.0,  # raw_gap_sum
        0.0,  # abs_gap_sum
        0,    # positive_gap_count
        0,    # negative_gap_count
        0.0,  # decision_confidence_sum
        0.0,  # outcome_score_sum
        0.0,  # realized_multiple_sum
        0.0,  # realized_multiple_sq_sum
        0.0,  # investment_gap_sum
        0,    # investment_obs_count
        0.0,  # innovate_explore_gap_sum
        0,    # innovate_explore_obs_count
        0.0,  # ai_gap_sum
        0,    # ai_obs_count
        0.0,  # non_ai_gap_sum
        0,    # non_ai_obs_count
        Float64[]
    )
end

"""
Entrepreneurial agent with emergent AI adoption behavior.

The agent learns how to respond to uncertainty, selects AI tools adaptively,
and makes strategic decisions about investing, innovating, exploring, or
maintaining resources.
"""
mutable struct EmergentAgent
    id::Int
    resources::AgentResources
    config::EmergentConfig
    operating_cost_estimate::Float64  # Estimated operating cost per round

    # Primary sector (assigned at creation based on NVCA sector weights)
    primary_sector::String

    # Agent traits
    traits::Dict{String,Float64}
    uncertainty_tolerance::Float64
    innovativeness::Float64
    competence::Float64
    ai_trust::Float64
    trait_momentum::Float64

    # AI state
    current_ai_level::String
    fixed_ai_level::Union{String,Nothing}
    ai_learning::AILearningProfile
    ai_usage_count::Int
    ai_tier_history::Vector{String}

    # AI subscription tracking (for cost charging)
    subscription_accounts::Dict{String,Int}  # tier → rounds remaining
    subscription_rates::Dict{String,Float64}  # tier → cost per round
    subscription_deferral_remaining::Dict{String,Int}  # tier → grace rounds
    last_subscription_charge::Float64
    last_tier_review_round::Int  # round of most-recent choose_ai_level eval

    # Performance state
    alive::Bool
    survival_rounds::Int
    insolvency_rounds::Int
    equity_streak::Int  # consecutive rounds below sector equity ratio (equity_failure mechanism)
    total_invested::Float64
    total_returned::Float64
    success_count::Int              # total successes (invest + innovate)
    failure_count::Int              # total failures (invest + innovate)
    innovation_count::Int           # total innovation attempts (not just successes)
    # Channel-specific success/failure counters. success_count/failure_count
    # are kept as combined totals across both channels; these four split the
    # totals into innovation vs investment so analysis does not conflate an
    # "any success" rate with a channel-specific rate.
    innovation_success_count::Int
    innovation_failure_count::Int
    investment_success_count::Int
    investment_failure_count::Int
    failure_round::Union{Int,Nothing}  # Track when agent fails (for Kaplan-Meier survival analysis)
    failure_reason::Union{String,Nothing}  # Why agent failed (capital/insolvency)

    # Decision state
    uncertainty_response::UncertaintyResponseProfile
    action_history::Vector{String}
    last_action::String
    last_outcome::Dict{String,Any}
    recent_outcomes::Vector{Dict{String,Any}}
    action_biases::Dict{String,Float64}
    confidence_outcome_diagnostics::ConfidenceOutcomeDiagnostics

    # Portfolio
    active_investments::Vector{Dict{String,Any}}

    # Emergent uncertainty metrics (agent-level)
    uncertainty_metrics::AgentUncertaintyMetrics

    # RNG
    rng::AbstractRNG

    # Cumulative count of investments made per sector over the whole run.
    # Used downstream to compute per-tier sector-concentration HHI (the direct
    # correlated-commitment / sector-convergence diagnostic).
    sector_investment_counts::Dict{String,Int}
end

function EmergentAgent(
    id::Int,
    config::EmergentConfig;
    initial_capital::Union{Float64,Nothing} = nothing,
    primary_sector::Union{String,Nothing} = nothing,
    fixed_ai_level::Union{String,Nothing} = nothing,
    rng::AbstractRNG = Random.default_rng()
)
    # Select primary sector based on NVCA-weighted probabilities
    sector = if isnothing(primary_sector)
        _sample_sector_weighted(config, rng)
    else
        primary_sector
    end

    # Sample initial capital. Precedence:
    # 1. Explicit `initial_capital` kwarg (per-agent override)
    # 2. config.INITIAL_CAPITAL scalar when USE_UNIFORM_INITIAL_CAPITAL=true
    #    (scripts set the flag + the scalar to force identical starting capital)
    # 3. sector_profile.initial_capital_range (default sector-specific sample)
    # 4. config.INITIAL_CAPITAL_RANGE (fallback if sector profile lacks the field)
    #
    # An explicit flag (rather than a sentinel value of INITIAL_CAPITAL) opts
    # into uniform capital, so scripts can set the scalar to the default value
    # and still get uniform starting capital unambiguously.
    use_uniform_cap = getfield_default(config, :USE_UNIFORM_INITIAL_CAPITAL, false)
    capital = if !isnothing(initial_capital)
        initial_capital
    elseif use_uniform_cap
        config.INITIAL_CAPITAL
    else
        sector_profile = get(config.SECTOR_PROFILES, sector, nothing)
        if !isnothing(sector_profile) && hasproperty(sector_profile, :initial_capital_range)
            rand(rng, Uniform(sector_profile.initial_capital_range...))
        else
            rand(rng, Uniform(config.INITIAL_CAPITAL_RANGE...))
        end
    end

    # Sample traits
    traits = sample_all_traits(config; rng=rng)
    action_biases = _sample_action_biases(config, rng)

    # Create agent resources with sector-boosted knowledge
    resources = AgentResources(capital)
    # Boost knowledge in primary sector
    resources.knowledge[sector] = min(1.0, get(resources.knowledge, sector, 0.1) + 0.2)

    # Initialize operating cost estimate from sector profile
    sector_profile = get(config.SECTOR_PROFILES, sector, nothing)
    initial_cost_estimate = if !isnothing(sector_profile) && hasproperty(sector_profile, :operational_cost_range)
        (sector_profile.operational_cost_range[1] + sector_profile.operational_cost_range[2]) / 2.0
    else
        config.BASE_OPERATIONAL_COST
    end

    return EmergentAgent(
        id,
        resources,
        config,
        initial_cost_estimate,  # operating_cost_estimate
        sector,  # primary_sector
        traits,
        get(traits, "uncertainty_tolerance", 0.5),
        get(traits, "innovativeness", 0.5),
        get(traits, "competence", 0.5),
        get(traits, "ai_trust", 0.5),
        get(traits, "trait_momentum", 0.7),
        # current_ai_level mirrors fixed_ai_level for fixed-tier agents so direct
        # reads (analysis scripts, telemetry) that introspect the field classify
        # them correctly without relying on get_ai_level() to mask a "none".
        something(fixed_ai_level, "none"),
        fixed_ai_level,
        AILearningProfile(),
        0,  # ai_usage_count
        String[],  # ai_tier_history
        Dict{String,Int}(),  # subscription_accounts
        Dict{String,Float64}(),  # subscription_rates
        Dict{String,Int}(),  # subscription_deferral_remaining
        0.0,  # last_subscription_charge
        -100,  # last_tier_review_round (negative sentinel ensures first-round eval fires)
        true,  # alive
        0,  # survival_rounds
        0,  # insolvency_rounds
        0,  # equity_streak (equity_failure mechanism)
        0.0,  # total_invested
        0.0,  # total_returned
        0,  # success_count
        0,  # failure_count
        0,  # innovation_count
        0,  # innovation_success_count
        0,  # innovation_failure_count
        0,  # investment_success_count
        0,  # investment_failure_count
        nothing,  # failure_round
        nothing,  # failure_reason
        UncertaintyResponseProfile(
            rng=rng,
            learning_rate=config.LEARNING_RATE,
            initial_response_variance=config.INITIAL_RESPONSE_VARIANCE,
        ),
        String[],  # action_history
        "maintain",  # last_action
        Dict{String,Any}(),  # last_outcome
        Dict{String,Any}[],  # recent_outcomes
        action_biases,  # action_biases
        ConfidenceOutcomeDiagnostics(),  # confidence_outcome_diagnostics
        Dict{String,Any}[],  # active_investments
        AgentUncertaintyMetrics(),  # uncertainty_metrics (emergent, agent-level)
        rng,
        Dict{String,Int}(),  # sector_investment_counts
    )
end

function _sample_action_biases(config::EmergentConfig, rng::AbstractRNG)::Dict{String,Float64}
    sigma = max(0.0, Float64(getfield_default(config, :ACTION_BIAS_SIGMA, 0.05)))
    actions = ("invest", "innovate", "explore", "maintain")
    biases = Dict{String,Float64}()
    for action in actions
        biases[action] = sigma > 0.0 ? sigma * randn(rng) : 0.0
    end

    center = mean(values(biases))
    for action in actions
        biases[action] -= center
    end
    return biases
end

"""
Sample a sector based on NVCA-weighted probabilities.
"""
function _sample_sector_weighted(config::EmergentConfig, rng::AbstractRNG)::String
    sectors = collect(keys(config.SECTOR_WEIGHTS))
    weights = [config.SECTOR_WEIGHTS[s] for s in sectors]

    # Normalize weights
    total = sum(weights)
    if total <= 0
        return rand(rng, sectors)
    end

    probs = weights ./ total
    r = rand(rng)
    cumsum = 0.0
    for (i, prob) in enumerate(probs)
        cumsum += prob
        if r <= cumsum
            return sectors[i]
        end
    end
    return sectors[end]
end

"""
Get the agent's current capital.
"""
function get_capital(agent::EmergentAgent)::Float64
    return agent.resources.capital
end

"""
Set the agent's capital.
"""
function set_capital!(agent::EmergentAgent, value::Float64)
    agent.resources.capital = max(0.0, value)
end

"""
Release a dead agent's in-flight capital from opportunity capacity tracking.

`opp.total_invested` must reflect current outstanding capital (the invariant
maintained at maturity in `process_matured_investments!`). Without this
release, capital deployed by an agent who dies before maturity stays in
`total_invested` forever, inflating capacity saturation — and therefore the
convex crowding penalty on every future investor — on precisely the
opportunities where agents died. The investment records are kept (telemetry
reads portfolio size at death) but marked so the release cannot double-fire.
"""
function _release_inflight_capital_at_death!(agent::EmergentAgent)
    for investment in agent.active_investments
        get(investment, "capital_released_at_death", false) && continue
        opp = get(investment, "opportunity", nothing)
        if !isnothing(opp) && hasfield(typeof(opp), :total_invested)
            opp.total_invested = max(0.0, opp.total_invested - Float64(investment["amount"]))
        end
        investment["capital_released_at_death"] = true
    end
end

"""
Check if agent is alive and above survival threshold.
Uses sector-specific survival thresholds calibrated from BLS/Fed data.
"""
function check_survival!(agent::EmergentAgent, round::Int)::Bool
    if !agent.alive
        return false
    end

    # Use sector-specific survival threshold (BLS/Fed calibrated)
    survival_threshold = _get_sector_survival_threshold(agent)

    # Survival floors are objective constraints, not perceived reserves. AI may
    # influence upstream choices, but it should not make insolvency thresholds
    # easier by label or trust alone.
    liquidity_floor = survival_threshold

    # Diagnostic toggle: when SURVIVAL_COUNTS_INFLIGHT is true, deployed-but-
    # not-yet-matured investment capital counts toward solvency (net worth =
    # cash + in-flight) instead of gating purely on cash-on-hand. Separates
    # genuine insolvency from illiquidity.
    effective_capital = agent.resources.capital
    if getfield_default(agent.config, :SURVIVAL_COUNTS_INFLIGHT, true)
        in_flight = sum(inv["amount"] for inv in agent.active_investments; init=0.0)
        effective_capital += in_flight
    end

    if effective_capital < liquidity_floor
        agent.insolvency_rounds += 1
        if agent.insolvency_rounds >= agent.config.INSOLVENCY_GRACE_ROUNDS
            agent.alive = false
            agent.failure_round = round  # Track when agent fails (for Kaplan-Meier)
            agent.failure_reason = "liquidity_failure"  # Failed due to insufficient capital
            _release_inflight_capital_at_death!(agent)
            return false
        end
    else
        agent.insolvency_rounds = 0
        agent.survival_rounds = round
    end

    # Equity-failure check: the second leg of the failure mechanism alongside
    # liquidity failure above. An agent fails when capital_ratio (current /
    # initial) stays below the sector equity ratio for INSOLVENCY_GRACE_ROUNDS
    # consecutive rounds. Without this leg, an agent above the liquidity floor
    # would survive indefinitely no matter how much equity it had burned.
    initial_equity = max(agent.resources.performance.initial_equity, 1.0)
    capital_ratio = effective_capital / initial_equity
    sector_equity_ratio = _get_sector_equity_ratio(agent)
    ratio_floor = sector_equity_ratio
    if capital_ratio < ratio_floor
        agent.equity_streak += 1
        if agent.equity_streak >= agent.config.INSOLVENCY_GRACE_ROUNDS
            agent.alive = false
            agent.failure_round = round
            agent.failure_reason = "equity_failure"
            _release_inflight_capital_at_death!(agent)
            return false
        end
    else
        agent.equity_streak = 0
    end

    return true
end

"""
Get the survival threshold for an agent based on their primary sector.
Calibrated from BLS Business Employment Dynamics and Fed SBCS 2024.
"""
function _get_sector_survival_threshold(agent::EmergentAgent)::Float64
    # When USE_UNIFORM_SURVIVAL_THRESHOLD is true, return the scalar
    # SURVIVAL_THRESHOLD and bypass sector profiles. Otherwise sector profiles
    # supply the threshold ($700K-$2.6M depending on sector), which would
    # otherwise override a scalar that a script intended to apply uniformly.
    if getfield_default(agent.config, :USE_UNIFORM_SURVIVAL_THRESHOLD, false)
        return agent.config.SURVIVAL_THRESHOLD
    end
    sector_profile = get(agent.config.SECTOR_PROFILES, agent.primary_sector, nothing)
    if !isnothing(sector_profile) && hasproperty(sector_profile, :survival_threshold)
        return sector_profile.survival_threshold
    end
    # Fallback to global threshold
    return agent.config.SURVIVAL_THRESHOLD
end

"""
Get the minimum capital-retention ratio (capital / initial_equity) below
which the agent is considered equity-impaired. Falls back to global
SURVIVAL_CAPITAL_RATIO if the sector profile doesn't expose one.
"""
function _get_sector_equity_ratio(agent::EmergentAgent)::Float64
    sector_profile = get(agent.config.SECTOR_PROFILES, agent.primary_sector, nothing)
    if !isnothing(sector_profile) && hasproperty(sector_profile, :survival_equity_ratio)
        return sector_profile.survival_equity_ratio
    end
    return agent.config.SURVIVAL_CAPITAL_RATIO
end

"""
Get the AI level to use (fixed or adaptive).
"""
function get_ai_level(agent::EmergentAgent)::String
    if !isnothing(agent.fixed_ai_level)
        return agent.fixed_ai_level
    end
    return agent.current_ai_level
end

function _investment_derivative_probability(
    agent::EmergentAgent,
    opportunity::Opportunity
)::Float64
    # Derivative-vs-novel is a property of the opportunity and market context,
    # not of the AI tier label. AI can change which opportunities agents find
    # and rank, but the same action in the same market should not be relabeled
    # because the investor used a higher-quality model.
    competition_signal = clamp(Float64(opportunity.competition) / 2.0, 0.0, 1.0)
    path_signal = clamp(Float64(opportunity.path_dependency), 0.0, 1.0)
    novelty_signal = clamp(Float64(opportunity.novelty_score), 0.0, 1.0)
    creator_signal = (!isnothing(opportunity.created_by) && opportunity.created_by == agent.id) ? 1.0 : 0.0
    innovation_origin_signal = isnothing(opportunity.origin_innovation_id) ? 0.0 : 1.0
    combination_signal = isnothing(opportunity.combination_signature) ? 0.0 : 1.0

    derivative_pressure = 0.20 + 0.35 * competition_signal + 0.20 * path_signal
    novelty_evidence = (
        0.45 * novelty_signal +
        0.20 * creator_signal +
        0.20 * innovation_origin_signal +
        0.10 * combination_signal
    )

    return clamp(derivative_pressure - novelty_evidence, 0.02, 0.98)
end

function _base_sector_name(sector::AbstractString)::String
    return base_sector_name(sector)
end

function _recent_sector_innovation_signals(
    market::MarketEnvironment,
    base_sector::String,
    round::Int
)::NamedTuple
    recent = Innovation[]
    min_round = max(0, round - 12)
    for innovation in market.innovations
        innov_sector = isnothing(innovation.sector) ? "" : _base_sector_name(String(innovation.sector))
        if innov_sector == base_sector && innovation.round_created >= min_round
            push!(recent, innovation)
        end
    end

    n_recent = length(recent)
    if n_recent == 0
        return (
            innovation_density = 0.0,
            new_combination_rate = 0.0,
            signature_diversity = 0.0,
        )
    end

    signatures = Set{String}()
    new_combinations = 0
    for innovation in recent
        innovation.is_new_combination && (new_combinations += 1)
        if !isnothing(innovation.combination_signature)
            push!(signatures, String(innovation.combination_signature))
        end
    end

    return (
        innovation_density = clamp(1.0 - exp(-n_recent / 4.0), 0.0, 1.0),
        new_combination_rate = clamp(new_combinations / n_recent, 0.0, 1.0),
        signature_diversity = clamp(length(signatures) / n_recent, 0.0, 1.0),
    )
end

function _niche_modifier_scores(
    agent::EmergentAgent,
    market::MarketEnvironment,
    sector::String,
    visible_sectors::Vector{String},
    round::Int;
    sector_novelty::Float64,
    sector_uncertainty::Float64,
    search_effort::Float64,
    niche_creation_evidence::Float64,
    current_knowledge::Float64,
    knowledge_gap::Float64,
    exploration_tendency::Float64,
    market_awareness::Float64,
)::Dict{String,Float64}
    base_sector = _base_sector_name(sector)
    innovation = _recent_sector_innovation_signals(market, base_sector, round)
    search_signal = clamp(1.0 - exp(-max(0.0, search_effort)), 0.0, 1.0)
    evidence_signal = clamp(niche_creation_evidence / 0.35, 0.0, 1.0)
    sector_diversity = isempty(agent.config.SECTORS) ? 0.0 :
        clamp(length(unique(visible_sectors)) / length(agent.config.SECTORS), 0.0, 1.0)
    knowledge_span = isempty(agent.config.SECTORS) ? 0.0 :
        clamp(count(s -> get(agent.resources.knowledge, s, 0.0) > 0.20, agent.config.SECTORS) /
              length(agent.config.SECTORS), 0.0, 1.0)

    digital_fit = base_sector in ("tech", "service", "retail") ? 1.0 : 0.35
    sustainable_fit = base_sector in ("manufacturing", "retail", "service") ? 0.85 : 0.45
    local_fit = base_sector in ("retail", "service") ? 1.0 : 0.45

    return Dict{String,Float64}(
        "standard" => 0.08 +
            0.25 * (1.0 - sector_novelty) * (1.0 - sector_uncertainty) +
            0.18 * (1.0 - evidence_signal),
        "local" => 0.10 +
            0.28 * local_fit +
            0.22 * (1.0 - sector_diversity) +
            0.15 * current_knowledge +
            0.15 * (1.0 - search_signal),
        "budget" => 0.10 +
            0.28 * knowledge_gap +
            0.18 * (1.0 - current_knowledge) +
            0.16 * (1.0 - sector_novelty) +
            0.10 * (1.0 - search_signal),
        "digital" => 0.10 +
            0.30 * search_signal +
            0.20 * sector_diversity +
            0.18 * digital_fit +
            0.12 * innovation.innovation_density,
        "sustainable" => 0.10 +
            0.22 * sustainable_fit +
            0.20 * sector_novelty +
            0.14 * sector_uncertainty +
            0.14 * knowledge_span +
            0.10 * innovation.signature_diversity,
        "specialized" => 0.10 +
            0.30 * sector_novelty +
            0.24 * sector_uncertainty +
            0.18 * current_knowledge +
            0.16 * innovation.new_combination_rate +
            0.08 * exploration_tendency,
        # Market-segment premium, not the AI capability tier.
        "premium" => 0.10 +
            0.24 * current_knowledge +
            0.18 * market_awareness +
            0.16 * search_signal +
            0.12 * (1.0 - sector_uncertainty) +
            0.10 * innovation.innovation_density,
    )
end

function _sample_niche_modifier(scores::Dict{String,Float64}, rng::AbstractRNG)::String
    order = ["specialized", "digital", "sustainable", "premium", "local", "budget", "standard"]
    total = sum(max(0.0, get(scores, modifier, 0.0)) for modifier in order)
    total <= 0.0 && return "standard"

    draw = rand(rng) * total
    cumulative = 0.0
    for modifier in order
        cumulative += max(0.0, get(scores, modifier, 0.0))
        if draw <= cumulative
            return modifier
        end
    end
    return last(order)
end

function _select_niche_taxonomy(
    agent::EmergentAgent,
    market::MarketEnvironment,
    sector::String,
    visible_sectors::Vector{String},
    round::Int;
    sector_novelty::Float64,
    sector_uncertainty::Float64,
    search_effort::Float64,
    niche_creation_evidence::Float64,
    current_knowledge::Float64,
    knowledge_gap::Float64,
    exploration_tendency::Float64,
    market_awareness::Float64,
)::NamedTuple
    scores = _niche_modifier_scores(
        agent,
        market,
        sector,
        visible_sectors,
        round;
        sector_novelty=sector_novelty,
        sector_uncertainty=sector_uncertainty,
        search_effort=search_effort,
        niche_creation_evidence=niche_creation_evidence,
        current_knowledge=current_knowledge,
        knowledge_gap=knowledge_gap,
        exploration_tendency=exploration_tendency,
        market_awareness=market_awareness,
    )
    modifier = _sample_niche_modifier(scores, agent.rng)
    base_sector = _base_sector_name(sector)
    return (
        base_sector = base_sector,
        modifier = modifier,
        niche_sector = "$(base_sector)_$(modifier)",
        scores = scores,
    )
end

"""
Execute an action and return the outcome.
"""
function execute_action!(
    agent::EmergentAgent,
    action::String,
    market::MarketEnvironment,
    round::Int;
    opportunity::Union{Opportunity,Nothing} = nothing,
    estimated_return::Union{Float64,Nothing} = nothing,
    # (design note): the RAW instrument estimate
    # (Information.estimated_return) alongside the decision-basis
    # estimated_return, which may be strategy-shaded under S1. Accuracy and
    # learning consumers judge the instrument against THIS value; sizing and
    # scoring keep using the (possibly shaded) decision value.
    instrument_estimated_return::Union{Float64,Nothing} = nothing,
    ai_info::Union{Information,Nothing} = nothing,
    innovation_engine::Union{InnovationEngine,Nothing} = nothing,
    market_conditions::Union{MarketConditions,Nothing} = nothing,
    uncertainty_perception::Union{Perception,Nothing} = nothing,
    # confidence and signal_score flow from the decision layer (perception +
    # evaluate_portfolio_opportunities) into invest sizing.
    confidence::Union{Float64,Nothing} = nothing,
    signal_score::Union{Float64,Nothing} = nothing,
    # A2 directed creation (open-action extension): the agent's perceived
    # sector density, computed in make_decision! from its visible set only
    # when ENABLE_DIRECTED_CREATION is on; `nothing` (default) leaves
    # innovation sector selection on the reactive deterministic path.
    sector_density::Union{Dict{String,Float64},Nothing} = nothing
)::Dict{String,Any}
    raw_ai_level = get_ai_level(agent)
    behavior_tier = behavior_ai_level(agent.config, raw_ai_level)
    outcome = Dict{String,Any}(
        "action" => action,
        "agent_id" => agent.id,
        "round" => round,
        "ai_level_used" => raw_ai_level,
        "ai_behavior_level" => behavior_tier,
        "ai_used" => counts_as_ai_use(agent.config, raw_ai_level),
        "capital_before" => get_capital(agent)
    )

    if action == "invest" && !isnothing(opportunity)
        outcome = _execute_invest!(agent, opportunity, market, round, outcome;
            estimated_return=estimated_return,
            instrument_estimated_return=instrument_estimated_return,
            ai_info=ai_info,
            uncertainty_perception=uncertainty_perception,
            confidence=confidence, signal_score=signal_score)
    elseif action == "innovate"
        outcome = _execute_innovate!(agent, market, round, outcome;
            innovation_engine=innovation_engine,
            market_conditions=market_conditions,
            uncertainty_perception=uncertainty_perception,
            sector_density=sector_density)
    elseif action == "explore"
        outcome = _execute_explore!(agent, market, round, outcome)
    else  # maintain
        outcome = _execute_maintain!(agent, round, outcome)
    end

    # Record action
    push!(agent.action_history, action)
    agent.last_action = action
    agent.last_outcome = outcome

    outcome["capital_after"] = get_capital(agent)
    if action in ("innovate", "explore")
        record_recent_outcome!(agent, outcome; action=action)
    end
    return outcome
end

function _outcome_float(outcome::Dict{String,Any}, keys)::Union{Float64,Nothing}
    for key in keys
        if haskey(outcome, key)
            value = outcome[key]
            if value isa Number
                return Float64(value)
            end
            try
                return Float64(value)
            catch
                continue
            end
        end
    end
    return nothing
end

function record_recent_outcome!(
    agent::EmergentAgent,
    outcome::Dict{String,Any};
    action::Union{String,Nothing} = nothing
)::Vector{Dict{String,Any}}
    action_name = isnothing(action) ? String(get(outcome, "action", "")) : action
    if isempty(action_name)
        return agent.recent_outcomes
    end

    recent = Dict{String,Any}(
        "action" => action_name,
        "success" => Bool(get(outcome, "success", false)),
        "ai_used" => Bool(get(outcome, "ai_used", false)),
    )

    amount = _outcome_float(outcome, ("investment_amount", "amount", "rd_spend", "explore_cost"))
    if !isnothing(amount)
        recent["amount"] = amount
        if action_name == "invest"
            recent["investment_amount"] = amount
        end
    end

    returned = _outcome_float(outcome, ("capital_returned", "innovation_return", "recovery"))
    if !isnothing(returned)
        recent["capital_returned"] = returned
    end

    multiple = _outcome_float(outcome, ("return_multiple", "cash_multiple"))
    if isnothing(multiple) && !isnothing(amount) && !isnothing(returned) && amount > 0
        multiple = returned / amount
    end
    if !isnothing(multiple)
        recent["cash_multiple"] = multiple
        if action_name == "invest"
            recent["return_multiple"] = multiple
        end
    end

    # F2d: forward the censored-salvage marker (pivot records) so analysis can
    # split policy-determined recovery multiples from market-determined ones.
    if haskey(outcome, "censored_salvage")
        recent["censored_salvage"] = Bool(outcome["censored_salvage"])
    end

    push!(agent.recent_outcomes, recent)
    max_history = max(1, Int(getfield_default(agent.config, :RECENT_OUTCOME_MEMORY, 24)))
    while length(agent.recent_outcomes) > max_history
        popfirst!(agent.recent_outcomes)
    end

    return agent.recent_outcomes
end

function _execute_invest!(
    agent::EmergentAgent,
    opportunity::Opportunity,
    market::MarketEnvironment,
    round::Int,
    outcome::Dict{String,Any};
    estimated_return::Union{Float64,Nothing} = nothing,
    # F1: raw Information.estimated_return at decision time (see
    # execute_action!). Falls back to the decision-basis estimate when not
    # provided (legacy/test callers and the no-S1 case, where the two
    # coincide).
    instrument_estimated_return::Union{Float64,Nothing} = nothing,
    ai_info::Union{Information,Nothing} = nothing,
    uncertainty_perception::Union{Perception,Nothing} = nothing,
    # confidence and signal_score scale the dollar amount so that agents who
    # perceive less uncertainty (higher confidence) and higher expected quality
    # (higher signal_score) deploy more capital per bet. When either is
    # `nothing`, sizing falls back to the flat rule below.
    confidence::Union{Float64,Nothing} = nothing,
    signal_score::Union{Float64,Nothing} = nothing
)::Dict{String,Any}
    capital = get_capital(agent)
    max_fraction = agent.config.MAX_INVESTMENT_FRACTION
    target_fraction = hasfield(typeof(agent.config), :TARGET_INVESTMENT_FRACTION) ?
        agent.config.TARGET_INVESTMENT_FRACTION : max_fraction
    min_fraction = hasfield(typeof(agent.config), :MIN_FUNDING_FRACTION) ?
        agent.config.MIN_FUNDING_FRACTION : 0.25
    # High-conviction bets may exceed the standard max_fraction cap when both
    # confidence AND signal_score are well above baseline. Real founders
    # sometimes deploy 10-30% of capital on conviction plays, and the standard
    # 3.7% cap suppresses the right tail that produces venture-scale outcomes.
    max_high_conviction_fraction = hasfield(typeof(agent.config), :MAX_HIGH_CONVICTION_FRACTION) ?
        agent.config.MAX_HIGH_CONVICTION_FRACTION : 0.10
    high_conviction_threshold = hasfield(typeof(agent.config), :HIGH_CONVICTION_THRESHOLD) ?
        agent.config.HIGH_CONVICTION_THRESHOLD : 1.2

    max_invest = capital * max_fraction

    if isnothing(confidence) || isnothing(signal_score)
        # Flat sizing, used by internal callers that don't have the evaluation
        # context handy (tests, replay paths).
        invest_amount = min(max_invest, opportunity.capital_requirements)
    else
        # Confidence × signal sizing:
        #   desired = capital · target_fraction · confidence · signal_score
        #   amount  = min(desired, max_invest)
        # Confidence is clipped to [0.15, 0.95] to match the perception clip.
        c = clamp(confidence, 0.15, 0.95)
        s = max(0.0, signal_score)
        desired = capital * target_fraction * c * s
        # Lift the max-cap when both confidence and signal_score are high enough
        # to qualify as a high-conviction bet. The product (c × s) gates this so
        # small confidence or a weak signal alone cannot trigger oversizing.
        effective_max = if (c * s) > high_conviction_threshold
            capital * max_high_conviction_fraction
        else
            max_invest
        end
        invest_amount = min(desired, effective_max)

        # Minimum funding floor: if the opportunity needs a minimum stake and
        # the agent can afford it, don't under-size below the stake floor.
        required = opportunity.capital_requirements
        min_required = required > 0.0 ? required * min_fraction : 0.0
        if required > 0.0 && invest_amount < min_required && capital >= min_required
            invest_amount = min(max_invest, min_required)
        end

        invest_amount = min(invest_amount, capital)
        invest_amount = max(0.0, invest_amount)
    end

    if invest_amount <= 0 || invest_amount > capital
        outcome["success"] = false
        outcome["reason"] = "insufficient_capital"
        return outcome
    end

    # Deduct capital
    set_capital!(agent, capital - invest_amount)
    agent.total_invested += invest_amount

    # Track total investment on opportunity (for capacity constraints)
    if hasfield(typeof(opportunity), :total_invested)
        opportunity.total_invested += invest_amount
    end
    capacity_saturation_at_entry = if hasfield(typeof(opportunity), :capacity) &&
                                      hasfield(typeof(opportunity), :total_invested) &&
                                      opportunity.capacity > 0.0
        opportunity.total_invested / opportunity.capacity
    else
        0.0
    end

    # In production estimated_return is always provided by the calling
    # information path (calculate_investment_utility passes the AI-tier-noisy
    # estimate). A missing kwarg means a test/probe path; fall back to a neutral
    # 1.0 (break-even, "no expected premium") rather than leaking the hidden
    # ground truth opportunity.latent_return_potential into the recorded outcome.
    est_ret = isnothing(estimated_return) ? 1.0 : estimated_return
    # F1 raw/decision split: "estimated_return" is the DECISION basis (S1 may
    # have shaded it); "instrument_estimated_return" is what the instrument
    # actually said (Information.estimated_return). Forecast-accuracy and
    # learning consumers must judge the instrument against the raw value —
    # judging it against its own strategy-shaded transform manufactures
    # forecast error the instrument never made.
    instrument_est_ret = isnothing(instrument_estimated_return) ? est_ret :
        instrument_estimated_return
    ai_tier = get_ai_level(agent)
    behavior_tier = behavior_ai_level(agent.config, ai_tier)

    # Record investment with estimated return for uncertainty tracking.
    # Persist AI metadata (hallucination, confidence, domain) and sector into
    # active_investments so they survive to maturity. Downstream telemetry reads
    # these at maturity; without persisting them here every matured investment
    # would fall back to defaults and route all telemetry to the same bucket.
    investment = Dict{String,Any}(
        "opportunity_id" => opportunity.id,
        "opportunity" => opportunity,
        "amount" => invest_amount,
        "round_invested" => round,
        "maturity_round" => round + opportunity.time_to_maturity,
        "ai_level" => behavior_tier,
        "ai_label" => ai_tier,
        "estimated_return" => est_ret,
        "instrument_estimated_return" => instrument_est_ret,
        "competition_at_entry" => hasfield(typeof(opportunity), :competition) ? opportunity.competition : 0.0,
        "capacity_saturation_at_entry" => capacity_saturation_at_entry,
        "sector" => opportunity.sector,
        "ai_contains_hallucination" => !isnothing(ai_info) ? ai_info.contains_hallucination : false,
        "ai_confidence" => !isnothing(ai_info) ? Float64(ai_info.confidence) : 0.5,
        "ai_analysis_domain" => !isnothing(ai_info) ? something(ai_info.domain, "market_analysis") : "market_analysis",
        # Persist decision_confidence: the agent's perceived confidence at
        # decision time, distinct from ai_confidence (the AI's stated confidence
        # in its information). Paradox observation compares the agent's
        # confidence against realized ROI, so it needs this signal rather than
        # the AI's confidence in its own output.
        "decision_confidence" => isnothing(confidence) ? 0.5 : Float64(confidence),
        "uncertainty_levels_at_decision" => _perception_uncertainty_levels(uncertainty_perception),
        # Perceived market crowding at commitment (the agent's OWN
        # decision-time observable, perception.practical_indeterminism.
        # crowding_pressure). Stored so the A1 pivot review can compare
        # like-with-like (commitment-time vs current perceived crowding)
        # without reconstructing past perception. Pure stored telemetry when
        # ENABLE_PIVOT=false — nothing reads it.
        "perceived_crowding_at_commitment" => isnothing(uncertainty_perception) ?
            NaN :
            clamp(Float64(uncertainty_perception.practical_indeterminism.crowding_pressure), 0.0, 1.0),
    )
    push!(agent.active_investments, investment)
    # Cumulative per-sector investment count (per-tier sector-convergence metric).
    agent.sector_investment_counts[opportunity.sector] =
        get(agent.sector_investment_counts, opportunity.sector, 0) + 1

    # Track performance
    record_deployment!(agent.resources.performance, "invest", invest_amount;
                       ai_level=behavior_tier, round_num=round)

    outcome["success"] = true
    outcome["amount"] = invest_amount
    outcome["opportunity_id"] = opportunity.id
    outcome["chosen_opportunity_obj"] = opportunity
    outcome["maturity_round"] = investment["maturity_round"]

    # Track whether investing in a derivative known pattern vs a novel
    # opportunity. The probability is based on opportunity/market features
    # only; AI affects this indirectly through discovery and ranking.
    derivative_prob = _investment_derivative_probability(agent, opportunity)
    is_derivative = rand(agent.rng) < derivative_prob

    # Record information quality as diagnostic metadata only.
    ai_config = get(agent.config.AI_LEVELS, behavior_tier, nothing)
    info_quality = isnothing(ai_config) ? 0.25 : Float64(ai_config.info_quality)
    outcome["invested_derivative"] = is_derivative
    outcome["derivative_probability"] = derivative_prob
    outcome["competition_at_investment"] = opportunity.competition
    outcome["info_quality_used"] = info_quality
    outcome["estimated_return"] = est_ret
    outcome["instrument_estimated_return"] = instrument_est_ret
    outcome["ai_behavior_level"] = behavior_tier
    outcome["uncertainty_levels_at_decision"] = get(investment, "uncertainty_levels_at_decision", Dict{String,Float64}())

    # Propagate InformationSystem metadata for diagnostic telemetry. Behavioral
    # uncertainty pressure uses observable post-maturity disconfirmation, not
    # these hidden hallucination / actual-accuracy fields.
    if !isnothing(ai_info)
        outcome["ai_contains_hallucination"] = ai_info.contains_hallucination
        outcome["ai_confidence"] = ai_info.confidence
        outcome["ai_actual_accuracy"] = ai_info.actual_accuracy
        outcome["ai_analysis_domain"] = ai_info.domain
    end

    # Flag for uncertainty.jl filters that count "AI-using" actions.
    # Usually ai_level_used != "none" implies AI use, but sham-label placebo
    # runs intentionally keep labels while suppressing AI behavior, so this
    # defers to counts_as_ai_use rather than testing the label directly.
    outcome["ai_used"] = counts_as_ai_use(agent.config, ai_tier)

    # Track for emergent agentic novelty (derivative = following, not novel = creating)
    record_creative_action!(agent.uncertainty_metrics; derivative_adoption=is_derivative)

    return outcome
end

function _execute_innovate!(
    agent::EmergentAgent,
    market::MarketEnvironment,
    round::Int,
    outcome::Dict{String,Any};
    innovation_engine::Union{InnovationEngine,Nothing} = nothing,
    market_conditions::Union{MarketConditions,Nothing} = nothing,
    uncertainty_perception::Union{Perception,Nothing} = nothing,
    # A2 directed creation: forwarded to attempt_innovation! → sector
    # selection; `nothing` keeps the reactive deterministic mapping.
    sector_density::Union{Dict{String,Float64},Nothing} = nothing
)::Dict{String,Any}
    capital = get_capital(agent)
    ai_tier = get_ai_level(agent)
    behavior_tier = behavior_ai_level(agent.config, ai_tier)
    outcome["ai_used"] = counts_as_ai_use(agent.config, ai_tier)
    outcome["ai_behavior_level"] = behavior_tier

    # Innovation cost (R&D spend)
    rd_spend = min(
        capital * agent.config.INNOVATION_BASE_SPEND_RATIO,
        agent.config.INNOVATION_MAX_SPEND
    )

    if rd_spend > capital * 0.5
        rd_spend = capital * 0.2  # Cap R&D if capital constrained
    end

    set_capital!(agent, capital - rd_spend)
    outcome["rd_spend"] = rd_spend
    outcome["ai_tier_used"] = ai_tier

    # Charge AI analysis when innovation uses an AI behavior tier. In fixed
    # legacy pricing this is Basic's per-use fee; in hybrid/token pricing it is
    # task-specific usage for any non-none tier.
    ai_call_charge = ai_analysis_call_cost(agent, behavior_tier; action="innovation_potential")
    if ai_call_charge > 0
        set_capital!(agent, get_capital(agent) - ai_call_charge)
    end
    outcome["ai_analysis_cost"] = ai_call_charge
    outcome["ai_action_analysis_cost"] = ai_call_charge
    outcome["ai_token_cost"] = _ai_cost_model(agent.config) in ("hybrid", "token") ? ai_call_charge : 0.0

    # =========================================================================
    # USE KNOWLEDGE-BASED INNOVATION SYSTEM (if available)
    # =========================================================================
    if !isnothing(innovation_engine)
        # If the caller didn't pass a MarketConditions snapshot, derive one from
        # the live market handle. attempt_innovation! requires a MarketConditions
        # argument, so a placeholder Dict would not type-check.
        mkt_cond = isnothing(market_conditions) ? get_market_conditions(market) : market_conditions

        # Attempt to create an innovation through knowledge recombination
        innovation = attempt_innovation!(
            innovation_engine,
            agent,
            mkt_cond,
            round;
            ai_level=behavior_tier,
            uncertainty_perception=uncertainty_perception,
            decision_perception=uncertainty_perception,
            sector_density=sector_density,
            rng=agent.rng
        )

        if !isnothing(innovation)
            # Innovation was created through knowledge recombination
            # Now evaluate its success and calculate returns

            # Get market innovations for competition calculation
            market_innovations = collect(values(innovation_engine.innovations))

            # Evaluate innovation success based on quality, novelty, scarcity, competition
            success, impact, cash_multiple = evaluate_innovation_success!(
                innovation_engine,
                innovation,
                mkt_cond,
                market_innovations;
                rng=agent.rng
            )

            # Calculate actual returns based on R&D spend and knowledge-derived multiplier
            innovation_return = rd_spend * cash_multiple

            set_capital!(agent, get_capital(agent) + innovation_return)

            # innovation_count counts innovations CREATED (success+failure);
            # innovation_success_count is the success-only sub-counter.
            agent.innovation_count += 1

            if success
                agent.success_count += 1
                agent.innovation_success_count += 1  # channel-specific
                # learn_from_success! adds the innovation's components to the
                # agent's knowledge set and, for high-quality innovations,
                # creates derived knowledge that enters the shared base. The
                # agentic scarcity mechanism depends on this growth: without new
                # components, the scarcity signal saturates near zero as existing
                # components saturate in usage.
                innovation.success = true  # field is read by create_derived_knowledge
                learn_from_success!(innovation_engine.knowledge_base, agent.id, innovation)
            else
                agent.failure_count += 1
                agent.innovation_failure_count += 1  # channel-specific
                innovation.success = false
                learn_from_failure!(innovation_engine.knowledge_base, agent.id, innovation; rng=agent.rng)
            end

            # Record detailed outcome
            outcome["success"] = success
            outcome["innovation_return"] = innovation_return
            outcome["cash_multiple"] = cash_multiple
            outcome["market_impact"] = impact
            outcome["innovation_id"] = innovation.id
            outcome["innovation_type"] = innovation.type
            outcome["innovation_quality"] = innovation.quality
            outcome["innovation_novelty"] = innovation.novelty
            outcome["innovation_scarcity"] = something(innovation.scarcity, 0.5)
            outcome["is_new_combination"] = innovation.is_new_combination
            outcome["knowledge_components"] = innovation.knowledge_components
            outcome["ai_assisted"] = innovation.ai_assisted
            outcome["ai_domains_used"] = innovation.ai_domains_used
            # Propagate remaining Innovation fields so simulation.jl can fully
            # reconstruct the Innovation object without losing semantic content.
            outcome["innovation_sector"] = innovation.sector
            outcome["combination_signature"] = innovation.combination_signature

            # Track for emergent agentic novelty
            record_creative_action!(agent.uncertainty_metrics; new_combination=innovation.is_new_combination)

            # Record knowledge recombination metrics for analysis
            push!(agent.uncertainty_metrics.innovation_qualities, innovation.quality)
            push!(agent.uncertainty_metrics.innovation_novelties, innovation.novelty)
            push!(agent.uncertainty_metrics.innovation_scarcities, something(innovation.scarcity, 0.5))

            # Get AI info parameters for tracking
            ai_config = get(agent.config.AI_LEVELS, behavior_tier, nothing)
            outcome["info_quality_used"] = isnothing(ai_config) ? 0.25 : Float64(ai_config.info_quality)
            outcome["info_breadth_used"] = isnothing(ai_config) ? 0.20 : Float64(ai_config.info_breadth)
        else
            # Failed to create innovation (couldn't find good knowledge combination)
            # Partial recovery of R&D spend
            recovery = rd_spend * agent.config.INNOVATION_FAIL_RECOVERY_RATIO
            set_capital!(agent, get_capital(agent) + recovery)

            outcome["success"] = false
            outcome["recovery"] = recovery
            outcome["failure_reason"] = "no_viable_combination"
            outcome["is_new_combination"] = false

            # Count this as an innovation attempt: no Innovation object was
            # created, but the agent did select :innovate and burn R&D. Omitting
            # this increment would leave innovation_count below the count of
            # innovate action selections.
            agent.innovation_count += 1
            agent.innovation_failure_count += 1
            agent.failure_count += 1
        end
    else
        # =====================================================================
        # FALLBACK: Simple probability-based innovation (legacy behavior)
        # This path is used when innovation_engine is not provided
        # =====================================================================
        base_prob = agent.config.INNOVATION_PROBABILITY
        competence_factor = agent.competence
        innovativeness_factor = agent.innovativeness

        success_prob = base_prob * (0.5 + 0.5 * competence_factor) * (0.7 + 0.3 * innovativeness_factor)
        success = rand(agent.rng) < success_prob

        # innovation_count counts attempts; innovation_success_count is the
        # success-only sub-counter.
        agent.innovation_count += 1

        if success
            base_return = rd_spend * agent.config.INNOVATION_SUCCESS_BASE_RETURN
            multiplier = rand(agent.rng, Uniform(agent.config.INNOVATION_SUCCESS_RETURN_MULTIPLIER...))
            innovation_return = base_return * multiplier

            set_capital!(agent, get_capital(agent) + innovation_return)
            agent.innovation_success_count += 1
            agent.success_count += 1

            outcome["success"] = true
            outcome["innovation_return"] = innovation_return

            # Simple new combination tracking. This legacy fallback is used only
            # when no InnovationEngine is supplied; keep AI info fields diagnostic
            # and do not give direct tier-driven novelty probability shifts.
            ai_config = get(agent.config.AI_LEVELS, behavior_tier, nothing)
            info_quality = isnothing(ai_config) ? 0.25 : Float64(ai_config.info_quality)
            info_breadth = isnothing(ai_config) ? 0.20 : Float64(ai_config.info_breadth)

            new_combo_base = 0.22
            new_combo_prob = new_combo_base * (0.7 + 0.6 * agent.innovativeness)
            is_new_combo = rand(agent.rng) < new_combo_prob

            outcome["is_new_combination"] = is_new_combo
            outcome["info_breadth_used"] = info_breadth
            outcome["info_quality_used"] = info_quality
            outcome["fallback_mode"] = true

            record_creative_action!(agent.uncertainty_metrics; new_combination=is_new_combo)
        else
            recovery = rd_spend * agent.config.INNOVATION_FAIL_RECOVERY_RATIO
            set_capital!(agent, get_capital(agent) + recovery)

            outcome["success"] = false
            outcome["recovery"] = recovery
            outcome["fallback_mode"] = true

            # Mirror the engine path's failure accounting. innovation_count was
            # already incremented above the if-branch; this path just needs the
            # failure-side counters.
            agent.innovation_failure_count += 1
            agent.failure_count += 1
        end
    end

    record_deployment!(agent.resources.performance, "innovate", rd_spend;
                       ai_level=behavior_tier, round_num=round)
    # Record the realized cash back (innovation return, or partial R&D recovery
    # on failure) so innovate ROIC reflects actual performance. Without this,
    # compute_roic("innovate") is exactly -1.0 forever after the first attempt,
    # turning the loss-ratio penalty in calculate_innovation_utility into a
    # permanent step function instead of a learning signal.
    realized_cash = Float64(get(outcome, "innovation_return", get(outcome, "recovery", 0.0)))
    if realized_cash > 0
        record_return!(agent.resources.performance, "innovate", realized_cash;
                       ai_level=behavior_tier, round_num=round)
    end

    return outcome
end

function _execute_explore!(
    agent::EmergentAgent,
    market::MarketEnvironment,
    round::Int,
    outcome::Dict{String,Any}
)::Dict{String,Any}
    ai_tier = get_ai_level(agent)
    behavior_tier = behavior_ai_level(agent.config, ai_tier)

    # Exploration cost (minimal)
    explore_cost = get_capital(agent) * 0.01
    set_capital!(agent, get_capital(agent) - explore_cost)

    # Build visible search context without calling get_opportunities_for_agent,
    # which can create private discoveries as a side effect. Exploration cost
    # and effort should reflect what the agent can observe before this action's
    # discovery draw.
    private_ids = agent.resources.private_opportunity_ids
    visible_opps = Opportunity[]
    for opp in market.opportunities
        if opp.discovered || opp.created_by == agent.id || (opp.id in private_ids)
            push!(visible_opps, opp)
        end
    end
    visible_sectors = unique([opp.sector for opp in visible_opps if !isnothing(opp.sector)])
    visible_count = length(visible_opps)
    sector_diversity = isempty(agent.config.SECTORS) ? 0.0 :
        clamp(Float64(length(visible_sectors)) / Float64(length(agent.config.SECTORS)), 0.0, 1.0)
    visible_complexity = isempty(visible_opps) ? 0.5 :
        mean(clamp(Float64(opp.complexity), 0.0, 2.0) for opp in visible_opps)
    visible_novelty = isempty(visible_opps) ? 0.35 :
        mean(clamp(Float64(opp.novelty_score), 0.0, 1.0) for opp in visible_opps)
    visible_uncertainty = isempty(visible_opps) ? 0.35 :
        mean(clamp(Float64(opp.combination_uncertainty), 0.0, 1.0) for opp in visible_opps)
    search_context_complexity = clamp(visible_complexity + 0.5 * sector_diversity, 0.0, 2.0)

    token_usage = estimate_ai_token_usage(agent, behavior_tier;
        action="exploration",
        n_candidates=Float64(max(visible_count, 1)),
        context_complexity=search_context_complexity,
        context_novelty=visible_novelty,
        context_uncertainty=visible_uncertainty,
    )
    ai_call_charge = ai_analysis_call_cost(agent, behavior_tier;
        action="exploration",
        n_candidates=Float64(max(visible_count, 1)),
        context_complexity=search_context_complexity,
        context_novelty=visible_novelty,
        context_uncertainty=visible_uncertainty,
    )
    if ai_call_charge > 0
        set_capital!(agent, get_capital(agent) - ai_call_charge)
    end
    search_effort = log1p(get(token_usage, "effective_tokens", 0.0) / 8_000.0)

    outcome["ai_analysis_cost"] = ai_call_charge
    outcome["ai_action_analysis_cost"] = ai_call_charge
    outcome["ai_token_cost"] = _ai_cost_model(agent.config) in ("hybrid", "token") ? ai_call_charge : 0.0
    outcome["ai_effective_search_tokens"] = get(token_usage, "effective_tokens", 0.0)
    outcome["ai_search_effort"] = search_effort
    outcome["ai_search_visible_opportunities"] = visible_count
    outcome["ai_search_sector_diversity"] = sector_diversity
    outcome["ai_search_context_complexity"] = search_context_complexity
    outcome["ai_search_context_novelty"] = visible_novelty
    outcome["ai_search_context_uncertainty"] = visible_uncertainty

    # Knowledge gain
    discovery_prob = agent.config.DISCOVERY_PROBABILITY
    discovered = rand(agent.rng) < discovery_prob

    if discovered
        # Increase knowledge in a sector the agent can actually observe.
        sector_pool = visible_sectors
        if isempty(sector_pool)
            sector_pool = agent.config.SECTORS
        end
        sector = rand(agent.rng, sector_pool)
        sector_opps = [opp for opp in visible_opps if opp.sector == sector]
        sector_novelty = isempty(sector_opps) ? 0.35 : mean(clamp(Float64(opp.novelty_score), 0.0, 1.0) for opp in sector_opps)
        sector_uncertainty = isempty(sector_opps) ? 0.35 : mean(clamp(Float64(opp.combination_uncertainty), 0.0, 1.0) for opp in sector_opps)
        current_knowledge = sector_familiarity(
            agent.resources.knowledge,
            sector;
            config=agent.config,
            default=0.1,
        )
        knowledge_gap = max(0.0, 1.0 - current_knowledge)

        # Knowledge gain now emerges from action context: search effort paid for
        # by the chosen tier, visible opportunity uncertainty, agent traits, and
        # the remaining knowledge gap in the selected sector.
        exploration_tendency = clamp(Float64(get(agent.traits, "exploration_tendency", 0.5)), 0.0, 1.0)
        market_awareness = clamp(Float64(get(agent.traits, "market_awareness", 0.5)), 0.0, 1.0)
        trait_search = (
            0.45 * exploration_tendency +
            0.35 * market_awareness +
            0.20 * agent.competence
        )
        knowledge_multiplier = 0.75 +
            0.25 * trait_search +
            0.12 * search_effort +
            0.18 * knowledge_gap +
            0.10 * sector_uncertainty
        knowledge_gain = rand(agent.rng, Uniform(0.05, 0.15)) * knowledge_multiplier

        current_exact_knowledge = get(agent.resources.knowledge, sector, 0.0)
        agent.resources.knowledge[sector] = min(1.0, current_exact_knowledge + knowledge_gain)
        agent.resources.knowledge_last_used[sector] = round
        base_sector = base_sector_name(sector, agent.config.SECTORS)
        if base_sector != sector
            base_gain = 0.25 * knowledge_gain
            agent.resources.knowledge[base_sector] = min(
                1.0,
                get(agent.resources.knowledge, base_sector, 0.0) + base_gain,
            )
            agent.resources.knowledge_last_used[base_sector] = round
        end

        outcome["success"] = true
        outcome["discovered_sector"] = sector
        outcome["knowledge_gain"] = knowledge_gain
        outcome["base_sector"] = base_sector

        # Niche discovery combines serendipity (exploratory traits and novelty
        # in the visible environment) with systematic search effort. AI tiers can
        # still matter strongly, but through the analysis they actually buy.
        serendipity_factor = 0.08 *
            exploration_tendency *
            (1.0 + 0.5 * sector_novelty) *
            (1.0 + 0.2 * knowledge_gap)
        systematic_factor = 0.055 *
            search_effort *
            (0.6 + 0.4 * market_awareness) *
            (1.0 + sector_uncertainty)
        niche_discovery_base = 0.08 + serendipity_factor + systematic_factor

        # Uncertainty tolerance still affects niche discovery (risk-takers explore edges)
        niche_discovery_prob = niche_discovery_base * (0.6 + 0.8 * agent.uncertainty_tolerance)
        discovered_niche = rand(agent.rng) < min(niche_discovery_prob, 1.0)
        niche_creation_evidence = knowledge_gain +
            0.025 * search_effort +
            0.035 * sector_novelty +
            0.025 * sector_uncertainty
        created_niche = discovered_niche && niche_creation_evidence > 0.14
        outcome["discovered_niche"] = discovered_niche
        outcome["created_niche"] = created_niche
        outcome["serendipity_factor"] = serendipity_factor
        outcome["systematic_factor"] = systematic_factor
        outcome["sector_novelty_context"] = sector_novelty
        outcome["sector_uncertainty_context"] = sector_uncertainty
        outcome["niche_creation_evidence"] = niche_creation_evidence
        # Tag the outcome with exploration_type == "niche_discovery" so the
        # simulation loop fires create_niche_opportunity. The downstream trigger
        # keys on this string, not on the created_niche Bool alone.
        if created_niche
            taxonomy = _select_niche_taxonomy(
                agent,
                market,
                sector,
                visible_sectors,
                round;
                sector_novelty=sector_novelty,
                sector_uncertainty=sector_uncertainty,
                search_effort=search_effort,
                niche_creation_evidence=niche_creation_evidence,
                current_knowledge=current_knowledge,
                knowledge_gap=knowledge_gap,
                exploration_tendency=exploration_tendency,
                market_awareness=market_awareness,
            )
            outcome["exploration_type"] = "niche_discovery"
            outcome["niche_modifier"] = taxonomy.modifier
            outcome["niche_sector"] = taxonomy.niche_sector
            outcome["niche_modifier_scores"] = taxonomy.scores
            # Telemetry flag for uncertainty.jl niche-creation accounting. That
            # consumer also looks for "new_opportunity_id", which is set later by
            # the simulation loop after create_niche_opportunity succeeds (the
            # new opportunity's id isn't known until creation).
            outcome["created_opportunity"] = true
        end

        # Track for emergent agentic novelty
        record_creative_action!(agent.uncertainty_metrics; niche_discovered=created_niche)
    else
        outcome["success"] = false
        outcome["discovered_niche"] = false
        outcome["created_niche"] = false
    end

    outcome["explore_cost"] = explore_cost
    outcome["ai_used"] = counts_as_ai_use(agent.config, ai_tier)
    outcome["ai_behavior_level"] = behavior_tier
    record_deployment!(agent.resources.performance, "explore", explore_cost + ai_call_charge;
                       ai_level=behavior_tier, round_num=round)

    return outcome
end

function _execute_maintain!(
    agent::EmergentAgent,
    round::Int,
    outcome::Dict{String,Any}
)::Dict{String,Any}
    # Maintenance action - preserve capital
    outcome["success"] = true
    outcome["action_type"] = "preserve"
    return outcome
end

"""
Process matured investments and return outcomes.
"""
function process_matured_investments!(
    agent::EmergentAgent,
    market::MarketEnvironment,
    round::Int;
    market_conditions::Union{MarketConditions,Nothing} = nothing
)::Vector{Dict{String,Any}}
    if !agent.alive
        return Dict{String,Any}[]
    end

    matured_outcomes = Dict{String,Any}[]
    remaining_investments = Dict{String,Any}[]

    # Use provided market_conditions or fetch from market
    market_conditions = isnothing(market_conditions) ? get_market_conditions(market) : market_conditions

    for investment in agent.active_investments
        if investment["maturity_round"] <= round
            # Investment matures
            opp = get(investment, "opportunity", nothing)
            if isnothing(opp)
                # Skip malformed investment - push to remaining and continue
                push!(remaining_investments, investment)
                continue
            end
            invested_amount = Float64(investment["amount"])
            invested_label = String(get(investment, "ai_label", get(investment, "ai_level", get_ai_level(agent))))
            tier_used = behavior_ai_level(agent.config, String(get(investment, "ai_level", invested_label)))
            ai_used_at_commitment = counts_as_ai_use(agent.config, invested_label)

            # Calculate realized return using the tier in force when capital was
            # committed, not the agent's current tier at maturity.
            ret_multiple = realized_return(opp, market_conditions, tier_used; rng=agent.rng)
            capital_returned = invested_amount * ret_multiple

            # Apply to agent capital
            set_capital!(agent, get_capital(agent) + capital_returned)
            agent.total_returned += capital_returned

            competition_at_entry = Float64(get(investment, "competition_at_entry", 0.0))
            capacity_saturation_at_entry = Float64(get(investment, "capacity_saturation_at_entry", 0.0))
            competition_at_maturity = hasfield(typeof(opp), :competition) ? Float64(opp.competition) : 0.0
            capacity_saturation_at_maturity = if hasfield(typeof(opp), :capacity) &&
                                                  hasfield(typeof(opp), :total_invested) &&
                                                  opp.capacity > 0.0
                Float64(opp.total_invested) / Float64(opp.capacity)
            else
                0.0
            end
            crowding_exposure = max(
                competition_at_entry,
                capacity_saturation_at_entry,
                competition_at_maturity,
                capacity_saturation_at_maturity,
            )

            # Release invested capital from opportunity capacity tracking
            # (total_invested should reflect current outstanding capital, not lifetime cumulative)
            if hasfield(typeof(opp), :total_invested)
                opp.total_invested = max(0.0, opp.total_invested - invested_amount)
            end

            success = ret_multiple >= 1.0 + max(0.0, agent.config.INVESTMENT_SUCCESS_ROI_THRESHOLD)
            if success
                agent.success_count += 1
                agent.investment_success_count += 1  # channel-specific
            else
                agent.failure_count += 1
                agent.investment_failure_count += 1  # channel-specific
            end

            # Append to tier-specific ROI history, a sliding window of the last
            # 12 outcomes per tier. choose_ai_level reads this so a tier is
            # credited only for the outcomes it actually generated, not for the
            # agent's overall ROI.
            if haskey(agent.ai_learning.tier_roi_history, tier_used)
                push!(agent.ai_learning.tier_roi_history[tier_used], ret_multiple - 1.0)
                while length(agent.ai_learning.tier_roi_history[tier_used]) > 12
                    popfirst!(agent.ai_learning.tier_roi_history[tier_used])
                end
            end

            # Record return
            record_return!(agent.resources.performance, "invest", capital_returned;
                          ai_level=tier_used, round_num=round)

            # Track emergent uncertainty metrics. F1 (design note
            # 2026-06-09): estimation accuracy is judged against the RAW
            # instrument estimate (instrument_estimated_return; legacy
            # fallback to the decision-basis estimated_return). When NO
            # estimate exists at all, the outcome is excluded from
            # accuracy/disconfirmation scoring — the old `ret_multiple`
            # fallback scored missing forecasts as perfect, diluting the
            # channel. NaN makes record_investment_outcome! skip only the
            # estimation-error push (return/competition tracking is
            # estimate-independent and still recorded).
            instrument_estimate = _stored_instrument_estimate(investment)
            record_investment_outcome!(
                agent.uncertainty_metrics,
                isnothing(instrument_estimate) ? NaN : instrument_estimate,
                ret_multiple,
                crowding_exposure
            )
            # Decision-basis estimate retained for legacy consumers of the
            # matured record's "estimated_return" key (sizing semantics).
            estimated_return = Float64(get(investment, "estimated_return",
                isnothing(instrument_estimate) ? ret_multiple : instrument_estimate))

            invested_tier = tier_used
            matured_outcome = Dict{String,Any}(
                "investment" => investment,
                "investment_amount" => invested_amount,
                "capital_returned" => capital_returned,
                "return_multiple" => ret_multiple,
                "success" => success,
                "round_matured" => round,
                "estimated_return" => estimated_return,
                # F1 raw/decision split: the RAW instrument estimate (what the
                # InformationSystem said at decision time) plus an explicit
                # presence marker. Forecast-accuracy consumers
                # (_apply_matured_ai_learning!, the ai_accurate gate in
                # step!) read THIS, never the possibly-S1-shaded
                # "estimated_return"; when has_return_estimate is false the
                # outcome is excluded from disconfirmation scoring.
                "instrument_estimated_return" =>
                    isnothing(instrument_estimate) ? estimated_return : instrument_estimate,
                "has_return_estimate" => !isnothing(instrument_estimate),
                "competition_at_entry" => competition_at_entry,
                "competition_at_maturity" => competition_at_maturity,
                "capacity_saturation_at_entry" => capacity_saturation_at_entry,
                "capacity_saturation_at_maturity" => capacity_saturation_at_maturity,
                "crowding_exposure" => crowding_exposure,
                # Emit the tier in force at investment time so update_tier_belief!
                # attributes the outcome to the tier that actually made the
                # investment, not the agent's current tier.
                "ai_level" => invested_tier,
                "ai_label" => invested_label,
                # Emit ai_used so update_state_from_outcome! gates AI-trust
                # learning correctly; without it update_ai_trust! would never
                # fire for matured outcomes even when the investment used AI.
                "ai_used" => ai_used_at_commitment,
                # Forward AI metadata + sector from active_investments to the
                # matured outcome so downstream readers see real data instead of
                # defaults (otherwise hallucination detection, domain, confidence,
                # and sector all collapse to a single default bucket).
                # decision_confidence (distinct from ai_confidence) is forwarded
                # too so paradox observation reads the agent's confidence at
                # decision time, not the AI's confidence in its information.
                "ai_contains_hallucination" => Bool(get(investment, "ai_contains_hallucination", false)),
                "ai_confidence" => Float64(get(investment, "ai_confidence", 0.5)),
                "ai_analysis_domain" => String(get(investment, "ai_analysis_domain", "market_analysis")),
                "sector" => let s = get(investment, "sector", nothing)
                    isnothing(s) ? "unknown" : String(s)
                end,
                "decision_confidence" => Float64(get(investment, "decision_confidence", 0.5)),
                "uncertainty_levels_at_decision" => get(investment, "uncertainty_levels_at_decision", Dict{String,Float64}()),
            )
            record_recent_outcome!(agent, matured_outcome; action="invest")
            push!(matured_outcomes, matured_outcome)
        else
            push!(remaining_investments, investment)
        end
    end

    agent.active_investments = remaining_investments
    return matured_outcomes
end

"""
Shared observable forecast-disconfirmation scorer + recorder ( (design note)
2026-06-09). Used by BOTH resolution paths of an investment — maturity
(`_apply_matured_ai_learning!`, simulation.jl) and pivot
(`review_and_pivot_investments!`) — so a prediction is scored by exactly one
rule regardless of how the position closed.

`predicted_return` must be the RAW instrument estimate (F1:
"instrument_estimated_return", falling back to "estimated_return" for legacy
records). When it is `nothing` (no estimate was ever stored) the outcome is
EXCLUDED from disconfirmation exposure recording — scoring a missing forecast
as if the instrument predicted the realized outcome (the old ret_multiple
fallback) diluted the channel with manufactured perfect forecasts.

Returns the `observable_ai_disconfirmation` NamedTuple, or `nothing` when the
outcome was excluded. Recording into the environment is skipped when
`uncertainty_env === nothing` (direct-call/test paths); the score is still
returned so agent-side learning can proceed.
"""
function _score_and_record_forecast_disconfirmation!(
    uncertainty_env::Union{KnightianUncertaintyEnvironment,Nothing},
    agent::EmergentAgent,
    round_num::Int;
    ai_tier::String,
    domain::String,
    investment_amount::Float64,
    capital_returned::Float64,
    predicted_return::Union{Float64,Nothing},
    ai_confidence::Float64,
    success::Bool,
)
    isnothing(predicted_return) && return nothing
    config = agent.config
    ai_assisted = counts_as_ai_use(config, ai_tier)
    recent_scores = get(agent.ai_learning.accuracy_estimates, domain, Float64[])
    recent_accuracy = isempty(recent_scores) ? 0.5 :
        mean(recent_scores[max(1, length(recent_scores) - 4):end])
    analytical = Float64(get(agent.traits, "analytical_ability", 0.5))
    disconfirmation = observable_ai_disconfirmation(
        investment_amount=investment_amount,
        capital_returned=capital_returned,
        predicted_return=predicted_return,
        ai_confidence=ai_confidence,
        success=success,
        recent_accuracy=recent_accuracy,
        analytical_ability=analytical,
    )
    if !isnothing(uncertainty_env)
        record_observable_ai_disconfirmation!(
            uncertainty_env,
            round_num,
            agent.id,
            behavior_ai_level(config, ai_tier),
            domain;
            suspected_ai_failure=disconfirmation.suspected_ai_failure,
            disconfirmation_score=disconfirmation.disconfirmation_score,
            severity=disconfirmation.severity,
            return_error=disconfirmation.return_error,
            ai_confidence=ai_confidence,
            accuracy_score=disconfirmation.accuracy_score,
            success=success,
            ai_assisted=ai_assisted,
        )
    end
    return disconfirmation
end

"""
F1 fallback chain for the instrument's decision-time prediction: the RAW
instrument estimate when stored, else the decision-basis estimate (legacy
records predating the raw/decision split — identical unless S1 shaded it),
else `nothing` (no estimate at all ⇒ caller must EXCLUDE the outcome from
forecast-accuracy scoring; never fall back to the realized multiple).
"""
function _stored_instrument_estimate(record::Dict{String,Any})::Union{Float64,Nothing}
    raw = get(record, "instrument_estimated_return", nothing)
    raw isa Number && isfinite(Float64(raw)) && return Float64(raw)
    legacy = get(record, "estimated_return", nothing)
    legacy isa Number && isfinite(Float64(legacy)) && return Float64(legacy)
    return nothing
end

"""
A1 open-action pivot review (the design notes
§Open-action extension). Once per round, an ENABLE_PIVOT agent reviews its
in-flight investments and liquidates any whose decision-relevant expected
residual value has fallen below today's recoverable value plus a small
redeployment margin (exact rule: `pivot_trigger`, open_action.jl). Returns
`(n_pivots, recovered_total, committed_total, pivoted_opportunity_ids)`.

Call site: make_decision!, FIRST among the decision-economics steps — before
estimate_operational_costs and ALL action-utility calculations (design note
F3c, 2026-06-09) — so every utility sees post-pivot capital/portfolio state
consistently and recovered capital is genuinely redeployable this round.
The trigger's instrument input is the decision-time-STORED commitment
estimate (the F1 raw instrument value): a same-round re-query of the
information cache was an inconsistent privilege, since holdings outside the
agent's current visible set never had a fresh estimate to read. Simulation
Phase 2 (process_matured_investments!) has already run this round, so no
investment reviewed here can simultaneously mature — a position is paid out
at maturity OR recovered at pivot, never both (investments with
maturity_round ≤ round are skipped defensively for direct-call/test paths).
The review consumes no RNG.

A pivot mirrors maturity (design note):

  LEDGER          recovered = committed × haircut(progress) credited to
                  capital and total_returned; record_return! books it into
                  the invest channel; compile_round_stats folds pivot
                  recoveries into capital_returned["invest"].
  DISCONFIRMATION an observable forecast-disconfirmation exposure is
                  recorded against the commitment-time INSTRUMENT estimate
                  (raw, F1 fallback chain) with recovered/amount as the
                  realization — same scorer as maturity
                  (_score_and_record_forecast_disconfirmation!).
  CALIBRATION     record_confidence_outcome_observation! with the stored
                  decision confidence vs the realized recovery multiple,
                  mirroring the maturity-path call (simulation.jl Phase 2).
  TIER EVIDENCE   recovered/amount − 1 enters tier_roi_history[tier] and
                  update_tier_belief! — the position going bad IS
                  tier-relevant evidence; the haircut LEVEL is
                  policy-determined, so the records are flagged
                  "censored_salvage" => true for analysis splits
                  (documented tradeoff, approved).
  BOOLEANS        success_count/failure_count are NOT incremented and the
                  pivot record is EXCLUDED from recent_success_rate /
                  ai_success_rate (its cash multiple still enters
                  return_evidence): a censored salvage is neither a success
                  nor a maturity-grade failure in the Boolean channels.

Capital conservation mirrors maturity/death: the opportunity's
total_invested is reduced by the committed amount exactly once (the record is
marked with the same "capital_released_at_death" flag
_release_inflight_capital_at_death! checks, so a later death can never
double-release, and the record is removed from active_investments so maturity
processing can never double-pay).

When ENABLE_PIVOT=false this returns before ANY computation (bit-identical
default, pinned by test_open_action_space.jl and the open-action debug
counter). Config validation happens at initialize!
(validate_open_action_config); the previous per-agent-per-round re-validation
was removed as hot-path hygiene (design note).
"""
function review_and_pivot_investments!(
    agent::EmergentAgent,
    round::Int,
    perception::Perception;
    uncertainty_env::Union{KnightianUncertaintyEnvironment,Nothing} = nothing,
)::Tuple{Int,Float64,Float64,Set{String}}
    # Default-off short-circuit BEFORE any other work.
    agent.config.ENABLE_PIVOT || return (0, 0.0, 0.0, Set{String}())
    (!agent.alive || isempty(agent.active_investments)) &&
        return (0, 0.0, 0.0, Set{String}())

    crowding_now = clamp(
        Float64(perception.practical_indeterminism.crowding_pressure), 0.0, 1.0)

    n_pivots = 0
    recovered_total = 0.0
    committed_total = 0.0
    pivoted_ids = Set{String}()
    # Deferred survivor vector (design note): allocated only when the
    # first pivot actually fires; the overwhelmingly common no-pivot round
    # allocates nothing.
    investments = agent.active_investments
    remaining::Union{Nothing,Vector{Dict{String,Any}}} = nothing

    for idx in eachindex(investments)
        investment = investments[idx]
        opp = get(investment, "opportunity", nothing)
        amount = Float64(get(investment, "amount", 0.0))
        maturity_round = Int(get(investment, "maturity_round", round))
        round_invested = Int(get(investment, "round_invested", round))
        # Never pivot a malformed, already-released, zero-amount, or already-
        # due investment (due ones belong to maturity processing; in the
        # production flow Phase 2 has already removed them this round).
        if isnothing(opp) || amount <= 0.0 || maturity_round <= round ||
           get(investment, "capital_released_at_death", false)
            remaining === nothing || push!(remaining, investment)
            continue
        end

        time_to_maturity = max(1, maturity_round - round_invested)
        progress = clamp((round - round_invested) / time_to_maturity, 0.0, 1.0)

        # Commitment-time perceived crowding; legacy/test records without the
        # stored field measure no deterioration (conservative hold bias).
        crowding_at_commit = Float64(get(investment,
            "perceived_crowding_at_commitment", NaN))
        isfinite(crowding_at_commit) || (crowding_at_commit = crowding_now)

        # Instrument input: the stored commitment-time estimate, RAW per F1
        # (instrument_estimated_return, falling back to the decision-basis
        # estimated_return only for legacy records — never to a realized
        # multiple). The trigger and the disconfirmation exposure below
        # therefore compare in the same (raw-instrument) space.
        stored_est = _stored_instrument_estimate(investment)
        est_multiple = isnothing(stored_est) ? 1.0 : stored_est

        should_pivot, _, recoverable_per_dollar = pivot_trigger(
            agent.config;
            progress=progress,
            crowding_now=crowding_now,
            crowding_at_commit=crowding_at_commit,
            decision_confidence=Float64(get(investment, "decision_confidence", 0.5)),
            est_multiple=est_multiple,
        )
        if !should_pivot
            remaining === nothing || push!(remaining, investment)
            continue
        end
        if remaining === nothing
            # First pivot this round: snapshot the survivors seen so far.
            remaining = investments[1:idx-1]
        end

        # ── Execute the pivot (capital conservation mirrors maturity/death) ──
        recovered = amount * recoverable_per_dollar
        recovery_multiple = recovered / amount
        set_capital!(agent, get_capital(agent) + recovered)
        agent.total_returned += recovered

        # Release the committed capital from opportunity capacity tracking
        # exactly once; the A5 death-release marker makes the release
        # idempotent against any later death-path scan of a stale reference.
        if hasfield(typeof(opp), :total_invested)
            opp.total_invested = max(0.0, opp.total_invested - amount)
        end
        investment["capital_released_at_death"] = true
        investment["pivoted"] = true
        investment["pivot_round"] = round
        # Tier-evidence flag (F2d): the haircut level is policy-determined,
        # so every record carrying pivot evidence is marked for analysis
        # splits.
        investment["censored_salvage"] = true
        opp_id = String(get(investment, "opportunity_id", opp.id))
        push!(pivoted_ids, opp_id)

        invested_label = String(get(investment, "ai_label",
            get(investment, "ai_level", get_ai_level(agent))))
        tier_used = behavior_ai_level(agent.config,
            String(get(investment, "ai_level", invested_label)))
        record_return!(agent.resources.performance, "invest", recovered;
                       ai_level=tier_used, round_num=round)

        # F2b DISCONFIRMATION: the commitment-time instrument estimate is
        # scored against the realized recovery, mirroring maturity. A record
        # with no stored estimate is excluded (stored_est === nothing), per
        # the F1 no-estimate rule.
        _score_and_record_forecast_disconfirmation!(
            uncertainty_env, agent, round;
            ai_tier=invested_label,
            domain=String(get(investment, "ai_analysis_domain", "market_analysis")),
            investment_amount=amount,
            capital_returned=recovered,
            predicted_return=stored_est,
            ai_confidence=Float64(get(investment, "ai_confidence", 0.5)),
            success=false,
        )

        # F2c CALIBRATION: mirror the maturity-path confidence-outcome
        # observation (simulation.jl Phase 2; channel="investment" because a
        # pivot resolves an investment).
        record_confidence_outcome_observation!(
            agent,
            Float64(get(investment, "decision_confidence", 0.5)),
            recovery_multiple;
            ai_used=counts_as_ai_use(agent.config, invested_label),
            channel="investment")

        # F2d TIER EVIDENCE: the realized recovery multiple updates the tier
        # that made the investment, mirroring maturity (the position going
        # bad is tier-relevant evidence; the haircut level is policy, hence
        # the censored_salvage flag on the records).
        if haskey(agent.ai_learning.tier_roi_history, tier_used)
            push!(agent.ai_learning.tier_roi_history[tier_used],
                  recovery_multiple - 1.0)
            while length(agent.ai_learning.tier_roi_history[tier_used]) > 12
                popfirst!(agent.ai_learning.tier_roi_history[tier_used])
            end
        end
        update_tier_belief!(agent.ai_learning, tier_used, recovery_multiple)

        # Honest learning record: a pivot realizes recovered/amount < 1. The
        # success=false Boolean is recorded but EXCLUDED from the recent/ai
        # success-rate channels (F2e, _recent_outcome_experience_stats); the
        # cash multiple still feeds return_evidence.
        # agent.success_count/failure_count are deliberately NOT incremented —
        # those counters stay maturity-only.
        record_recent_outcome!(agent, Dict{String,Any}(
            "action" => "pivot",
            "success" => false,
            "ai_used" => counts_as_ai_use(agent.config, invested_label),
            "amount" => amount,
            "capital_returned" => recovered,
            "censored_salvage" => true,
        ); action="pivot")

        n_pivots += 1
        recovered_total += recovered
        committed_total += amount
    end

    if remaining !== nothing
        agent.active_investments = remaining
    end
    return (n_pivots, recovered_total, committed_total, pivoted_ids)
end

"""
Get a snapshot of the agent's current state.
"""
function snapshot(agent::EmergentAgent, round::Int)::Dict{String,Any}
    conf_diag = agent.confidence_outcome_diagnostics
    return Dict{String,Any}(
        "id" => agent.id,
        "round" => round,
        "capital" => get_capital(agent),
        "alive" => agent.alive,
        "ai_level" => get_ai_level(agent),
        "survival_rounds" => agent.survival_rounds,
        "success_count" => agent.success_count,
        "failure_count" => agent.failure_count,
        "innovation_count" => agent.innovation_count,
        "total_invested" => agent.total_invested,
        "total_returned" => agent.total_returned,
        "active_investments" => length(agent.active_investments),
        "uncertainty_tolerance" => agent.uncertainty_tolerance,
        "innovativeness" => agent.innovativeness,
        "competence" => agent.competence,
        "ai_trust" => agent.ai_trust,
        "confidence_outcome_weighted_gap" => conf_diag.weighted_gap,
        "confidence_outcome_obs_count" => conf_diag.obs_count,
        "confidence_outcome_raw_gap_mean" => _safe_ratio(conf_diag.raw_gap_sum, conf_diag.obs_count),
        "confidence_outcome_abs_gap_mean" => _safe_ratio(conf_diag.abs_gap_sum, conf_diag.obs_count),
    )
end

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# Note: stable_sigmoid is defined in innovation.jl

# ============================================================================
# AI PERFORMANCE METRICS
# ============================================================================

"""
Compute AI performance metrics for AI level selection.
"""
function compute_ai_performance_metrics(agent::EmergentAgent)::Dict{String,Any}
    perf = agent.resources.performance
    metrics = Dict{String,Any}()

    # Basic ROI metrics
    overall_roic = compute_roic(perf)
    invest_roic = compute_roic(perf, "invest")
    innovate_roic = compute_roic(perf, "innovate")
    explore_roic = compute_roic(perf, "explore")

    metrics["overall_roic"] = overall_roic
    metrics["invest_roic"] = invest_roic
    metrics["innovate_roic"] = innovate_roic
    metrics["explore_roic"] = explore_roic

    # Cash flow metrics
    deployed_total = get(perf.deployed_by_action, "overall", 0.0)
    returned_total = get(perf.returned_by_action, "overall", 0.0)
    metrics["overall_cash_flow_total"] = returned_total - deployed_total

    # Recent activity — count actual invest/innovate/explore actions in the
    # last 20 rounds as a true sliding window. choose_ai_level reads this to
    # estimate per-use AI cost, so it must reflect recent activity: a cumulative
    # count would let a long-idle agent look as active as a recently-active one
    # and bias per-use tier adoption estimates.
    window_size = 20
    recent_slice = agent.action_history[max(1, end - window_size + 1):end]
    recent_ai_using = count(a -> a in ("invest", "innovate", "explore"), recent_slice)
    metrics["recent_ai_activity"] = max(1.0, Float64(recent_ai_using) / 5.0)

    # ROI gain — net return on deployed capital this window.
    if deployed_total > 0
        metrics["roi_gain"] = (returned_total - deployed_total) / deployed_total
    else
        metrics["roi_gain"] = 0.0
    end

    metrics["recent_roi_gain"] = metrics["roi_gain"]
    metrics["recent_cash_flow_total"] = metrics["overall_cash_flow_total"]

    return metrics
end

# ============================================================================
# AI LEVEL SELECTION
# ============================================================================

"""
Estimate AI cost for a given level and expected calls.
"""
function _ai_cost_model(config::EmergentConfig)::String
    model = lowercase(strip(String(getfield_default(config, :AI_COST_MODEL, "hybrid"))))
    return model in ("hybrid", "token") ? model : "fixed"
end

function _config_map_value(config::EmergentConfig, field::Symbol, key::String, default::Float64)::Float64
    table = getfield_default(config, field, Dict{String,Float64}())
    if table isa AbstractDict
        return Float64(get(table, key, default))
    end
    return default
end

function _ai_subscription_cost(agent::EmergentAgent, ai_config::AILevelConfig)::Float64
    # Pure token mode is a refutation/calibration option: marginal usage costs
    # remain, but recurring seat/subscription fees are disabled.
    if _ai_cost_model(agent.config) == "token"
        return 0.0
    end
    return Float64(ai_config.cost) * Float64(agent.config.AI_COST_INTENSITY)
end

function _ai_task_key(action::String)::String
    key = lowercase(strip(action))
    if key in ("invest", "investment", "portfolio", "opportunity")
        return "market_analysis"
    elseif key in ("innovate", "innovation")
        return "innovation_potential"
    elseif key in ("explore", "discovery")
        return "exploration"
    elseif key in ("review", "tier_choice")
        return "tier_review"
    end
    return key
end

"""
Estimate token-equivalent AI usage for one analytical task.

These are abstract compute/search units used for within-model costing. They are
not a claim about literal provider tokens; tier depth, opportunity complexity,
novelty, uncertainty, and candidate-set breadth determine how much analysis is
performed before a decision.
"""
function estimate_ai_token_usage(
    agent::EmergentAgent,
    ai_level::String;
    action::String = "market_analysis",
    opportunity::Union{Opportunity,Nothing} = nothing,
    expected_calls::Float64 = 1.0,
    n_candidates::Float64 = 1.0,
    context_complexity::Union{Float64,Nothing} = nothing,
    context_novelty::Union{Float64,Nothing} = nothing,
    context_uncertainty::Union{Float64,Nothing} = nothing
)::Dict{String,Float64}
    usage = Dict{String,Float64}(
        "input_tokens" => 0.0,
        "output_tokens" => 0.0,
        "reasoning_tokens" => 0.0,
        "effective_tokens" => 0.0,
        "calls" => 0.0,
    )
    if ai_level == "none"
        return usage
    end

    tier = behavior_ai_level(agent.config, ai_level)
    if tier == "none" || !haskey(agent.config.AI_LEVELS, tier)
        return usage
    end

    calls = max(expected_calls, 0.0)
    if calls <= 0.0
        return usage
    end

    task = _ai_task_key(action)
    depth = max(0.0, _config_map_value(agent.config, :AI_TOKEN_DEPTH_MULTIPLIER, tier, 1.0))
    base_input = max(0.0, _config_map_value(agent.config, :AI_TOKEN_ACTION_BASE_INPUT, task, 4_000.0))
    base_output = max(0.0, _config_map_value(agent.config, :AI_TOKEN_ACTION_BASE_OUTPUT, task, 1_000.0))
    reasoning_factor = max(0.0, _config_map_value(agent.config, :AI_TOKEN_ACTION_REASONING_FACTOR, task, 0.5))

    complexity = !isnothing(context_complexity) ? clamp(Float64(context_complexity), 0.0, 2.0) :
        isnothing(opportunity) ? 0.5 : clamp(Float64(opportunity.complexity), 0.0, 2.0)
    novelty = !isnothing(context_novelty) ? clamp(Float64(context_novelty), 0.0, 1.0) :
        isnothing(opportunity) ? 0.35 : clamp(Float64(opportunity.novelty_score), 0.0, 1.0)
    uncertainty = !isnothing(context_uncertainty) ? clamp(Float64(context_uncertainty), 0.0, 1.0) :
        isnothing(opportunity) ? 0.35 : clamp(Float64(opportunity.combination_uncertainty), 0.0, 1.0)

    complexity_mult = max(0.0, Float64(getfield_default(agent.config, :AI_TOKEN_COMPLEXITY_MULTIPLIER, 0.35)))
    difficulty_scaling = max(0.0, Float64(getfield_default(agent.config, :AI_DIFFICULTY_COST_SCALING, 0.0)))
    novelty_mult = max(0.0, Float64(getfield_default(agent.config, :AI_TOKEN_NOVELTY_MULTIPLIER, 0.25)))
    uncertainty_mult = max(0.0, Float64(getfield_default(agent.config, :AI_TOKEN_UNCERTAINTY_MULTIPLIER, 0.20)))
    candidate_mult = max(0.0, Float64(getfield_default(agent.config, :AI_TOKEN_CANDIDATE_MULTIPLIER, 0.08)))
    output_complexity_mult = max(0.0, Float64(getfield_default(agent.config, :AI_TOKEN_OUTPUT_COMPLEXITY_MULTIPLIER, 0.15)))

    candidate_factor = 1.0 + candidate_mult * log1p(max(n_candidates, 1.0) - 1.0)
    difficulty_factor = 1.0 +
        (complexity_mult + difficulty_scaling) * complexity +
        novelty_mult * novelty +
        uncertainty_mult * uncertainty

    input_tokens = calls * base_input * depth * candidate_factor * difficulty_factor
    output_tokens = calls * base_output * sqrt(max(depth, 0.0)) * (1.0 + output_complexity_mult * complexity)
    reasoning_tokens = calls * base_input * reasoning_factor * depth * difficulty_factor
    effective_tokens = input_tokens + output_tokens + reasoning_tokens

    usage["input_tokens"] = input_tokens
    usage["output_tokens"] = output_tokens
    usage["reasoning_tokens"] = reasoning_tokens
    usage["effective_tokens"] = effective_tokens
    usage["calls"] = calls
    return usage
end

function ai_token_call_cost(
    agent::EmergentAgent,
    ai_level::String;
    action::String = "market_analysis",
    opportunity::Union{Opportunity,Nothing} = nothing,
    expected_calls::Float64 = 1.0,
    n_candidates::Float64 = 1.0,
    context_complexity::Union{Float64,Nothing} = nothing,
    context_novelty::Union{Float64,Nothing} = nothing,
    context_uncertainty::Union{Float64,Nothing} = nothing
)::Float64
    tier = behavior_ai_level(agent.config, ai_level)
    if tier == "none"
        return 0.0
    end

    usage = estimate_ai_token_usage(agent, tier;
        action=action,
        opportunity=opportunity,
        expected_calls=expected_calls,
        n_candidates=n_candidates,
        context_complexity=context_complexity,
        context_novelty=context_novelty,
        context_uncertainty=context_uncertainty,
    )

    input_price = max(0.0, _config_map_value(agent.config, :AI_TOKEN_INPUT_PRICE_PER_MILLION, tier, 0.0))
    output_price = max(0.0, _config_map_value(agent.config, :AI_TOKEN_OUTPUT_PRICE_PER_MILLION, tier, 0.0))
    reasoning_price = max(0.0, _config_map_value(agent.config, :AI_TOKEN_REASONING_PRICE_PER_MILLION, tier, 0.0))

    raw_cost = (
        usage["input_tokens"] * input_price +
        usage["output_tokens"] * output_price +
        usage["reasoning_tokens"] * reasoning_price
    ) / 1_000_000.0

    return raw_cost * Float64(agent.config.AI_COST_INTENSITY)
end

function ai_analysis_call_cost(
    agent::EmergentAgent,
    ai_level::String;
    opportunity::Union{Opportunity,Nothing} = nothing,
    expected_calls::Float64 = 1.0,
    action::String = "market_analysis",
    n_candidates::Float64 = 1.0,
    context_complexity::Union{Float64,Nothing} = nothing,
    context_novelty::Union{Float64,Nothing} = nothing,
    context_uncertainty::Union{Float64,Nothing} = nothing
)::Float64
    if ai_level == "none"
        return 0.0
    end

    tier = behavior_ai_level(agent.config, ai_level)
    if tier == "none"
        return 0.0
    end

    ai_config = get(agent.config.AI_LEVELS, tier, nothing)
    if isnothing(ai_config)
        return 0.0
    end

    calls = max(expected_calls, 0.0)
    if calls <= 0.0
        return 0.0
    end

    if _ai_cost_model(agent.config) in ("hybrid", "token")
        return ai_token_call_cost(agent, tier;
            action=action,
            opportunity=opportunity,
            expected_calls=calls,
            n_candidates=n_candidates,
            context_complexity=context_complexity,
            context_novelty=context_novelty,
            context_uncertainty=context_uncertainty,
        )
    end

    cost_intensity = Float64(agent.config.AI_COST_INTENSITY)
    difficulty_scaling = max(0.0, Float64(getfield_default(agent.config, :AI_DIFFICULTY_COST_SCALING, 0.0)))
    complexity = isnothing(opportunity) ? 0.5 : clamp(Float64(opportunity.complexity), 0.0, 2.0)
    per_use_cost = Float64(ai_config.per_use_cost) * cost_intensity

    if ai_config.cost_type == "per_use"
        return per_use_cost * calls * (1.0 + difficulty_scaling * complexity)
    elseif ai_config.cost_type == "subscription"
        return per_use_cost * calls * difficulty_scaling * complexity
    end

    return 0.0
end

function estimate_ai_cost(
    agent::EmergentAgent,
    ai_level::String;
    expected_calls::Float64 = 1.0,
    action::String = "market_analysis",
    n_candidates::Float64 = 1.0,
    opportunity::Union{Opportunity,Nothing} = nothing,
    context_complexity::Union{Float64,Nothing} = nothing,
    context_novelty::Union{Float64,Nothing} = nothing,
    context_uncertainty::Union{Float64,Nothing} = nothing
)::Float64
    if ai_level == "none"
        return 0.0
    end

    tier = behavior_ai_level(agent.config, ai_level)
    if tier == "none"
        return 0.0
    end

    ai_config = get(agent.config.AI_LEVELS, tier, nothing)
    if isnothing(ai_config)
        return 0.0
    end

    cost_type = ai_config.cost_type
    call_cost = ai_analysis_call_cost(agent, tier;
        expected_calls=expected_calls,
        action=action,
        n_candidates=n_candidates,
        opportunity=opportunity,
        context_complexity=context_complexity,
        context_novelty=context_novelty,
        context_uncertainty=context_uncertainty,
    )

    if cost_type == "subscription"
        # Hybrid mode: monthly seat fee plus task-specific usage. Fixed mode
        # preserves legacy behavior because call_cost is zero for subscription
        # tiers unless AI_DIFFICULTY_COST_SCALING is explicitly enabled.
        return _ai_subscription_cost(agent, ai_config) + call_cost
    elseif cost_type == "per_use"
        return call_cost
    end

    return call_cost
end

function strategic_anticipation_multiplier(
    agent::EmergentAgent,
    opp::Opportunity,
    ai_level::String,
    info_quality::Float64,
    perception::Perception,
    early_signal_count::Int
)::Float64
    enabled = getfield_default(agent.config, :STRATEGIC_ANTICIPATION_ENABLED, false)
    strength = Float64(getfield_default(agent.config, :STRATEGIC_ANTICIPATION_STRENGTH, 0.0))
    if !enabled || strength <= 0.0 || ai_level == "none"
        return 1.0
    end

    recursion = clamp(Float64(perception.competitive_recursion.level), 0.0, 1.0)
    crowding = clamp(Float64(get(perception.crowding_metrics, "crowding_index", 0.25)), 0.0, 1.0)
    comp_scale = max(1.0, Float64(getfield_default(agent.config, :DISRUPTION_COMPETITION_THRESHOLD, 10.0)))
    visible_competition = clamp(Float64(opp.competition) / comp_scale, 0.0, 1.0)
    signal_pressure = clamp(Float64(early_signal_count) / 5.0, 0.0, 1.0)

    same_signal_risk = clamp(
        0.50 * recursion + 0.25 * visible_competition + 0.15 * crowding + 0.10 * signal_pressure,
        0.0,
        1.0
    )
    analytical = clamp(Float64(get(agent.traits, "analytical_ability", 0.5)), 0.0, 1.0)
    tolerance = clamp(Float64(get(agent.traits, "uncertainty_tolerance", 0.5)), 0.0, 1.0)
    anticipation = strength * (0.5 + 0.5 * clamp(info_quality, 0.0, 1.0)) *
        (0.75 + 0.25 * analytical) * (1.0 - 0.25 * tolerance)

    return clamp(1.0 - anticipation * same_signal_risk, 0.35, 1.05)
end

function select_portfolio_evaluation(
    agent::EmergentAgent,
    evaluations::Vector{OpportunityEvaluation},
    ai_level::String,
    perception::Perception
)::Union{OpportunityEvaluation,Nothing}
    isempty(evaluations) && return nothing

    enabled = getfield_default(agent.config, :STRATEGIC_DIVERSIFICATION_ENABLED, false)
    strength = Float64(getfield_default(agent.config, :STRATEGIC_DIVERSIFICATION_STRENGTH, 0.0))
    if !enabled || strength <= 0.0 || ai_level == "none"
        return evaluations[1]
    end

    ai_config = get(agent.config.AI_LEVELS, ai_level, nothing)
    info_quality = isnothing(ai_config) ? 0.25 : Float64(ai_config.info_quality)
    top_k = max(1, Int(getfield_default(agent.config, :STRATEGIC_DIVERSIFICATION_TOP_K, 5)))
    score_band = clamp(Float64(getfield_default(agent.config, :STRATEGIC_DIVERSIFICATION_SCORE_BAND, 0.10)), 0.0, 1.0)

    best = evaluations[1]
    best_score = max(best.final_score, 1e-9)
    n = min(top_k, length(evaluations))
    candidates = OpportunityEvaluation[]
    floor_score = best_score * (1.0 - score_band)
    for ev in @view evaluations[1:n]
        if ev.final_score >= floor_score
            push!(candidates, ev)
        end
    end
    length(candidates) <= 1 && return best

    recursion = clamp(Float64(perception.competitive_recursion.level), 0.0, 1.0)
    crowding = clamp(Float64(get(perception.crowding_metrics, "crowding_index", 0.25)), 0.0, 1.0)
    comp_scale = max(1.0, Float64(getfield_default(agent.config, :DISRUPTION_COMPETITION_THRESHOLD, 10.0)))
    visible_competition = mean(clamp(ev.competition_at_evaluation / comp_scale, 0.0, 1.0) for ev in candidates)
    same_signal_pressure = clamp(0.55 * recursion + 0.25 * crowding + 0.20 * visible_competition, 0.0, 1.0)
    diversify_prob = clamp(strength * (0.5 + 0.5 * info_quality) * same_signal_pressure, 0.0, 1.0)

    if rand(agent.rng) >= diversify_prob
        return best
    end

    temperature = max(1e-6, best_score * (0.08 + 0.35 * diversify_prob))
    weights = Float64[]
    for ev in candidates
        score_weight = exp((ev.final_score - best_score) / temperature)
        crowd_weight = (1.0 + max(0.0, ev.competition_at_evaluation))^(-diversify_prob)
        push!(weights, max(score_weight * crowd_weight, 1e-9))
    end

    total = sum(weights)
    draw = rand(agent.rng) * total
    cumulative = 0.0
    for (idx, weight) in enumerate(weights)
        cumulative += weight
        if draw <= cumulative
            return candidates[idx]
        end
    end

    return candidates[end]
end

"""
Start a subscription schedule for a given AI level.

Charges the documented monthly subscription cost once per round. The simulation
runs on a monthly cadence (N_ROUNDS = 120 months = 10 years; operational costs
and discount rates throughout are explicitly monthly), so each round is one
month of subscription and a \$3500/month tier charges \$3500 per round (no
quarterly conversion).

In hybrid cost mode this subscription charge is combined with task-specific
token-equivalent usage costs. In pure token mode, subscription schedules are
not started.
"""
function start_subscription_schedule!(
    agent::EmergentAgent,
    ai_level::String,
    ai_config::AILevelConfig
)
    monthly_cost = _ai_subscription_cost(agent, ai_config)

    if monthly_cost <= 0
        return
    end

    # Set up subscription to charge EVERY round (no amortization)
    # Each round = one month per config.N_ROUNDS documentation
    agent.subscription_accounts[ai_level] = 1  # Active subscription
    agent.subscription_rates[ai_level] = monthly_cost  # Monthly cost per round
    agent.subscription_deferral_remaining[ai_level] = 0
end

"""
Charge one subscription installment for a given AI level.
Deducts the documented monthly cost from agent capital each round.
Returns the amount charged.
"""
function charge_subscription_installment!(
    agent::EmergentAgent,
    ai_level::String
)::Float64
    active = get(agent.subscription_accounts, ai_level, 0)
    rate = get(agent.subscription_rates, ai_level, 0.0)

    if active <= 0 || rate <= 0
        return 0.0
    end

    # Check deferral (credit line grace period)
    deferral = get(agent.subscription_deferral_remaining, ai_level, 0)
    if deferral > 0
        agent.subscription_deferral_remaining[ai_level] = deferral - 1
        return 0.0
    end

    # CHARGE THE SUBSCRIPTION - deduct documented monthly cost from capital
    set_capital!(agent, get_capital(agent) - rate)

    # Subscription remains active (charge EVERY round, no amortization)
    # subscription_accounts[ai_level] stays at 1 to indicate active subscription

    return rate
end

"""
Apply subscription charges for all active AI subscriptions.
Called once per round to charge all subscription installments.
Returns total amount charged.
"""
function apply_subscription_carry!(
    agent::EmergentAgent,
    round::Int
)::Float64
    total = 0.0

    # Agents pay subscription costs from round 1, with no credit-line grace
    # period. Planning (estimate_ai_cost, choose_ai_level) already charges the
    # full monthly cost from round 1, so waiving early charges here would create
    # a plan-vs-bill inconsistency that distorts early-round tier selection.
    # Runway comes from initial capital (the series-A/B 24-36 month calibration),
    # not from free AI.
    for level in collect(keys(agent.subscription_accounts))
        total += charge_subscription_installment!(agent, level)
    end

    agent.last_subscription_charge = total
    return total
end

"""
Cancel the subscription schedule for a specific AI level.
"""
function cancel_subscription_schedule!(agent::EmergentAgent, ai_level::String)
    delete!(agent.subscription_accounts, ai_level)
    delete!(agent.subscription_rates, ai_level)
    delete!(agent.subscription_deferral_remaining, ai_level)
end

"""
Ensure subscription schedule is active for the given AI level.
Call this whenever an agent adopts or switches to a subscription-based AI tier.

For emergent-mode agents, this also cancels any active subscription to a
DIFFERENT tier, so an agent who drops from premium to advanced stops being
charged for premium and never accumulates simultaneous subscriptions across
tiers.
"""
function ensure_subscription_schedule!(
    agent::EmergentAgent,
    ai_level::String
)
    ai_config = get(agent.config.AI_LEVELS, ai_level, nothing)

    if isnothing(ai_config)
        return
    end

    # Cancel any active subscription to a DIFFERENT tier.
    for active_tier in collect(keys(agent.subscription_accounts))
        if active_tier != ai_level && get(agent.subscription_accounts, active_tier, 0) > 0
            cancel_subscription_schedule!(agent, active_tier)
        end
    end

    if ai_config.cost_type != "subscription" || ai_config.cost <= 0
        return
    end

    # Start schedule if not already active
    if get(agent.subscription_accounts, ai_level, 0) <= 0
        start_subscription_schedule!(agent, ai_level, ai_config)
    end
end

"""
Compute penalty for switching AI levels.

The penalty is asymmetric: downgrades carry larger friction than upgrades.
Real downgrades involve data/integration loss and the social signal of
"giving up on the expensive tool", whereas upgrades have a learning-curve cost
but typically come with vendor onboarding. Following the loss-aversion
literature, the per-step downgrade cost is set at roughly 2× the per-step
upgrade cost.
"""
function compute_ai_switch_penalty(
    agent::EmergentAgent,
    current_level::String,
    proposed_level::String
)::Float64
    if proposed_level == current_level
        return 0.0
    end

    order = ["none", "basic", "advanced", "premium"]
    current_idx = findfirst(==(current_level), order)
    proposed_idx = findfirst(==(proposed_level), order)

    if isnothing(current_idx) || isnothing(proposed_idx)
        return 0.1
    end

    distance = abs(proposed_idx - current_idx)
    # Asymmetric base penalty: downgrades cost more per step than upgrades.
    is_downgrade = proposed_idx < current_idx
    per_step = is_downgrade ? 0.10 : 0.05
    base_penalty = per_step * distance + 0.03

    # Experience with proposed level reduces penalty
    experience = count(==(proposed_level), agent.ai_tier_history)
    experience_modifier = 1.0 / (1.0 + 0.15 * experience)

    # Fatigue from frequent switching
    n_switches = 0
    for i in 2:length(agent.ai_tier_history)
        if agent.ai_tier_history[i] != agent.ai_tier_history[i-1]
            n_switches += 1
        end
    end
    fatigue_penalty = 0.02 * max(0, n_switches - 3)

    learning_relief = clamp(log1p(experience) * 0.015, 0.0, 0.1)

    return max(0.0, (base_penalty * experience_modifier) + fatigue_penalty - learning_relief)
end

"""
Choose AI level by scoring each tier on Bayesian beliefs, performance metrics,
peer signals, and cost, then selecting the highest-scoring tier. The scoring
model combines:

  - Cost: cost_term is scaled to the agent's monthly operating cost so price is
    felt as multiples of burn rate (e.g. the frontier tier's cost_term is
    comparable in magnitude to its trust_term).
  - Neutral priors: tier posteriors are read from ai_learning.tier_beliefs,
    which start at a neutral 0.50 for every tier (DEFAULT_TIER_PRIORS in
    models.jl), so tier preferences emerge from data rather than a prior bias.
  - Status-quo tie-breaking: ties are resolved in favor of the agent's current
    tier rather than the lowest-indexed tier.
  - Tier-specific ROI: roi_term uses each tier's own roi_history (maintained in
    ai_learning.tier_roi_history by process_matured_investments!), so a tier is
    credited only for the outcomes it generated.
  - Sticky re-evaluation: the tier is re-decided only every
    AI_TIER_REVIEW_INTERVAL rounds, matching how infrequently real subscribers
    reconsider a subscription.
  - Peer effects: observation of neighbors' tier choices shifts the score
    through peer_term.
"""
function choose_ai_level(
    agent::EmergentAgent;
    neighbor_signals::Dict{String,Any} = Dict{String,Any}(),
    current_round::Int = 0,
    visible_opportunities::Float64 = NaN
)::String
    # If fixed AI level, return it
    if !isnothing(agent.fixed_ai_level)
        return agent.fixed_ai_level
    end

    # Sticky re-evaluation: defer the first tier review until investments have
    # matured (initial_freeze rounds), then review at the ongoing cadence.
    #
    #   * Without an initial freeze, the first review fires at round 1-3 while
    #     tier_roi_history is empty (no investments matured yet). The decision
    #     then devolves into cost-vs-peer-signal cascades, locking in a
    #     winner-take-all distribution before any agent has tier-specific data.
    #   * With the initial freeze (12 rounds = one investment maturity cycle),
    #     agents stay at their starting tier for the first cycle, build tier_roi
    #     data, then make an informed first switch.
    #
    # A negative last_tier_review_round sentinel means "never reviewed": wait
    # until initial_freeze rounds have elapsed, then respect review_interval.
    review_interval = max(1, getfield_default(agent.config, :AI_TIER_REVIEW_INTERVAL, 3))
    initial_freeze = max(0, getfield_default(agent.config, :AI_TIER_INITIAL_FREEZE_ROUNDS, 12))
    if agent.last_tier_review_round < 0
        if current_round < initial_freeze
            return agent.current_ai_level
        end
    else
        rounds_since_review = current_round - agent.last_tier_review_round
        if rounds_since_review < review_interval && current_round > 0
            return agent.current_ai_level
        end
    end

    base_trust = clamp(get(agent.traits, "ai_trust", 0.5), 0.0, 1.0)
    uncertainty_tolerance = clamp(get(agent.traits, "uncertainty_tolerance", 0.5), 0.0, 1.0)
    avoidance = 1.0 - uncertainty_tolerance

    metrics = compute_ai_performance_metrics(agent)
    order = ["none", "basic", "advanced", "premium"]

    # Neutral priors: read posteriors only from ai_learning.tier_beliefs, which
    # initializes uniformly at alpha=2, beta=2 (posterior mean 0.5) for every
    # tier (DEFAULT_TIER_PRIORS), so no tier is favored before any data arrives.
    posterior_means = Dict{String,Float64}()
    for tier in order
        beliefs = agent.ai_learning.tier_beliefs
        if haskey(beliefs, tier)
            belief = beliefs[tier]
            total = belief["alpha"] + belief["beta"]
            posterior_means[tier] = total > 0 ? belief["alpha"] / total : 0.5
        else
            posterior_means[tier] = 0.5
        end
    end

    # Compute cost ratios relative to the agent's monthly burn so cost is felt
    # at burn-rate scale. The burn estimate matches what estimate_operational_costs
    # actually charges: sector-derived agent.operating_cost_estimate, scaled by
    # OPS_COST_INTENSITY, with the same active-investment competition multiplier
    # (1 + 0.2 × avg_competition). The multiplier formula is replicated here
    # rather than calling estimate_operational_costs because that function needs
    # `market` in scope, while this multiplier depends only on
    # agent.active_investments.
    raw_burn = agent.operating_cost_estimate * agent.config.OPS_COST_INTENSITY
    if !isempty(agent.active_investments)
        comps = Float64[]
        for inv in agent.active_investments
            opp = get(inv, "opportunity", nothing)
            if !isnothing(opp) && opp isa Opportunity
                push!(comps, opp.competition)
            end
        end
        if !isempty(comps)
            raw_burn *= 1.0 + mean(comps) * 0.2
        end
    end
    operating_cost = max(raw_burn, 1.0)
    recent_activity = max(1.0, Float64(get(metrics, "recent_ai_activity", 1.0)))

    # ── Usage planning: price the action mix the agent ACTUALLY executes ──
    # Billing charges "market_analysis" once per visible opportunity per round
    # (the round-scoped info cache dedupes calculate_investment_utility and
    # evaluate_portfolio_opportunities to one charge per opportunity), plus
    # task charges on innovate ("innovation_potential") and explore
    # ("exploration"). Posted prices are not Knightian uncertainty: an agent
    # may misjudge returns, but a planner that misreads the published price
    # sheet is mispricing, not bounded rationality. A misconfigured planner that priced ≤4
    # cheap "tier_review" calls and underestimated frontier usage cost by an
    # order of magnitude, systematically biasing emergent tier adoption
    # toward expensive tiers.
    #
    # Visibility is tier-dependent (info_breadth), so each candidate tier's
    # expected analysis volume is the agent's observed visible count scaled by
    # the published breadth ratio — knowable product information, like price.
    effective_current = behavior_ai_level(agent.config, agent.current_ai_level)
    current_cfg = get(agent.config.AI_LEVELS, effective_current,
                      get(agent.config.AI_LEVELS, "none", nothing))
    current_breadth = isnothing(current_cfg) ? 0.2 : max(Float64(current_cfg.info_breadth), 0.05)
    visible_now = isfinite(visible_opportunities) ?
        max(visible_opportunities, 1.0) : recent_activity

    # Recent action mix over roughly one review window: how often does this
    # agent actually run innovate/explore tasks per round?
    mix_window = agent.action_history[max(1, end - 11):end]
    n_mix = max(length(mix_window), 1)
    share_innovate = count(==("innovate"), mix_window) / n_mix
    share_explore = count(==("explore"), mix_window) / n_mix

    cost_ratios = Dict{String,Float64}()

    for tier in order
        if tier == "none"
            cost_ratios[tier] = 0.0
            continue
        end
        cfg = get(agent.config.AI_LEVELS, tier, nothing)
        if isnothing(cfg)
            cost_ratios[tier] = 0.0
            continue
        end
        base_cost = _ai_subscription_cost(agent, cfg)
        # Expected per-round analysis volume at THIS tier's visibility.
        breadth_ratio = max(Float64(cfg.info_breadth), 0.05) / current_breadth
        v_est = clamp(visible_now * breadth_ratio, 1.0, 400.0)
        call_cost = ai_analysis_call_cost(agent, tier;
            expected_calls=v_est,
            action="market_analysis",
            n_candidates=v_est,
        )
        if share_innovate > 0.0
            call_cost += share_innovate * ai_analysis_call_cost(agent, tier;
                action="innovation_potential")
        end
        if share_explore > 0.0
            call_cost += share_explore * ai_analysis_call_cost(agent, tier;
                action="exploration",
                n_candidates=v_est,
            )
        end

        total_cost = if cfg.cost_type == "subscription"
            # Match actual billing (start_subscription_schedule!): a subscription
            # charges the full monthly cost per round in fixed and hybrid modes,
            # so the planner's per-round cost includes the full base_cost (not an
            # amortized fraction). Under-estimating this would bias emergent-mode
            # tier choice toward expensive tiers. In hybrid pricing, usage cost
            # is added through token-equivalent work, matching the marginal
            # charges deducted by the decision paths.
            base_cost + call_cost
        elseif cfg.cost_type == "per_use"
            # Use the per_use_cost field (call_cost) directly rather than the
            # subscription `cost` field, so per-use tiers are priced at their
            # documented per-use rate.
            call_cost
        else
            call_cost
        end

        cost_ratios[tier] = total_cost / operating_cost
    end

    current_level = agent.current_ai_level
    if !(current_level in order)
        current_level = "none"
    end

    # Extract performance signals
    net_cash_total = Float64(get(metrics, "overall_cash_flow_total", 0.0))
    initial_equity = max(agent.resources.performance.initial_equity, 1.0)
    # Back out the subscription already paid THIS round when computing
    # capital_health: it is a sunk cost and should not bias the forward-looking
    # decision.
    effective_capital = get_capital(agent) + agent.last_subscription_charge
    capital_health = (effective_capital / initial_equity) - 1.0

    # Peer signals
    peer_roi_signal = Float64(get(neighbor_signals, "peer_roi_gap", 0.0))
    adoption_pressure = Float64(get(neighbor_signals, "ai_adoption_pressure", 0.0))
    peer_distribution = get(neighbor_signals, "ai_distribution", Dict{String,Float64}())

    # Tier capability is captured solely by the principled config fields —
    # AI_LEVELS (cost, info_quality, info_breadth, hallucination, per_use_cost)
    # and AI_DOMAIN_CAPABILITIES — with no asymmetric tier-specific tunings
    # (reserve haircuts, tier skepticism factors, or per-tier cost subsidies)
    # applied to the score below.

    # Score each tier
    scores = Dict{String,Float64}()
    for tier in order
        posterior = posterior_means[tier]
        # Trust term: posterior expectation × base_trust (no tier attenuation)
        trust_term = posterior * base_trust

        # Tier-specific ROI: use this tier's own roi_history when available
        # (maintained by process_matured_investments!).
        tier_roi_window = get(agent.ai_learning.tier_roi_history, tier, Float64[])
        tier_specific_roi = if !isempty(tier_roi_window)
            mean(tier_roi_window)
        else
            # No tier-specific data yet; use 0 (neutral) rather than letting a
            # global ROI signal leak across tiers.
            0.0
        end
        # Weight 0.75 on the posterior-scaled tier-specific ROI (net gain).
        roi_term = posterior * 0.75 * tier_specific_roi

        cash_term = 0.2 * clamp(net_cash_total / initial_equity, -1.5, 1.5)

        # Peer effects: peer adoption is a social-proof signal for tier choice.
        # The 0.10 weight on peer_distribution keeps it meaningful without
        # producing winner-take-all herding into a single tier.
        peer_term = 0.10 * peer_roi_signal + 0.15 * adoption_pressure +
                   0.10 * Float64(get(peer_distribution, tier, 0.0))

        # Experience with tier
        tier_experience = count(==(tier), agent.ai_tier_history)
        learning_relief = clamp(log1p(tier_experience) * 0.02, 0.0, 0.12)

        # Cost term. Because the cost ratio is scaled to monthly burn (above),
        # the 0.6 coefficient makes cost_term comparable in magnitude to
        # trust_term (e.g. ≈ 0.6 × 0.156 = 0.09 for the frontier tier) — large
        # enough to matter without forcing a corner solution where the most
        # expensive tier goes extinct.
        cost_term = 0.6 * get(cost_ratios, tier, 0.0) * (1.0 - learning_relief)

        switch_penalty = max(0.0, compute_ai_switch_penalty(agent, current_level, tier) - learning_relief)

        # Gumbel noise for stochastic selection
        noise = -log(-log(rand(agent.rng))) * 0.02  # Gumbel(0,1) * 0.02

        total_score = (
            trust_term +
            roi_term +
            cash_term +
            0.25 * capital_health +
            peer_term -
            cost_term -
            switch_penalty -
            0.05 * avoidance +
            noise
        )

        scores[tier] = total_score
    end

    # Select the tier with the highest score, using status-quo tie-breaking:
    # ties go to the agent's CURRENT tier rather than the lowest-index tier,
    # which would otherwise impose a systematic bias toward cheaper tiers. The
    # loop seeds best_tier with the current level so equal scores never displace
    # it.
    best_tier = current_level
    best_score = scores[current_level]
    for (i, tier) in enumerate(order)
        score = scores[tier]
        if score > best_score
            best_score = score
            best_tier = tier
        end
    end

    # Update tracking
    push!(agent.ai_tier_history, best_tier)
    agent.current_ai_level = best_tier
    agent.last_tier_review_round = current_round  # sticky re-eval bookkeeping

    # Start subscription schedule if this is a subscription tier
    ensure_subscription_schedule!(agent, best_tier)

    return best_tier
end

# ============================================================================
# UTILITY CALCULATIONS
# ============================================================================

function _perception_uncertainty_levels(
    perception::Union{Perception,Nothing}
)::Dict{String,Float64}
    if isnothing(perception)
        return Dict{String,Float64}()
    end

    return Dict{String,Float64}(
        "actor_ignorance" => Float64(perception.actor_ignorance.level),
        "practical_indeterminism" => Float64(perception.practical_indeterminism.level),
        "agentic_novelty" => Float64(perception.agentic_novelty.level),
        "competitive_recursion" => Float64(perception.competitive_recursion.level),
    )
end

function _response_adjusted_uncertainty(
    agent::EmergentAgent,
    uncertainty_type::String,
    uncertainty_level::Float64
)::Float64
    bounded_level = clamp(uncertainty_level, 0.0, 1.0)
    response_factor = get_response_factor(
        agent.uncertainty_response,
        uncertainty_type,
        bounded_level;
        explore=false,
        rng=agent.rng,
    )
    return clamp(2.0 * (1.0 - response_factor), 0.0, 1.0)
end

"""
Calculate investment utility given opportunities and perception.
"""
function calculate_investment_utility(
    agent::EmergentAgent,
    opportunities::Vector{Opportunity},
    market_conditions::MarketConditions,
    perception::Perception;
    ai_level::String = "none",
    info_system::Union{InformationSystem,Nothing} = nothing
)::Float64
    if isempty(opportunities)
        return 0.0
    end

    effective_ai_level = behavior_ai_level(agent.config, ai_level)

    # Get the best opportunity score using the agent's decision-time
    # information. AI capability enters through get_information /
    # evaluate_opportunity_basic and perception, not through a second generic
    # utility uplift. Score against the agent's PERCEIVED return
    # (estimated_return from the InformationSystem), falling back to neutral
    # break-even in diagnostic paths, so the should-I-invest-at-all decision is
    # not aware of hidden ground truth.
    #
    # Each fresh get_information call is a chargeable analytical task under the
    # active cost backend, so this loop deducts AI analysis cost the same way
    # evaluate_portfolio_opportunities does — otherwise utility evaluation would
    # give some tiers free AI while ranking charged for it.
    # AGI strategy ladder gate (the design notes).
    # Computed ONCE per call; when inactive (reactive default, or this tier
    # outside STRATEGY_TIERS) every hook below short-circuits before any
    # strategy computation, leaving the reactive path bit-identical.
    ladder_active = strategy_active(agent.config, effective_ai_level)
    # G1 two-pass S2: per-opportunity edges + the choice-set mean computed
    # ONCE over the full candidate pool (strategy_edge_context,
    # src/strategy.jl); the per-candidate multiplier below is then a
    # dictionary lookup centered on that mean (mean multiplier ≡ 1 per
    # choice set — re-ranking signal only, no level confound).
    ladder_edges = ladder_active ?
        strategy_edge_context(agent.config, agent.resources.knowledge,
                              opportunities) : nothing

    max_score = 0.0
    for opp in opportunities
        # Propagate the visible Information fields (est_return, est_uncertainty,
        # confidence) so evaluate_opportunity_basic can use tier-aware
        # uncertainty + confidence in its scoring; the hidden hallucination flag
        # is not passed into the evaluator. est_return initializes to a neutral
        # 1.0 (break-even); production overwrites it from info.estimated_return,
        # and the fallback only matters when info_system === nothing (test
        # paths), where leaking ground truth would falsify diagnostics.
        est_return = 1.0
        est_uncertainty = nothing
        conf = nothing
        strategy_info::Union{Information,Nothing} = nothing
        if !isnothing(info_system)
            # Only deduct per-use cost on cache MISS. Re-reading a cached info
            # object is not a new API call — the agent already paid for it.
            cached_before = haskey(info_system.information_cache,
                                   (opp.id, effective_ai_level, agent.id))
            info = get_information(info_system, opp, effective_ai_level;
                                   agent_id=agent.id,
                                   agent_traits=agent.traits,
                                   sector_knowledge=agent.resources.knowledge,
                                   rng=agent.rng)
            ai_call_charge = ai_analysis_call_cost(agent, effective_ai_level;
                opportunity=opp,
                action="market_analysis",
                n_candidates=Float64(length(opportunities)),
            )
            if !cached_before && ai_call_charge > 0
                set_capital!(agent, get_capital(agent) - ai_call_charge)
            end
            est_return = info.estimated_return
            est_uncertainty = info.estimated_uncertainty
            conf = info.confidence
            strategy_info = info
            # The agent does not read the hidden contains_hallucination flag.
            # Reliability enters only through the visible estimate, uncertainty,
            # confidence, and the eventual matured outcomes.
        end

        # S1 hook (AGI strategy ladder): anticipatory congestion shade on the
        # expected return BEFORE scoring — the strategist discounts what every
        # rival's instrument also ranks highly, instead of only reacting to the
        # recursion penalty after the fact. The reactive 0.25 recursion penalty
        # below stays untouched (S1 adds the anticipatory component; the
        # design doc forbids double-counting via a retuned coefficient).
        if ladder_active
            est_return = strategy_shaded_return(
                agent.config, est_return, strategy_info, perception)
        end

        base_score = evaluate_opportunity_basic(agent, opp, market_conditions;
                                                estimated_return=est_return,
                                                estimated_uncertainty=est_uncertainty,
                                                confidence=conf,
                                                ai_level=effective_ai_level)

        # S2/S3b hook (AGI strategy ladder): private-edge re-weighting and
        # low-confidence penalty softening, mirrored at the ranking site in
        # evaluate_portfolio_opportunities so the invest-vs-not utility and the
        # chosen opportunity move together.
        if ladder_active
            base_score *= strategy_score_multiplier(
                agent.config, ladder_edges, opp, strategy_info, perception)
        end

        max_score = max(max_score, base_score)
    end

    scaled_score = stable_sigmoid(max_score - 1.0)

    # Extract uncertainty levels from perception (typed field access, no dict alloc)
    actor_unc = perception.actor_ignorance.level
    practical_unc = perception.practical_indeterminism.level
    agentic_unc = perception.agentic_novelty.level
    recursive_unc = perception.competitive_recursion.level
    decision_conf = clamp(perception.decision_confidence, 0.1, 0.95)
    actor_pressure = _response_adjusted_uncertainty(agent, "actor_ignorance", actor_unc)
    agentic_pressure = _response_adjusted_uncertainty(agent, "agentic_novelty", agentic_unc)
    recursive_pressure = _response_adjusted_uncertainty(agent, "competitive_recursion", recursive_unc)

    # Capital metrics
    capital_ratio = get_capital(agent) / max(agent.resources.performance.initial_equity, 1.0)
    liquidity_boost = clamp(capital_ratio - 0.8, 0.0, 1.5)
    opportunity_boost = clamp(length(opportunities) / 6.0, 0.0, 1.0)

    # Ignorance adjustment: a linear response in [0.2, 1.0] so the
    # tier-differentiated Knightian-ignorance perception translates into utility.
    # A tier with low perceived ignorance earns a real utility premium and one
    # with high perceived ignorance a real penalty; a sigmoid in this operating
    # range (agents typically perceive 0.05–0.25) would flatten that difference.
    ignorance_adjustment = clamp(1.0 - actor_pressure * 0.8, 0.2, 1.0)

    value = scaled_score * ignorance_adjustment * decision_conf

    # Performance adjustments
    perf = agent.resources.performance
    invest_roic = compute_roic(perf, "invest")

    # Idle (liquid) capital. get_capital already excludes in-flight investments:
    # they were debited from agent.resources.capital at commitment time, so
    # subtracting locked capital again would double-count it and understate
    # liquidity by exactly the deployed amount.
    idle_capital = max(0.0, get_capital(agent))
    idle_ratio = idle_capital / max(agent.resources.performance.initial_equity, 1.0)

    value *= 0.7 + 0.3 * liquidity_boost
    value += 0.2 * opportunity_boost

    roic_multiplier = clamp(1.0 + 0.35 * invest_roic, 0.4, 1.4)
    value *= roic_multiplier

    value += max(0.0, idle_ratio) * 0.25

    # Portfolio size penalty
    if length(agent.active_investments) >= 4
        value -= 0.07 * (length(agent.active_investments) - 3)
    end

    # Uncertainty hooks
    value += 0.15 * agentic_pressure
    # The competitive_recursion coefficient (0.25) feeds the equilibrium-trap
    # mechanism into the pre-decision utility, not just the post-maturation
    # return via the capital-saturation convexity. Under high-quality AI the
    # environment reports near-maximal recursion (~0.98) because high-tier
    # agents identify the same top opportunities and pile in; a small
    # coefficient would let agents see the crowded niches without behaviorally
    # pivoting away from them.
    value -= 0.25 * recursive_pressure

    # Risk tolerance
    risk_tolerance = Float64(get(agent.traits, "uncertainty_tolerance", 0.5))
    value *= clamp(0.85 + 0.3 * risk_tolerance, 0.5, 1.4)
    value -= (1.0 - risk_tolerance) * 0.12

    # Recent action penalty
    recent_invests = count(==("invest"), agent.action_history[max(1, end-9):end])
    value -= 0.1 * recent_invests / 10.0

    return clamp(value, 0.0, 1.0)
end

"""
Calculate innovation utility.

AI capability affects innovation through perception, knowledge access,
combination generation, and explicit robustness/refutation knobs rather than
through direct post-perception utility bonuses.
"""
function calculate_innovation_utility(
    agent::EmergentAgent,
    market_conditions::MarketConditions,
    perception::Perception;
    ai_level::String = "none",
    # AGI strategy ladder S3a shift (the design notes),
    # computed by the caller (make_decision!) from the agent's own information
    # cache. Default 0.0 = reactive; applied behind a != 0.0 gate below so the
    # reactive path performs the identical FP operations.
    strategy_shift::Float64 = 0.0
)::Float64
    base_drive = get(agent.traits, "innovativeness", 0.5) * 0.5 + 0.05

    # Extract signals (typed field access)
    practical_unc = perception.practical_indeterminism.level
    agentic_unc = perception.agentic_novelty.level
    practical_pressure = _response_adjusted_uncertainty(agent, "practical_indeterminism", practical_unc)
    agentic_pressure = _response_adjusted_uncertainty(agent, "agentic_novelty", agentic_unc)

    indeterminism_bonus = practical_pressure * 0.4
    innovation_opportunity = agentic_pressure * 0.25

    # Capital constraints
    capital_ratio = get_capital(agent) / max(agent.resources.performance.initial_equity, 1.0)
    liquidity_penalty = clamp(1.0 - capital_ratio, 0.0, 1.5)

    # R&D burden
    rd_deployed = get(agent.resources.performance.deployed_by_action, "innovate", 0.0)
    rd_burden = clamp(rd_deployed / max(agent.resources.performance.initial_equity, 1.0), 0.0, 2.0)

    innovate_roic = compute_roic(agent.resources.performance, "innovate")
    loss_ratio = max(0.0, -innovate_roic)

    risk_tolerance = Float64(get(agent.traits, "uncertainty_tolerance", 0.5))
    avoidance = 1.0 - risk_tolerance

    raw_score = (
        base_drive +
        indeterminism_bonus +
        innovation_opportunity +
        0.25 * liquidity_penalty -
        0.3 * rd_burden -
        0.4 * loss_ratio +
        0.2 * (risk_tolerance - 0.5) -
        0.15 * avoidance
    )

    # S3a (AGI strategy ladder): complement-seeking shift toward creative
    # action when shared signals dominate and the agent's own instrument is
    # unsure about much of the visible set. Gated (not an unconditional +0.0)
    # to keep the reactive path bit-identical.
    if strategy_shift != 0.0
        raw_score += strategy_shift
    end

    return clamp(stable_sigmoid(raw_score - 0.6), 0.02, 0.98)
end

"""
Calculate exploration utility.
"""
function calculate_exploration_utility(
    agent::EmergentAgent,
    perception::Perception;
    ai_level::String = "none",
    # AGI strategy ladder S3a shift — see calculate_innovation_utility.
    strategy_shift::Float64 = 0.0
)::Float64
    base_tendency = get(agent.traits, "exploration_tendency", 0.3)

    # Extract signals (typed field access)
    actor_unc = perception.actor_ignorance.level
    agentic_unc = perception.agentic_novelty.level
    recursive_unc = perception.competitive_recursion.level
    actor_pressure = _response_adjusted_uncertainty(agent, "actor_ignorance", actor_unc)
    agentic_pressure = _response_adjusted_uncertainty(agent, "agentic_novelty", agentic_unc)
    recursive_pressure = _response_adjusted_uncertainty(agent, "competitive_recursion", recursive_unc)

    ignorance_drive = actor_pressure * 0.35

    # Knowledge deficit
    avg_knowledge = fast_mean(values(agent.resources.knowledge))
    knowledge_deficit = clamp(1.0 - avg_knowledge, 0.0, 1.0)

    novelty_gap = max(0.0, 0.5 - agentic_pressure)

    # Capital slack. get_capital already excludes in-flight investments (debited
    # at commitment), so locked capital must not be subtracted again.
    capital_slack = max(0.0, get_capital(agent) / max(agent.resources.performance.initial_equity, 1.0))

    rich_knowledge_penalty = clamp(avg_knowledge - 0.25, 0.0, 1.0)

    ai_trust_level = Float64(get(agent.traits, "ai_trust", 0.5))
    trust_penalty = max(0.0, ai_trust_level - 0.5) * 0.45

    recursion_penalty = max(0.0, recursive_pressure - 0.5) * 0.2
    # No tier-specific modification - let behavior emerge naturally

    # Exploration momentum from the agent's recent discovery rate vs. the
    # configured baseline. Exploration returns knowledge rather than cash, so
    # cash ROIC is structurally -1.0 here and would pin this term at a constant
    # penalty rather than a momentum signal.
    explore_events = [o for o in agent.recent_outcomes if get(o, "action", "") == "explore"]
    momentum_bonus = if isempty(explore_events)
        0.0
    else
        recent_discovery_rate = count(o -> get(o, "success", false), explore_events) /
            length(explore_events)
        clamp(recent_discovery_rate - agent.config.DISCOVERY_PROBABILITY, -0.5, 0.5) * 0.2
    end

    risk_tolerance = Float64(get(agent.traits, "uncertainty_tolerance", 0.5))
    avoidance = 1.0 - risk_tolerance

    # Recent explore penalty
    recent_explores = count(==("explore"), agent.action_history[max(1, end-4):end])
    recent_explore_penalty = recent_explores > 2 ? 0.25 : 0.0

    raw_score = (
        base_tendency * 0.35 +
        ignorance_drive +
        knowledge_deficit * 0.25 +
        0.2 * novelty_gap +
        0.08 * capital_slack +
        momentum_bonus -
        0.35 * rich_knowledge_penalty -
        trust_penalty -
        recursion_penalty +
        0.18 * avoidance -
        0.08 * risk_tolerance +
        0.18 * recursive_pressure -
        recent_explore_penalty
    )

    # S3a (AGI strategy ladder): complement-seeking shift toward exploration
    # of oracle-blind territory. Gated to keep the reactive path bit-identical.
    if strategy_shift != 0.0
        raw_score += strategy_shift
    end

    return clamp(stable_sigmoid(raw_score - 0.45), 0.05, 0.95)
end

"""
Calculate maintain utility.
"""
function calculate_maintain_utility(
    agent::EmergentAgent,
    market_conditions::MarketConditions,
    perception::Perception;
    estimated_cost::Float64 = 0.0
)::Float64
    cost = estimated_cost > 0 ? estimated_cost : agent.config.BASE_OPERATIONAL_COST
    capital_buffer_in_rounds = get_capital(agent) / (cost + 1e-9)
    buffer_pressure = stable_sigmoid(2.0 - capital_buffer_in_rounds)

    # Diversification (use active investments as proxy)
    sectors_in_portfolio = String[]
    for inv in agent.active_investments
        opp = get(inv, "opportunity", nothing)
        if !isnothing(opp) && !isnothing(opp.sector)
            push!(sectors_in_portfolio, opp.sector)
        end
    end
    n_sectors = length(unique(sectors_in_portfolio))
    diversification = min(1.0, n_sectors / 4.0)

    practical_unc = perception.practical_indeterminism.level
    practical_pressure = _response_adjusted_uncertainty(agent, "practical_indeterminism", practical_unc)
    uncertainty_penalty = practical_pressure * 0.3

    idle_penalty = min(0.2, (get_capital(agent) / max(agent.config.INITIAL_CAPITAL, 1.0)) * 0.1)

    avg_knowledge = fast_mean(values(agent.resources.knowledge))
    knowledge_penalty = clamp(avg_knowledge - 0.25, 0.0, 1.0) * 0.2
    surplus_buffer_penalty = clamp(capital_buffer_in_rounds - 3.0, 0.0, 3.0) * 0.1

    maintain_utility = 0.2 + buffer_pressure * 0.5 + diversification * 0.3 - uncertainty_penalty - idle_penalty
    maintain_utility -= knowledge_penalty + surplus_buffer_penalty

    risk_tolerance = Float64(get(agent.traits, "uncertainty_tolerance", 0.5))
    avoidance = 1.0 - risk_tolerance
    maintain_utility += 0.2 * avoidance - 0.1 * risk_tolerance

    return clamp(maintain_utility, 0.05, 0.95)
end

# ============================================================================
# OPPORTUNITY EVALUATION
# ============================================================================

"""
Basic opportunity evaluation.

The agent's *perceived* return (`estimated_return`) drives the score. This is
the canonical input, set by `evaluate_portfolio_opportunities` from
`InformationSystem.get_information`, so the information-quality mechanism (each
tier perceives returns with tier-specific noise) actually governs ranking.
When `estimated_return` is omitted, the function falls back to neutral
break-even (`1.0`) rather than `opportunity.latent_return_potential`, so
diagnostic / test paths do not leak hidden ground truth into scoring.
"""
function evaluate_opportunity_basic(
    agent::EmergentAgent,
    opportunity::Opportunity,
    market_conditions::MarketConditions;
    estimated_return::Union{Float64,Nothing} = nothing,
    estimated_uncertainty::Union{Float64,Nothing} = nothing,
    confidence::Union{Float64,Nothing} = nothing,
    ai_level::Union{String,Nothing} = nothing,
    # This function deliberately takes no contains_hallucination argument: it
    # must not read the hidden-truth flag. Reliability effects reach scoring only
    # via realized outcomes and learned disconfirmation.
)::Float64
    # Expected profit margin: use the AI-tier-aware estimate when provided.
    # The fallback is a neutral 1.0 (break-even) rather than the hidden
    # opportunity.latent_return_potential. Production callers always pass
    # estimated_return; the fallback only matters in test/probe paths, where
    # leaking the latent value would invalidate scoring.
    expected_return = isnothing(estimated_return) ? 1.0 : estimated_return
    expected_margin = expected_return - 1.0

    # Use the AI-tier-aware uncertainty estimate when provided. A higher-quality
    # AI produces not only a better return estimate but also a better assessment
    # of HOW uncertain that estimate is, narrowing perceived uncertainty;
    # low-quality AI (or no AI) falls back to opp.complexity as a crude proxy.
    # Ignoring estimated_uncertainty would understate the high tiers'
    # info-quality advantage.
    perceived_uncertainty = if !isnothing(estimated_uncertainty)
        # Blend AI-estimated uncertainty with opp.complexity (which captures
        # intrinsic difficulty independent of AI assessment).
        0.6 * estimated_uncertainty + 0.4 * opportunity.complexity
    else
        opportunity.complexity
    end
    uncertainty_adjusted = expected_margin * (1.0 - perceived_uncertainty * 0.5)

    score = uncertainty_adjusted * (1.0 + get(agent.traits, "uncertainty_tolerance", 0.5))

    # Confidence weighting: the agent's conviction in this estimate scales the
    # score's magnitude, so low-confidence signals influence decisions less.
    # (uncertainty_tolerance captures the agent's general appetite for ambiguity;
    # confidence captures how much they trust THIS specific AI call.) Defaults to
    # 0.5 when no confidence info is available.
    conf = isnothing(confidence) ? 0.5 : clamp(confidence, 0.05, 0.99)
    # At conf = 0.5, no change. At conf = 0.99, +15% score. At conf = 0.05, -25%.
    score *= 0.75 + 0.5 * conf

    # Do not apply a second tier-level reliability multiplier here. AI
    # reliability is already represented upstream by estimated_return,
    # estimated_uncertainty, confidence, hallucinated/noisy information draws,
    # and downstream by observable disconfirmation after outcomes mature.

    # Regime multiplier (const lookup — see _REGIME_MULTIPLIER at top of file)
    score *= get(_REGIME_MULTIPLIER, market_conditions.regime, 1.0)

    # Sector knowledge
    sector_knowledge = sector_familiarity(
        agent.resources.knowledge,
        opportunity.sector;
        opportunity=opportunity,
        config=agent.config,
        default=0.1,
    )
    score *= 1.0 + sector_knowledge * 0.5

    return max(0.1, score)
end

"""
Apply uncertainty adjustments to opportunity score.
"""
function apply_uncertainty_adjustments(
    agent::EmergentAgent,
    base_score::Float64,
    opportunity::Opportunity,
    perception::Perception
)::Float64
    adjusted = base_score

    # Competition adjustment. The penalty scales with intolerance (1 - tolerance):
    # uncertainty_tolerance captures appetite for ambiguity, so a high-tolerance
    # agent is LESS bothered by competition and takes a smaller penalty.
    if opportunity.competition > 0.5
        tolerance = get(agent.traits, "uncertainty_tolerance", 0.5)
        intolerance = 1.0 - tolerance
        adjusted *= 1.0 - (intolerance * opportunity.competition * 0.5)
    end

    # Social proof (herding tendency)
    analytical = get(agent.traits, "analytical_ability", 0.5)
    social_sensitivity = 1.0 - analytical
    social_bonus = 1.0 + opportunity.competition * social_sensitivity * 0.25
    adjusted *= social_bonus

    # Creator bonus
    if !isnothing(opportunity.created_by)
        adjusted *= 1.1
        if opportunity.created_by == agent.id
            adjusted *= 1.2
        end
    end

    return max(0.01, adjusted)
end

"""
Evaluate portfolio opportunities and return ranked list.

Sequential Decision Signals:
If early_signals is provided (from sequential decision making), opportunities that
early investors chose receive a boost. The effect scales with info_quality - Frontier AI
agents trust visible signals more (rational herding / information cascades).
"""
function evaluate_portfolio_opportunities(
    agent::EmergentAgent,
    opportunities::Vector{Opportunity},
    market_conditions::MarketConditions,
    perception::Perception;
    ai_level::String = "none",
    info_system::Union{InformationSystem,Nothing} = nothing,
    early_signals::Union{Dict{String,Int},Nothing} = nothing,
    signal_weight::Float64 = 0.15
)::Vector{OpportunityEvaluation}
    if isempty(opportunities)
        return OpportunityEvaluation[]
    end

    effective_ai_level = behavior_ai_level(agent.config, ai_level)

    evaluations = OpportunityEvaluation[]
    opp_pool = copy(opportunities)

    # ─────────────────────────────────────────────────────────────────
    # Compute AI-tier-aware per-opportunity estimates that drive final_score.
    #
    # When `info_system` is provided (production path), use
    # `get_information(opp, ai_level)` to produce hallucination-injected,
    # tier-noisy estimates from the InformationSystem. When `info_system` is
    # nothing (test/diagnostic path), use the inline tier-noise model below.
    # Either way the estimates feed `evaluate_opportunity_basic`, so ranking is
    # governed by tier-specific information quality rather than hidden ground
    # truth.
    #
    # Herding emerges naturally from information quality:
    # - High-quality-AI agents have low noise → rank similar opportunities best
    # - No-AI agents have high noise → rank diverse opportunities best
    # The optional novelty-generalization stress is off in baseline and enabled
    # only by explicit robustness/refutation cells.
    # ─────────────────────────────────────────────────────────────────

    estimated_returns = Dict{String,Float64}()
    # Stash Information objects for per-opp propagation into invest-action
    # outcomes. The ai_contains_hallucination / ai_confidence /
    # ai_actual_accuracy fields are diagnostics only; behavioral uncertainty
    # responds to observable post-maturity disconfirmation, not these fields.
    info_cache_local = Dict{String,Information}()

    if !isnothing(info_system)
        # Production path: full InformationSystem with hallucination chain.
        # Deduct AI analysis cost for each fresh get_information call under the
        # configured backend, so the per-use cost is actually charged against
        # agent.capital rather than only appearing in planning estimates.
        for opp in opp_pool
            # Deduct per-use cost on cache MISS only (see calculate_investment_utility
            # for rationale — prevents double-charging across utility + ranking).
            cached_before = haskey(info_system.information_cache,
                                   (opp.id, effective_ai_level, agent.id))
            info = get_information(info_system, opp, effective_ai_level;
                                   agent_id=agent.id,
                                   agent_traits=agent.traits,
                                   sector_knowledge=agent.resources.knowledge,
                                   rng=agent.rng)
            info_cache_local[opp.id] = info
            est = info.estimated_return
            if !isnothing(early_signals)
                signal_count = get(early_signals, opp.id, 0)
                if signal_count > 0
                    signal_boost = signal_count * signal_weight * 0.1
                    est *= (1.0 + signal_boost)
                end
            end
            estimated_returns[opp.id] = est

            ai_call_charge = ai_analysis_call_cost(agent, effective_ai_level;
                opportunity=opp,
                action="market_analysis",
                n_candidates=Float64(length(opp_pool)),
            )
            if !cached_before && ai_call_charge > 0
                set_capital!(agent, get_capital(agent) - ai_call_charge)
            end
        end
    else
        # Diagnostic/test path: inline tier-noise estimate (no hallucinations).
        # Tier noise (scaled by 1 - info_quality and drawn from agent.rng) is
        # applied to every opportunity, including the single-candidate case, so
        # tier info_quality affects the perceived return and preserves the
        # model's tier-noise-driven decision variance even with one candidate.
        ai_config = get(agent.config.AI_LEVELS, effective_ai_level, nothing)
        info_quality = isnothing(ai_config) ? 0.25 : Float64(ai_config.info_quality)
        base_noise_scale = 0.5 * (1.0 - info_quality)
        inversion_factor = getfield_default(agent.config, :NOVELTY_NOISE_INVERSION_FACTOR, 0.0)
        for opp in opp_pool
            opp_novelty = hasfield(typeof(opp), :novelty_score) ? opp.novelty_score : 0.0
            novelty_noise_penalty = opp_novelty * info_quality * inversion_factor
            effective_noise = base_noise_scale + novelty_noise_penalty
            noise = randn(agent.rng) * effective_noise
            est = opp.latent_return_potential + noise
            if !isnothing(early_signals)
                signal_count = get(early_signals, opp.id, 0)
                if signal_count > 0
                    signal_boost = signal_count * signal_weight * 0.1
                    est *= (1.0 + signal_boost)
                end
            end
            estimated_returns[opp.id] = est
        end
    end

    # ─────────────────────────────────────────────────────────────────
    # Evaluate each opportunity using the AI-tier-aware estimate
    # ─────────────────────────────────────────────────────────────────

    ai_config_eval = get(agent.config.AI_LEVELS, effective_ai_level, nothing)
    info_quality_eval = isnothing(ai_config_eval) ? 0.25 : Float64(ai_config_eval.info_quality)
    info_quality_boosts = getfield_default(agent.config, :AI_INFORMATION_QUALITY_BOOSTS, Dict{String,Float64}())
    info_quality_eval = clamp(info_quality_eval + Float64(get(info_quality_boosts, effective_ai_level, 0.0)), 0.0, 0.99)

    # AGI strategy ladder gate (the design notes): when
    # inactive the hooks below short-circuit before any strategy computation
    # (bit-identical reactive requirement). Hooked HERE (the ranking loop) so
    # the CHOSEN opportunity changes under S1/S2/S3, not just the
    # invest-vs-not utility.
    ladder_active = strategy_active(agent.config, effective_ai_level)
    # G1 two-pass S2 (mirrors calculate_investment_utility): edges + the
    # choice-set mean over the FULL ranked pool, computed once; the multiplier
    # below centers each candidate on that mean (mean multiplier ≡ 1).
    ladder_edges = ladder_active ?
        strategy_edge_context(agent.config, agent.resources.knowledge,
                              opp_pool) : nothing

    for opp in opp_pool
        # estimated_returns is populated for every opp earlier in this function;
        # the neutral 1.0 (break-even) fallback only fires if the dict somehow
        # misses the opp.id (probe path), where leaking the hidden
        # latent_return_potential would falsify scoring.
        est_return = get(estimated_returns, opp.id, 1.0)
        # Pull the full Information object when available so scoring uses
        # AI-tier-aware uncertainty + confidence. The hidden
        # contains_hallucination flag is not threaded through; reliability comes
        # from visible signal quality and matured disconfirmation.
        est_unc = nothing
        conf = nothing
        strategy_info::Union{Information,Nothing} = nothing
        if haskey(info_cache_local, opp.id)
            inf = info_cache_local[opp.id]
            est_unc = inf.estimated_uncertainty
            conf = inf.confidence
            strategy_info = inf
        end
        # S1 hook (AGI strategy ladder): shade the decision-time expected
        # return BEFORE scoring. Applied to `est_return` itself (not as a
        # post-multiplier) so the shaded value flows through the full scoring
        # pipeline AND into the OpportunityEvaluation record — the agent's
        # strategy-adjusted belief is also what sizes the commitment.
        if ladder_active
            est_return = strategy_shaded_return(
                agent.config, est_return, strategy_info, perception)
        end
        base_score = evaluate_opportunity_basic(agent, opp, market_conditions;
                                                estimated_return=est_return,
                                                estimated_uncertainty=est_unc,
                                                confidence=conf,
                                                ai_level=effective_ai_level)
        final_score = apply_uncertainty_adjustments(agent, base_score, opp, perception)
        signal_count = isnothing(early_signals) ? 0 : Int(get(early_signals, opp.id, 0))
        final_score *= strategic_anticipation_multiplier(
            agent,
            opp,
            effective_ai_level,
            info_quality_eval,
            perception,
            signal_count
        )
        # S2/S3b hook (AGI strategy ladder): private-edge re-ranking and
        # low-confidence penalty softening — changes WHICH opportunity tops
        # the sort, mirroring the calculate_investment_utility hook.
        if ladder_active
            final_score *= strategy_score_multiplier(
                agent.config, ladder_edges, opp, strategy_info, perception)
        end

        inf = get(info_cache_local, opp.id, nothing)
        eval_record = OpportunityEvaluation(;
            opportunity = opp,
            final_score = final_score,
            ai_level_used = effective_ai_level,
            estimated_return = est_return,
            competition_at_evaluation = hasfield(typeof(opp), :competition) ? opp.competition : 0.0,
            ai_info = inf,
            ai_contains_hallucination = inf === nothing ? false : inf.contains_hallucination,
            ai_confidence = inf === nothing ? 0.0 : Float64(inf.confidence),
            ai_actual_accuracy = inf === nothing ? 0.0 : Float64(something(inf.actual_accuracy, 0.0)),
        )
        push!(evaluations, eval_record)
    end

    # Single sort by final_score (which now reflects AI-tier-aware estimate)
    sort!(evaluations, by=e -> e.final_score, rev=true)

    return evaluations
end

# ============================================================================
# DECISION METHODS
# ============================================================================

"""
Make a complete decision including AI level selection and action choice.
"""
function make_decision!(
    agent::EmergentAgent,
    opportunities::Vector{Opportunity},
    market_conditions::MarketConditions,
    market::MarketEnvironment,
    round_num::Int;
    uncertainty_env::Union{KnightianUncertaintyEnvironment,Nothing} = nothing,
    neighbor_agents::Vector{EmergentAgent} = EmergentAgent[],
    ai_level_override::Union{String,Nothing} = nothing,
    innovation_engine::Union{InnovationEngine,Nothing} = nothing,
    info_system::Union{InformationSystem,Nothing} = nothing,
    early_signals::Union{Dict{String,Int},Nothing} = nothing,
    signal_weight::Float64 = 0.15
)::Dict{String,Any}
    if !agent.alive
        return Dict{String,Any}(
            "action" => "maintain",
            "agent_id" => agent.id,
            "round" => round_num,
            "success" => false,
            "reason" => "agent_inactive"
        )
    end

    # Choose AI level
    ai_level = if !isnothing(ai_level_override)
        ai_level_override
    elseif !isnothing(agent.fixed_ai_level)
        agent.fixed_ai_level
    else
        neighbor_signals = collect_neighbor_signals(agent, neighbor_agents)
        # current_round drives the sticky re-evaluation gating in choose_ai_level.
        # visible_opportunities feeds the usage-cost planner: billing charges
        # market_analysis per visible opportunity per round, so the planner
        # needs the agent's actual visibility to price candidate tiers.
        choose_ai_level(agent; neighbor_signals=neighbor_signals,
                        current_round=round_num,
                        visible_opportunities=Float64(length(opportunities)))
    end
    decision_ai_level = behavior_ai_level(agent.config, ai_level)

    # Build perception
    perception = if !isnothing(uncertainty_env)
        agent_traits = agent.traits
        visible_opps = opportunities
        perceive_uncertainty(
            uncertainty_env,
            agent_traits,
            visible_opps,
            market_conditions;
            ai_level=decision_ai_level,
            agent_knowledge=Set(keys(agent.resources.knowledge)),
            sector_knowledge=agent.resources.knowledge,
            action_history=agent.action_history,
            ai_learning_profile=agent.ai_learning,
            agent_id=agent.id,
            agent_resources=agent.resources,
            recent_outcomes=agent.recent_outcomes
        )
    else
        # Test/diagnostic fallback. See empty_perception in models.jl.
        empty_perception()
    end
    # A1 open-action pivot review (the design notes).
    # Sited FIRST among the decision-economics steps ( (design note)
    # 2026-06-09) — BEFORE estimate_operational_costs and ALL action-utility
    # calculations — so every utility sees post-pivot capital and portfolio
    # state consistently (previously invest utility saw pre-pivot state while
    # the later steps saw post-pivot state). The trigger reads the
    # decision-time-STORED commitment estimates (the F1 instrument value), not
    # this round's information cache: a same-round re-query was an
    # inconsistent privilege anyway, since holdings outside the visible set
    # never had one. ENABLE_PIVOT=false returns before any computation
    # (bit-identical default). The pivoted ids are excluded from this round's
    # invest commitment below (no pivot→re-enter haircut churn).
    pivot_count, pivot_recovered, pivot_committed, pivoted_opportunity_ids =
        review_and_pivot_investments!(agent, round_num, perception;
                                      uncertainty_env=uncertainty_env)
    decision_experience_stats = _recent_outcome_experience_stats(agent.recent_outcomes)

    # Calculate utilities for each action. estimate_operational_costs gives a
    # perceived cost reflecting the three components actually charged: the
    # sector-specific base (agent.operating_cost_estimate), the OPS_COST_INTENSITY
    # knob, and the competition multiplier from the agent's own active
    # investments. The round-local severity multiplier (bounded [0.7, 1.9]) is a
    # market-wide stochastic variable not visible at decision time, so it is
    # excluded here: agents react to expected burn, with realized burn
    # fluctuating around it via severity.
    estimated_cost = estimate_operational_costs(agent, market)

    capital_before_decision_analysis = get_capital(agent)
    invest_utility = calculate_investment_utility(agent, opportunities, market_conditions, perception; ai_level=decision_ai_level, info_system=info_system)
    decision_ai_analysis_cost = max(0.0, capital_before_decision_analysis - get_capital(agent))
    # S3a hook (AGI strategy ladder, the design notes):
    # complement-seeking explore/innovate shift, computed AFTER
    # calculate_investment_utility so the agent's own Information objects for
    # this round's visible set are already in the cache (re-reading paid-for
    # information: no new charge, no RNG). Reactive mode short-circuits to 0.0
    # before any strategy computation (bit-identical default).
    strategy_action_shift = strategy_active(agent.config, decision_ai_level) ?
        strategy_complement_action_shift(
            agent.config, opportunities, info_system, decision_ai_level,
            agent.id, perception) :
        0.0
    # A2 directed creation: the agent's seat-level perceived sector density,
    # computed from its visible set only when the channel is enabled (the
    # disabled path performs no computation and threads `nothing`, leaving the
    # innovation path bit-identical).
    sector_density = agent.config.ENABLE_DIRECTED_CREATION ?
        perceived_sector_density(opportunities) : nothing
    innovate_utility = calculate_innovation_utility(agent, market_conditions, perception; ai_level=decision_ai_level, strategy_shift=strategy_action_shift)
    explore_utility = calculate_exploration_utility(agent, perception; ai_level=decision_ai_level, strategy_shift=strategy_action_shift)
    maintain_utility = calculate_maintain_utility(agent, market_conditions, perception; estimated_cost=estimated_cost)

    # Select action
    utilities = Dict(
        "invest" => invest_utility,
        "innovate" => innovate_utility,
        "explore" => explore_utility,
        "maintain" => maintain_utility
    )

    # Can't invest without opportunities
    if isempty(opportunities)
        utilities["invest"] = 0.0
    end

    # Persistent idiosyncratic action tastes. These are sampled once at agent
    # creation and centered, so they add heterogeneity without changing the
    # population mean preference for any action.
    biased_utilities = Dict{String,Float64}()
    for (action, util) in utilities
        biased_utilities[action] = util + get(agent.action_biases, action, 0.0)
    end
    if isempty(opportunities)
        biased_utilities["invest"] = 0.0
    end

    # Add noise and select
    noisy_utilities = Dict{String,Float64}()
    for (action, util) in biased_utilities
        noise = agent.config.ACTION_SELECTION_NOISE * randn(agent.rng)
        noisy_utilities[action] = max(0.0, util + noise)
    end

    # With no visible opportunities, investing is impossible — remove it from
    # the choice set entirely. Merely zeroing the utility leaves softmax weight
    # exp(0/T) on "invest", and the sampled-but-impossible action would execute
    # maintain semantics while being recorded as an invest in round stats.
    if isempty(opportunities)
        delete!(noisy_utilities, "invest")
    end

    # Softmax selection. This is the SINGLE action-selection chokepoint:
    # every live agent's invest/innovate/explore/maintain choice is sampled
    # from this softmax each round.
    #
    # Emergence audit P10 (DECISION_TEMPERATURE = T): T divides the utilities
    # pre-softmax by multiplying the calibrated base temperature —
    # exp(u / (base * T)) ≡ exp((u / T) / base). T > 1 flattens the choice
    # distribution toward uniform (a mixed-equilibrium conjecture:
    # noisy humans approximate the mixed-strategy congestion equilibrium);
    # T < 1 sharpens toward argmax. BIT-IDENTITY at T = 1.0: the default path
    # short-circuits before the multiply (dividing by literal 1.0 would be
    # FP-identical anyway, but the guard makes exactness structural) and
    # consumes no extra RNG. Validated > 0 here at first use (loud-failure
    # convention) and at initialize!.
    temperature = agent.config.ACTION_SELECTION_TEMPERATURE
    decision_temperature = agent.config.DECISION_TEMPERATURE
    if decision_temperature != 1.0
        (decision_temperature > 0.0) || error(
            "Invalid DECISION_TEMPERATURE=$(decision_temperature) at first " *
            "use (make_decision! softmax). Require > 0.")
        temperature *= decision_temperature
    end
    total = sum(exp(u / temperature) for u in values(noisy_utilities))
    if total > 0
        probs = Dict(a => exp(u / temperature) / total for (a, u) in noisy_utilities)
    else
        probs = Dict("maintain" => 1.0, "invest" => 0.0, "innovate" => 0.0, "explore" => 0.0)
    end

    # Sample action
    r = rand(agent.rng)
    cumsum = 0.0
    chosen_action = "maintain"
    for action in ["invest", "innovate", "explore", "maintain"]
        cumsum += get(probs, action, 0.0)
        if r <= cumsum
            chosen_action = action
            break
        end
    end

    # Execute chosen action
    outcome = if chosen_action == "invest" && !isempty(opportunities)
        evals = evaluate_portfolio_opportunities(
            agent, opportunities, market_conditions, perception;
            ai_level=decision_ai_level,
            info_system=info_system,
            early_signals=early_signals,
            signal_weight=signal_weight
        )
        # Same-round hygiene (design note): an opportunity
        # the agent just liquidated this round cannot be re-entered in the
        # same round — re-committing at full price seconds after paying the
        # liquidation haircut is pure churn, not a decision.
        if !isempty(pivoted_opportunity_ids)
            evals = filter(e -> !(e.opportunity.id in pivoted_opportunity_ids), evals)
        end
        if !isempty(evals)
            selected_eval = select_portfolio_evaluation(agent, evals, decision_ai_level, perception)
            best_eval = isnothing(selected_eval) ? evals[1] : selected_eval
            best_opp = best_eval.opportunity
            estimated_return = best_eval.estimated_return
            ai_info_for_invest = best_eval.ai_info
            # F1: the RAW instrument estimate is what the InformationSystem
            # actually said (best_eval.estimated_return may be S1-shaded).
            # When no Information object exists (diagnostic path) the two
            # coincide by construction.
            instrument_estimated_return = ai_info_for_invest isa Information ?
                Float64(ai_info_for_invest.estimated_return) : best_eval.estimated_return
            # Extract confidence (from perception) and signal_score (the
            # evaluation's final_score) so _execute_invest! can size the bet by them.
            decision_conf = Float64(get(perception, "decision_confidence", 0.5))
            final_score = best_eval.final_score
            execute_action!(agent, "invest", market, round_num;
                opportunity=best_opp,
                estimated_return=estimated_return,
                instrument_estimated_return=instrument_estimated_return,
                ai_info=ai_info_for_invest isa Information ? ai_info_for_invest : nothing,
                innovation_engine=innovation_engine,
                market_conditions=market_conditions,
                uncertainty_perception=perception,
                confidence=decision_conf,
                signal_score=final_score)
        else
            execute_action!(agent, "maintain", market, round_num;
                innovation_engine=innovation_engine,
                market_conditions=market_conditions,
                uncertainty_perception=perception)
        end
    else
        execute_action!(agent, chosen_action, market, round_num;
            innovation_engine=innovation_engine,
            market_conditions=market_conditions,
            uncertainty_perception=perception,
            sector_density=sector_density)
    end

    # A1 pivot telemetry (open-action extension): per-decision counts summed
    # into compile_round_stats' per-round pivot columns.
    if agent.config.ENABLE_PIVOT
        outcome["pivot_count"] = pivot_count
        outcome["pivot_recovered"] = pivot_recovered
        outcome["pivot_committed"] = pivot_committed
    end

    outcome["ai_level_used"] = ai_level
    outcome["ai_behavior_level"] = decision_ai_level
    outcome["ai_used"] = counts_as_ai_use(agent.config, ai_level)
    outcome["utilities"] = utilities
    outcome["biased_utilities"] = biased_utilities
    outcome["noisy_utilities"] = noisy_utilities
    outcome["action_probabilities"] = probs
    outcome["action_biases"] = copy(agent.action_biases)
    outcome["visible_opportunity_ids"] = [opp.id for opp in opportunities]
    outcome["n_visible_opportunities"] = length(opportunities)
    outcome["experience_stats"] = decision_experience_stats
    outcome["confidence_diagnostics"] = copy(perception.confidence_diagnostics)
    action_ai_analysis_cost = Float64(get(outcome, "ai_analysis_cost", 0.0))
    outcome["ai_decision_analysis_cost"] = decision_ai_analysis_cost
    outcome["ai_action_analysis_cost"] = Float64(get(outcome, "ai_action_analysis_cost", action_ai_analysis_cost))
    outcome["ai_total_analysis_cost"] = decision_ai_analysis_cost + action_ai_analysis_cost
    decision_token_cost = _ai_cost_model(agent.config) in ("hybrid", "token") ? decision_ai_analysis_cost : 0.0
    action_token_cost = Float64(get(outcome, "ai_token_cost", 0.0))
    outcome["ai_decision_token_cost"] = decision_token_cost
    outcome["ai_total_token_cost"] = decision_token_cost + action_token_cost
    outcome["perception"] = perception
    # Alias under the key the uncertainty module reads when computing knowledge
    # gaps; without it the knowledge_gaps merge would always see an empty Dict.
    outcome["perception_at_decision"] = perception

    return outcome
end

# ============================================================================
# NEIGHBOR SIGNALS AND SOCIAL INFLUENCE
# ============================================================================

"""
Collect signals from neighbor agents for social influence.
"""
function collect_neighbor_signals(
    agent::EmergentAgent,
    neighbors::Vector{EmergentAgent}
)::Dict{String,Any}
    signals = Dict{String,Any}()

    if isempty(neighbors)
        return signals
    end

    alive_neighbors = filter(n -> n.alive && n.id != agent.id, neighbors)
    if isempty(alive_neighbors)
        return signals
    end

    # AI distribution among neighbors
    ai_counts = Dict{String,Int}()
    for n in alive_neighbors
        level = get_ai_level(n)
        ai_counts[level] = get(ai_counts, level, 0) + 1
    end
    total_neighbors = length(alive_neighbors)
    ai_distribution = Dict(k => v / total_neighbors for (k, v) in ai_counts)
    signals["ai_distribution"] = ai_distribution

    # AI adoption pressure (fraction using non-none AI)
    ai_users = count(n -> get_ai_level(n) != "none", alive_neighbors)
    signals["ai_adoption_pressure"] = ai_users / total_neighbors

    # Peer ROI comparison
    agent_roi = compute_roic(agent.resources.performance)
    neighbor_rois = [compute_roic(n.resources.performance) for n in alive_neighbors]
    avg_neighbor_roi = isempty(neighbor_rois) ? 0.0 : mean(neighbor_rois)
    signals["peer_roi_gap"] = avg_neighbor_roi - agent_roi

    # Average capital
    signals["avg_neighbor_capital"] = mean(get_capital(n) for n in alive_neighbors)

    return signals
end

# ============================================================================
# OUTCOME PROCESSING AND TRAIT EVOLUTION
# ============================================================================

function update_uncertainty_response_from_outcome!(
    agent::EmergentAgent,
    outcome::Dict{String,Any}
)::Nothing
    levels_any = get(outcome, "uncertainty_levels_at_decision", nothing)
    levels = Dict{String,Float64}()

    if levels_any isa Dict
        for (k, v) in levels_any
            if v isa Number
                levels[string(k)] = Float64(v)
            end
        end
    elseif haskey(outcome, "perception")
        perception = outcome["perception"]
        if perception isa Perception
            levels = _perception_uncertainty_levels(perception)
        end
    end

    isempty(levels) && return nothing

    amount = _outcome_float(outcome, ("investment_amount", "amount", "rd_spend", "explore_cost"))
    returned = _outcome_float(outcome, ("capital_returned", "innovation_return", "recovery"))
    multiple = _outcome_float(outcome, ("return_multiple", "cash_multiple"))
    if isnothing(multiple) && !isnothing(amount) && !isnothing(returned) && amount > 0
        multiple = returned / amount
    elseif isnothing(multiple)
        multiple = Bool(get(outcome, "success", false)) ? 1.5 : 0.5
    end

    success = Bool(get(outcome, "success", multiple >= 1.0 + max(0.0, agent.config.INVESTMENT_SUCCESS_ROI_THRESHOLD)))
    profile = agent.uncertainty_response
    for uncertainty_type in keys(profile.response_weights)
        haskey(levels, uncertainty_type) || continue
        push!(profile.outcome_history[uncertainty_type], (Float64(levels[uncertainty_type]), Float64(multiple), success))
        while length(profile.outcome_history[uncertainty_type]) > profile.memory_limit
            popfirst!(profile.outcome_history[uncertainty_type])
        end
        update_response_weight!(profile, uncertainty_type, Dict{String,Any}())
    end

    return nothing
end

"""
Update agent state from an outcome.
"""
function update_state_from_outcome!(
    agent::EmergentAgent,
    outcome::Dict{String,Any};
    ai_was_accurate::Union{Bool,Nothing} = nothing
)
    # This function performs no capital bookkeeping: the caller applies capital
    # changes before calling it (in the matured-investment path,
    # process_matured_investments! has already credited capital_returned).
    # Crediting capital here as well would double-count every matured investment.
    # Its job is purely state evolution: AI-trust learning and trait evolution.

    # Track AI accuracy
    ai_used = get(outcome, "ai_used", false)
    if ai_used && !isnothing(ai_was_accurate)
        update_ai_trust!(agent, ai_was_accurate, outcome)
    end

    # Update traits from experience
    was_successful = get(outcome, "success", false)
    evolve_traits_from_experience!(agent, was_successful, ai_used; ai_was_accurate=ai_was_accurate)
    update_uncertainty_response_from_outcome!(agent, outcome)

    # Store outcome
    agent.last_outcome = outcome
end

"""
Update AI trust based on accuracy.
"""
function update_ai_trust!(
    agent::EmergentAgent,
    ai_was_accurate::Bool,
    outcome::Dict{String,Any}
)
    current_trust = agent.ai_trust
    base_rate = agent.config.AI_TRUST_ADJUSTMENT_RATE

    # Attribution based on analytical ability
    attribution = 1.0 - get(agent.traits, "analytical_ability", 0.5)

    # Surprise factor for inaccurate high-confidence predictions
    surprise = 1.0
    confidence = get(outcome, "ai_confidence", 0.5)
    if !ai_was_accurate && confidence > 0.75
        surprise = 1.5
    end

    effective_rate = base_rate * attribution * surprise

    if ai_was_accurate
        agent.ai_trust = current_trust + (1.0 - current_trust) * effective_rate
    else
        agent.ai_trust = current_trust - current_trust * effective_rate
    end

    agent.ai_trust = clamp(agent.ai_trust, 0.0, 1.0)
    agent.traits["ai_trust"] = agent.ai_trust
end

"""
Evolve agent traits based on experience.
"""
function evolve_traits_from_experience!(
    agent::EmergentAgent,
    successful_action::Bool,
    used_ai::Bool;
    ai_was_accurate::Union{Bool,Nothing} = nothing
)
    # Competence evolution
    target_competence = successful_action ? 1.0 : 0.0
    current_competence = get(agent.traits, "competence", 0.5)

    base_momentum = agent.trait_momentum
    cognitive_mult = get(agent.traits, "cognitive_style", 0.8)
    effective_momentum = base_momentum * cognitive_mult

    new_competence = effective_momentum * current_competence + (1.0 - effective_momentum) * target_competence
    agent.traits["competence"] = clamp(new_competence, 0.0, 1.0)
    agent.competence = agent.traits["competence"]

    # Innovativeness evolves toward successful innovation
    if agent.last_action == "innovate"
        current_innov = agent.innovativeness
        target_innov = successful_action ? min(1.0, current_innov + 0.1) : max(0.0, current_innov - 0.05)
        agent.innovativeness = 0.9 * current_innov + 0.1 * target_innov
        agent.traits["innovativeness"] = agent.innovativeness
    end
end

_safe_ratio(numerator::Real, denominator::Integer)::Float64 =
    denominator > 0 ? Float64(numerator) / denominator : 0.0

_safe_ratio(numerator::Real, denominator::Real)::Float64 =
    denominator > 0 ? Float64(numerator) / Float64(denominator) : 0.0

_safe_std(values::AbstractVector{<:Real})::Float64 =
    length(values) < 2 ? 0.0 : std(Float64.(values))

"""
Record a confidence-outcome observation.

The rolling `weighted_gap` is retained as a smoothed diagnostic, while the
companion accumulators store raw unweighted gaps for calibration analysis.
"""
function record_confidence_outcome_observation!(
    agent::EmergentAgent,
    decision_confidence::Float64,
    realized_roi::Float64;
    ai_used::Bool = false,
    channel::String = "unknown"
)
    diag = agent.confidence_outcome_diagnostics
    decision_conf = clamp(decision_confidence, 0.05, 0.99)
    realized_multiple = max(realized_roi, 0.0)
    roi_value = clamp(realized_multiple, 0.0, 3.0)

    # Convert realized multiple to the same bounded score used by confidence.
    roi_score = 1.0 / (1.0 + exp(-3.0 * (roi_value - 1.0)))
    gap = decision_conf - roi_score

    diag.obs_count += 1
    diag.raw_gap_sum += gap
    diag.abs_gap_sum += abs(gap)
    diag.positive_gap_count += gap > 0.0 ? 1 : 0
    diag.negative_gap_count += gap < 0.0 ? 1 : 0
    diag.decision_confidence_sum += decision_conf
    diag.outcome_score_sum += roi_score
    diag.realized_multiple_sum += realized_multiple
    diag.realized_multiple_sq_sum += realized_multiple^2
    push!(diag.realized_multiples, realized_multiple)

    channel_key = lowercase(channel)
    if channel_key == "investment"
        diag.investment_gap_sum += gap
        diag.investment_obs_count += 1
    elseif channel_key in ("innovate", "explore", "innovate_explore", "immediate")
        diag.innovate_explore_gap_sum += gap
        diag.innovate_explore_obs_count += 1
    end

    if ai_used
        diag.ai_gap_sum += gap
        diag.ai_obs_count += 1
    else
        diag.non_ai_gap_sum += gap
        diag.non_ai_obs_count += 1
    end

    # Preserve the original smoothed diagnostic semantics, but expose it under
    # confidence-outcome naming so the sign is not mistaken for a causal effect.
    weight = ai_used ? 1.2 : 0.8
    inertia = 0.85
    new_signal = inertia * diag.weighted_gap + (1.0 - inertia) * gap * weight
    diag.weighted_gap = clamp(new_signal, -1.0, 1.0)
end

"""
Aggregate confidence-outcome diagnostics over a set of agents.
"""
function confidence_outcome_stats(agents::AbstractVector{<:EmergentAgent})::Dict{String,Any}
    observed = [a for a in agents if a.confidence_outcome_diagnostics.obs_count > 0]
    weighted_gaps = Float64[
        a.confidence_outcome_diagnostics.weighted_gap for a in observed
    ]

    obs_count = 0
    raw_gap_sum = 0.0
    abs_gap_sum = 0.0
    positive_count = 0
    negative_count = 0
    decision_confidence_sum = 0.0
    outcome_score_sum = 0.0
    realized_multiple_sum = 0.0
    realized_multiple_sq_sum = 0.0
    investment_gap_sum = 0.0
    investment_obs_count = 0
    innovate_explore_gap_sum = 0.0
    innovate_explore_obs_count = 0
    ai_gap_sum = 0.0
    ai_obs_count = 0
    non_ai_gap_sum = 0.0
    non_ai_obs_count = 0
    realized_multiples = Float64[]

    for agent in agents
        diag = agent.confidence_outcome_diagnostics
        obs_count += diag.obs_count
        raw_gap_sum += diag.raw_gap_sum
        abs_gap_sum += diag.abs_gap_sum
        positive_count += diag.positive_gap_count
        negative_count += diag.negative_gap_count
        decision_confidence_sum += diag.decision_confidence_sum
        outcome_score_sum += diag.outcome_score_sum
        realized_multiple_sum += diag.realized_multiple_sum
        realized_multiple_sq_sum += diag.realized_multiple_sq_sum
        investment_gap_sum += diag.investment_gap_sum
        investment_obs_count += diag.investment_obs_count
        innovate_explore_gap_sum += diag.innovate_explore_gap_sum
        innovate_explore_obs_count += diag.innovate_explore_obs_count
        ai_gap_sum += diag.ai_gap_sum
        ai_obs_count += diag.ai_obs_count
        non_ai_gap_sum += diag.non_ai_gap_sum
        non_ai_obs_count += diag.non_ai_obs_count
        append!(realized_multiples, diag.realized_multiples)
    end

    realized_multiple_mean = _safe_ratio(realized_multiple_sum, obs_count)
    realized_multiple_var = obs_count > 1 ?
        max(0.0, realized_multiple_sq_sum / obs_count - realized_multiple_mean^2) :
        0.0

    return Dict{String,Any}(
        "confidence_outcome_observed_agents" => length(observed),
        "confidence_outcome_agent_coverage" => isempty(agents) ? 0.0 :
            length(observed) / length(agents),
        "confidence_outcome_observations" => obs_count,
        "confidence_outcome_weighted_gap_mean" => isempty(weighted_gaps) ? 0.0 : mean(weighted_gaps),
        "confidence_outcome_weighted_gap_std" => _safe_std(weighted_gaps),
        "confidence_outcome_weighted_abs_gap_mean" => isempty(weighted_gaps) ? 0.0 : mean(abs.(weighted_gaps)),
        "confidence_outcome_raw_gap_mean" => _safe_ratio(raw_gap_sum, obs_count),
        "confidence_outcome_abs_gap_mean" => _safe_ratio(abs_gap_sum, obs_count),
        "confidence_outcome_positive_gap_rate" => _safe_ratio(positive_count, obs_count),
        "confidence_outcome_negative_gap_rate" => _safe_ratio(negative_count, obs_count),
        "confidence_outcome_decision_confidence_mean" => _safe_ratio(decision_confidence_sum, obs_count),
        "confidence_outcome_score_mean" => _safe_ratio(outcome_score_sum, obs_count),
        "confidence_outcome_investment_gap_mean" => _safe_ratio(investment_gap_sum, investment_obs_count),
        "confidence_outcome_investment_observations" => investment_obs_count,
        "confidence_outcome_innovate_explore_gap_mean" => _safe_ratio(innovate_explore_gap_sum, innovate_explore_obs_count),
        "confidence_outcome_innovate_explore_observations" => innovate_explore_obs_count,
        "confidence_outcome_ai_gap_mean" => _safe_ratio(ai_gap_sum, ai_obs_count),
        "confidence_outcome_ai_observations" => ai_obs_count,
        "confidence_outcome_non_ai_gap_mean" => _safe_ratio(non_ai_gap_sum, non_ai_obs_count),
        "confidence_outcome_non_ai_observations" => non_ai_obs_count,
        "confidence_outcome_realized_multiple_mean" => realized_multiple_mean,
        "confidence_outcome_realized_multiple_std" => sqrt(realized_multiple_var),
        "confidence_outcome_realized_multiple_p50" => isempty(realized_multiples) ? 0.0 : quantile(realized_multiples, 0.50),
        "confidence_outcome_realized_multiple_p90" => isempty(realized_multiples) ? 0.0 : quantile(realized_multiples, 0.90),
        "confidence_outcome_realized_multiple_p95" => isempty(realized_multiples) ? 0.0 : quantile(realized_multiples, 0.95),
    )
end

# ============================================================================
# OPERATIONAL COSTS
# ============================================================================

"""
Estimate operational costs including competition pressure.
"""
function estimate_operational_costs(
    agent::EmergentAgent,
    market::MarketEnvironment
)::Float64
    # Use the sector-specific cost initialized in the agent constructor (the
    # sector midpoint of operational_cost_range), not the global
    # BASE_OPERATIONAL_COST, so sector heterogeneity in operating costs is
    # preserved in the production charge path (e.g. tech agents are charged more
    # than service agents).
    base_cost = agent.operating_cost_estimate

    # Apply OPS_COST_INTENSITY scaling (default 1.0). This is the single
    # operating-cost knob: scaling op costs as a calibration/refutation lever is
    # done by setting OPS_COST_INTENSITY (e.g. 0.5 or 0.25), since production
    # reads agent.operating_cost_estimate rather than BASE_OPERATIONAL_COST.
    base_cost *= agent.config.OPS_COST_INTENSITY

    # Competition pressure from active investments
    if !isempty(agent.active_investments)
        competitions = Float64[]
        for inv in agent.active_investments
            opp = get(inv, "opportunity", nothing)
            if !isnothing(opp) && opp isa Opportunity
                push!(competitions, opp.competition)
            end
        end
        if !isempty(competitions)
            avg_competition = mean(competitions)
            base_cost *= 1.0 + avg_competition * 0.2
        end
    end

    return base_cost
end
