"""
Core dataclasses used across the Glimpse ABM simulation.

This module defines the fundamental data structures that represent the
entities and interactions in the agent-based model of entrepreneurship
under Knightian uncertainty with AI augmentation.
"""

using Random
using Statistics

# ============================================================================
# INFORMATION
# ============================================================================

"""
Information about an opportunity - Enhanced for AI learning.
"""
mutable struct Information
    estimated_return::Float64
    estimated_uncertainty::Float64
    confidence::Float64
    insights::Vector{String}
    hidden_factors::Dict{String,Float64}
    domain::Union{String,Nothing}
    actual_accuracy::Union{Float64,Nothing}
    contains_hallucination::Bool
    bias_applied::Float64
    overconfidence_factor::Float64
end

function Information(;
    estimated_return::Float64,
    estimated_uncertainty::Float64,
    confidence::Float64,
    insights::Vector{String} = String[],
    hidden_factors::Dict{String,Float64} = Dict{String,Float64}(),
    domain::Union{String,Nothing} = nothing,
    actual_accuracy::Union{Float64,Nothing} = nothing,
    contains_hallucination::Bool = false,
    bias_applied::Float64 = 0.0,
    overconfidence_factor::Float64 = 1.0
)
    Information(
        estimated_return, estimated_uncertainty, confidence, insights,
        hidden_factors, domain, actual_accuracy, contains_hallucination,
        bias_applied, overconfidence_factor
    )
end

"""
Calculate information quality score.
"""
function quality_score(info::Information)::Float64
    base_quality = info.confidence * (1.0 - abs(get(info.hidden_factors, "bias", 0.0)))
    return base_quality / max(1.0, info.overconfidence_factor)
end

# ============================================================================
# OPPORTUNITY
# ============================================================================

"""
Market opportunity with latent characteristics that manifest through action.

This struct operationalizes the entrepreneurial concept that opportunities
are not simply "discovered" but are enacted through the interplay of
entrepreneurial action and market conditions.
"""
mutable struct Opportunity
    id::String
    latent_return_potential::Float64
    latent_failure_potential::Float64
    complexity::Float64
    discovered::Bool
    created_by::Union{Int,Nothing}
    discovery_round::Union{Int,Nothing}
    config::Union{EmergentConfig,Nothing}
    competition::Float64
    lifecycle_stage::String
    path_dependency::Float64
    sector::Union{String,Nothing}
    capital_requirements::Float64
    capital_history::Vector{Float64}
    realized_returns::Vector{Float64}
    time_to_maturity::Int
    market_share::Float64
    entry_barriers::Float64
    age::Int
    base_failure_potential::Union{Float64,Nothing}
    truly_unknown::Bool
    required_discovery_threshold::Float64
    combination_uncertainty::Float64
    combination_signature::Union{String,Nothing}
    market_impact::Float64
    origin_innovation_id::Union{String,Nothing}
    crowding_penalty::Float64
    component_scarcity::Float64
    # New fields for paradox mechanisms
    novelty_score::Float64              # 0.0 = established, 1.0 = very novel
    total_invested::Float64             # Track total investment for capacity constraints
    capacity::Float64                   # Max investment capacity
    disrupted_count::Int                # How many times disrupted by innovations
end

function Opportunity(;
    id::String,
    latent_return_potential::Float64 = 1.5,
    latent_failure_potential::Float64 = 0.7,
    complexity::Float64 = 0.5,
    discovered::Bool = false,
    created_by::Union{Int,Nothing} = nothing,
    discovery_round::Union{Int,Nothing} = nothing,
    config::Union{EmergentConfig,Nothing} = nothing,
    competition::Float64 = 0.0,
    lifecycle_stage::String = "emerging",
    path_dependency::Float64 = 0.0,
    sector::Union{String,Nothing} = nothing,
    capital_requirements::Float64 = 10000.0,
    capital_history::Vector{Float64} = Float64[],
    realized_returns::Vector{Float64} = Float64[],
    time_to_maturity::Int = 12,
    market_share::Float64 = 1.0,
    entry_barriers::Float64 = 1.0,
    age::Int = 0,
    base_failure_potential::Union{Float64,Nothing} = nothing,
    truly_unknown::Bool = false,
    required_discovery_threshold::Float64 = 0.0,
    combination_uncertainty::Float64 = 0.5,
    combination_signature::Union{String,Nothing} = nothing,
    market_impact::Float64 = 0.0,
    origin_innovation_id::Union{String,Nothing} = nothing,
    crowding_penalty::Float64 = 0.0,
    component_scarcity::Float64 = 0.5,
    # New fields for paradox mechanisms
    novelty_score::Float64 = 0.0,       # 0.0 = established, 1.0 = very novel
    total_invested::Float64 = 0.0,      # Track total investment
    capacity::Float64 = 0.0,            # 0 = sentinel: sample from config
    disrupted_count::Int = 0,           # Disruption counter
    # Seedable RNG for reproducible capacity sampling. Default falls back to
    # global rand() for backwards compatibility with ad-hoc Opportunity
    # construction, but production paths (market._create_realistic_opportunity,
    # create_niche_opportunity, spawn_opportunity_from_innovation!) MUST pass
    # the market RNG for bit-reproducibility.
    rng::Random.AbstractRNG = Random.default_rng()
)
    opp = Opportunity(
        id, latent_return_potential, latent_failure_potential, complexity,
        discovered, created_by, discovery_round, config, competition,
        lifecycle_stage, path_dependency, sector, capital_requirements,
        capital_history, realized_returns, time_to_maturity, market_share,
        entry_barriers, age, base_failure_potential, truly_unknown,
        required_discovery_threshold, combination_uncertainty,
        combination_signature, market_impact, origin_innovation_id,
        crowding_penalty, component_scarcity,
        novelty_score, total_invested, capacity, disrupted_count
    )

    # Initialize base_failure_potential if not set
    if isnothing(opp.base_failure_potential)
        opp.base_failure_potential = opp.latent_failure_potential
    end

    # Sample capacity from config IF the caller didn't pass an explicit value
    # (capacity==0.0 is the sentinel for "please sample"). Callers that need
    # a specific capacity — e.g., tests probing the crowding penalty, or
    # spawn_opportunity_from_innovation! which computes its own — pass
    # capacity=… and keep their explicit value. Uses the passed RNG so
    # reproducibility is preserved across seeds.
    if opp.capacity <= 0.0 && !isnothing(config) && hasfield(typeof(config), :OPPORTUNITY_BASE_CAPACITY)
        variance = hasfield(typeof(config), :OPPORTUNITY_CAPACITY_VARIANCE) ? config.OPPORTUNITY_CAPACITY_VARIANCE : 0.3
        opp.capacity = config.OPPORTUNITY_BASE_CAPACITY * (1.0 + (rand(rng) - 0.5) * 2 * variance)
    end

    return opp
end

"""
Convert latent return potential into a realized investment multiple.

This method operationalizes the core insight that entrepreneurial returns
are not predetermined but emerge through the interaction of opportunity
characteristics, market conditions, competitive dynamics, and execution
contingencies.
"""
function realized_return(
    opp::Opportunity,
    market_conditions::MarketConditions,
    investor_tier::Union{String,Nothing} = nothing;
    rng::Random.AbstractRNG = Random.default_rng()
)::Float64
    # Base multiple from latent potential
    base_multiple = clamp(coalesce(opp.latent_return_potential, 1.0), 0.3, 10.0)

    # Regime effects
    regime_return = Float64(market_conditions.regime_return_multiplier)
    base_multiple *= regime_return

    risk_signal = clamp(opp.latent_failure_potential, 0.05, 0.95)
    regime_failure = Float64(market_conditions.regime_failure_multiplier)
    risk_signal = clamp(risk_signal * regime_failure, 0.05, 0.95)

    # Lifecycle multipliers
    lifecycle_multipliers = Dict(
        "emerging" => 1.12,
        "growing" => 1.03,
        "mature" => 0.9,
        "declining" => 0.65,
    )
    lifecycle_factor = get(lifecycle_multipliers, opp.lifecycle_stage, 1.0)
    resilience = clamp(0.75 + 0.25 * (1.0 - risk_signal), 0.55, 1.45)

    # Sector clearing dynamics. sector_demand_adjustments (from
    # market.jl:get_demand_adjustments) is the single source of truth for the
    # demand/supply clearing-ratio signal: it already carries the correct sign
    # and is clamped to sane bounds, so realized_return reads it once here and
    # applies it multiplicatively.
    sector_key = coalesce(opp.sector, "unknown")

    sector_adjustments = market_conditions.sector_demand_adjustments
    sector_adjust = get(sector_adjustments, sector_key, nothing)
    if isa(sector_adjust, Dict)
        base_multiple *= Float64(get(sector_adjust, "return", 1.0))
        risk_signal = clamp(risk_signal * Float64(get(sector_adjust, "failure", 1.0)), 0.05, 0.95)
    end

    config = opp.config

    execution_multiplier = 1.0
    if !isnothing(config) && !isnothing(investor_tier) &&
       hasfield(typeof(config), :AI_EXECUTION_SUCCESS_MULTIPLIERS)
        execution_multiplier = max(
            0.01,
            Float64(get(config.AI_EXECUTION_SUCCESS_MULTIPLIERS, String(investor_tier), 1.0))
        )
        risk_signal = clamp(risk_signal / sqrt(execution_multiplier), 0.02, 0.95)
    end

    base_mean = base_multiple * lifecycle_factor * resilience

    # Uncertainty state effects
    unc_state = market_conditions.uncertainty_state
    agentic_state = if isa(unc_state, Dict)
        get(unc_state, "agentic_novelty", unc_state)
    else
        Dict{String,Any}()
    end

    novelty_signal = if isa(agentic_state, Dict)
        Float64(get(agentic_state, "novelty_potential", get(agentic_state, "level", 0.5)))
    else
        0.5
    end

    stored_scarcity = coalesce(opp.component_scarcity, 0.5)
    scarcity_signal = if isa(agentic_state, Dict)
        Float64(get(agentic_state, "component_scarcity", stored_scarcity))
    else
        stored_scarcity
    end
    scarcity_signal = clamp(0.5 * scarcity_signal + 0.5 * stored_scarcity, 0.0, 1.0)

    reuse_pressure = if isa(agentic_state, Dict)
        Float64(get(agentic_state, "reuse_pressure", 0.0))
    else
        0.0
    end

    # Scarcity ceiling
    scarcity_ceiling = 4.5 + 1.4 * scarcity_signal
    base_mean = clamp(base_mean, 0.2, min(15.0, scarcity_ceiling))

    # =========================================================================
    # CROWDING PENALTY
    # =========================================================================
    # The penalty is applied multiplicatively to the realized Pareto draw
    # below, not to base_mean: applied to base_mean it is absorbed by the
    # x_min floor on the draw, so it must hit the realized return directly to
    # propagate faithfully.
    convex_crowding_penalty = 0.0

    use_capacity_convexity = !isnothing(config) &&
        hasfield(typeof(config), :USE_CAPACITY_CONVEXITY_CROWDING) &&
        config.USE_CAPACITY_CONVEXITY_CROWDING

    if use_capacity_convexity
        # =====================================================================
        # CAPITAL-SATURATION CONVEXITY CROWDING
        # =====================================================================
        #   penalty    = λ · max(0, S/K_sat − 1)^γ
        #   net_return = base_return · exp(−penalty)
        #   S          = opp.total_invested / opp.capacity   (capital saturation)
        #
        # Crowding is keyed on dollar saturation, not competitor headcount: ten
        # $10k investments and one $10M investment impose materially different
        # crowding, and dollar saturation is the economically correct signal.
        # =====================================================================
        K_sat = hasfield(typeof(config), :CROWDING_CAPACITY_RATIO_K) ? config.CROWDING_CAPACITY_RATIO_K : 1.5
        γ = hasfield(typeof(config), :CROWDING_CONVEXITY_GAMMA) ? config.CROWDING_CONVEXITY_GAMMA : 2.0
        λ = hasfield(typeof(config), :CROWDING_STRENGTH_LAMBDA) ? config.CROWDING_STRENGTH_LAMBDA : 0.50

        # Capital saturation: how full is the opportunity relative to capacity?
        saturation = (hasfield(typeof(opp), :capacity) && opp.capacity > 0.0) ?
            (opp.total_invested / opp.capacity) : 0.0

        # Market-wide crowding adds a smaller rising-tide contribution.
        crowding_metrics = market_conditions.crowding_metrics
        crowding_index = Float64(get(crowding_metrics, "crowding_index", 0.25))
        effective_sat = saturation + crowding_index * 0.3

        excess = max(0.0, effective_sat / K_sat - 1.0)
        convex_crowding_penalty = λ * (excess ^ γ)

        # Scarcity and novelty adjustments.
        scarcity_bonus = 0.10 * scarcity_signal
        novelty_relief = max(0.0, novelty_signal - 0.5) * 0.15
        base_mean *= (1.0 + scarcity_bonus + novelty_relief)

    else
        # =====================================================================
        # ALTERNATIVE CROWDING MODEL (combination-HHI + square-root competition)
        # =====================================================================
        # Used only when USE_CAPACITY_CONVEXITY_CROWDING is false (the default
        # is true, i.e. the capital-saturation model above). Penalizes return
        # via knowledge-combination concentration and opportunity-level
        # competition rather than dollar saturation.
        combo_hhi = Float64(get(market_conditions.extras, "combo_hhi", 0.0))
        reuse_penalty = coalesce(opp.crowding_penalty, 0.0)
        crowding_penalty_legacy = (0.25 * combo_hhi + 0.2 * reuse_penalty + 0.1 * reuse_pressure) * max(0.0, 1.0 - scarcity_signal)
        scarcity_bonus = 0.15 * scarcity_signal
        novelty_relief = max(0.0, novelty_signal - 0.5) * 0.3

        structural_adjustment = clamp(
            1.0 - crowding_penalty_legacy + scarcity_bonus + novelty_relief,
            0.5,
            1.45
        )
        base_mean *= structural_adjustment

        # Market-level crowding
        crowding_metrics = market_conditions.crowding_metrics
        crowding_index = Float64(get(crowding_metrics, "crowding_index", 0.25))

        crowd_threshold = isnothing(config) ? 0.35 : config.RETURN_DEMAND_CROWDING_THRESHOLD
        if crowding_index > crowd_threshold
            crowd_penalty_extra = clamp(1.0 - 0.25 * (crowding_index - crowd_threshold), 0.5, 1.0)
            base_mean *= crowd_penalty_extra
        end

        # Opportunity-level competition penalty (square-root form)
        opp_comp_threshold = isnothing(config) ? 0.2 : config.OPPORTUNITY_COMPETITION_THRESHOLD
        opp_comp_penalty_rate = isnothing(config) ? 0.5 : config.OPPORTUNITY_COMPETITION_PENALTY
        opp_comp_floor = isnothing(config) ? 0.1 : config.OPPORTUNITY_COMPETITION_FLOOR

        if opp.competition > opp_comp_threshold
            excess_competition = opp.competition - opp_comp_threshold
            sqrt_factor = sqrt(excess_competition)
            competition_penalty = 1.0 - opp_comp_penalty_rate * sqrt_factor
            base_mean *= clamp(competition_penalty, opp_comp_floor, 1.0)
        end
    end


    # =========================================================================
    # POWER LAW RETURN DISTRIBUTION
    # =========================================================================
    # Venture returns empirically follow a power law (Pareto) distribution:
    # - Most investments return near the minimum (many failures/zombies)
    # - A small number of "home runs" drive portfolio returns
    # - Matches VC empirics: ~65% return <1×, ~25% return 1-3×, ~10% are outliers
    #
    # We use a Pareto distribution with:
    # - x_m (minimum): scales with opportunity quality (base_mean)
    # - α (shape): from config.POWER_LAW_SHAPE_A (typically 2.0-2.5)
    #
    # Lower α = heavier tails = more extreme outliers
    # α ≈ 2.0: Very heavy tails (early-stage VC)
    # α ≈ 2.5: Moderate tails (growth equity)
    # α ≈ 3.0: Lighter tails (late-stage/buyout)
    # =========================================================================

    volatility = Float64(market_conditions.volatility)
    regime = market_conditions.regime

    # Power law shape parameter - lower = heavier tails
    alpha = isnothing(config) ? 2.5 : config.POWER_LAW_SHAPE_A

    # Adjust alpha based on regime (more extreme outcomes in volatile regimes)
    regime_alpha_adjust = Dict(
        "crisis" => -0.4,    # Heavier tails in crisis
        "recession" => -0.2,
        "normal" => 0.0,
        "growth" => -0.1,    # Slightly heavier in growth (more unicorns)
        "boom" => -0.3       # Heavier in boom (bubbles create outliers)
    )
    alpha += get(regime_alpha_adjust, regime, 0.0)
    alpha = clamp(alpha, 1.5, 4.0)  # Keep alpha in reasonable range

    # Adjust alpha based on volatility (more volatile = heavier tails)
    alpha -= volatility * 0.3
    alpha = clamp(alpha, 1.5, 4.0)

    # The minimum return (x_m) scales with opportunity quality
    # Higher quality opportunities have higher floor returns
    # Calibrated so uncrowded median ≈ 0.95×, mean ≈ 1.2× (matches VC empirics:
    # ~55% loss rate, mean > median due to power-law tail, portfolio mean ~1.07×)
    x_min = clamp(base_mean * 0.24, 0.15, 5.0)

    # Sample from Pareto distribution: X = x_m / U^(1/α)
    u = rand(rng)
    # Avoid division by zero and extreme values
    u = clamp(u, 1e-6, 1.0 - 1e-6)
    pareto_draw = x_min / (u ^ (1.0 / alpha))

    # Pareto draw already scales with base_mean via x_min — no additional quality scaling needed
    scaled_return = pareto_draw

    if execution_multiplier != 1.0
        execution_return_boost = clamp(1.0 + 0.15 * log(execution_multiplier), 0.5, 2.0)
        scaled_return *= execution_return_boost
    end

    # Apply the convex-crowding penalty to the realized return after the Pareto
    # draw. The draw's x_min floor collapses penalties applied to base_mean, so
    # the penalty must multiply the realized return to fire faithfully.
    if convex_crowding_penalty > 0.0
        scaled_return *= exp(-convex_crowding_penalty)
    end

    # Downside risk adjustment — risk signal plus a hot-market variance term.
    # Hot markets (demand > supply) expose the investor to more contested-
    # entry outcomes, so downside widens. Capped at 2× baseline because
    # further ratio growth doesn't add meaningfully more variance at the
    # individual-opportunity level (the expected-value boost from
    # sector_demand_adjustments is already doing that work).
    sector_key_local = coalesce(opp.sector, "unknown")
    raw_ratio = if !isempty(market_conditions.sector_clearing_index)
        Float64(get(market_conditions.sector_clearing_index, sector_key_local,
                    market_conditions.aggregate_clearing_ratio))
    else
        Float64(market_conditions.aggregate_clearing_ratio)
    end
    hot_market_pressure = clamp(raw_ratio - 1.0, 0.0, 2.0)
    downside_weight = isnothing(config) ? 0.65 : config.DOWNSIDE_OVERSUPPLY_WEIGHT
    downside = clamp(0.2 + risk_signal * 0.4 + downside_weight * hot_market_pressure, 0.0, 2.0)

    # Beta shock for additional variance
    shock = rand(rng, Beta(1.5, 2.5))
    scaled_return *= clamp(1.0 - downside * shock * 0.5, 0.3, 1.2)

    # Idiosyncratic execution/market noise. This preserves the structural mean
    # but lets otherwise similar opportunities realize different outcomes.
    return_noise_scale = !isnothing(config) && hasfield(typeof(config), :RETURN_NOISE_SCALE) ?
        max(0.0, Float64(config.RETURN_NOISE_SCALE)) : 0.0
    if return_noise_scale > 0.0
        log_noise = return_noise_scale * randn(rng)
        scaled_return *= exp(log_noise - 0.5 * return_noise_scale^2)
    end

    # Final bounds. The 200× ceiling preserves the venture-scale Pareto
    # right-tail (real venture data shows the top ~0.1% of investments
    # returning 100–1000×); the capital-saturation convexity still constrains
    # crowded niches, so reaching the tail requires uncrowded + scarce + novel
    # opportunities. The 150 coefficient lets the headroom formula actually
    # reach 200× when scarcity and novelty both max out (20 + 150·1.2 = 200).
    scarcity_headroom = 20.0 + 150.0 * max(0.0, scarcity_signal + novelty_signal - 0.8)
    upper_bound = clamp(scarcity_headroom, 5.0, 200.0)
    # Lower bound is a return multiple in [0,1]: 0.0 = total loss, 1.0 = break-even.
    lower_bound = isnothing(config) ? 0.0 : clamp(config.RETURN_LOWER_BOUND, 0.0, 1.0)

    return clamp(scaled_return, lower_bound, upper_bound)
end

"""
Update opportunity lifecycle based on adoption rate.
"""
function update_lifecycle!(opp::Opportunity, adoption_rate::Float64)
    if opp.lifecycle_stage == "emerging" && adoption_rate > 0.2
        opp.lifecycle_stage = "growing"
    elseif opp.lifecycle_stage == "growing" && adoption_rate > 0.6
        opp.lifecycle_stage = "mature"
    elseif opp.lifecycle_stage == "mature" && adoption_rate > 0.8
        opp.lifecycle_stage = "declining"
    end
end

# ============================================================================
# INNOVATION
# ============================================================================

"""
Innovation artifact produced by the innovation engine.
"""
mutable struct Innovation
    id::String
    type::String
    knowledge_components::Vector{String}
    novelty::Float64
    quality::Float64
    round_created::Int
    creator_id::Int
    success::Union{Bool,Nothing}
    market_impact::Union{Float64,Nothing}
    ai_assisted::Bool
    ai_domains_used::Vector{String}
    sector::Union{String,Nothing}
    combination_signature::Union{String,Nothing}
    cash_multiple::Union{Float64,Nothing}
    scarcity::Union{Float64,Nothing}
    is_new_combination::Bool
    ai_level_used::String
end

function Innovation(;
    id::String,
    type::String,
    knowledge_components::Vector{String},
    novelty::Float64,
    quality::Float64,
    round_created::Int,
    creator_id::Int,
    success::Union{Bool,Nothing} = nothing,
    market_impact::Union{Float64,Nothing} = nothing,
    ai_assisted::Bool = false,
    ai_domains_used::Vector{String} = String[],
    sector::Union{String,Nothing} = nothing,
    combination_signature::Union{String,Nothing} = nothing,
    cash_multiple::Union{Float64,Nothing} = nothing,
    scarcity::Union{Float64,Nothing} = nothing,
    is_new_combination::Bool = false,
    ai_level_used::String = "none"
)
    Innovation(
        id, type, knowledge_components, novelty, quality, round_created,
        creator_id, success, market_impact, ai_assisted, ai_domains_used,
        sector, combination_signature, cash_multiple, scarcity,
        is_new_combination, ai_level_used
    )
end

"""
Estimate market potential based on innovation attributes.
"""
function calculate_potential(innov::Innovation, market_conditions::MarketConditions)::Float64
    base_potential = innov.quality * (0.5 + 0.5 * innov.novelty)

    type_modifiers = Dict(
        "incremental" => 0.7,
        "architectural" => 0.9,
        "radical" => 1.2,
        "disruptive" => 1.5,
    )
    type_factor = get(type_modifiers, innov.type, 1.0)

    regime = market_conditions.regime
    market_factor = if regime == "growth"
        1.2
    elseif regime == "crisis"
        innov.type == "disruptive" ? 1.5 : 0.7
    else
        1.0
    end

    return base_potential * type_factor * market_factor
end

# ============================================================================
# KNOWLEDGE
# ============================================================================

"""
Represents a piece of knowledge or capability.
"""
mutable struct Knowledge
    id::String
    domain::String
    level::Float64
    discovered_round::Int
    discovered_by::Union{Int,Nothing}
    parent_knowledge::Vector{String}
end

function Knowledge(;
    id::String,
    domain::String,
    level::Float64,
    discovered_round::Int,
    discovered_by::Union{Int,Nothing} = nothing,
    parent_knowledge::Vector{String} = String[]
)
    Knowledge(id, domain, level, discovered_round, discovered_by, parent_knowledge)
end

"""
Calculate compatibility between two knowledge pieces.
"""
function compatibility_with(k1::Knowledge, k2::Knowledge)::Float64
    if k1.domain == k2.domain
        return 0.8 + 0.2 * (1.0 - abs(k1.level - k2.level))
    end

    domain_adjacency = Dict(
        "technology" => ["process", "market"],
        "market" => ["technology", "business_model"],
        "process" => ["technology", "business_model"],
        "business_model" => ["market", "process"],
    )

    adjacent = get(domain_adjacency, k1.domain, String[])
    if k2.domain in adjacent
        return 0.4 + 0.2 * (1.0 - abs(k1.level - k2.level))
    end

    return 0.1 + 0.1 * (1.0 - abs(k1.level - k2.level))
end

# ============================================================================
# AI ANALYSIS
# ============================================================================

"""
Enhanced AI analysis with potential errors and biases.
"""
struct AIAnalysis
    estimated_return::Float64
    estimated_uncertainty::Float64
    confidence::Float64
    insights::Vector{String}
    hidden_factors::Dict{String,Float64}
    actual_accuracy::Float64
    contains_hallucination::Bool
    bias_applied::Float64
    domain::String
    false_insights::Vector{String}
    overconfidence_factor::Float64
end

function AIAnalysis(;
    estimated_return::Float64,
    estimated_uncertainty::Float64,
    confidence::Float64,
    insights::Vector{String} = String[],
    hidden_factors::Dict{String,Float64} = Dict{String,Float64}(),
    actual_accuracy::Float64,
    contains_hallucination::Bool,
    bias_applied::Float64,
    domain::String,
    false_insights::Vector{String} = String[],
    overconfidence_factor::Float64
)
    AIAnalysis(
        estimated_return, estimated_uncertainty, confidence, insights,
        hidden_factors, actual_accuracy, contains_hallucination,
        bias_applied, domain, false_insights, overconfidence_factor
    )
end

# ============================================================================
# AI LEARNING PROFILE
# ============================================================================

"""
Agent's learned understanding of AI capabilities, tracked per domain.

The `hallucination_experiences` field records agent-attributed suspected AI
disconfirmation — inferred from observable forecast error, confidence,
realized return, and recent misses — not privileged knowledge of AI errors.
"""
mutable struct AILearningProfile
    domain_trust::Dict{String,Float64}
    accuracy_estimates::Dict{String,Vector{Float64}}        # per-domain prediction accuracy track record
    # Counts agent-attributed suspected AI disconfirmation per domain (inferred
    # from observable outcomes, not privileged knowledge of AI errors).
    hallucination_experiences::Dict{String,Int}
    usage_count::Dict{String,Int}                           # per-domain AI call counter
    # Bayesian beliefs about AI tier effectiveness (alpha, beta of a Beta distribution)
    tier_beliefs::Dict{String,Dict{String,Float64}}
    # Per-tier rolling ROI window (sliding window of the last 12 matured
    # outcomes, maintained by process_matured_investments!). Lets choose_ai_level
    # weight tier-specific performance rather than a single global ROI signal.
    tier_roi_history::Dict{String,Vector{Float64}}
end

const DEFAULT_DOMAINS = ["market_analysis", "technical_assessment", "uncertainty_evaluation", "innovation_potential"]
const AI_TIERS = ["none", "basic", "advanced", "premium"]

# Neutral priors for all tiers: every tier starts at a Beta(2,2) prior mean of
# 0.50 and agents update beliefs from realized outcomes, so tier preferences
# emerge from experience rather than from a hardcoded prior bias.
const DEFAULT_TIER_PRIORS = Dict(
    "none" => Dict("alpha" => 2.0, "beta" => 2.0),      # Prior mean = 0.50 (neutral)
    "basic" => Dict("alpha" => 2.0, "beta" => 2.0),     # Prior mean = 0.50 (neutral)
    "advanced" => Dict("alpha" => 2.0, "beta" => 2.0),  # Prior mean = 0.50 (neutral)
    "premium" => Dict("alpha" => 2.0, "beta" => 2.0)    # Prior mean = 0.50 (neutral)
)

function AILearningProfile()
    # Initialize tier beliefs with default priors
    tier_beliefs = Dict{String,Dict{String,Float64}}()
    for tier in AI_TIERS
        tier_beliefs[tier] = copy(DEFAULT_TIER_PRIORS[tier])
    end

    AILearningProfile(
        Dict(d => 0.5 for d in DEFAULT_DOMAINS),
        Dict(d => Float64[] for d in DEFAULT_DOMAINS),  # accuracy_estimates
        Dict(d => 0 for d in DEFAULT_DOMAINS),          # suspected AI disconfirmation events
        Dict(d => 0 for d in DEFAULT_DOMAINS),          # usage_count
        tier_beliefs,
        Dict(t => Float64[] for t in AI_TIERS),  # tier_roi_history
    )
end

"""
Update trust based on AI accuracy.
"""
function update_trust!(profile::AILearningProfile, domain::String, was_accurate::Bool; magnitude::Float64 = 0.1)
    if was_accurate
        profile.domain_trust[domain] = min(1.0, profile.domain_trust[domain] + magnitude)
    else
        profile.domain_trust[domain] = max(0.0, profile.domain_trust[domain] - magnitude * 1.5)
    end
end

"""
Update Bayesian beliefs about AI tier effectiveness based on realized return multiplier.

Uses continuous evidence derived from the return multiplier.
Higher returns provide stronger positive evidence, lower returns provide negative evidence.
The evidence is computed as: stable_sigmoid((multiplier - 1.0) * 2.5), clamped to [0.02, 0.98].
"""
function update_tier_belief!(profile::AILearningProfile, tier::String, realized_multiplier::Float64)
    if !haskey(profile.tier_beliefs, tier)
        # Initialize with default prior if tier not found
        prior = get(DEFAULT_TIER_PRIORS, tier, Dict("alpha" => 2.0, "beta" => 2.0))
        profile.tier_beliefs[tier] = copy(prior)
    end

    belief = profile.tier_beliefs[tier]

    # Compute continuous evidence from the realized multiplier.
    multiplier = clamp(realized_multiplier, 0.0, 3.0)
    evidence = clamp(stable_sigmoid((multiplier - 1.0) * 2.5), 0.02, 0.98)

    belief["alpha"] += evidence
    belief["beta"] += (1.0 - evidence)

    # Cap beliefs to prevent extreme certainty (keep some exploration)
    max_strength = 50.0
    total = belief["alpha"] + belief["beta"]
    if total > max_strength
        scale = max_strength / total
        belief["alpha"] *= scale
        belief["beta"] *= scale
    end
end

"""
Update Bayesian beliefs about AI tier effectiveness based on binary success/failure.

This is a convenience method that converts success to a multiplier estimate:
- Success: multiplier = 1.5 (positive evidence)
- Failure: multiplier = 0.5 (negative evidence)
"""
function update_tier_belief!(profile::AILearningProfile, tier::String, was_successful::Bool)
    # Convert binary outcome to estimated multiplier for continuous update
    estimated_multiplier = was_successful ? 1.5 : 0.5
    update_tier_belief!(profile, tier, estimated_multiplier)
end

"""
Get the posterior mean belief about a tier's effectiveness.
"""
function get_tier_belief_mean(profile::AILearningProfile, tier::String)::Float64
    if !haskey(profile.tier_beliefs, tier)
        return 0.5  # Neutral prior
    end
    belief = profile.tier_beliefs[tier]
    total = belief["alpha"] + belief["beta"]
    return total > 0 ? belief["alpha"] / total : 0.5
end

"""
Determine whether to use AI for a specific domain.
"""
function should_use_ai_for_domain(
    profile::AILearningProfile,
    domain::String,
    ai_cost::Float64,
    capital_buffer::Float64;
    cost_type::String = "per_use"
)::Bool
    trust = get(profile.domain_trust, domain, 0.5)
    # Usage-based exploration bonus: under-used domains get a fixed bonus that
    # decays to zero once the agent has enough experience with them.
    usage = get(profile.usage_count, domain, 0)
    exploration_bonus = usage < 5 ? 0.3 : 0.0

    cost_factor = if cost_type == "subscription"
        1.0
    else
        available_buffer = max(capital_buffer - ai_cost, 1.0)
        cost_ratio = ai_cost / available_buffer
        max(0.0, 1.0 - cost_ratio)
    end

    threshold = 0.4 - exploration_bonus
    decision_score = (trust + exploration_bonus) * cost_factor
    return decision_score > threshold
end

"""
Get adjusted confidence based on learned AI trust.
"""
function get_adjusted_confidence(profile::AILearningProfile, ai_confidence::Float64, domain::String)::Float64
    trust = get(profile.domain_trust, domain, 0.5)
    # `hallucination_experiences` is retained as the compatibility field name,
    # but its behavioral content is observable suspected AI disconfirmation.
    hall_count = get(profile.hallucination_experiences, domain, 0)
    usage = get(profile.usage_count, domain, 1)
    hallucination_rate = hall_count / max(1, usage)
    adjustment = trust * (1.0 - hallucination_rate)
    return ai_confidence * adjustment
end

# ============================================================================
# PERFORMANCE TRACKER
# ============================================================================

"""
Tracks capital deployment and returns by action and AI tier for ROE/ROIC metrics.
"""
mutable struct PerformanceTracker
    initial_equity::Float64
    deployed_by_action::Dict{String,Float64}
    returned_by_action::Dict{String,Float64}
    deployments_by_ai::Dict{String,Dict{String,Float64}}
    returns_by_ai::Dict{String,Dict{String,Float64}}
    roi_events::Vector{Dict{String,Any}}
end

function PerformanceTracker(initial_equity::Float64)
    PerformanceTracker(
        initial_equity,
        Dict{String,Float64}(),
        Dict{String,Float64}(),
        Dict{String,Dict{String,Float64}}(),
        Dict{String,Dict{String,Float64}}(),
        Dict{String,Any}[]
    )
end

function normalize_action(action::Union{String,Nothing})::String
    return isnothing(action) ? "overall" : lowercase(action)
end

function normalize_ai_level_tracker(ai_level::Union{String,Nothing})::String
    return isnothing(ai_level) ? "none" : lowercase(strip(ai_level))
end

"""
Record capital deployment for an action.
"""
function record_deployment!(
    tracker::PerformanceTracker,
    action::String,
    amount::Float64;
    ai_level::String = "none",
    round_num::Union{Int,Nothing} = nothing
)
    if amount <= 0
        return
    end

    action_key = normalize_action(action)
    ai_key = normalize_ai_level_tracker(ai_level)

    tracker.deployed_by_action[action_key] = get(tracker.deployed_by_action, action_key, 0.0) + amount
    tracker.deployed_by_action["overall"] = get(tracker.deployed_by_action, "overall", 0.0) + amount

    if !haskey(tracker.deployments_by_ai, action_key)
        tracker.deployments_by_ai[action_key] = Dict{String,Float64}()
    end
    tracker.deployments_by_ai[action_key][ai_key] = get(tracker.deployments_by_ai[action_key], ai_key, 0.0) + amount

    if !haskey(tracker.deployments_by_ai, "overall")
        tracker.deployments_by_ai["overall"] = Dict{String,Float64}()
    end
    tracker.deployments_by_ai["overall"][ai_key] = get(tracker.deployments_by_ai["overall"], ai_key, 0.0) + amount

    if !isnothing(round_num)
        push!(tracker.roi_events, Dict{String,Any}(
            "round" => round_num,
            "type" => "deployment",
            "action" => action_key,
            "ai_level" => ai_key,
            "amount" => amount
        ))
    end
end

"""
Record capital return for an action.
"""
function record_return!(
    tracker::PerformanceTracker,
    action::String,
    amount::Float64;
    ai_level::String = "none",
    round_num::Union{Int,Nothing} = nothing
)
    amount = max(0.0, amount)

    action_key = normalize_action(action)
    ai_key = normalize_ai_level_tracker(ai_level)

    tracker.returned_by_action[action_key] = get(tracker.returned_by_action, action_key, 0.0) + amount
    tracker.returned_by_action["overall"] = get(tracker.returned_by_action, "overall", 0.0) + amount

    if !haskey(tracker.returns_by_ai, action_key)
        tracker.returns_by_ai[action_key] = Dict{String,Float64}()
    end
    tracker.returns_by_ai[action_key][ai_key] = get(tracker.returns_by_ai[action_key], ai_key, 0.0) + amount

    if !haskey(tracker.returns_by_ai, "overall")
        tracker.returns_by_ai["overall"] = Dict{String,Float64}()
    end
    tracker.returns_by_ai["overall"][ai_key] = get(tracker.returns_by_ai["overall"], ai_key, 0.0) + amount

    if !isnothing(round_num)
        push!(tracker.roi_events, Dict{String,Any}(
            "round" => round_num,
            "type" => "return",
            "action" => action_key,
            "ai_level" => ai_key,
            "amount" => amount
        ))
    end
end

"""
Compute Return on Invested Capital for an action.
"""
function compute_roic(tracker::PerformanceTracker, action::Union{String,Nothing} = nothing)::Float64
    key = normalize_action(action)
    deployed = get(tracker.deployed_by_action, key, 0.0)
    if deployed <= 0
        return 0.0
    end
    returned = get(tracker.returned_by_action, key, 0.0)
    return (returned - deployed) / deployed
end

"""
Compute Return on Equity.
"""
function compute_roe(tracker::PerformanceTracker, current_capital::Float64)::Float64
    denominator = max(tracker.initial_equity, 1e-9)
    return (current_capital - tracker.initial_equity) / denominator
end

# ============================================================================
# PORTFOLIO
# ============================================================================

"""
Investment record within a portfolio.
"""
mutable struct InvestmentRecord
    opportunity_id::String
    amount::Float64
    sector::String
    entry_round::Int
    maturation_round::Int
    time_to_maturity::Int
    returns::Float64
    matured::Bool
    success::Bool
    raw_success::Bool
    defaulted::Bool
    ai_level_used::String
    estimated_return_at_investment::Float64
    estimated_uncertainty_at_investment::Float64
    ai_confidence_at_investment::Float64
    decision_confidence::Float64
    realized_multiplier::Float64
    realized_roi::Float64
    counterfactual_roi::Float64
end

function InvestmentRecord(;
    opportunity_id::String,
    amount::Float64,
    sector::String,
    entry_round::Int,
    maturation_round::Int,
    time_to_maturity::Int,
    ai_level_used::String = "none",
    estimated_return::Float64 = 0.0,
    estimated_uncertainty::Float64 = 0.5,
    ai_confidence::Float64 = 0.5,
    decision_confidence::Float64 = 0.5
)
    InvestmentRecord(
        opportunity_id, amount, sector, entry_round, maturation_round,
        time_to_maturity, 0.0, false, false, false, false, ai_level_used,
        estimated_return, estimated_uncertainty, ai_confidence,
        clamp(decision_confidence, 0.05, 0.99), 0.0, 0.0, 0.0
    )
end

"""
Manages an agent's collection of investments with proper maturation tracking.
"""
mutable struct Portfolio
    active_investments::Dict{String,InvestmentRecord}
    pending_investments::Dict{String,InvestmentRecord}
    matured_investments::Vector{InvestmentRecord}
    past_investments::Vector{Dict{String,Any}}
    total_invested::Float64
    locked_capital::Float64
    diversification_score::Float64
    config::Union{EmergentConfig,Nothing}
end

function Portfolio(config::Union{EmergentConfig,Nothing} = nothing)
    Portfolio(
        Dict{String,InvestmentRecord}(),
        Dict{String,InvestmentRecord}(),
        InvestmentRecord[],
        Dict{String,Any}[],
        0.0,
        0.0,
        0.0,
        config
    )
end

"""
Add a new investment to the portfolio.
"""
function add_investment!(
    portfolio::Portfolio,
    opp_id::String,
    amount::Float64,
    sector::String,
    round_num::Int,
    time_to_maturity::Int;
    ai_level_used::String = "none",
    estimated_return::Float64 = 0.0,
    estimated_uncertainty::Float64 = 0.5,
    ai_confidence::Float64 = 0.5,
    decision_confidence::Float64 = 0.5
)
    maturation_round = round_num + time_to_maturity

    investment = InvestmentRecord(
        opportunity_id = opp_id,
        amount = amount,
        sector = sector,
        entry_round = round_num,
        maturation_round = maturation_round,
        time_to_maturity = time_to_maturity,
        ai_level_used = ai_level_used,
        estimated_return = estimated_return,
        estimated_uncertainty = estimated_uncertainty,
        ai_confidence = ai_confidence,
        decision_confidence = decision_confidence
    )

    # Handle duplicate keys
    investment_key = opp_id
    if haskey(portfolio.pending_investments, investment_key)
        suffix = 1
        while haskey(portfolio.pending_investments, "$(opp_id)#$(suffix)")
            suffix += 1
        end
        investment_key = "$(opp_id)#$(suffix)"
    end

    portfolio.pending_investments[investment_key] = investment
    portfolio.active_investments[investment_key] = investment
    portfolio.total_invested += amount
    portfolio.locked_capital += amount

    update_diversification!(portfolio)
end

"""
Check for matured investments and process outcomes.
"""
function check_matured_investments!(
    portfolio::Portfolio,
    current_round::Int,
    market_conditions::MarketConditions;
    rng::Random.AbstractRNG = Random.default_rng()
)::Vector{Dict{String,Any}}
    newly_matured = Dict{String,Any}[]

    for (investment_key, investment) in collect(portfolio.pending_investments)
        if current_round >= investment.maturation_round
            # Calculate failure probability
            raw_risk = 0.5  # Default risk
            risk = clamp(raw_risk, 0.05, 0.95)

            # Market adjustments
            regime_failure = market_conditions.regime_failure_multiplier
            base_failure = 0.05 + 0.5 * risk * regime_failure

            # Crowding adjustments
            crowding_metrics = market_conditions.crowding_metrics
            crowd_idx = get(crowding_metrics, "crowding_index", 0.25)
            crowd_threshold = 0.35
            crowd_multiplier = crowd_idx > crowd_threshold ? 1.0 + 0.25 * (crowd_idx - crowd_threshold) : 1.0

            failure_chance = clamp(base_failure * crowd_multiplier, 0.05, 0.9)
            success = rand(rng) >= failure_chance

            amount = investment.amount
            if success
                # Successful investment
                realized_multiplier = 1.0 + 0.2 * rand(rng)  # Simple return model
                capital_returned = amount * realized_multiplier
                investment.defaulted = false
                investment.raw_success = true
            else
                # Failed investment - partial recovery
                recovery_ratio = clamp(0.2 + 0.2 * rand(rng), 0.0, 0.4)
                realized_multiplier = recovery_ratio
                capital_returned = amount * recovery_ratio
                investment.defaulted = rand(rng) < 0.2 * risk
                investment.raw_success = false
            end

            investment.returns = capital_returned
            investment.matured = true
            investment.realized_multiplier = realized_multiplier
            investment.realized_roi = realized_multiplier - 1.0
            investment.success = success && investment.realized_roi >= 0.0

            # Counterfactual ROI
            investment.counterfactual_roi = investment.realized_roi

            # Move to matured
            push!(portfolio.matured_investments, investment)
            delete!(portfolio.pending_investments, investment_key)
            delete!(portfolio.active_investments, investment_key)
            portfolio.locked_capital -= investment.amount

            push!(newly_matured, Dict{String,Any}(
                "opportunity_id" => investment.opportunity_id,
                "investment_amount" => investment.amount,
                "capital_returned" => capital_returned,
                "success" => investment.success,
                "raw_success" => investment.raw_success,
                "defaulted" => investment.defaulted,
                "net_return" => capital_returned - investment.amount,
                "ai_level_used" => investment.ai_level_used,
                "realized_multiplier" => realized_multiplier,
                "decision_confidence" => investment.decision_confidence
            ))
        end
    end

    if !isempty(newly_matured)
        update_diversification!(portfolio)
    end

    return newly_matured
end

"""
Update portfolio diversification score.
"""
function update_diversification!(portfolio::Portfolio)
    all_investments = merge(portfolio.pending_investments, portfolio.active_investments)

    if isempty(all_investments)
        portfolio.diversification_score = 0.0
        return
    end

    sector_amounts = Dict{String,Float64}()
    total_amount = 0.0

    for inv in values(all_investments)
        amount = inv.amount
        if amount <= 0
            continue
        end
        sector_amounts[inv.sector] = get(sector_amounts, inv.sector, 0.0) + amount
        total_amount += amount
    end

    if total_amount <= 0 || isempty(sector_amounts)
        portfolio.diversification_score = 0.0
        return
    end

    # Compute entropy-based diversification
    proportions = [amt / total_amount for amt in values(sector_amounts)]
    n_sectors = length(sector_amounts)
    max_entropy = n_sectors > 1 ? log(n_sectors) : 1.0  # Use actual sector count

    if max_entropy <= 0
        portfolio.diversification_score = 0.0
        return
    end

    entropy = -sum(p * log(p + 1e-12) for p in proportions)
    diversity = entropy / max_entropy

    portfolio.diversification_score = clamp(diversity, 0.0, 1.0)
end

"""
Get available capital for new investments.
"""
function get_available_capital(portfolio::Portfolio, total_capital::Float64)::Float64
    return max(0.0, total_capital - portfolio.locked_capital)
end

"""
Check if portfolio has pending investments.
"""
function has_pending_investments(portfolio::Portfolio)::Bool
    return !isempty(portfolio.pending_investments)
end

"""
Get portfolio size (number of active investments).
"""
function portfolio_size(portfolio::Portfolio)::Int
    return length(portfolio.active_investments)
end

# ============================================================================
# UNCERTAINTY RESPONSE PROFILE
# ============================================================================

"""
Tracks how an agent learns to respond to different uncertainty types.

Implements adaptive uncertainty response mechanism where entrepreneurs develop
heterogeneous response patterns to each Knightian uncertainty dimension.
"""
mutable struct UncertaintyResponseProfile
    exploration_rate::Float64
    memory_limit::Int
    learning_rate::Float64
    response_weights::Dict{String,Float64}
    outcome_history::Dict{String,Vector{Tuple{Float64,Float64,Bool}}}
end

function UncertaintyResponseProfile(;
    rng::Random.AbstractRNG = Random.default_rng(),
    learning_rate::Float64 = 0.05,
    initial_response_variance::Float64 = 0.3
)
    half_width = clamp(initial_response_variance, 0.0, 0.5)
    low = clamp(0.5 - half_width, 0.0, 1.0)
    high = clamp(0.5 + half_width, 0.0, 1.0)
    sample_weight() = low + (high - low) * rand(rng)

    UncertaintyResponseProfile(
        0.15,
        60,
        learning_rate,  # Use configurable learning rate
        Dict{String,Float64}(
            "actor_ignorance" => sample_weight(),
            "practical_indeterminism" => sample_weight(),
            "agentic_novelty" => sample_weight(),
            "competitive_recursion" => sample_weight()
        ),
        Dict{String,Vector{Tuple{Float64,Float64,Bool}}}(
            "actor_ignorance" => Tuple{Float64,Float64,Bool}[],
            "practical_indeterminism" => Tuple{Float64,Float64,Bool}[],
            "agentic_novelty" => Tuple{Float64,Float64,Bool}[],
            "competitive_recursion" => Tuple{Float64,Float64,Bool}[]
        )
    )
end

"""
Get response factor for an uncertainty type.
"""
function get_response_factor(
    profile::UncertaintyResponseProfile,
    uncertainty_type::String,
    uncertainty_level::Float64;
    explore::Bool = true,
    rng::Random.AbstractRNG = Random.default_rng()
)::Float64
    if explore && rand(rng) < profile.exploration_rate
        experimental_weight = rand(rng)
        return 1.0 - (uncertainty_level * experimental_weight)
    end

    weight = get(profile.response_weights, uncertainty_type, 0.5)
    return 1.0 - (uncertainty_level * weight)
end

"""
Update response profile from outcome.
"""
function update_from_outcome!(
    profile::UncertaintyResponseProfile,
    uncertainty_perception::Dict{String,Any},
    action::String,
    outcome::Dict{String,Any},
    market_conditions::Dict{String,Any}
)
    success = get(outcome, "success", false)
    investment_amount = max(get(outcome, "investment_amount", 1.0), 1.0)
    returns = get(outcome, "capital_returned", 0.0) / investment_amount

    for u_type in keys(profile.response_weights)
        if haskey(uncertainty_perception, u_type)
            u_data = uncertainty_perception[u_type]
            level = if isa(u_data, Dict)
                get(u_data, "level", get(u_data, "$(split(u_type, "_")[1])_level", 0.0))
            else
                0.0
            end

            push!(profile.outcome_history[u_type], (level, returns, success))

            # Trim to memory limit
            if length(profile.outcome_history[u_type]) > profile.memory_limit
                popfirst!(profile.outcome_history[u_type])
            end

            update_response_weight!(profile, u_type, market_conditions)
        end
    end
end

"""
Update response weight for a specific uncertainty type.
"""
function update_response_weight!(
    profile::UncertaintyResponseProfile,
    uncertainty_type::String,
    market_conditions::Dict{String,Any}
)
    history = profile.outcome_history[uncertainty_type]
    if isempty(history)
        return
    end

    recent = history[max(1, end - profile.memory_limit + 1):end]

    # Separate low and high uncertainty experiences
    low_uncertainty = [(r, s) for (l, r, s) in recent if l < 0.33]
    high_uncertainty = [(r, s) for (l, r, s) in recent if l >= 0.67]

    # Compute mean performance with prior
    function mean_with_prior(data, prior_mean = 1.0, prior_weight = 2.0)
        if isempty(data)
            return prior_mean
        end
        returns = [r for (r, _) in data]
        m = mean(returns)
        weight = length(returns)
        return (m * weight + prior_mean * prior_weight) / (weight + prior_weight)
    end

    perf_low = mean_with_prior(low_uncertainty)
    perf_high = mean_with_prior(high_uncertainty, perf_low)

    # Compute variability
    all_returns = [r for (_, r, _) in recent]
    variability = length(all_returns) > 1 ? std(all_returns) : 0.0

    base_weight = profile.response_weights[uncertainty_type]
    target_weight = base_weight

    if perf_high < perf_low * 0.85
        target_weight = min(1.0, base_weight + profile.learning_rate * (perf_low - perf_high))
    elseif perf_low < perf_high * 0.9
        target_weight = max(0.0, base_weight - profile.learning_rate * (perf_high - perf_low))
    else
        target_weight = base_weight * (1 - variability * 0.2)
    end

    stability = clamp(1.0 - variability, 0.3, 1.0)
    updated_weight = stability * target_weight + (1 - stability) * base_weight
    profile.response_weights[uncertainty_type] = clamp(updated_weight, 0.0, 1.0)
end

# ============================================================================
# OPPORTUNITY EVALUATION
# ============================================================================

"""
Per-opportunity scoring record produced by `evaluate_portfolio_opportunities`
and consumed by `make_decision!` when selecting the best opportunity to invest
in. Implemented as a typed struct (rather than a dict) because this is the
largest per-run allocation hotspot; the Base.get/getindex/haskey shims below
preserve dict-style reads for diagnostic and test code.
"""
struct OpportunityEvaluation
    opportunity::Opportunity
    final_score::Float64
    ai_level_used::String
    estimated_return::Float64
    competition_at_evaluation::Float64
    ai_info::Union{Information,Nothing}
    ai_contains_hallucination::Bool
    ai_confidence::Float64
    ai_actual_accuracy::Float64
end

# Keyword constructor with safe defaults for the optional AI fields. The
# previous dict producer omitted these entries entirely when no Information
# object was available; storing typed zeros plus an explicit `ai_info ===
# nothing` flag makes that condition checkable without `haskey`.
function OpportunityEvaluation(;
    opportunity::Opportunity,
    final_score::Float64,
    ai_level_used::String,
    estimated_return::Float64,
    competition_at_evaluation::Float64 = 0.0,
    ai_info::Union{Information,Nothing} = nothing,
    ai_contains_hallucination::Bool = false,
    ai_confidence::Float64 = 0.0,
    ai_actual_accuracy::Float64 = 0.0,
)
    OpportunityEvaluation(
        opportunity, final_score, ai_level_used, estimated_return,
        competition_at_evaluation, ai_info, ai_contains_hallucination,
        ai_confidence, ai_actual_accuracy,
    )
end

# Legacy Dict-like read access — same pattern as MarketConditions
# (market_conditions.jl:55-74). Returns the typed field if present; falls
# through to the supplied default only if the key is genuinely unknown.
function Base.get(ev::OpportunityEvaluation, key::AbstractString, default)
    sym = Symbol(key)
    return hasfield(OpportunityEvaluation, sym) ? getfield(ev, sym) : default
end

function Base.haskey(ev::OpportunityEvaluation, key::AbstractString)
    return hasfield(OpportunityEvaluation, Symbol(key))
end

function Base.getindex(ev::OpportunityEvaluation, key::AbstractString)
    sym = Symbol(key)
    hasfield(OpportunityEvaluation, sym) || throw(KeyError(key))
    return getfield(ev, sym)
end

# ============================================================================
# PERCEPTION
# ============================================================================
#
# Typed payload returned by uncertainty.jl::perceive_uncertainty (and its
# nested sub-structs), used instead of nested dicts because the perception
# payload is constructed once per agent per round and is a major allocation
# hotspot. Field names match the diagnostic keys so the Base.get/getindex/
# haskey shims (defined below) keep diagnostic and test code working. The shim
# follows the same contract as MarketConditions: an existing field always wins
# over the supplied default (so a present-but-zero field is never masked).

struct KnowledgeSignal
    confidence::Float64
    ignorance_level::Float64
    knowledge_gaps::Dict{String,Float64}
    gap_pressure::Float64
    info_quality::Float64
    info_breadth::Float64
    personal_knowledge::Float64
    knowledge_span::Float64
    discovery_momentum::Float64
    volatility::Float64
end

struct ActorIgnorance
    ignorance_level::Float64
    level::Float64                              # alias for ignorance_level
    knowledge_gaps::Dict{String,Float64}
    info_sufficiency::Float64
    discovery_momentum::Float64
    volatility::Float64
    gap_pressure::Float64
end

struct ExecutionRisk
    risk_level::Float64
    timing_pressure::Dict{String,Float64}
    market_regime_risk::Float64
    volatility::Float64
    crowding_pressure::Float64
    ai_pressure::Float64
    ai_herding_intensity::Float64
end

struct PracticalIndeterminism
    indeterminism_level::Float64
    level::Float64                              # alias for indeterminism_level
    timing_criticality::Dict{String,Float64}
    timing_pressure::Dict{String,Float64}
    market_regime_risk::Float64
    regime_stability::Float64
    volatility::Float64
    crowding_pressure::Float64
    ai_pressure::Float64
    ai_herding_intensity::Float64
end

struct InnovationSignal
    novelty_potential::Float64
    creative_confidence::Float64
    component_scarcity::Float64
    new_possibility_rate::Float64
    innovation_births::Int
    new_possibilities::Int
    combo_rate::Float64
    reuse_pressure::Float64
    adoption_rate::Float64
end

struct AgenticNovelty
    novelty_potential::Float64
    level::Float64                              # alias for novelty_potential
    creative_confidence::Float64
    disruption_potential::Dict{String,Float64}
    volatility::Float64
    new_possibility_rate::Float64
    innovation_births::Int
    new_possibilities::Int
    component_scarcity::Float64
    drivers::Dict{String,Any}                   # generic — derived from novelty_state
    combo_rate::Float64
    reuse_pressure::Float64
    adoption_rate::Float64
    tier_combo_rate::Float64
    tier_reuse_pressure::Float64
    tier_new_possibility_rate::Float64
    tier_adoption_rate::Float64
    global_combo_rate::Float64
    global_reuse_pressure::Float64
    tier_novelty_adjustment::Float64
end

struct CompetitionSignal
    pressure_level::Float64
    herding_awareness::Float64
    herding_pressure::Dict{String,Float64}
    ai_usage_share::Float64
    capital_crowding::Float64
    ai_herding_intensity::Float64
end

struct CompetitiveRecursion
    recursion_level::Float64
    level::Float64                              # alias for recursion_level
    herding_awareness::Float64
    herding_pressure::Dict{String,Float64}
    ai_herding_patterns::Dict{String,Float64}
    ai_herding_intensity::Float64
    strategic_opacity::Float64
    volatility::Float64
    ai_usage_share::Float64
end

struct Perception
    knowledge_signal::KnowledgeSignal
    actor_ignorance::ActorIgnorance
    execution_risk::ExecutionRisk
    practical_indeterminism::PracticalIndeterminism
    innovation_signal::InnovationSignal
    agentic_novelty::AgenticNovelty
    competition_signal::CompetitionSignal
    competitive_recursion::CompetitiveRecursion
    crowding_metrics::Dict{String,Float64}      # copy of market.crowding_metrics
    volatility_metric::Float64
    action_profile::Dict{String,Float64}
    decision_confidence::Float64
    confidence_diagnostics::Dict{String,Float64}
end

# Generate Base.get/getindex/haskey shims for every Perception-family struct
# in a single loop, matching the silent-zero-prevention pattern used by
# MarketConditions. Centralized here to avoid 27 hand-written method bodies.
const _PERCEPTION_TYPES = (
    KnowledgeSignal, ActorIgnorance, ExecutionRisk, PracticalIndeterminism,
    InnovationSignal, AgenticNovelty, CompetitionSignal, CompetitiveRecursion,
    Perception,
)

for T in _PERCEPTION_TYPES
    @eval function Base.get(s::$T, key::AbstractString, default)
        sym = Symbol(key)
        return hasfield($T, sym) ? getfield(s, sym) : default
    end
    @eval function Base.haskey(s::$T, key::AbstractString)
        return hasfield($T, Symbol(key))
    end
    @eval function Base.getindex(s::$T, key::AbstractString)
        sym = Symbol(key)
        hasfield($T, sym) || throw(KeyError(key))
        return getfield(s, sym)
    end
end

"""
Construct a neutral Perception with all "level"/confidence fields at 0.5 and
all dict-valued fields empty. Useful for diagnostic tests that need to call
`evaluate_portfolio_opportunities` without spinning up a full
KnightianUncertaintyEnvironment. Production code uses
`perceive_uncertainty` instead.
"""
function empty_perception()::Perception
    _empty = Dict{String,Float64}()
    Perception(
        KnowledgeSignal(0.5, 0.5, _empty, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ActorIgnorance(0.5, 0.5, _empty, 0.0, 0.0, 0.0, 0.0),
        ExecutionRisk(0.5, _empty, 0.0, 0.0, 0.0, 0.0, 0.0),
        PracticalIndeterminism(0.5, 0.5, _empty, _empty, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        InnovationSignal(0.5, 0.5, 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0),
        AgenticNovelty(0.5, 0.5, 0.5, _empty, 0.0, 0.0, 0, 0, 0.0,
                       Dict{String,Any}(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        CompetitionSignal(0.5, 0.0, _empty, 0.0, 0.0, 0.0),
        CompetitiveRecursion(0.5, 0.5, 0.0, _empty, _empty, 0.0, 0.0, 0.0, 0.0),
        Dict{String,Float64}(),
        0.0,
        Dict{String,Float64}(),
        0.5,
        Dict{String,Float64}(
            "decision_confidence" => 0.5,
            "return_evidence" => 0.0,
            "confidence_saturation_hit" => 0.0,
        ),
    )
end
