"""
Market environment for GlimpseABM.jl

This module implements the market dynamics that create the context for
entrepreneurial action under Knightian uncertainty.
"""

using Random
using Statistics
using Distributions

# ============================================================================
# MARKET ENVIRONMENT
# ============================================================================

"""
Dynamic market with emergent properties and realistic return dynamics.

This struct manages the market context in which entrepreneurial agents
operate, including opportunity generation, macroeconomic regime
transitions, supply-demand clearing, and crowding dynamics.
"""
mutable struct MarketEnvironment
    config::EmergentConfig
    n_agents::Int
    opportunities::Vector{Opportunity}
    opportunities_by_sector::Dict{String,Vector{Opportunity}}
    opportunity_map::Dict{String,Opportunity}
    current_round::Int
    market_regime::String
    volatility::Float64
    trend::Float64
    market_momentum::Float64
    price_history::Vector{Float64}
    volume_history::Vector{Float64}
    competition_levels::Dict{String,Float64}
    total_investment_by_round::Vector{Float64}
    failure_rate_by_round::Vector{Float64}
    new_ventures_by_round::Vector{Int}
    innovations::Vector{Innovation}
    exploration_activity::Int
    sectors::Vector{String}
    sector_profiles::Dict{String,SectorProfile}
    regime_return_multiplier::Float64
    regime_failure_multiplier::Float64
    tier_invest_share::Dict{String,Float64}
    sector_clearing_index::Dict{String,Float64}
    aggregate_clearing_ratio::Float64
    sector_demand_adjustments::Dict{String,Dict{String,Float64}}
    branch_params::Dict{String,Dict{String,Any}}
    crowding_metrics::Dict{String,Float64}
    rng::AbstractRNG
end

function MarketEnvironment(
    config::EmergentConfig;
    rng::AbstractRNG = Random.default_rng()
)
    initialize!(config)  # Ensure SECTORS is populated

    market = MarketEnvironment(
        config,
        config.N_AGENTS,
        Opportunity[],
        Dict{String,Vector{Opportunity}}(),
        Dict{String,Opportunity}(),
        0,
        "normal",
        config.MARKET_VOLATILITY,
        0.0,
        0.0,
        Float64[],
        Float64[],
        Dict{String,Float64}(),
        Float64[],
        Float64[],
        Int[],
        Innovation[],
        0,
        config.SECTORS,
        config.SECTOR_PROFILES,
        1.0,
        1.0,
        Dict("none" => 0.0, "basic" => 0.0, "advanced" => 0.0, "premium" => 0.0),
        Dict{String,Float64}(),
        1.0,
        Dict{String,Dict{String,Float64}}(),
        Dict{String,Dict{String,Any}}(),
        Dict{String,Float64}(),
        rng
    )

    # Initialize branch parameters for each sector
    for (sector, profile) in market.sector_profiles
        _initialize_branch!(market, sector, profile)
    end

    # Initialize opportunities
    _initialize_opportunities!(market)

    return market
end

"""
Initialize branch parameters for a sector.
"""
function _initialize_branch!(market::MarketEnvironment, name::String, profile::SectorProfile)
    log_mu = profile.return_log_mu
    log_sigma = clamp(profile.return_log_sigma * market.config.LOG_SIGMA_MULT, 0.05, market.config.LOG_SIGMA_CAP)
    failure_mean = (profile.failure_range[1] + profile.failure_range[2]) / 2
    failure_sigma = clamp(mean(profile.failure_volatility_range), 0.01, 0.5)

    market.branch_params[name] = Dict{String,Any}(
        "name" => name,
        "root" => split(name, "_")[1],
        "profile" => profile,
        "log_mu" => log_mu,
        "log_sigma" => log_sigma,
        "failure_mean" => failure_mean,
        "failure_sigma" => failure_sigma,
        "depth" => 0
    )
end

"""
Sample opportunity characteristics from a branch.
"""
function _sample_branch_characteristics(
    market::MarketEnvironment,
    branch_name::String;
    quality_roll::Union{Float64,Nothing} = nothing
)::Tuple{Float64,Float64,Float64,Int}
    params = get(market.branch_params, branch_name, nothing)
    if isnothing(params)
        # Create default params if branch doesn't exist
        profile = get(market.sector_profiles, branch_name, first(values(market.sector_profiles)))
        _initialize_branch!(market, branch_name, profile)
        params = market.branch_params[branch_name]
    end

    profile = params["profile"]
    log_mu = params["log_mu"]
    log_sigma = params["log_sigma"]

    quality_roll = isnothing(quality_roll) ? rand(market.rng) : quality_roll

    # Quality adjustments
    if quality_roll > 0.92
        log_mu += 0.18
        log_sigma *= 0.9
    elseif quality_roll < 0.28
        log_mu -= 0.12
        log_sigma *= 1.08
    end
    log_sigma = clamp(log_sigma, 0.05, market.config.LOG_SIGMA_CAP)

    # Sample latent return
    latent_return = exp(log_mu + log_sigma * randn(market.rng))
    latent_return = clamp(latent_return, profile.return_range[1], profile.return_range[2] * market.config.RETURN_RANGE_MAX_MULT)

    # Sample failure probability
    failure_mean = params["failure_mean"]
    failure_sigma = params["failure_sigma"]
    if quality_roll > 0.85
        failure_mean *= 0.92
    elseif quality_roll < 0.3
        failure_mean *= 1.08
    end

    latent_failure = failure_mean + failure_sigma * randn(market.rng)
    latent_failure = clamp(latent_failure, profile.failure_range[1], profile.failure_range[2])

    # Sample capital and maturity
    capital_req = rand(market.rng, Uniform(profile.capital_range...))
    maturity = rand(market.rng, profile.maturity_range[1]:profile.maturity_range[2])

    return latent_return, latent_failure, capital_req, maturity
end

"""
Create a realistic opportunity in a given sector.
"""
function _create_realistic_opportunity(
    market::MarketEnvironment,
    opp_id::String,
    sector::String;
    innovator_capability::Float64 = 0.5
)::Opportunity
    quality_roll = rand(market.rng)
    latent_return, latent_failure, capital_req, maturity = _sample_branch_characteristics(
        market, sector; quality_roll=quality_roll
    )

    # Regime effects are applied only at realization (in realized_return),
    # not at creation, to avoid double-counting regime multipliers.

    # Global sanity clamp only. The floor must sit BELOW every sector's
    # calibrated failure_range minimum (service is 0.033); a floor of 0.1
    # would censor the calibrated sector heterogeneity (it pinned every
    # service-sector opportunity at exactly 0.1).
    latent_return = clamp(latent_return, 0.5, market.config.RETURN_CLAMP_MAX)
    latent_failure = clamp(latent_failure, 0.02, 0.95)

    return Opportunity(
        id=opp_id,
        latent_return_potential=latent_return,
        latent_failure_potential=latent_failure,
        complexity=rand(market.rng, Uniform(0.3, 1.2)),
        discovered=true,
        discovery_round=0,
        config=market.config,
        sector=sector,
        capital_requirements=capital_req,
        time_to_maturity=maturity,
        rng=market.rng,
    )
end

"""
Initialize the initial set of opportunities.
"""
function _initialize_opportunities!(market::MarketEnvironment)
    n_opportunities = get_scaled_opportunities(market.config, market.n_agents)

    for i in 1:n_opportunities
        sector = rand(market.rng, market.sectors)
        opp = _create_realistic_opportunity(market, "initial_$(sector)_$(i)", sector)
        push!(market.opportunities, opp)
        market.opportunity_map[opp.id] = opp

        # Index by sector
        if !haskey(market.opportunities_by_sector, sector)
            market.opportunities_by_sector[sector] = Opportunity[]
        end
        push!(market.opportunities_by_sector[sector], opp)
    end
end

"""
Step the market environment forward one round.
"""
function step!(
    market::MarketEnvironment,
    round::Int,
    agent_actions::Vector{Dict{String,Any}},
    innovations::Vector{Innovation};
    matured_outcomes::Vector{Dict{String,Any}} = Dict{String,Any}[]
)::MarketConditions
    market.current_round = round
    # Do NOT clear market.sector_demand_adjustments here. realized_return reads
    # them from market_conditions during Phase 4 (process_matured_investments!),
    # which runs BEFORE market.step! this round, so clearing here would leave
    # the dict empty for every matured investment. The adjustments are a
    # one-round-stale cache: populated during Phase 5.5 (update_clearing_metrics!)
    # from this round's flow + clearing data, then read on the NEXT round's
    # Phase 4. One month of staleness is acceptable for a monthly sim, since
    # market-wide demand state doesn't fluctuate that fast.

    # Record innovations
    append!(market.innovations, innovations)

    # Calculate total investment
    invest_actions = filter(a -> get(a, "action", "") == "invest", agent_actions)
    total_investment = isempty(invest_actions) ? 0.0 : sum(
        Float64(get(a, "amount", get(a, "capital_deployed", 0.0)))
        for a in invest_actions
    )
    push!(market.total_investment_by_round, total_investment)

    # Update competition levels (uses COMPETITION_SCALE_FACTOR for robustness testing)
    # Normalize competition delta to preserve calibrated total pressure per opportunity.
    #
    # Original calibration at N=1K with reference_population=100:
    #   delta = 0.1 * (100/1000) = 0.01 per agent
    #   ~25 agents invest per opp → total pressure = 25 × 0.01 = 0.25 ✓
    #
    # For larger N, scale delta by 1/√(N/1000) from the calibrated baseline:
    #   N=10K:  delta = 0.01/√10 = 0.00316, ~76 agents/opp → pressure = 0.24
    #   N=100K: delta = 0.01/√100 = 0.001, ~446 agents/opp → pressure = 0.45
    #
    # Total pressure rises moderately with N (more agents per opp) but stays
    # in the same order of magnitude. The √N scaling keeps per-opportunity
    # pressure roughly invariant to population: 1/N scaling would let pressure
    # vanish at large N, while no scaling would let it explode.
    calibrated_delta = 0.01  # delta at N=1000 (calibrated value — do not change)
    scale_from_ref = Float64(market.config.N_AGENTS) / 1000.0
    population_normalized_delta = calibrated_delta / sqrt(max(1.0, scale_from_ref))

    for action in agent_actions
        if get(action, "action", "") == "invest"
            opp = get(action, "chosen_opportunity_obj", nothing)
            if !isnothing(opp) && isa(opp, Opportunity)
                update_opportunity_competition!(market, opp, population_normalized_delta)
            end
        elseif get(action, "action", "") == "explore"
            market.exploration_activity += 1
        end
    end

    # Regime transitions happen exactly once per round, in update_macro_regime!
    # (called from update_market_dynamics! in Phase 5.5). A second stochastic
    # transition here would apply the NBER-calibrated monthly transition matrix
    # ~1.1x per month, systematically shortening regime durations relative to
    # the documented calibration.

    # Calculate crowding metrics
    total_actions = length(agent_actions)
    if total_actions > 0
        action_counts = Dict{String,Int}()
        for action in agent_actions
            act_type = get(action, "action", "maintain")
            action_counts[act_type] = get(action_counts, act_type, 0) + 1
        end

        share_invest = get(action_counts, "invest", 0) / total_actions
        share_innovate = get(action_counts, "innovate", 0) / total_actions
        share_explore = get(action_counts, "explore", 0) / total_actions
        share_maintain = get(action_counts, "maintain", 0) / total_actions

        crowding_index = share_invest^2 + share_innovate^2 + share_explore^2 + share_maintain^2

        # Calculate behavioral AI usage share. Placebo labels can remain
        # non-none for grouping while action["ai_used"] suppresses AI dynamics.
        ai_usage_share = count(action_counts_as_ai_use, agent_actions) / max(1, total_actions)

        market.crowding_metrics = Dict(
            "crowding_index" => crowding_index,
            "share_invest" => share_invest,
            "share_innovate" => share_innovate,
            "share_explore" => share_explore,
            "share_maintain" => share_maintain,
            "ai_usage_share" => ai_usage_share
        )
    end

    # Calculate tier investment shares
    invest_actions = [a for a in agent_actions if get(a, "action", "") == "invest"]
    if !isempty(invest_actions)
        tier_counts = Dict("none" => 0, "basic" => 0, "advanced" => 0, "premium" => 0)
        for action in invest_actions
            tier = action_behavior_ai_level(action)
            if haskey(tier_counts, tier)
                tier_counts[tier] += 1
            end
        end
        n_invest = length(invest_actions)
        for tier in keys(tier_counts)
            market.tier_invest_share[tier] = tier_counts[tier] / n_invest
        end
    end

    return get_market_conditions(market)
end

"""
Construct the typed `MarketConditions` payload for this round.

`uncertainty_state` is passed as a kwarg and injected at construction so the
returned struct can stay immutable (the uncertainty snapshot is part of the
payload, not mutated onto it afterward).
"""
function get_market_conditions(market::MarketEnvironment;
                               uncertainty_state::Dict{String,Any}=Dict{String,Any}()
                              )::MarketConditions
    comp_values = [o.competition for o in market.opportunities if o.discovered]
    avg_competition = isempty(comp_values) ? 0.0 : mean(comp_values)

    # Copy dict fields so MarketConditions is a true snapshot. The struct is
    # immutable but its Dict contents are not: storing the market's live dicts
    # by reference would let a later round's update to `tier_invest_share` or
    # `sector_demand_adjustments` silently leak into MarketConditions instances
    # already handed out. Shallow copies suffice for Dict{String,Float64};
    # sector_demand_adjustments is nested, so it needs deepcopy.
    return MarketConditions(
        market.market_regime,
        market.volatility,
        market.regime_return_multiplier,
        market.regime_failure_multiplier,
        market.current_round,
        copy(market.tier_invest_share),
        copy(market.sector_clearing_index),
        market.aggregate_clearing_ratio,
        copy(market.crowding_metrics),
        deepcopy(market.sector_demand_adjustments),
        avg_competition,
        # deepcopy uncertainty_state so the snapshot is isolated from the live
        # KnightianUncertaintyEnvironment, including its nested dicts. A
        # same-pointer store would let a later round-end update to the env
        # mutate every outstanding MarketConditions.
        deepcopy(uncertainty_state),
        Dict{String,Any}(),  # extras — empty by default
    )
end

"""
Get available opportunities for an agent.
"""
function get_available_opportunities(market::MarketEnvironment)::Vector{Opportunity}
    return [opp for opp in market.opportunities if opp.discovered]
end

"""
Add a new opportunity to the market.
"""
function add_opportunity!(market::MarketEnvironment, opp::Opportunity)
    push!(market.opportunities, opp)
    market.opportunity_map[opp.id] = opp

    if !isnothing(opp.sector)
        if !haskey(market.opportunities_by_sector, opp.sector)
            market.opportunities_by_sector[opp.sector] = Opportunity[]
        end
        push!(market.opportunities_by_sector[opp.sector], opp)
    end
end

# ============================================================================
# SECTOR-SPECIFIC COMPETITION INTENSITY
# ============================================================================

"""
Get sector-specific competition intensity based on Census HHI data.

Competition intensity values calibrated from Census Bureau Economic Census:
- Tech: 1.2 (HHI 1500-2500, moderate concentration)
- Retail: 0.7 (HHI 500-1000, fragmented)
- Service: 0.9 (HHI 800-1500, moderately fragmented)
- Manufacturing: 1.4 (HHI 1800-3000, concentrated)

Returns base intensity scaled by global COMPETITION_SCALE_FACTOR for robustness testing.
"""
function get_sector_competition_intensity(market::MarketEnvironment, sector::String)::Float64
    profile = get(market.sector_profiles, sector, nothing)
    if isnothing(profile)
        base_sector = base_sector_name(sector, market.config.SECTORS)
        profile = get(market.sector_profiles, base_sector, nothing)
    end
    base_intensity = if !isnothing(profile) && hasproperty(profile, :competition_intensity)
        profile.competition_intensity
    else
        1.0  # Default intensity
    end

    # Apply global competition scale factor for robustness testing
    return base_intensity * market.config.COMPETITION_SCALE_FACTOR
end

"""
Apply sector-specific competition intensity to opportunity competition updates.
"""
function update_opportunity_competition!(market::MarketEnvironment, opp::Opportunity, delta::Float64)
    # Counterfactual mode: disable competition dynamics
    if market.config.DISABLE_COMPETITION_DYNAMICS
        opp.competition = 0.0
        return
    end

    # Normal mode: update competition based on agent activity
    #
    # Competition is modeled as a "competitive pressure index" that can exceed 1.0.
    # This is analogous to Herfindahl-Hirschman Index (HHI) which ranges 0-10,000,
    # or price impact in market microstructure which scales with trade size.
    #
    # Interpretation:
    # - competition = 0.0: No other investors (monopoly-like position)
    # - competition = 1.0: Normal competitive pressure (~10 investors)
    # - competition = 3.0: Severe crowding (~30 investors, 3x normal)
    #
    # This unbounded representation is essential for the AI Information Paradox:
    # - When 33 Frontier AI agents pile into ONE opportunity, competition ~3.3
    # - When 15 Human agents invest (spread across opportunities), competition ~1.5
    # - The penalty difference reflects the real economic cost of crowding
    #
    # Theoretical basis: Cournot competition (profits ∝ 1/n²) and
    # market microstructure (Kyle 1985: price impact ∝ √order_flow)
    intensity = get_sector_competition_intensity(market, opp.sector)

    # Per-investment update: pure accumulation. Time-decay (× 0.9) is applied
    # ONCE per round inside manage_opportunities!, not here per-investment.
    # Decaying on every invest call would cap accumulated competition at ~0.7
    # regardless of investment volume, effectively disabling crowding penalties.
    opp.competition = max(0.0, opp.competition + delta * intensity)
end

"""
Return how much observable evidence an opportunity emits.

This is an opportunity property, not an AI-tier penalty. Frontier AI can search
the signal space much more intensely, but opportunities with little external
trace remain hard to infer because there is little evidence to condition on.
"""
function opportunity_signal_strength(opp::Opportunity)::Float64
    complexity = clamp(Float64(opp.complexity) / 2.0, 0.0, 1.0)
    novelty = clamp(Float64(opp.novelty_score), 0.0, 1.0)
    combo_unc = clamp(Float64(opp.combination_uncertainty), 0.0, 1.0)
    threshold = clamp(Float64(opp.required_discovery_threshold), 0.0, 1.0)

    intrinsic_legibility = (
        0.35 * (1.0 - complexity) +
        0.25 * (1.0 - novelty) +
        0.20 * (1.0 - combo_unc) +
        0.20 * (1.0 - threshold)
    )

    capacity = max(Float64(opp.capacity), Float64(opp.capital_requirements) * 10.0, 1.0)
    invested_trace = 1.0 - exp(-max(0.0, Float64(opp.total_invested)) / capacity)
    competition_trace = 1.0 - exp(-max(0.0, Float64(opp.competition)) / 1.5)
    impact_trace = 1.0 - exp(-max(0.0, Float64(opp.market_impact)) / 3.0)
    market_trace = 0.45 * invested_trace + 0.35 * competition_trace + 0.20 * impact_trace

    # Constant lifecycle term pinned at the "emerging" value (0.30): the staged
    # opportunity lifecycle was excised as unreachable — every opportunity
    # stayed "emerging" in practice, so this
    # term always contributed 0.15 * 0.30.
    lifecycle_signal = 0.30

    signal = 0.60 * intrinsic_legibility + 0.25 * market_trace + 0.15 * lifecycle_signal

    if opp.truly_unknown
        signal *= 0.08
    end
    if threshold > 0.0
        signal *= (1.0 - 0.70 * threshold)
    end

    return max(0.0, signal)
end

function _agent_search_capacity(config::EmergentConfig, agent, ai_level::String)::Float64
    profile = get(config.AI_LEVELS, ai_level, config.AI_LEVELS["none"])
    info_quality = Float64(profile.info_quality)
    info_breadth = Float64(profile.info_breadth)

    exploration = clamp(Float64(get(agent.traits, "exploration_tendency", 0.5)), 0.0, 1.0)
    awareness = clamp(Float64(get(agent.traits, "market_awareness", 0.5)), 0.0, 1.0)
    human_attention = 0.20 + 0.40 * exploration + 0.40 * awareness

    # Exponential search capacity makes frontier systems near-omniscient when
    # evidence exists, while preserving the constraint that no-evidence cases
    # cannot be solved by label alone.
    ai_capacity = exp(4.0 * ((info_quality + info_breadth) - 0.45))
    return human_attention * ai_capacity
end

function opportunity_discovery_probability(
    market::MarketEnvironment,
    agent,
    opp::Opportunity,
    ai_level::String
)::Float64
    sector_knowledge = sector_familiarity(
        agent.resources.knowledge,
        opp.sector;
        opportunity=opp,
        config=market.config,
        default=0.0,
    )
    if sector_knowledge <= 0.0
        return 0.0
    end

    signal = opportunity_signal_strength(opp)
    search_capacity = _agent_search_capacity(market.config, agent, ai_level)
    knowledge_fit = 0.05 + 0.95 * clamp(sector_knowledge, 0.0, 1.0)
    intensity = 0.012 * signal * search_capacity * knowledge_fit

    if !isfinite(intensity)
        return 1.0
    elseif intensity <= 0.0
        return 0.0
    end
    return 1.0 - exp(-intensity)
end

function _public_diffusion_probability(opp::Opportunity, demand_count::Int)::Float64
    if demand_count <= 0 && opp.total_invested <= 0.0 && opp.competition <= 0.0 && opp.market_impact <= 0.0
        return 0.0
    end

    capacity = max(Float64(opp.capacity), Float64(opp.capital_requirements) * 10.0, 1.0)
    invested_trace = 1.0 - exp(-max(0.0, Float64(opp.total_invested)) / capacity)
    competition_trace = 1.0 - exp(-max(0.0, Float64(opp.competition)) / 1.5)
    demand_trace = 1.0 - exp(-max(0.0, Float64(demand_count)) / 3.0)
    impact_trace = 1.0 - exp(-max(0.0, Float64(opp.market_impact)) / 3.0)

    trace = 0.35 * invested_trace + 0.30 * competition_trace + 0.25 * demand_trace + 0.10 * impact_trace
    opacity = opportunity_signal_strength(opp)
    intensity = 0.35 * trace * (0.25 + opacity)

    if !isfinite(intensity)
        return 1.0
    elseif intensity <= 0.0
        return 0.0
    end
    return 1.0 - exp(-intensity)
end

# ============================================================================
# BRANCH FEEDBACK AND DEMAND ADJUSTMENTS
# ============================================================================

"""
Modifier effects for branch parameters.
"""
const MODIFIER_EFFECTS = Dict{String,Dict{String,Float64}}(
    "premium" => Dict("log_mu" => 0.12, "log_sigma" => -0.05, "failure_mean" => -0.05),
    "budget" => Dict("log_mu" => -0.15, "log_sigma" => 0.02, "failure_mean" => 0.04),
    "digital" => Dict("log_mu" => 0.05, "log_sigma" => -0.02, "failure_mean" => -0.02),
    "local" => Dict("log_mu" => -0.05, "log_sigma" => 0.03, "failure_mean" => 0.01),
    "specialized" => Dict("log_mu" => 0.08, "log_sigma" => 0.04, "failure_mean" => -0.01),
    "sustainable" => Dict("log_mu" => 0.04, "log_sigma" => 0.0, "failure_mean" => -0.02)
)

"""
Apply feedback to branch parameters based on observed ROI.
"""
function apply_branch_feedback!(market::MarketEnvironment, branch_name::String, mean_roi::Float64)
    params = get(market.branch_params, branch_name, nothing)
    if isnothing(params)
        return
    end

    profile = params["profile"]
    rate = market.config.BRANCH_FEEDBACK_RATE
    feedback = clamp(mean_roi, -1.0, 1.0) * rate

    # Adjust log_mu based on ROI
    log_range = (log(profile.return_range[1]), log(profile.return_range[2]))
    params["log_mu"] = clamp(params["log_mu"] + feedback, log_range[1], log_range[2])

    # Adjust failure_mean inversely
    failure_range = profile.failure_range
    params["failure_mean"] = clamp(
        params["failure_mean"] - feedback * 0.5,
        failure_range[1], failure_range[2]
    )
end

"""
Create child branch parameters with drift.
"""
function create_child_params(
    market::MarketEnvironment,
    name::String,
    parent_params::Dict{String,Any};
    modifier::Union{String,Nothing} = nothing
)::Dict{String,Any}
    root_sector = parent_params["root"]
    profile = market.sector_profiles[root_sector]

    # Apply drift
    mu_drift = market.config.BRANCH_LOG_MEAN_DRIFT * randn(market.rng)
    sigma_drift = market.config.BRANCH_LOG_SIGMA_DRIFT * randn(market.rng)
    failure_drift = market.config.BRANCH_FAILURE_DRIFT * randn(market.rng)

    # Apply modifier effects
    if !isnothing(modifier)
        effect = get(MODIFIER_EFFECTS, lowercase(modifier), Dict{String,Float64}())
        mu_drift += get(effect, "log_mu", 0.0)
        sigma_drift += get(effect, "log_sigma", 0.0)
        failure_drift += get(effect, "failure_mean", 0.0)
    end

    log_mu = parent_params["log_mu"] + mu_drift
    log_sigma = max(0.05, parent_params["log_sigma"] + sigma_drift)

    log_mu = clamp(log_mu, log(profile.return_range[1]), log(profile.return_range[2]))
    failure_mean = clamp(
        parent_params["failure_mean"] + failure_drift,
        profile.failure_range[1], profile.failure_range[2]
    )
    failure_sigma = clamp(
        parent_params["failure_sigma"] * rand(market.rng, Uniform(0.85, 1.15)),
        0.01, 0.5
    )

    return Dict{String,Any}(
        "name" => name,
        "root" => root_sector,
        "profile" => profile,
        "log_mu" => log_mu,
        "log_sigma" => log_sigma,
        "failure_mean" => failure_mean,
        "failure_sigma" => failure_sigma,
        "depth" => get(parent_params, "depth", 0) + 1
    )
end

"""
Ensure a branch exists, creating it if necessary.
"""
function ensure_branch!(market::MarketEnvironment, branch_name::String)::Dict{String,Any}
    if haskey(market.branch_params, branch_name)
        return market.branch_params[branch_name]
    end

    parts = split(branch_name, "_")
    base = String(parts[1])

    if !haskey(market.branch_params, base)
        profile = get(market.sector_profiles, base, nothing)
        if isnothing(profile)
            error("Unknown sector '$base'")
        end
        _initialize_branch!(market, base, profile)
    end

    parent = base
    for depth in 2:length(parts)
        child = join(parts[1:depth], "_")
        if haskey(market.branch_params, child)
            parent = child
            continue
        end
        parent_params = market.branch_params[parent]
        new_params = create_child_params(market, child, parent_params; modifier=String(parts[depth]))
        market.branch_params[child] = new_params
        parent = child
    end

    return market.branch_params[branch_name]
end

"""
Get demand adjustments for a sector based on clearing metrics.
"""
function get_demand_adjustments(market::MarketEnvironment, sector::String)::Dict{String,Float64}
    if haskey(market.sector_demand_adjustments, sector)
        return market.sector_demand_adjustments[sector]
    end

    # Normalize the sector ratio by the aggregate so sectors are compared
    # RELATIVE to market-wide supply/demand pressure, not by absolute
    # magnitude. At large N the raw demand/supply ratio can reach 50+ in every
    # sector because capital_requirements (the supply proxy) is a fundraising
    # gate floor that doesn't scale with population flow; absolute ratios would
    # saturate all sectors at the clamp and flatten inter-sector heterogeneity.
    # The normalized ratio (sector / aggregate) preserves relative hot/cold
    # differences even under extreme absolute values.
    raw_sector_ratio = get(market.sector_clearing_index, sector, market.aggregate_clearing_ratio)
    if !isfinite(raw_sector_ratio)
        raw_sector_ratio = market.aggregate_clearing_ratio
    end
    agg_ratio = market.aggregate_clearing_ratio
    if !isfinite(agg_ratio) || agg_ratio <= 0.0
        clearing_ratio = raw_sector_ratio  # fall back to raw when aggregate is zero (early rounds)
    else
        clearing_ratio = raw_sector_ratio / agg_ratio
    end
    # Still cap the normalized signal to a reasonable economic range —
    # extreme outlier sectors shouldn't blow up the formula.
    clearing_ratio = clamp(clearing_ratio, 0.1, 5.0)

    # Crowding calculations
    crowd_threshold = market.config.RETURN_DEMAND_CROWDING_THRESHOLD
    flow_share = 0.0
    if haskey(market.crowding_metrics, "share_invest")
        flow_share = get(market.crowding_metrics, "share_invest", 0.25)
    end

    crowd_excess = max(0.0, flow_share - crowd_threshold)
    crowd_relief = max(0.0, crowd_threshold - flow_share)

    penalty_strength = market.config.RETURN_DEMAND_CROWDING_PENALTY

    # Convex crowding penalty and relief
    return_penalty = 1.0 - penalty_strength * crowd_excess^2
    return_penalty *= 1.0 + 0.35 * crowd_relief^2

    failure_pressure = 1.0 + market.config.FAILURE_DEMAND_PRESSURE * crowd_excess^2
    failure_pressure *= 1.0 - 0.2 * crowd_relief^2

    # Supply/demand adjustments. clearing_ratio = demand / supply, so:
    #   ratio > 1 → demand exceeds supply (hot market) → returns up, but also
    #     HIGHER failure rate, because many who tried couldn't secure entry.
    #     (Giving hot markets lower failure would let the realized_return clamp
    #     make oversubscribed sectors effectively immortal.)
    #   ratio < 1 → supply exceeds demand (soft market) → returns down, failure
    #     also somewhat elevated (a tough environment, even if capacity exists).
    if clearing_ratio > 1.0
        excess_demand = clearing_ratio - 1.0
        return_penalty *= 1.0 + market.config.RETURN_UNDERSUPPLY_BONUS * excess_demand
        failure_pressure *= 1.0 + 0.20 * excess_demand
    else
        excess_supply = 1.0 - clearing_ratio
        return_penalty *= 1.0 - market.config.RETURN_OVERSUPPLY_PENALTY * excess_supply
        failure_pressure *= 1.0 + 0.30 * excess_supply
    end

    # Clamp both signals to sane positive ranges. The compound formulas above
    # can otherwise drive return_penalty well above 1 and failure_pressure
    # negative in crowded sectors; a negative failure signal would be
    # clamp-saved downstream, leaving oversubscribed sectors functionally
    # immortal.
    return_penalty  = clamp(return_penalty,  0.10, 3.00)
    failure_pressure = clamp(failure_pressure, 0.10, 3.00)

    adjustments = Dict{String,Float64}(
        "return" => return_penalty,
        "failure" => failure_pressure
    )
    market.sector_demand_adjustments[sector] = adjustments
    return adjustments
end

# ============================================================================
# MARKET DYNAMICS
# ============================================================================

"""
Update market dynamics based on agent actions.
"""
function update_market_dynamics!(
    market::MarketEnvironment,
    agent_actions::Vector{Dict{String,Any}},
    total_investment::Float64,
    ai_invest_share::Float64
)
    # Calculate average investment quality
    invest_actions = filter(a -> get(a, "action", "") == "invest", agent_actions)
    qualities = Float64[]
    for action in invest_actions
        # Actions emit the agent's perceived return under "estimated_return"
        # (agents.jl:719). Older callers that packaged decision details under
        # "chosen_opportunity_details" are still supported as a fallback.
        expected = get(action, "estimated_return", nothing)
        if isnothing(expected)
            expected = get(action, "expected_return", nothing)
        end
        if isnothing(expected)
            details = get(action, "chosen_opportunity_details", Dict())
            expected = get(details, "estimated_return", nothing)
        end
        if !isnothing(expected) && isfinite(expected)
            push!(qualities, Float64(expected))
        end
    end

    avg_quality = isempty(qualities) ? 1.0 : mean(qualities)
    quality_adjustment = (avg_quality - 1.0) * 0.1

    # AI activity adjustment (paradox: high AI share can degrade realized quality)
    ai_activity = clamp(ai_invest_share - 0.5, -0.5, 0.5)
    quality_adjustment -= ai_activity * 0.05

    # Investment activity
    investment_activity = clamp(total_investment / (market.n_agents * 250_000), 0, 2)

    # Update momentum
    market.market_momentum = market.market_momentum * 0.8 +
        0.2 * (investment_activity + quality_adjustment + ai_activity * 0.1)

    # Black swan probability: base rate modulated by AI investment share.
    # The former boom-streak escalation (BOOM_TAIL_UNCERTAINTY_EXPONENT) was
    # excised because its investment_activity
    # normalizer ($250K/agent/round) was unreachable under bet-sizing caps,
    # so boom_streak was identically 0 and the exponent never applied.
    base_prob = market.config.BLACK_SWAN_PROBABILITY
    dynamic_black_swan = base_prob * (1 + ai_invest_share * 0.3)

    crowding_index = get(market.crowding_metrics, "crowding_index", 0.25)

    signals = Dict{String,Float64}(
        "investment_activity" => investment_activity,
        "ai_activity" => ai_activity,
        "crowding_index" => crowding_index,
        "quality_adjustment" => quality_adjustment,
        "momentum" => market.market_momentum
    )

    update_macro_regime!(market, signals, dynamic_black_swan)
end

"""
Update macro regime based on market signals.
"""
function update_macro_regime!(
    market::MarketEnvironment,
    signals::Dict{String,Float64},
    black_swan_prob::Float64
)
    states = market.config.MACRO_REGIME_STATES
    transition_map = market.config.MACRO_REGIME_TRANSITIONS
    current_state = haskey(transition_map, market.market_regime) ? market.market_regime : states[1]

    # Base transition probabilities
    base_probs = [get(get(transition_map, current_state, Dict()), state, 0.0) for state in states]
    if sum(base_probs) <= 0
        base_probs = ones(length(states))
    end

    adjustments = zeros(length(states))
    idx_map = Dict(state => i for (i, state) in enumerate(states))

    invest = get(signals, "investment_activity", 1.0)
    momentum = get(signals, "momentum", 0.0)
    crowding = get(signals, "crowding_index", 0.25)
    ai_activity = get(signals, "ai_activity", 0.0)
    quality = get(signals, "quality_adjustment", 0.0)

    # Growth/boom boost
    if invest > 1.1 || momentum > 0.8 || quality > 0
        boost = 0.05 * max(invest - 1.1, 0.0) + 0.04 * max(momentum - 0.8, 0.0) + 0.03 * max(quality, 0.0)
        if haskey(idx_map, "growth")
            adjustments[idx_map["growth"]] += boost
        end
        if haskey(idx_map, "boom")
            adjustments[idx_map["boom"]] += boost * 0.6
        end
    end

    # NOTE: the former recession/crisis drag
    # branch (`invest < 0.9 || momentum < -0.6`) was excised. Its
    # investment_activity normalizer ($250K/agent/round) was unreachable under
    # bet-sizing caps (realistic values 0.1-0.3), so the branch fired EVERY
    # round — a permanent, unintended downward regime tilt layered on top of
    # the NBER-calibrated transition matrix. Removing it is behavior-changing:
    # regime conditions are slightly less recessionary on average; the
    # recalibration pass absorbs the shift.

    # Crowding effects
    crowd_threshold = market.config.RETURN_DEMAND_CROWDING_THRESHOLD
    if crowding > crowd_threshold
        penalty = 0.05 * (crowding - crowd_threshold)
        if haskey(idx_map, "recession")
            adjustments[idx_map["recession"]] += penalty
        end
        if haskey(idx_map, "crisis")
            adjustments[idx_map["crisis"]] += penalty * 0.6
        end
    else
        relief = 0.03 * (crowd_threshold - crowding)
        if haskey(idx_map, "normal")
            adjustments[idx_map["normal"]] += relief
        end
        if haskey(idx_map, "growth")
            adjustments[idx_map["growth"]] += relief * 0.5
        end
    end

    # AI activity can increase crisis risk
    if ai_activity > 0.3 && haskey(idx_map, "crisis")
        adjustments[idx_map["crisis"]] += 0.02 * ai_activity
    end

    # Black swan effect
    if haskey(idx_map, "crisis")
        adjustments[idx_map["crisis"]] += black_swan_prob
    end

    # Compute final probabilities
    probs = base_probs .+ adjustments
    probs = max.(probs, 0.0)
    total = sum(probs)

    if !isfinite(total) || total <= 0
        probs = ones(length(states)) / length(states)
    else
        probs ./= total
    end

    # Sample new state
    cumsum_probs = cumsum(probs)
    r = rand(market.rng)
    idx = clamp(searchsortedfirst(cumsum_probs, r), 1, length(states))
    new_state = states[idx]

    market.market_regime = new_state

    # Update modifiers
    market.regime_return_multiplier = get(market.config.MACRO_REGIME_RETURN_MODIFIERS, new_state, 1.0)
    market.regime_failure_multiplier = get(market.config.MACRO_REGIME_FAILURE_MODIFIERS, new_state, 1.0)

    target_trend = get(market.config.MACRO_REGIME_TREND, new_state, 0.0)
    target_volatility = get(market.config.MACRO_REGIME_VOLATILITY, new_state, market.volatility)

    market.trend = clamp(0.7 * market.trend + 0.3 * target_trend, -1.0, 1.0)
    market.volatility = clamp(0.6 * market.volatility + 0.4 * target_volatility, 0.05, 1.0)

    # Reset momentum on crisis
    if new_state == "crisis"
        market.market_momentum = min(market.market_momentum, -1.5)
    end
end

# ============================================================================
# OPPORTUNITY MANAGEMENT (creation, aging, removal)
# ============================================================================

"""
Manage opportunity creation, aging, and removal.
"""
function manage_opportunities!(
    market::MarketEnvironment,
    round::Int,
    opportunity_demand::Dict{String,Int},
    total_investment::Float64
)
    # Target opportunity count scaled to agent count.
    target_cap = get_scaled_opportunities(market.config, market.n_agents)

    # Calculate desired new opportunities
    discovered_count = count(o -> o.discovered, market.opportunities)
    young_discovered = count(o -> o.discovered && o.age < 5, market.opportunities)
    poisson_new = rand(market.rng, Poisson(max(0.0, discovered_count * 0.02)))
    desired_new = max(0, target_cap - discovered_count + poisson_new - young_discovered)
    push!(market.new_ventures_by_round, desired_new)

    # Create new opportunities with sector balancing
    for _ in 1:desired_new
        # Calculate sector probabilities (favor underrepresented sectors)
        sector_counts = Dict{String,Int}()
        for opp in market.opportunities
            sector_counts[opp.sector] = get(sector_counts, opp.sector, 0) + 1
        end

        total_opps = length(market.opportunities)
        if total_opps > 0
            probs = Float64[]
            for sector in market.sectors
                count = get(sector_counts, sector, 0)
                push!(probs, (total_opps - count) / total_opps)
            end
            if sum(probs) > 0
                probs ./= sum(probs)
            else
                probs = ones(length(market.sectors)) / length(market.sectors)
            end

            # Sample sector
            cumsum_probs = cumsum(probs)
            r = rand(market.rng)
            idx = clamp(searchsortedfirst(cumsum_probs, r), 1, length(market.sectors))
            sector = market.sectors[idx]
        else
            sector = rand(market.rng, market.sectors)
        end

        new_opp = _create_realistic_opportunity(
            market, "market_$(round)_$(rand(market.rng, 1000:9999))", sector
        )
        new_opp.discovery_round = round
        push!(market.opportunities, new_opp)
        market.opportunity_map[new_opp.id] = new_opp
        _index_opportunity!(market, new_opp)
    end

    # Private search does not broadcast opportunities. Public discovery emerges
    # from observable market traces: investment demand, crowding, deployed
    # capital, or realized impact.
    for opp in market.opportunities
        opp.discovered && continue
        demand_count = get(opportunity_demand, opp.id, 0)
        diffusion_prob = _public_diffusion_probability(opp, demand_count)
        if rand(market.rng) < diffusion_prob
            opp.discovered = true
            opp.discovery_round = round
        end
    end

    # Age all opportunities and decay competition. (The staged opportunity
    # lifecycle that was once updated here was excised as
    # unreachable.)
    for opp in market.opportunities
        opp.age += 1
        opp.competition = max(0.0, opp.competition * (1.0 - market.config.COMPETITION_DECAY_RATE))
    end

    # Remove dead opportunities
    dead_opps = Opportunity[]
    for opp in market.opportunities
        if opp.competition < 0.01 && opp.age > 20
            push!(dead_opps, opp)
        elseif opp.age > 10 && get(opportunity_demand, opp.id, 0) == 0
            push!(dead_opps, opp)
        end
    end

    for opp in dead_opps
        _remove_opportunity!(market, opp)
    end

    _rebuild_sector_index!(market)
end

"""
Index an opportunity by sector.
"""
function _index_opportunity!(market::MarketEnvironment, opp::Opportunity)
    sector = isnothing(opp.sector) ? "unknown" : opp.sector
    if !haskey(market.opportunities_by_sector, sector)
        market.opportunities_by_sector[sector] = Opportunity[]
    end
    push!(market.opportunities_by_sector[sector], opp)
end

"""
Remove an opportunity from the market.
"""
function _remove_opportunity!(market::MarketEnvironment, opp::Opportunity)
    filter!(o -> o.id != opp.id, market.opportunities)

    # Remove from sector index
    sector = isnothing(opp.sector) ? "unknown" : opp.sector
    if haskey(market.opportunities_by_sector, sector)
        filter!(o -> o.id != opp.id, market.opportunities_by_sector[sector])
    end

    # Remove from map
    delete!(market.opportunity_map, opp.id)
end

"""
Rebuild the sector index from scratch.
"""
function _rebuild_sector_index!(market::MarketEnvironment)
    empty!(market.opportunities_by_sector)
    for opp in market.opportunities
        _index_opportunity!(market, opp)
    end
    market.opportunity_map = Dict(opp.id => opp for opp in market.opportunities)
end

# ============================================================================
# CLEARING METRICS
# ============================================================================

"""
Estimate sector capacity for capital absorption.
"""
function estimate_sector_capacity(market::MarketEnvironment)::Dict{String,Float64}
    # Sector-level per-round *fundraising absorption* capacity: how much new
    # capital the sector can soak up in one round before clearing pressure
    # drives returns down / failure up. This is deliberately NOT the same as
    # `opp.capacity` (used by realized_return's convexity penalty): opp.capacity
    # is a per-opportunity *lifetime* absorption limit at the venture level,
    # whereas this returns a per-sector *per-round* flow rate driving
    # market-wide supply/demand dynamics. The two operate at different units
    # (sector-round vs opp-lifetime) and are intentionally distinct — unifying
    # them around opp.capacity produces systematically cold markets, because
    # opp count grows unboundedly as innovations spawn new opportunities. The
    # demand-adjustment clamp bounds any downstream over/undershoot regardless.
    capacity = Dict{String,Float64}()
    for opp in market.opportunities
        sector = isnothing(opp.sector) ? "unknown" : opp.sector
        req = opp.capital_requirements
        maturity = max(1, opp.time_to_maturity)
        capacity[sector] = get(capacity, sector, 0.0) + req / maturity
    end

    if isempty(capacity)
        capacity["unknown"] = Float64(market.n_agents) * 50_000.0
    end

    return capacity
end

"""
Update clearing metrics from agent actions.
"""
function update_clearing_metrics!(market::MarketEnvironment, agent_actions::Vector{Dict{String,Any}})
    # Clear stale adjustments at ENTRY so the fresh recompute at the bottom of
    # this function actually recomputes. get_demand_adjustments checks `haskey`
    # first and returns cached values, so without this clear the adjustments
    # would freeze at whatever was first computed.
    empty!(market.sector_demand_adjustments)

    sector_flows = Dict{String,Float64}()
    tier_flows = Dict{String,Float64}()

    for action in agent_actions
        if get(action, "action", "") != "invest"
            continue
        end

        # Extract sector
        details = get(action, "chosen_opportunity_details", Dict())
        sector = get(details, "sector", nothing)
        if isnothing(sector)
            opp = get(action, "chosen_opportunity_obj", nothing)
            if !isnothing(opp) && isa(opp, Opportunity)
                sector = opp.sector
            end
        end
        sector = isnothing(sector) ? "unknown" : string(sector)

        # Extract capital
        capital = Float64(get(action, "capital_deployed", get(action, "amount", 0.0)))
        if capital <= 0
            continue
        end

        sector_flows[sector] = get(sector_flows, sector, 0.0) + capital

        # Track by behavioral tier; sham labels collapse to no-AI.
        tier = action_behavior_ai_level(action)
        tier_flows[tier] = get(tier_flows, tier, 0.0) + capital
    end

    # Calculate clearing index
    capacity = estimate_sector_capacity(market)
    clearing_index = Dict{String,Float64}()
    for sector in union(keys(capacity), keys(sector_flows))
        supply = max(get(capacity, sector, 0.0), 1.0)
        demand = get(sector_flows, sector, 0.0)
        clearing_index[sector] = demand / supply
    end

    total_capacity = max(sum(values(capacity)), 1.0)
    total_demand = sum(values(sector_flows))

    market.sector_clearing_index = clearing_index
    market.aggregate_clearing_ratio = total_demand / total_capacity

    # Update tier invest shares: rebuild from scratch each round so a tier with
    # zero demand this round reads 0.0, not its stale share from a prior round.
    # Writing only the tiers present in tier_flows would freeze every other
    # tier's share at its last non-zero value across a no-invest round.
    for tier in keys(market.tier_invest_share)
        market.tier_invest_share[tier] = 0.0
    end
    if total_demand > 0
        for (tier, flow) in tier_flows
            market.tier_invest_share[tier] = flow / total_demand
        end
    end

    # Compute fresh sector_demand_adjustments for every sector from THIS
    # round's clearing metrics + crowding flow; these are read on the NEXT
    # round by realized_return via get_market_conditions.
    #
    # Iterate over all sectors present in clearing_index, not just
    # market.sectors: niche-branch sectors (from create_niche_opportunity! for
    # innovation-derived niches) appear in clearing_index but would otherwise
    # be absent from sector_demand_adjustments, leaving niche opps with a
    # clearing signal in the downside term but no return/failure multiplier —
    # inconsistent with base-sector opps.
    all_sectors = union(market.sectors, keys(clearing_index))
    for sector in all_sectors
        get_demand_adjustments(market, sector)
    end
end

# ============================================================================
# OPPORTUNITY ACCESS AND PERCEPTION
# ============================================================================

"""
Get opportunities available to an agent based on AI level and traits.
Note: agent is typed as Any to avoid circular dependency with agents.jl
"""
function get_opportunities_for_agent(
    market::MarketEnvironment,
    agent
)::Vector{Opportunity}
    # Start with public opportunities, agent-created opportunities, and this
    # agent's private discoveries. Private discovery is agent-local: it should
    # not broadcast the opportunity to the whole market merely because one
    # high-capability agent found a signal.
    agent_id = hasproperty(agent, :id) ? agent.id : nothing
    private_ids = hasproperty(agent, :resources) && hasproperty(agent.resources, :private_opportunity_ids) ?
        agent.resources.private_opportunity_ids : Set{String}()
    visible_ids = Set{String}()
    visible_opps = Opportunity[]
    for opp in market.opportunities
        if opp.discovered ||
           (!isnothing(agent_id) && opp.created_by == agent_id) ||
           (opp.id in private_ids)
            if !(opp.id in visible_ids)
                push!(visible_opps, opp)
                push!(visible_ids, opp.id)
            end
        end
    end

    # AI-based private discovery of non-public opportunities.
    ai_level = behavior_ai_level(market.config, get_ai_level(agent))

    for opp in market.opportunities
        if opp.discovered
            continue
        end

        if (!isnothing(agent_id) && opp.created_by == agent_id) || (opp.id in private_ids)
            continue
        end

        discovery_prob = opportunity_discovery_probability(market, agent, opp, ai_level)
        if rand(market.rng) < discovery_prob
            push!(agent.resources.private_opportunity_ids, opp.id)
            if !(opp.id in visible_ids)
                push!(visible_opps, opp)
                push!(visible_ids, opp.id)
            end
        end
    end

    # Filter and rank by agent perspective
    perceived = get_perceived_opportunities(market, visible_opps, ai_level, agent)

    # The creator-visibility override must survive the perception filter.
    # get_perceived_opportunities keeps the top-N by relevance (sector
    # knowledge); if the agent has little knowledge in the niche's sector, that
    # tie-break can drop the agent's own creation, so a creator would not see
    # the niche it just created. Re-insert any visible opp with
    # created_by == agent.id that the perception filter dropped.
    if !isnothing(agent_id)
        perceived_ids = Set(o.id for o in perceived)
        for opp in visible_opps
            if opp.created_by == agent_id && !(opp.id in perceived_ids)
                push!(perceived, opp)
            end
        end
    end
    return perceived
end

"""
Filter and rank opportunities based on agent's perspective.
"""
function get_perceived_opportunities(
    market::MarketEnvironment,
    all_opportunities::Vector{Opportunity},
    ai_level::String,
    agent  # EmergentAgent - untyped to avoid circular dependency
)::Vector{Opportunity}
    if isempty(all_opportunities)
        return Opportunity[]
    end

    profile = get(market.config.AI_LEVELS, ai_level, market.config.AI_LEVELS["none"])
    info_quality = Float64(profile.info_quality)
    info_breadth = Float64(profile.info_breadth)

    knowledge_map = agent.resources.knowledge
    agent_rng = hasproperty(agent, :rng) ? agent.rng : market.rng

    # Stochastic visibility budget. Higher-tier AI can scan a much larger
    # fraction of public, data-rich opportunities, but visibility remains a
    # discovery process rather than a deterministic global top-k oracle.
    raw_budget = (3.0 + info_breadth * 60.0) * (1.0 + info_quality)

    # WEALTH_SCALED_COMPUTE robustness cell (robustness addendum, 2026-06-11):
    # founders purchase analysis breadth in proportion to their resources —
    # the rich-get-richer marginal-compute dynamic. The multiplier
    # clamp(capital/initial_equity, 0.25, 4.0)^scaling applies UNIFORMLY
    # across tiers (no tier-keyed term; the exact-equality counterfactual
    # discipline is preserved — any tier asymmetry in its effect must emerge
    # from tier differences in baseline budgets and billing). Billing charges
    # market_analysis per visible opportunity, so the extra visibility is
    # paid for at the margin under the hybrid/token backends. Default 0.0
    # skips this block entirely (bit-identical baseline).
    wealth_scaling = max(0.0, Float64(getfield_default(market.config, :WEALTH_COMPUTE_SCALING, 0.0)))
    if wealth_scaling > 0.0 && hasproperty(agent, :resources)
        initial_eq = hasproperty(agent.resources, :performance) ?
            max(Float64(agent.resources.performance.initial_equity), 1.0) : 1.0
        capital_ratio = max(Float64(agent.resources.capital), 0.0) / initial_eq
        raw_budget *= clamp(capital_ratio, 0.25, 4.0)^wealth_scaling
    end
    coverage = 1.0 - exp(-1.4 * raw_budget / max(1.0, length(all_opportunities)))
    target_visible = clamp(round(Int, length(all_opportunities) * coverage), 1, length(all_opportunities))

    # Score opportunities by relevance and discoverability. Tacit/novel/
    # endogenous opportunities can still be found, but they have lower sampling
    # weight unless the agent has relevant sector knowledge and breadth.
    weights = Float64[]
    for opp in all_opportunities
        relevance = 0.5
        if !isnothing(opp.sector)
            relevance += sector_familiarity(
                knowledge_map,
                opp.sector;
                opportunity=opp,
                config=market.config,
                default=0.0,
            ) * 0.5
        end

        opacity = (
            (opp.truly_unknown ? 0.35 : 0.0) +
            0.25 * clamp(opp.combination_uncertainty, 0.0, 1.0) +
            0.15 * clamp(opp.novelty_score, 0.0, 1.0) +
            0.10 * clamp(opp.complexity / 2.0, 0.0, 1.0)
        )
        opacity_penalty = clamp(1.0 - opacity * (1.0 - 0.75 * info_breadth), 0.05, 1.0)
        quality_weight = 0.75 + 0.5 * info_quality
        push!(weights, max(0.01, relevance * opacity_penalty * quality_weight))
    end

    if target_visible >= length(all_opportunities)
        return copy(all_opportunities)
    end

    keys = Float64[]
    for weight in weights
        u = clamp(rand(agent_rng), eps(Float64), 1.0 - eps(Float64))
        push!(keys, log(u) / weight)
    end
    selected_indices = sortperm(keys, rev=true)[1:target_visible]

    return [all_opportunities[i] for i in selected_indices]
end

# ============================================================================
# NICHE AND INNOVATION OPPORTUNITIES
# ============================================================================

"""
Create a niche opportunity discovered via exploration.
"""
function create_niche_opportunity(
    market::MarketEnvironment,
    niche_id::String,
    discoverer_id::Int,
    round_num::Int;
    sector_novelty_context::Float64 = 0.35,
    sector_uncertainty_context::Float64 = 0.35,
    search_effort::Float64 = 0.0,
    niche_creation_evidence::Float64 = 0.0
)::Opportunity
    # Parse niche_id
    if occursin("_", niche_id)
        parts = rsplit(niche_id, "_"; limit=2)
        base_sector = String(parts[1])
        modifier = String(parts[2])
    else
        base_sector = niche_id
        modifier = "standard"
    end

    modifier_effects = Dict{String,Dict{String,Float64}}(
        "premium" => Dict("return_mult" => 1.15, "uncertainty_mult" => 0.85, "capital_mult" => 1.8),
        "budget" => Dict("return_mult" => 0.85, "uncertainty_mult" => 1.15, "capital_mult" => 0.6),
        "sustainable" => Dict("return_mult" => 1.05, "uncertainty_mult" => 0.95, "capital_mult" => 1.1),
        "digital" => Dict("return_mult" => 1.08, "uncertainty_mult" => 1.0, "capital_mult" => 0.8),
        "local" => Dict("return_mult" => 0.92, "uncertainty_mult" => 0.85, "capital_mult" => 0.7),
        "specialized" => Dict("return_mult" => 1.12, "uncertainty_mult" => 1.05, "capital_mult" => 1.3)
    )

    mods = get(modifier_effects, modifier, Dict("return_mult" => 1.0, "uncertainty_mult" => 1.0, "capital_mult" => 1.0))

    branch_name = niche_id
    ensure_branch!(market, branch_name)
    latent_return, latent_failure, capital_req, maturity = _sample_branch_characteristics(market, branch_name)

    latent_return *= mods["return_mult"]
    latent_failure *= mods["uncertainty_mult"]
    capital_req *= mods["capital_mult"]

    modifier_novelty = get(Dict(
        "premium" => 0.40,
        "budget" => 0.25,
        "sustainable" => 0.50,
        "digital" => 0.55,
        "local" => 0.25,
        "specialized" => 0.65,
        "standard" => 0.35,
    ), modifier, 0.35)
    context_novelty = clamp(Float64(sector_novelty_context), 0.0, 1.0)
    context_uncertainty = clamp(Float64(sector_uncertainty_context), 0.0, 1.0)
    search_signal = clamp(1.0 - exp(-max(0.0, Float64(search_effort))), 0.0, 1.0)
    evidence_signal = clamp(Float64(niche_creation_evidence) / 0.35, 0.0, 1.0)
    discovery_evidence = clamp(0.55 * evidence_signal + 0.45 * search_signal, 0.0, 1.0)

    niche_novelty = clamp(
        0.20 + 0.35 * context_novelty + 0.25 * modifier_novelty + 0.20 * discovery_evidence,
        0.05,
        0.95,
    )
    niche_combination_uncertainty = clamp(
        0.18 + 0.45 * context_uncertainty + 0.22 * niche_novelty + 0.15 * (1.0 - discovery_evidence),
        0.05,
        0.95,
    )
    niche_discovery_threshold = clamp(
        0.08 + 0.32 * niche_novelty + 0.28 * niche_combination_uncertainty + 0.10 * (1.0 - discovery_evidence),
        0.0,
        0.85,
    )
    niche_component_scarcity = clamp(0.35 + 0.35 * niche_combination_uncertainty + 0.30 * niche_novelty, 0.0, 1.0)
    niche_truly_unknown = niche_novelty > 0.70 && niche_combination_uncertainty > 0.65 && discovery_evidence < 0.50
    niche_signature = "niche:$(base_sector):$(modifier)"

    # No creation-time demand adjustment here. The demand signal is applied
    # once at realization via sector_demand_adjustments in realized_return.
    # Baking it into latent_return/latent_failure at construction would apply
    # it twice for niche opps, whereas regular-path opps see it only at
    # realization.
    niche_opp = Opportunity(
        id="niche_$(branch_name)_$(round_num)_$(rand(market.rng, 1000:9999))",
        latent_return_potential=clamp(latent_return, 0.5, market.config.RETURN_CLAMP_MAX),
        latent_failure_potential=clamp(latent_failure, 0.02, 0.95),
        complexity=rand(market.rng, Uniform(0.4, 0.8)),
        # discovered=false: creator-only visibility is handled in
        # get_opportunities_for_agent via a `created_by == agent.id` override.
        # Setting discovered=true would leak the opp to every agent instantly,
        # whereas the model assumes gradual AI-tier-aware discovery, not broadcast.
        discovered=false,
        discovery_round=round_num,
        created_by=discoverer_id,
        sector=branch_name,
        capital_requirements=capital_req,
        time_to_maturity=maturity,
        novelty_score=niche_novelty,
        combination_uncertainty=niche_combination_uncertainty,
        required_discovery_threshold=niche_discovery_threshold,
        truly_unknown=niche_truly_unknown,
        combination_signature=niche_signature,
        component_scarcity=niche_component_scarcity,
        config=market.config,
        rng=market.rng,
    )

    push!(market.opportunities, niche_opp)
    market.opportunity_map[niche_opp.id] = niche_opp
    _index_opportunity!(market, niche_opp)

    return niche_opp
end

"""
Record innovation outcome and update opportunity characteristics.
"""
function record_innovation_outcome!(
    market::MarketEnvironment,
    opportunity::Opportunity,
    success::Bool,
    return_achieved::Float64
)
    if success
        opportunity.market_impact = return_achieved
        intrinsic_gain = clamp(return_achieved, 0.2, 3.5)

        # Adjust return potential. The 25.0 ceiling matches the
        # spawn_opportunity_from_innovation! cap; a lower cap would clip
        # high-novelty/scarcity spawned opps out of their legitimate 5-25×
        # range the first time this fires, eliminating most unicorn upside
        # immediately after spawn.
        opportunity.latent_return_potential = clamp(
            opportunity.latent_return_potential + 0.05 * intrinsic_gain,
            0.15, market.config.RETURN_CLAMP_MAX
        )
    else
        opportunity.market_impact = 0.0
        opportunity.latent_failure_potential = clamp(
            opportunity.latent_failure_potential * 1.05,
            0.05, 0.99
        )
    end
end

"""
Spawn a derivative opportunity from a successful innovation.
"""
function spawn_opportunity_from_innovation!(
    market::MarketEnvironment,
    innovation::Innovation,
    cash_multiple::Float64
)
    sector = isnothing(innovation.sector) ? "tech" : innovation.sector
    # Handle potential Nothing values with defaults
    scarcity = clamp(something(innovation.scarcity, 0.5), 0.0, 1.0)
    novelty = clamp(something(innovation.novelty, 0.5), 0.0, 1.0)

    scarcity_scale = 0.85 + scarcity * 0.7
    novelty_scale = 0.9 + novelty * 0.6

    intrinsic_multiple = clamp(
        1.05 + (cash_multiple - 1.0) * 0.6,
        0.8, 3.2
    ) * scarcity_scale * novelty_scale * (0.9 + clamp(innovation.quality, 0.0, 1.5) * 0.2)

    derived_multiplier = clamp(intrinsic_multiple, 0.65, 3.8)
    base_failure_signal = clamp(0.35 - scarcity * 0.2 - novelty * 0.1, 0.04, 0.9)

    branch_name = sector
    latent_return, latent_failure, capital_req, maturity = _sample_branch_characteristics(market, branch_name)
    latent_return *= derived_multiplier
    latent_failure = clamp(0.5 * latent_failure + 0.5 * base_failure_signal, 0.05, 0.95)

    # No spawn-time demand adjustment here. The same signal is applied at
    # realization via sector_demand_adjustments in realized_return. Baking it
    # into latent_return/latent_failure at construction would apply it twice
    # for spawned opps, whereas regular opps from _create_realistic_opportunity
    # see it only at realization.
    opp_id = "spawn_$(innovation.id)_$(rand(market.rng, 1000:9999))"

    # Novelty score inherited from innovation (novel opportunities are harder to evaluate)
    innov_novelty = clamp(something(innovation.novelty, 0.5), 0.0, 1.0)
    quality_signal = clamp(Float64(innovation.quality), 0.0, 1.5) / 1.5
    new_combination_signal = innovation.is_new_combination ? 1.0 : 0.0
    impact_trace = clamp(log1p(max(cash_multiple - 1.0, 0.0)) / log(6.0), 0.0, 1.0)
    derived_combination_uncertainty = clamp(
        0.20 + 0.35 * innov_novelty + 0.25 * scarcity +
        0.15 * new_combination_signal - 0.10 * quality_signal,
        0.05,
        0.95,
    )
    derived_discovery_threshold = clamp(
        0.08 + 0.30 * innov_novelty + 0.22 * derived_combination_uncertainty +
        0.12 * scarcity + 0.08 * new_combination_signal - 0.16 * impact_trace,
        0.0,
        0.85,
    )
    derived_truly_unknown = derived_discovery_threshold > 0.68 && impact_trace < 0.35

    opportunity = Opportunity(
        id=opp_id,
        latent_return_potential=clamp(latent_return, 0.5, market.config.RETURN_CLAMP_MAX),
        latent_failure_potential=clamp(latent_failure, 0.02, 0.95),
        complexity=clamp(0.3 + innovation.quality * 0.3, 0.3, 1.0),
        # discovered=false — creator visibility handled via the created_by
        # override in get_opportunities_for_agent (see create_niche_opportunity
        # for rationale).
        discovered=false,
        discovery_round=market.current_round,
        created_by=innovation.creator_id,
        config=market.config,
        sector=branch_name,
        capital_requirements=capital_req,
        time_to_maturity=maturity,
        novelty_score=innov_novelty,  # Novel opportunities from innovations
        combination_uncertainty=derived_combination_uncertainty,
        required_discovery_threshold=derived_discovery_threshold,
        truly_unknown=derived_truly_unknown,
        combination_signature=innovation.combination_signature,
        origin_innovation_id=innovation.id,
        component_scarcity=scarcity,
        # S2 per-opportunity edge (gated): a created niche requires exactly the
        # knowledge components the innovation recombined. RNG-neutral (no draw),
        # and empty when ENABLE_OPPORTUNITY_COMPONENTS is off (byte-identical).
        knowledge_components=market.config.ENABLE_OPPORTUNITY_COMPONENTS ?
            copy(innovation.knowledge_components) : String[],
        # Leave capacity at the constructor sentinel so spawned opportunities use
        # the same canonical niche-size distribution as other opportunities.
        rng=market.rng,
    )

    push!(market.opportunities, opportunity)
    market.opportunity_map[opportunity.id] = opportunity
    _index_opportunity!(market, opportunity)

    return opportunity
end

# ============================================================================
# DIVERSITY METRICS
# ============================================================================

"""
Compute combination and sector HHI diversity metrics.
"""
function get_combination_diversity_metrics(market::MarketEnvironment)::Tuple{Float64,Float64}
    combo_counts = Dict{String,Int}()
    sector_counts = Dict{String,Int}()

    for opp in market.opportunities
        signature = !isnothing(opp.combination_signature) ? opp.combination_signature : "sector_$(opp.sector)"
        combo_counts[signature] = get(combo_counts, signature, 0) + 1
        sector_counts[opp.sector] = get(sector_counts, opp.sector, 0) + 1
    end

    function compute_hhi(counter::Dict{String,Int})::Float64
        total = sum(values(counter))
        if total <= 0
            return 0.0
        end
        return sum((count / total)^2 for count in values(counter))
    end

    return (compute_hhi(combo_counts), compute_hhi(sector_counts))
end

"""
Clear old opportunities that have been around too long with low competition.
"""
function clear_old_opportunities!(market::MarketEnvironment, round::Int)
    filter!(market.opportunities) do opp
        age = round - opp.discovery_round
        !(age > 80 && opp.competition < 0.05)
    end
    _rebuild_sector_index!(market)
end
