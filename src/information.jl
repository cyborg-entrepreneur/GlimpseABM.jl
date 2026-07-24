"""
Information system components for GlimpseABM.jl

Provides AI-augmented information generation about opportunities,
including hallucination modeling and domain-specific accuracy.
"""

using Random
using Statistics

# ============================================================================
# INFORMATION SYSTEM
# ============================================================================

"""
System for generating information about opportunities.
"""
mutable struct InformationSystem
    config::EmergentConfig
    # Cache key (opp_id, ai_level, agent_id). agent_id is part of the key so
    # within-tier estimates differ across agents (no shared-noise herding).
    information_cache::Dict{Tuple{String,String,Int},Information}
    discovered_opportunities::Set{String}
    cache_hits::Int
    cache_misses::Int
    domain_performance::Dict{String,Vector{Float64}}
    # ── Emergence audit extension (P9, AI_ERROR_CORRELATION) ────────────────
    # Per-(opportunity, round, tier) common error draws. Keyed (opp_id,
    # ai_level); the round dimension is implicit because the cache is cleared
    # with the information cache at the start of every round (clear_cache!,
    # called from simulation step!). At rho = 0 (default) NOTHING is ever
    # inserted here — the blend in get_information short-circuits before any
    # common draw is consumed (bit-identity pin, test_emergence_audit.jl).
    common_error_cache::Dict{Tuple{String,String},Float64}
    # DEDICATED RNG stream for the common draws. Stream design: derived
    # deterministically from the simulation seed at construction
    # (MersenneTwister(seed ⊻ COMMON_ERROR_STREAM_SALT)) and consumed ONLY by
    # common_estimate_error!, so enabling rho > 0 never desynchronizes
    # market.rng or agent.rng draw sequences — the agent stream consumes
    # exactly one randn for the estimate error at every rho (idiosyncratic at
    # rho = 0, blend component at rho > 0), keeping all other draws aligned
    # across rho conditions under paired seeds.
    common_error_rng::Random.AbstractRNG
end

# Salt for deriving the dedicated common-error stream from the simulation
# seed. Any fixed constant works (the stream only has to be independent of
# the market/agent streams and reproducible from the seed); documented so
# the derivation is auditable.
const COMMON_ERROR_STREAM_SALT = UInt64(0x9E3779B97F4A7C15)

"""
Create a new InformationSystem.

`common_error_seed` seeds the dedicated common-error stream (emergence audit
P9 dial); production passes the simulation's `actual_seed` so rho > 0 runs
are reproducible per (seed, run). The default 0 only matters for diagnostic
paths that never enable the dial.
"""
function InformationSystem(config::EmergentConfig; common_error_seed::Integer=0)
    return InformationSystem(
        config,
        Dict{Tuple{String,String,Int},Information}(),
        Set{String}(),
        0,
        0,
        Dict{String,Vector{Float64}}(
            "market_analysis" => Float64[],
            "technical_assessment" => Float64[],
            "uncertainty_evaluation" => Float64[],
            "innovation_potential" => Float64[]
        ),
        Dict{Tuple{String,String},Float64}(),
        Random.MersenneTwister(UInt64(abs(common_error_seed)) ⊻ COMMON_ERROR_STREAM_SALT)
    )
end

"""
One standard-normal common draw per (opportunity, round, tier), shared by
every agent of that tier evaluating that opportunity that round (the round
key is implicit: the cache is cleared per round). Drawn from the dedicated
`common_error_rng` stream — see the field docs for the stream design.
Only reachable when AI_ERROR_CORRELATION > 0 (the rho = 0 default
short-circuits in get_information before calling this).
"""
function common_estimate_error!(sys::InformationSystem, opp_id::String,
                                ai_level::String)::Float64
    return get!(sys.common_error_cache, (opp_id, ai_level)) do
        randn(sys.common_error_rng)
    end
end

"""
Determine the analysis domain for an opportunity.
"""
function determine_domain(opp::Opportunity)::String
    if !isnothing(opp.created_by)
        return "innovation_potential"
    end
    if opp.complexity > 0.7
        return "technical_assessment"
    end
    base_sector = base_sector_name(opp.sector)
    if base_sector in ["tech", "manufacturing"]
        return "technical_assessment"
    end
    if opp.competition > 0.5
        return "market_analysis"
    end
    return "uncertainty_evaluation"
end

function novelty_generalization_pressure(
    config::EmergentConfig,
    opp::Opportunity,
    ai_level::String,
    info_breadth::Float64
)::Float64
    factor = max(0.0, Float64(getfield_default(config, :NOVELTY_NOISE_INVERSION_FACTOR, 0.0)))
    if factor <= 0.0 || ai_level == "none"
        return 0.0
    end

    novelty = max(0.0, Float64(opp.novelty_score))
    combination_uncertainty = max(0.0, Float64(opp.combination_uncertainty))
    data_poverty = max(0.0, 1.0 - info_breadth)
    stealth_bonus = opp.truly_unknown ? 0.35 : 0.0
    endogenous_bonus = isnothing(opp.created_by) ? 0.0 : 0.25

    opportunity_pressure =
        0.45 * novelty +
        0.30 * combination_uncertainty +
        0.15 * data_poverty +
        stealth_bonus +
        endogenous_bonus

    return factor * max(0.0, opportunity_pressure)
end

"""
Generate insights based on information breadth.
"""
function generate_insights(
    opp::Opportunity,
    breadth::Float64,
    domain::Union{String,Nothing}=nothing
)::Vector{String}
    insights = String[]

    if breadth > 0.4
        if opp.competition > 0.5
            push!(insights, "High competitive pressure detected")
        else
            push!(insights, "Limited competition currently")
        end
    end

    if breadth > 0.6
        if opp.path_dependency > 0.5
            push!(insights, "Strong first-mover advantages")
        end
        if opp.complexity > 0.7
            push!(insights, "Requires specialized capabilities")
        end
        push!(insights, "Sector: $(opp.sector)")
    end

    if breadth > 0.8
        push!(insights, "Hidden uncertainty factors: ~$(round((1-breadth)*opp.complexity*100, digits=1))%")
        if !isnothing(opp.created_by)
            push!(insights, "Novel opportunity (agent-created)")
        end
        if !isnothing(domain)
            push!(insights, "Analysis domain: $domain")
        end
    end

    return insights
end

"""
Generate false insights for hallucination modeling.
"""
function generate_false_insights(domain::String; rng::Random.AbstractRNG=Random.default_rng())::Vector{String}
    false_insight_templates = Dict(
        "market_analysis" => [
            "Untapped customer segment identified in emerging markets",
            "Significant demand spike detected through alternative data",
            "Competitor withdrawal likely within 3 quarters"
        ],
        "technical_assessment" => [
            "Breakthrough resolves key scalability challenges",
            "Prototype efficiency exceeds market benchmarks",
            "Regulatory path streamlined due to new standards"
        ],
        "uncertainty_evaluation" => [
            "Black swan likelihood reduced by recent policy changes",
            "Risk contagion limited to adjacent sectors",
            "Dominant uncertainty source neutralized by supplier agreement"
        ],
        "innovation_potential" => [
            "High cross-domain synergy with existing portfolio",
            "Productized knowledge base accelerates time-to-market",
            "Community adoption hurdle lower than industry peers"
        ]
    )

    domain_insights = get(false_insight_templates, domain, String[])
    n_false = min(2, length(domain_insights))

    if n_false > 0 && !isempty(domain_insights)
        return domain_insights[randperm(rng, length(domain_insights))[1:n_false]]
    end
    return String[]
end

"""
Get information about an opportunity for a given AI level.

Applies lognormal accuracy variance and a multi-step hallucination-rate
chain (base rate → stochastic modification → lognormal variance → intensity
scaling) to produce a noisy, possibly-hallucinated estimate.
"""
function get_information(
    sys::InformationSystem,
    opp::Opportunity,
    ai_level::String;
    agent_id::Union{Int,Nothing}=nothing,
    agent_traits::Union{Dict{String,Float64},Nothing}=nothing,
    sector_knowledge::Union{Dict{String,Float64},Nothing}=nothing,
    rng::Random.AbstractRNG=Random.default_rng()
)::Information
    # Cache key includes agent_id so each agent gets idiosyncratic noise.
    # Keying only on (opp.id, ai_level) would hand every agent in a tier THE
    # SAME noisy estimate per opportunity — collapsing within-tier heterogeneity
    # and producing artificial herding when many same-tier agents evaluate the
    # same opp. agent_id may be `nothing` for diagnostic / test paths, in which
    # case the per-tier cache is reused.
    cache_key = (opp.id, ai_level, something(agent_id, 0))

    # Check cache
    if haskey(sys.information_cache, cache_key)
        sys.cache_hits += 1
        return sys.information_cache[cache_key]
    end

    sys.cache_misses += 1

    # Get AI configuration
    ai_config = get(sys.config.AI_LEVELS, ai_level, sys.config.AI_LEVELS["none"])
    info_quality = Float64(ai_config.info_quality)
    info_breadth = Float64(ai_config.info_breadth)
    quality_boosts = getfield_default(sys.config, :AI_INFORMATION_QUALITY_BOOSTS, Dict{String,Float64}())
    info_quality = clamp(info_quality + Float64(get(quality_boosts, ai_level, 0.0)), 0.0, 0.99)
    domain = determine_domain(opp)
    domain_cap = get_ai_domain_capability(sys.config, ai_level, domain)
    generalization_pressure = novelty_generalization_pressure(sys.config, opp, ai_level, info_breadth)

    # Tier noise factor: higher info quality means lower noise.
    tier_noise = max(0.1, 1.2 - info_quality)

    # Base accuracy from domain capability (use struct field access)
    base_accuracy = Float64(domain_cap.accuracy)
    complementarity_strength = Float64(getfield_default(sys.config, :AI_COMPLEMENTARITY_STRENGTH, 0.0))
    complementarity_delta = 0.0
    if complementarity_strength > 0.0 && ai_level != "none" && !isnothing(agent_traits)
        sector_key = isnothing(opp.sector) ? "unknown" : opp.sector
        sector_k = isnothing(sector_knowledge) ? 0.1 : sector_familiarity(
            sector_knowledge,
            sector_key;
            opportunity=opp,
            config=sys.config,
            default=0.1,
        )
        analytical = Float64(get(agent_traits, "analytical_ability", 0.5))
        competence = Float64(get(agent_traits, "competence", 0.5))
        market_awareness = Float64(get(agent_traits, "market_awareness", 0.5))
        complementarity_delta = clamp(
            0.35 * analytical + 0.25 * competence + 0.20 * market_awareness + 0.20 * sector_k - 0.5,
            -0.5,
            0.5
        )
        base_accuracy = clamp(base_accuracy + 0.20 * complementarity_strength * complementarity_delta, 0.05, 0.99)
    end
    if generalization_pressure > 0.0
        base_accuracy = max(0.05, base_accuracy / (1.0 + 0.35 * generalization_pressure))
    end

    # Quality-dependent accuracy scaling: apply lognormal accuracy noise.
    accuracy_noise_scale = 0.12 + 0.08 * tier_noise
    accuracy_noise_loc = 1.0 - 0.35 * (1.0 - info_quality)
    if generalization_pressure > 0.0
        accuracy_noise_scale *= (1.0 + 0.50 * generalization_pressure)
        accuracy_noise_loc /= (1.0 + 0.12 * generalization_pressure)
    end
    accuracy_noise = exp(randn(rng) * accuracy_noise_scale) * accuracy_noise_loc
    actual_accuracy = clamp(base_accuracy * accuracy_noise, 0.35, 0.99)

    # Hallucination-rate chain.
    # Step 1: Get base hallucination rate from domain capability
    base_hallucination = Float64(domain_cap.hallucination_rate)
    if complementarity_strength > 0.0 && ai_level != "none"
        hallucination_scale = 1.0 -
            0.40 * complementarity_strength * max(0.0, complementarity_delta) +
            0.25 * complementarity_strength * max(0.0, -complementarity_delta)
        base_hallucination *= clamp(hallucination_scale, 0.5, 1.5)
    end
    if generalization_pressure > 0.0
        base_hallucination *= (1.0 + 0.75 * generalization_pressure)
    end

    # Step 2: Apply stochastic modification
    stochastic_rate = get_stochastic_hallucination_rate(base_hallucination, domain; rng=rng)

    # Step 3: Apply lognormal variance
    lognormal_mu = 0.0
    lognormal_sigma = 0.25 * tier_noise
    lognormal_factor = clamp(exp(lognormal_mu + lognormal_sigma * randn(rng)), 0.2, 3.0)
    hallucination_rate = stochastic_rate * lognormal_factor

    # Step 4: Apply intensity scaling (robustness parameter)
    hallucination_intensity = sys.config.HALLUCINATION_INTENSITY
    hallucination_rate = clamp(hallucination_rate * hallucination_intensity, 0.0, 1.0)

    # AI_BIAS_INTENSITY scales the systematic-bias channel the way
    # HALLUCINATION_INTENSITY scales hallucinations in Step 4 above:
    # 0.0 = bias-free, 1.0 = baseline.
    # Multiplication by exactly 1.0 is an identity in IEEE arithmetic, so the
    # default path stays bit-identical.
    bias = Float64(domain_cap.bias) *
           max(0.0, Float64(getfield_default(sys.config, :AI_BIAS_INTENSITY, 1.0)))
    generalization_noise_multiplier = 1.0 + 0.35 * generalization_pressure

    # Estimate return with noise (quality-scaled).
    #
    # AI_ERROR_CORRELATION = rho: the continuous estimate
    # error is eps = sqrt(rho)*eps_common(opp, round, tier) + sqrt(1-rho)*eps_idio
    # with eps_common, eps_idio independent standard normals, so
    # Var(eps) = rho + (1 - rho) = 1 at every rho — total error variance (and
    # therefore accuracy) is preserved by construction; the dial moves signal
    # COMMONALITY at fixed signal QUALITY. Full algebra + scope at the
    # AI_ERROR_CORRELATION definition (config.jl).
    #
    # At rho = 0 (default), the branch below short-circuits to a single
    # randn(rng): no common draw is consumed and the agent-rng draw order is
    # unchanged. At rho > 0
    # the agent rng STILL consumes exactly one randn (the idiosyncratic
    # component, even at rho = 1 where its weight is 0), keeping all
    # subsequent agent-rng draws aligned across rho conditions.
    #
    # SCOPE: only this return-estimate error is correlated; the hallucination
    # flag, accuracy noise, uncertainty noise, and overconfidence draws below
    # stay idiosyncratic, and the "none" tier (no AI instrument) is exempt —
    # conservative choices documented at the config field.
    rho = sys.config.AI_ERROR_CORRELATION
    return_eps = if rho == 0.0 || ai_level == "none"
        randn(rng)
    else
        (0.0 < rho <= 1.0) || error(
            "Invalid AI_ERROR_CORRELATION=$(rho) at first use " *
            "(get_information). Require 0 ≤ rho ≤ 1.")
        sqrt(rho) * common_estimate_error!(sys, opp.id, ai_level) +
            sqrt(1.0 - rho) * randn(rng)
    end
    return_noise = return_eps * (1 - actual_accuracy) * 0.5 * generalization_noise_multiplier
    estimated_return = opp.latent_return_potential + return_noise + bias * 0.3
    estimated_return = clamp(estimated_return,
        sys.config.OPPORTUNITY_RETURN_RANGE[1],
        sys.config.OPPORTUNITY_RETURN_RANGE[2])

    # Estimate uncertainty with noise (quality-scaled)
    uncertainty_noise = randn(rng) * (1 - actual_accuracy) * 0.3 * generalization_noise_multiplier
    estimated_uncertainty = opp.latent_failure_potential + uncertainty_noise - bias * 0.2
    estimated_uncertainty = clamp(estimated_uncertainty,
        sys.config.OPPORTUNITY_UNCERTAINTY_RANGE[1],
        sys.config.OPPORTUNITY_UNCERTAINTY_RANGE[2])

    # Check for hallucination
    contains_hallucination = rand(rng) < hallucination_rate
    if contains_hallucination
        if rand(rng) < 0.5
            estimated_return = rand(rng) * (sys.config.OPPORTUNITY_RETURN_RANGE[2] -
                sys.config.OPPORTUNITY_RETURN_RANGE[1]) + sys.config.OPPORTUNITY_RETURN_RANGE[1]
        else
            estimated_uncertainty = rand(rng) * (sys.config.OPPORTUNITY_UNCERTAINTY_RANGE[2] -
                sys.config.OPPORTUNITY_UNCERTAINTY_RANGE[1]) + sys.config.OPPORTUNITY_UNCERTAINTY_RANGE[1]
        end
    end

    # Compute confidence with quality-dependent scaling
    true_confidence = actual_accuracy * (1 - opp.complexity * (1 - actual_accuracy))
    if generalization_pressure > 0.0
        true_confidence /= (1.0 + 0.15 * generalization_pressure)
    end
    # Scale overconfidence by intensity parameter (for robustness testing)
    overconfidence_intensity = sys.config.OVERCONFIDENCE_INTENSITY
    base_overconfidence = info_quality < 0.5 ? (0.5 - info_quality) * 0.5 : 0.0
    overconfidence_factor = 1.0 + base_overconfidence * overconfidence_intensity
    stated_confidence = clamp(true_confidence * overconfidence_factor, 0.1, 0.95)
    if complementarity_strength > 0.0 && ai_level != "none" && !isnothing(agent_traits)
        trust = Float64(get(agent_traits, "ai_trust", 0.5))
        stated_confidence = clamp(
            stated_confidence * (1.0 + 0.25 * complementarity_strength * (trust - 0.5)),
            0.1,
            0.95
        )
    end

    # Generate insights
    info_breadth = ai_config.info_breadth
    insights = generate_insights(opp, info_breadth, domain)

    # Add false insights if hallucinating
    if contains_hallucination
        append!(insights, generate_false_insights(domain; rng=rng))
    end

    # Create information object
    info = Information(
        estimated_return=estimated_return,
        estimated_uncertainty=estimated_uncertainty,
        confidence=stated_confidence,
        insights=insights,
        hidden_factors=Dict{String,Float64}(
            "bias" => bias,
            "unknown_uncertainty" => (1 - info_breadth) * opp.complexity,
            "info_quality_used" => info_quality,
            "market_shift_sensitivity" => 1 - actual_accuracy,
            "hallucination_uncertainty" => Float64(contains_hallucination),
            "ai_complementarity_delta" => complementarity_delta,
            "novelty_generalization_pressure" => generalization_pressure
        ),
        domain=domain,
        actual_accuracy=actual_accuracy,
        contains_hallucination=contains_hallucination,
        bias_applied=bias,
        overconfidence_factor=overconfidence_factor
    )

    sys.information_cache[cache_key] = info
    return info
end

"""
Get human-only information (no AI assistance).
"""
function get_human_information(
    sys::InformationSystem,
    opp::Opportunity,
    agent_traits::Dict{String,Float64};
    rng::Random.AbstractRNG=Random.default_rng()
)::Information
    # Human quality based on traits
    quality = (
        get(agent_traits, "analytical_ability", 0.5) * 0.4 +
        get(agent_traits, "competence", 0.5) * 0.4 +
        get(agent_traits, "market_awareness", 0.5) * 0.2
    ) * 0.30

    breadth = (
        get(agent_traits, "exploration_tendency", 0.5) * 0.5 +
        get(agent_traits, "market_awareness", 0.5) * 0.5
    ) * 0.25

    # Estimate with higher noise for humans
    return_noise = randn(rng) * (1 - quality) * 0.7
    estimated_return = opp.latent_return_potential + return_noise

    uncertainty_noise = randn(rng) * (1 - quality) * 0.5
    estimated_uncertainty = opp.latent_failure_potential + uncertainty_noise

    confidence = clamp(get(agent_traits, "competence", 0.5) * 0.4, 0.05, 0.45)

    insights = generate_insights(opp, breadth, nothing)

    return Information(
        estimated_return=clamp(estimated_return,
            sys.config.OPPORTUNITY_RETURN_RANGE[1],
            sys.config.OPPORTUNITY_RETURN_RANGE[2]),
        estimated_uncertainty=clamp(estimated_uncertainty,
            sys.config.OPPORTUNITY_UNCERTAINTY_RANGE[1],
            sys.config.OPPORTUNITY_UNCERTAINTY_RANGE[2]),
        confidence=confidence,
        insights=insights,
        hidden_factors=Dict{String,Float64}(
            "human_bias" => Float64(0.5 - get(agent_traits, "analytical_ability", 0.5)),
            "unknowns" => Float64(1 - quality)
        ),
        domain=nothing,
        actual_accuracy=quality,
        contains_hallucination=false,
        bias_applied=0.0,
        overconfidence_factor=1.0
    )
end

"""
Clear the per-round information caches: the per-agent information cache AND
the per-(opportunity, tier) common-error cache (emergence audit P9 — the
common cache MUST clear with the info cache so eps_common is per-(opp,
round, tier), not per-(opp, tier) forever). Called at the start of every
simulation round (step!, simulation.jl).
"""
function clear_cache!(sys::InformationSystem)
    empty!(sys.information_cache)
    empty!(sys.common_error_cache)
    return nothing
end

"""
Update an agent's AI learning profile with realized investment outcome data.

Infer AI disconfirmation from observable post-maturity evidence.

This intentionally does not read hidden `contains_hallucination` truth.
Agents can observe realized return, forecast error, confidence, repeated
domain misses, and their own analytical ability; they cannot observe the
model's latent hallucination flag.
"""
function observable_ai_disconfirmation(;
    investment_amount::Float64,
    capital_returned::Float64,
    predicted_return::Float64,
    ai_confidence::Float64,
    success::Bool,
    recent_accuracy::Float64 = 0.5,
    analytical_ability::Float64 = 0.5,
)
    actual_return = capital_returned / max(1.0, investment_amount)
    return_error = abs(actual_return - predicted_return) / max(1.0, abs(predicted_return))
    accuracy_score = clamp(1.0 - return_error, 0.0, 1.0)

    # Smooth bounded evidence terms avoid hard caps while keeping the signal in
    # probability-like units. Inputs that are already model probabilities are
    # range-checked only to preserve invariants.
    confidence = isfinite(ai_confidence) ? clamp(ai_confidence, 0.0, 1.0) : 0.5
    analytical = isfinite(analytical_ability) ? clamp(analytical_ability, 0.0, 1.0) : 0.5
    recent = isfinite(recent_accuracy) ? clamp(recent_accuracy, 0.0, 1.0) : 0.5

    error_signal = 1.0 - exp(-return_error)
    confidence_signal = stable_sigmoid(5.0 * (confidence - 0.65))
    bad_outcome_signal = success ? 0.0 : 1.0 - exp(-max(0.0, 1.0 - actual_return))
    recent_miss_signal = max(0.0, 0.65 - recent) / 0.65

    evidence = (
        0.50 * error_signal +
        0.20 * error_signal * confidence_signal +
        0.10 * bad_outcome_signal +
        0.20 * recent_miss_signal
    )
    attribution_threshold = 0.40 - 0.10 * analytical
    disconfirmation_score = stable_sigmoid(8.0 * (evidence - attribution_threshold))
    suspected_ai_failure = evidence > attribution_threshold

    severity = disconfirmation_score * error_signal *
               (0.75 + 0.50 * confidence_signal) *
               (1.0 - 0.25 * analytical)

    return (
        actual_return=actual_return,
        return_error=return_error,
        accuracy_score=accuracy_score,
        evidence=evidence,
        attribution_threshold=attribution_threshold,
        disconfirmation_score=disconfirmation_score,
        suspected_ai_failure=suspected_ai_failure,
        severity=severity,
    )
end

"""
Update an agent's AI learning profile from a realized investment outcome.

Increments usage_count per AI call, computes realized-vs-predicted return
error, appends an accuracy score, increments hallucination_experiences for
observable suspected AI disconfirmation, and updates per-domain trust on
success or on high-confidence inaccuracy.

The `hallucination_experiences` field name is retained for compatibility,
but its writes come from observable inference, not the hidden
`contains_hallucination` truth.
"""
function update_agent_learning!(profile::AILearningProfile, domain::String,
                                investment_amount::Float64, capital_returned::Float64,
                                predicted_return::Float64, suspected_ai_failure::Bool,
                                ai_confidence::Float64, success::Bool)
    # Defensive defaults for new domains
    if !haskey(profile.usage_count, domain)
        profile.usage_count[domain] = 0
    end
    if !haskey(profile.accuracy_estimates, domain)
        profile.accuracy_estimates[domain] = Float64[]
    end
    if !haskey(profile.hallucination_experiences, domain)
        profile.hallucination_experiences[domain] = 0
    end
    if !haskey(profile.domain_trust, domain)
        profile.domain_trust[domain] = 0.5
    end

    profile.usage_count[domain] += 1

    actual_return = capital_returned / max(1.0, investment_amount)
    return_error = abs(actual_return - predicted_return) / max(1.0, abs(predicted_return))
    was_accurate = return_error < 0.3

    push!(profile.accuracy_estimates[domain], clamp(1.0 - return_error, 0.0, 1.0))
    while length(profile.accuracy_estimates[domain]) > 20
        popfirst!(profile.accuracy_estimates[domain])
    end

    if suspected_ai_failure
        profile.hallucination_experiences[domain] += 1
    end

    if success
        update_trust!(profile, domain, was_accurate; magnitude=0.1)
    elseif !was_accurate && ai_confidence > 0.7
        update_trust!(profile, domain, false; magnitude=0.2)
    end

    return was_accurate
end

# ============================================================================
# ENHANCED AI ANALYSIS
# ============================================================================

"""
Get enhanced AI analysis (vectorized computation).
"""
function get_enhanced_ai_analysis(
    latent_return_potential::Float64,
    latent_failure_potential::Float64,
    complexity::Float64,
    base_accuracy::Float64,
    base_quality::Float64,
    hallucination_rate::Float64,
    bias::Float64,
    return_range::Tuple{Float64,Float64},
    uncertainty_range::Tuple{Float64,Float64};
    rng::Random.AbstractRNG=Random.default_rng()
)
    actual_accuracy = clamp(randn(rng) * 0.1 + base_accuracy, 0.0, 1.0)

    return_noise = randn(rng) * (1 - actual_accuracy) * 0.5
    estimated_return = latent_return_potential + return_noise + bias * 0.3

    uncertainty_noise = randn(rng) * (1 - actual_accuracy) * 0.3
    estimated_uncertainty = latent_failure_potential + uncertainty_noise - bias * 0.2

    contains_hallucination = rand(rng) < hallucination_rate
    if contains_hallucination
        if rand(rng) < 0.5
            estimated_return = rand(rng) * (return_range[2] - return_range[1]) + return_range[1]
        else
            estimated_uncertainty = rand(rng) * (uncertainty_range[2] - uncertainty_range[1]) + uncertainty_range[1]
        end
    end

    estimated_return = clamp(estimated_return, return_range[1], return_range[2])
    estimated_uncertainty = clamp(estimated_uncertainty, uncertainty_range[1], uncertainty_range[2])

    true_confidence = actual_accuracy * (1 - complexity * (1 - actual_accuracy))
    overconfidence_factor = base_quality < 0.5 ? 1.0 + (0.5 - base_quality) * 0.5 : 1.0
    stated_confidence = clamp(true_confidence * overconfidence_factor, 0.1, 0.95)

    return (estimated_return, estimated_uncertainty, stated_confidence, contains_hallucination)
end

"""
Get stochastic hallucination rate with domain-specific adjustments.
"""
function get_stochastic_hallucination_rate(
    base_rate::Float64,
    domain::String;
    rng::Random.AbstractRNG=Random.default_rng()
)::Float64
    # Beta distribution parameters based on base rate
    if base_rate < 0.1
        alpha, beta = 2, 20
    elseif base_rate < 0.2
        alpha, beta = 3, 12
    else
        alpha, beta = 4, 8
    end

    stochastic_factor = rand(rng)
    stochastic_rate = base_rate * (0.5 + stochastic_factor)

    # Random streak effect
    if rand(rng) < 0.1
        streak = rand(rng) * 0.4 + 0.8  # 0.8 to 1.2
        stochastic_rate *= streak
    end

    return clamp(stochastic_rate, 0.0, 0.5)
end
