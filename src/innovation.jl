"""
Innovation system for GlimpseABM.jl

Handles the creation of new knowledge through combinations,
innovation tracking, and success evaluation.
"""

using Random
using Statistics

# ============================================================================
# CONSTANTS
# ============================================================================

const DOMAIN_TO_INDEX = Dict(
    "technology" => 1,
    "market" => 2,
    "process" => 3,
    "business_model" => 4
)

const ADJACENCY_MATRIX = [
    false true  true  false;
    true  false false true;
    true  false false true;
    false true  true  false
]

# ============================================================================
# COMBINATION TRACKER
# ============================================================================

"""
Tracks knowledge combinations used across the market.
"""
mutable struct CombinationTracker
    combination_history::Dict{String,Vector{String}}
    combination_success::Dict{String,Vector{Float64}}
    total_combinations::Int
end

"""
Create a new CombinationTracker.
"""
function CombinationTracker()
    return CombinationTracker(
        Dict{String,Vector{String}}(),
        Dict{String,Vector{Float64}}(),
        0
    )
end

"""
Check if a combination signature is new.
"""
function is_new_signature(tracker::CombinationTracker, combination_signature::Union{String,Nothing})::Bool
    if isnothing(combination_signature)
        return false
    end
    return !haskey(tracker.combination_history, combination_signature)
end

"""
Record a combination being used.
"""
function record_combination!(tracker::CombinationTracker, combination_signature::String, innovation_id::String)
    if !haskey(tracker.combination_history, combination_signature)
        tracker.combination_history[combination_signature] = String[]
    end
    push!(tracker.combination_history[combination_signature], innovation_id)
    tracker.total_combinations += 1
end

"""
Record the outcome of a combination.
"""
function record_outcome!(tracker::CombinationTracker, combination_signature::String, success::Float64)
    if !haskey(tracker.combination_success, combination_signature)
        tracker.combination_success[combination_signature] = Float64[]
    end
    push!(tracker.combination_success[combination_signature], success)
end

"""
Sample a random combination signature from history.
"""
function sample_signature(tracker::CombinationTracker; lookback::Int=100, rng::AbstractRNG=Random.default_rng())::Union{String,Nothing}
    if isempty(tracker.combination_history)
        return nothing
    end
    signatures = collect(keys(tracker.combination_history))
    if isempty(signatures)
        return nothing
    end
    signature = signatures[rand(rng, 1:length(signatures))]
    history = get(tracker.combination_history, signature, String[])
    if !isempty(history) && lookback > 0
        history = history[max(1, length(history)-lookback+1):end]
        if isempty(history)
            return signature
        end
    end
    return signature
end

"""
Get the reuse ratio for a combination signature.
"""
function get_reuse_ratio(tracker::CombinationTracker, combination_signature::Union{String,Nothing})::Float64
    if isnothing(combination_signature)
        return 0.0
    end
    history = get(tracker.combination_history, combination_signature, String[])
    if isempty(history) || tracker.total_combinations <= 0
        return 0.0
    end
    return clamp(length(history) / max(1, tracker.total_combinations), 0.0, 1.0)
end

# ============================================================================
# INNOVATION ENGINE
# ============================================================================

"""
Handles the creation of new knowledge through combinations.
"""
mutable struct InnovationEngine
    config::EmergentConfig
    knowledge_base::KnowledgeBase
    combination_tracker::CombinationTracker
    innovations::Dict{String,Innovation}
    innovation_history::Dict{Int,Vector{Innovation}}
    rd_investments::Dict{Int,Float64}
    ai_assisted_innovations::Set{String}
    innovation_success_by_ai::Dict{String,Vector{Bool}}
end

"""
Create a new InnovationEngine.
"""
function InnovationEngine(config::EmergentConfig, knowledge_base::KnowledgeBase, combination_tracker::CombinationTracker)
    return InnovationEngine(
        config,
        knowledge_base,
        combination_tracker,
        Dict{String,Innovation}(),
        Dict{Int,Vector{Innovation}}(),
        Dict{Int,Float64}(),
        Set{String}(),
        Dict{String,Vector{Bool}}("ai_assisted" => Bool[], "human_only" => Bool[])
    )
end

"""
Attempt to create an innovation.
"""
function attempt_innovation!(
    engine::InnovationEngine,
    agent::Any,  # EmergentAgent - using Any to avoid circular dependency
    market_conditions::MarketConditions,
    round::Int;
    ai_level::String="none",
    uncertainty_perception::Union{Perception,Nothing}=nothing,
    decision_perception::Union{Perception,Nothing}=nothing,
    rng::AbstractRNG=Random.default_rng()
)::Union{Innovation,Nothing}
    behavior_level = behavior_ai_level(engine.config, ai_level)

    # Get accessible knowledge
    accessible_knowledge = get_accessible_knowledge(
        engine.knowledge_base,
        agent.id,
        behavior_level;
        agent_traits=agent.traits,
        rng=rng
    )

    if length(accessible_knowledge) < 2
        return nothing
    end

    # Calculate innovation probability using sector-specific rate (NSF BRDIS/USPTO calibrated)
    sector_profile = get(engine.config.SECTOR_PROFILES, agent.primary_sector, nothing)
    base_prob = if !isnothing(sector_profile) && hasproperty(sector_profile, :innovation_probability)
        sector_profile.innovation_probability
    else
        engine.config.INNOVATION_PROBABILITY
    end
    # Use agent's innovativeness trait and competence field
    competence_score = (
        get(agent.traits, "innovativeness", 0.5) * 0.6 +
        (hasfield(typeof(agent), :competence) ? agent.competence : 0.5) * 0.4
    )

    avg_trust = 0.5
    dynamic_bonus = 0.0
    clarity_signal = 0.0

    if !isnothing(decision_perception)
        ignorance = get(get(decision_perception, "actor_ignorance", Dict()), "ignorance_level", 0.5)
        indeterminism = get(get(decision_perception, "practical_indeterminism", Dict()), "indeterminism_level", 0.5)
        clarity_signal = ((stable_sigmoid(1.0 - ignorance) + stable_sigmoid(1.0 - indeterminism)) / 2.0) - 0.5
    end

    # AI learning profile adjustments
    if counts_as_ai_use(engine.config, ai_level) && !isnothing(agent.ai_learning)
        learning_profile = agent.ai_learning
        trust_values = [
            get(learning_profile.domain_trust, dom, 0.5)
            for dom in ["technical_assessment", "innovation_potential"]
        ]
        avg_trust = isempty(trust_values) ? 0.5 : mean(trust_values)

        # Reliability signal: recent per-domain accuracy relative to the 0.5
        # neutral midpoint. Evaluates to 0 until accuracy_estimates is populated
        # (empty scores), then biases the AI bonus by the agent's observed
        # forecast accuracy in the relevant domains.
        reliability_signals = Float64[]
        for dom in ["technical_assessment", "innovation_potential"]
            scores = get(learning_profile.accuracy_estimates, dom, Float64[])
            if !isempty(scores)
                push!(reliability_signals, mean(scores[max(1, length(scores)-4):end]) - 0.5)
            end
        end
        reliability = isempty(reliability_signals) ? 0.0 : mean(reliability_signals)
        dynamic_bonus = (avg_trust - 0.5) * 0.2 + reliability * 0.25 + clarity_signal * 0.15
    end

    ai_bonus = clamp(dynamic_bonus, -0.2, 0.3)

    human_ingenuity_bonus = (
        get(agent.traits, "exploration_tendency", 0.5) * 0.15 +
        get(agent.traits, "market_awareness", 0.5) * 0.15 +
        get(agent.traits, "innovativeness", 0.5) * 0.15
    )

    innovation_prob = base_prob * 0.35 + competence_score * 0.45 + ai_bonus + human_ingenuity_bonus
    innovation_prob = clamp(innovation_prob, 0.05, 0.95)

    if rand(rng) > innovation_prob
        return nothing
    end

    # Determine AI domains used. Gate: an agent uses AI in a domain when its
    # learned trust in that domain exceeds the 0.45 threshold.
    ai_domains_used = String[]
    if counts_as_ai_use(engine.config, ai_level) && !isnothing(agent.ai_learning)
        learning_profile = agent.ai_learning
        for domain in ["technical_assessment", "innovation_potential"]
            trust = get(learning_profile.domain_trust, domain, 0.5)
            if trust > 0.45
                push!(ai_domains_used, domain)
            end
        end
    end

    # Determine innovation type
    # Use survival_rounds as a proxy for experience, or innovation_count
    experience_units = Float64(hasfield(typeof(agent), :survival_rounds) ? agent.survival_rounds : 0) +
                       Float64(hasfield(typeof(agent), :innovation_count) ? agent.innovation_count * 2 : 0)
    innovation_type = determine_innovation_type(
        engine,
        accessible_knowledge,
        agent.traits,
        market_conditions;
        ai_assisted=length(ai_domains_used) > 0,
        experience_units=experience_units,
        rng=rng
    )

    # Get component count
    n_components = get_component_count(innovation_type; rng=rng)

    # Check for reuse - use hasfield for struct access
    reuse_prob = hasfield(typeof(engine.config), :INNOVATION_REUSE_PROBABILITY) ?
        engine.config.INNOVATION_REUSE_PROBABILITY : 0.0
    lookback = hasfield(typeof(engine.config), :INNOVATION_REUSE_LOOKBACK) ?
        engine.config.INNOVATION_REUSE_LOOKBACK : 100
    selected_knowledge = nothing
    reuse_signature = nothing

    # Combination reuse is a market-history parameter. AI can still alter
    # novelty through the knowledge it makes accessible above, but should not
    # directly suppress reuse once the accessible set and reuse history are fixed.
    effective_reuse_prob = clamp(reuse_prob, 0.02, 0.75)

    if reuse_prob > 0 && rand(rng) < effective_reuse_prob
        reuse_signature = sample_signature(engine.combination_tracker; lookback=lookback, rng=rng)
        if !isnothing(reuse_signature)
            component_ids = split(reuse_signature, "||")
            knowledge_lookup = engine.knowledge_base.knowledge_pieces
            candidate = [knowledge_lookup[k_id] for k_id in component_ids if haskey(knowledge_lookup, k_id)]
            if length(candidate) >= n_components
                selected_knowledge = candidate[1:n_components]
            end
        end
    end

    if isnothing(selected_knowledge)
        selected_knowledge = select_knowledge_combination(
            engine,
            accessible_knowledge,
            n_components,
            agent.traits,
            agent.ai_learning,
            behavior_level;
            rng=rng
        )
    end

    if isnothing(selected_knowledge)
        return nothing
    end

    # Determine sector
    innovation_sector = determine_innovation_sector(engine, agent, selected_knowledge)

    # Create innovation
    innovation = create_innovation(
        engine,
        selected_knowledge,
        innovation_type,
        agent,
        round;
        ai_assisted=length(ai_domains_used) > 0,
        ai_domains_used=ai_domains_used,
        ai_level_used=ai_level,
        sector=innovation_sector,
        rng=rng
    )

    if !isnothing(reuse_signature)
        innovation.combination_signature = reuse_signature
        innovation.novelty = clamp(innovation.novelty * 0.6, 0.0, 1.0)
    end

    # Record usage
    record_usage!(engine.knowledge_base, innovation.knowledge_components)
    innovation.scarcity = get_combination_scarcity(engine.knowledge_base, innovation.knowledge_components)

    # Store innovation
    engine.innovations[innovation.id] = innovation
    if !haskey(engine.innovation_history, agent.id)
        engine.innovation_history[agent.id] = Innovation[]
    end
    push!(engine.innovation_history[agent.id], innovation)

    if innovation.ai_assisted
        push!(engine.ai_assisted_innovations, innovation.id)
    end

    return innovation
end

"""
Determine the type of innovation to create.
"""
function determine_innovation_type(
    engine::InnovationEngine,
    knowledge_pieces::Vector{Knowledge},
    agent_traits::Dict{String,Float64},
    market_conditions::MarketConditions;
    ai_assisted::Bool=false,
    experience_units::Float64=0.0,
    rng::AbstractRNG=Random.default_rng()
)::String
    base_probabilities = Dict{String,Float64}(
        "incremental" => 0.4 + max(experience_units, 0) * 0.01,
        "architectural" => 0.3 + get(agent_traits, "trait_momentum", 0.1) * 0.35,
        "radical" => 0.2 + get(agent_traits, "innovativeness", 0.5) * 0.25,
        "disruptive" => 0.1 + get(agent_traits, "exploration_tendency", 0.5) * 0.2
    )

    if market_conditions.regime == "crisis"
        base_probabilities["disruptive"] += 0.05
    end

    # Normalize probabilities
    total = sum(values(base_probabilities))
    for key in keys(base_probabilities)
        base_probabilities[key] /= total
    end

    types = collect(keys(base_probabilities))
    probabilities = [base_probabilities[t] for t in types]

    return weighted_choice(types, probabilities; rng=rng)
end

"""
Get the number of knowledge components for an innovation type.
"""
function get_component_count(innovation_type::String; rng::AbstractRNG=Random.default_rng())::Int
    if innovation_type == "incremental"
        return rand(rng) < 0.7 ? 2 : 3
    elseif innovation_type == "architectural"
        return rand(rng) < 0.6 ? 3 : 4
    elseif innovation_type == "radical"
        r = rand(rng)
        return r < 0.3 ? 3 : (r < 0.7 ? 4 : 5)
    else  # disruptive
        return rand(rng) < 0.5 ? 4 : 5
    end
end

"""
Select knowledge combination for an innovation.
"""
function select_knowledge_combination(
    engine::InnovationEngine,
    accessible_knowledge::Vector{Knowledge},
    n_components::Int,
    agent_traits::Dict{String,Float64},
    ai_learning_profile::Union{AILearningProfile,Nothing},
    ai_level::String;
    rng::AbstractRNG=Random.default_rng()
)::Union{Vector{Knowledge},Nothing}
    if isempty(accessible_knowledge) || length(accessible_knowledge) < n_components
        return nothing
    end

    accessible = copy(accessible_knowledge)
    num_accessible = length(accessible)

    # Build domain indices and levels
    domain_indices = Int[]
    domain_recognised = true
    for item in accessible
        idx = get(DOMAIN_TO_INDEX, item.domain, nothing)
        if isnothing(idx)
            domain_recognised = false
            break
        end
        push!(domain_indices, idx)
    end
    levels = [k.level for k in accessible]

    # Start with random first component
    first_idx = rand(rng, 1:num_accessible)
    knowledge_components = [accessible[first_idx]]
    selected_indices = [first_idx]
    remaining_indices = [idx for idx in 1:num_accessible if idx != first_idx]

    if domain_recognised
        selected_domains = [domain_indices[first_idx]]
        selected_levels = [levels[first_idx]]
    else
        selected_domains = nothing
        selected_levels = nothing
    end

    # Build combination iteratively
    while length(knowledge_components) < n_components && !isempty(remaining_indices)
        scores = Float64[]
        len_selected = length(selected_indices)

        if domain_recognised && !isnothing(selected_domains)
            for idx in remaining_indices
                domain_idx = domain_indices[idx]
                level_val = levels[idx]
                total = 0.0
                for (dom, lvl) in zip(selected_domains, selected_levels)
                    diff = abs(level_val - lvl)
                    if domain_idx == dom
                        comp = 0.8 + 0.2 * (1.0 - diff)
                    elseif ADJACENCY_MATRIX[domain_idx, dom]
                        comp = 0.4 + 0.2 * (1.0 - diff)
                    else
                        comp = 0.1 + 0.1 * (1.0 - diff)
                    end
                    total += comp
                end
                push!(scores, len_selected > 0 ? total / len_selected : 0.1)
            end
        else
            selected_objects = [accessible[s_idx] for s_idx in selected_indices]
            for idx in remaining_indices
                total = 0.0
                for other in selected_objects
                    total += get_compatibility(engine.knowledge_base, accessible[idx], other)
                end
                push!(scores, len_selected > 0 ? total / len_selected : 0.1)
            end
        end

        weights = max.(scores, 1e-6)
        total_weight = sum(weights)
        if total_weight <= 0
            break
        end
        weights ./= total_weight

        choice_pos = weighted_choice(collect(1:length(remaining_indices)), weights; rng=rng)
        next_idx = remaining_indices[choice_pos]
        deleteat!(remaining_indices, choice_pos)

        if next_idx in selected_indices
            continue
        end

        push!(selected_indices, next_idx)
        push!(knowledge_components, accessible[next_idx])

        if domain_recognised && !isnothing(selected_domains)
            push!(selected_domains, domain_indices[next_idx])
            push!(selected_levels, levels[next_idx])
        end
    end

    if length(knowledge_components) < 2
        return nothing
    end

    return knowledge_components
end

"""
Determine the sector for an innovation.
"""
function determine_innovation_sector(
    engine::InnovationEngine,
    agent::Any,  # EmergentAgent - using Any to avoid circular dependency
    selected_knowledge::Vector{Knowledge}
)::String
    # Use knowledge base domain-to-sector mapping
    domain_to_sector = engine.knowledge_base.domain_to_sector

    # Count domains
    domain_counts = Dict{String,Int}()
    for k in selected_knowledge
        domain_counts[k.domain] = get(domain_counts, k.domain, 0) + 1
    end

    # Find dominant domain
    dominant_domain = first(sort(collect(domain_counts), by=x->x[2], rev=true))[1]

    # Get available sectors
    available_sectors = collect(engine.config.SECTORS)
    default_sector = isempty(available_sectors) ? "tech" : available_sectors[1]

    return get(domain_to_sector, dominant_domain, default_sector)
end

"""
Create an innovation from selected knowledge.
"""
function create_innovation(
    engine::InnovationEngine,
    knowledge_pieces::Vector{Knowledge},
    innovation_type::String,
    agent::Any,  # EmergentAgent - using Any to avoid circular dependency
    round::Int;
    ai_assisted::Bool=false,
    ai_domains_used::Vector{String}=String[],
    ai_level_used::String="none",
    sector::Union{String,Nothing}=nothing,
    rng::AbstractRNG=Random.default_rng()
)::Innovation
    knowledge_levels = [k.level for k in knowledge_pieces]
    base_quality = mean(knowledge_levels) * 0.7 + rand(rng) * 0.2 + 0.2
    mix_penalty = std(knowledge_levels)
    quality = clamp(base_quality * (1 - mix_penalty * 0.3), 0.1, 1.0)

    # Explicit favorable-AI/refutation knob; baseline defaults to zero.
    if ai_assisted && hasfield(typeof(engine.config), :AI_QUALITY_BOOST)
        quality_boost = get(engine.config.AI_QUALITY_BOOST, ai_level_used, 0.0)
        quality += quality_boost
    end
    quality = clamp(quality + randn(rng) * 0.05, 0.0, 1.0)

    # Novelty comes from combining DIVERSE knowledge: the std of component
    # knowledge levels rewards diverse combinations with higher novelty.
    knowledge_levels = [k.level for k in knowledge_pieces]
    diversity = length(knowledge_levels) > 1 ? std(knowledge_levels) : 0.0
    # Scale diversity (typically 0-0.3) into the novelty range, plus base and noise.
    novelty = clamp(
        diversity * 2.0 + rand(rng) * 0.2 + 0.2,
        0.0,
        1.0
    )

    if isnothing(sector)
        sector = "tech"
    end

    innovation_id = "inn_$(agent.id)_$(round)_$(rand(rng, 1:999))"

    innovation = Innovation(
        id=innovation_id,
        type=innovation_type,
        knowledge_components=[k.id for k in knowledge_pieces],
        novelty=novelty,
        quality=quality,
        round_created=round,
        creator_id=agent.id,
        ai_assisted=ai_assisted,
        ai_domains_used=ai_domains_used,
        ai_level_used=ai_level_used,
        sector=sector
    )

    # Record combination
    combination_signature = join(sort(innovation.knowledge_components), "||")
    innovation.is_new_combination = is_new_signature(engine.combination_tracker, combination_signature)
    record_combination!(engine.combination_tracker, combination_signature, innovation.id)
    innovation.combination_signature = combination_signature

    return innovation
end

"""
Invest in R&D for an agent.
"""
function invest_in_rd!(engine::InnovationEngine, agent_id::Int, amount::Float64)
    engine.rd_investments[agent_id] = get(engine.rd_investments, agent_id, 0.0) + amount
end

"""
Evaluate innovation success.
"""
function evaluate_innovation_success!(
    engine::InnovationEngine,
    innovation::Innovation,
    market_conditions::MarketConditions,
    market_innovations::Vector{Innovation};
    rng::AbstractRNG=Random.default_rng()
)::Tuple{Bool,Float64,Float64}
    potential = calculate_potential(innovation, market_conditions)

    if isnothing(innovation.sector)
        innovation.sector = "tech"
    end

    # Find competing innovations — sector-filtered for O(K_sector) instead of O(K_total)
    min_round = innovation.round_created - 5
    sector = something(innovation.sector, "tech")
    competing_innovations = [
        inn for inn in market_innovations
        if inn.id != innovation.id &&
           inn.round_created >= min_round &&
           something(inn.sector, "") == sector
    ]

    # Get sector-specific competition intensity (Census HHI-calibrated)
    sector_profile = get(engine.config.SECTOR_PROFILES, innovation.sector, nothing)
    sector_competition_intensity = if !isnothing(sector_profile) && hasproperty(sector_profile, :competition_intensity)
        sector_profile.competition_intensity
    else
        1.0  # Default intensity
    end

    competition_factor = if !isempty(competing_innovations)
        competitor_strength = mean([c.quality * c.novelty for c in competing_innovations])
        # Apply sector-specific intensity to competition effects
        1 - min(0.5, competitor_strength * sector_competition_intensity)
    else
        1.0
    end

    # Market readiness factor
    readiness_factor = if innovation.novelty > 0.8
        0.7
    elseif innovation.novelty < 0.3
        0.8
    else
        1.0
    end

    scarcity = Float64(something(innovation.scarcity, 0.5))
    novelty = Float64(something(innovation.novelty, 0.5))
    scarcity_boost = 1.0 + (scarcity - 0.5) * 0.35

    base_success_prob = potential * competition_factor * readiness_factor * scarcity_boost

    # Apply AI-tier execution success multiplier (for refutation testing)
    # Default multipliers are 1.0 for all tiers; can be modified in config
    ai_tier = innovation.ai_level_used
    execution_multiplier = if hasfield(typeof(engine.config), :AI_EXECUTION_SUCCESS_MULTIPLIERS)
        get(engine.config.AI_EXECUTION_SUCCESS_MULTIPLIERS, ai_tier, 1.0)
    else
        1.0
    end

    success_prob = clamp(base_success_prob * execution_multiplier, 0.0, 0.95)
    success = rand(rng) < success_prob

    impact = 0.0
    cash_multiple = 0.0

    if success
        impact = innovation.quality * innovation.novelty * competition_factor

        if innovation.sector == "tech"
            impact *= 1.2
        elseif innovation.sector == "manufacturing"
            impact *= 0.9
        end

        impact = clamp(impact * (1.0 + (novelty - 0.5) * 0.25), 0.05, 2.5)

        base_multiple = 1.25 + (hasfield(typeof(engine.config), :INNOVATION_SUCCESS_BASE_RETURN) ?
            engine.config.INNOVATION_SUCCESS_BASE_RETURN : 0.25)

        # Use sector-specific innovation return multiplier (R&D intensity calibrated)
        sector_profile = get(engine.config.SECTOR_PROFILES, innovation.sector, nothing)
        mult_range = if !isnothing(sector_profile) && hasproperty(sector_profile, :innovation_return_multiplier)
            sector_profile.innovation_return_multiplier
        else
            (hasfield(typeof(engine.config), :INNOVATION_SUCCESS_RETURN_MULTIPLIER) ?
                engine.config.INNOVATION_SUCCESS_RETURN_MULTIPLIER : (1.8, 3.0))
        end

        low, high = if isa(mult_range, Tuple) && length(mult_range) >= 2
            Float64(mult_range[1]), Float64(mult_range[2])
        else
            Float64(mult_range), Float64(mult_range) + 1.0
        end

        if high < low
            low, high = high, low
        end

        impact_gain = rand(rng) * (high - low) + low
        scarcity_bonus = 1.0 + (scarcity - 0.5) * 0.8
        novelty_bonus = 1.0 + (novelty - 0.5) * 0.55
        cash_multiple = clamp((base_multiple + impact * impact_gain) * scarcity_bonus * novelty_bonus, 1.1, 8.5)
    else
        recovery_ratio = hasfield(typeof(engine.config), :INNOVATION_FAIL_RECOVERY_RATIO) ?
            engine.config.INNOVATION_FAIL_RECOVERY_RATIO : 0.15
        # Linear interpolation for recovery floor
        novelty_clamped = clamp(innovation.novelty, 0.05, 0.95)
        recovery_floor = 0.78 - (novelty_clamped - 0.05) / (0.95 - 0.05) * (0.78 - 0.42)
        salvage = max(recovery_ratio, recovery_floor - 0.12 * (scarcity - 0.5))
        cash_multiple = clamp(salvage, 0.25, 0.65)
    end

    innovation.success = success
    innovation.market_impact = impact

    if innovation.ai_assisted
        push!(engine.innovation_success_by_ai["ai_assisted"], success)
    else
        push!(engine.innovation_success_by_ai["human_only"], success)
    end

    return success, impact, cash_multiple
end

# Note: calculate_potential is defined in models.jl
