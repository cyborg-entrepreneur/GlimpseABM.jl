"""
GlimpseABM.jl — the GLIMPSE Agent-Based Model

A high-performance Julia implementation of the GLIMPSE ABM for studying
AI adoption and Knightian uncertainty in entrepreneurial ecosystems.

Theoretical Foundation
----------------------
This model operationalizes concepts from:

    Townsend, D. M., Hunt, R. A., Rady, J., Manocha, P., & Jin, J-H. (2025).
    Are the futures computable? Knightian uncertainty & artificial intelligence.
    Academy of Management Review, 50(2), 415-440.

The four dimensions of Knightian uncertainty modeled are:
1. Actor Ignorance - Information gaps about current states
2. Practical Indeterminism - Unpredictable execution outcomes
3. Agentic Novelty - Genuinely new possibilities from creative action
4. Competitive Recursion - Strategic interdependence effects

License: MIT
"""
module GlimpseABM

using Random
using Random: AbstractRNG  # Explicitly import to avoid ambiguity with RandomNumbers
using Statistics
using LinearAlgebra
using Distributions
using DataFrames
using Dates

# Action-dict field name constants (centralizes producer/consumer keys
# whose mismatch silently zeroes dataflow at module boundaries)
include("action_keys.jl")

# Core configuration and models
include("config.jl")
# MarketConditions typed payload — loaded before models.jl because
# realized_return consumes it.
include("market_conditions.jl")
include("models.jl")

# Utilities (loaded early for stable_sigmoid and other helpers)
include("utils.jl")

# Knowledge and information systems
include("knowledge.jl")
include("information.jl")

# Innovation system (before agents.jl - uses Any for agent type to avoid circular dep)
include("innovation.jl")

# AGI strategy ladder (the strategy-ladder design notes). After
# information.jl (reads Information / InformationSystem caches) and utils.jl
# (sector_familiarity, behavior_ai_level); before agents.jl, whose decision
# hooks call into it.
include("strategy.jl")

# Open-action extension (the strategy-ladder design notes, Open-action
# extension): A1 pivot trigger + A2 directed-creation density/weights. Same
# placement rationale as strategy.jl (reads Information; called from
# agents.jl/innovation.jl hooks). innovation.jl's directed branch references
# these functions at RUNTIME only, so the include order here is safe.
include("open_action.jl")

# Simulation components
include("market.jl")
include("uncertainty.jl")
include("agents.jl")
include("simulation.jl")

# I/O utilities
include("io.jl")

# Optional deterministic MT19937 (NumpyRNG) stream
include("numpy_rng.jl")

# Exports - Configuration
export EmergentConfig
export ActorIgnoranceProducerWeights
export PracticalIndeterminismProducerWeights
export AgenticNoveltyProducerWeights
export CompetitiveRecursionProducerWeights
export KnightianProducerWeights
export UncertaintyPerceptionWeights
export MarketConditions
export CalibrationProfile
export apply_calibration_profile
export get_calibration_profile
export load_calibration_profile
export CALIBRATION_LIBRARY
export initialize!, get_scaled_opportunities

# Exports - Models
export Opportunity
export Information
export Innovation
export Knowledge
export AIAnalysis
export AILearningProfile
export OpportunityEvaluation
export Perception, KnowledgeSignal, ActorIgnorance, ExecutionRisk
export PracticalIndeterminism, InnovationSignal, AgenticNovelty
export CompetitionSignal, CompetitiveRecursion
export empty_perception

# Exports - Knowledge and Information Systems
export KnowledgeBase
export InformationSystem
export InnovationEngine
export CombinationTracker
export get_accessible_knowledge
export get_information, update_agent_learning!, observable_ai_disconfirmation
export attempt_innovation!
export evaluate_innovation_success!
export get_component_scarcity_metric
export learn_from_success!, learn_from_failure!

# Exports - Simulation
export EmergentSimulation
export EmergentAgent
export MarketEnvironment
export KnightianUncertaintyEnvironment

# Exports - Functions
export run!
export step!
export initialize_agents!
export save_results
export load_results

# Exports - Agent distress tracking
export check_survival!
export get_capital, set_capital!, get_ai_level

# Exports - Emergent uncertainty (agent-level metrics)
export AgentUncertaintyMetrics
export get_emergent_uncertainty, compute_emergent_uncertainty
export emergent_uncertainty_observation_counts
export aggregate_emergent_uncertainty_by_tier
export record_investment_outcome!, record_creative_action!

# Exports - AI subscription charging
export ensure_subscription_schedule!, start_subscription_schedule!
export charge_subscription_installment!, apply_subscription_carry!

# Exports - Utility functions
export stable_sigmoid, safe_exp, safe_mean, fast_mean
export behavior_ai_level, counts_as_ai_use
export action_counts_as_ai_use, action_behavior_ai_level
export normalize_ai_label, compute_hhi, compute_gini
export perceive_uncertainty, measure_uncertainty_state!, get_uncertainty_state

# Exports — deterministic MT19937 (NumpyRNG) stream
export NumpyRNG, numpy_rand, numpy_randn, numpy_randint, numpy_seed!
export numpy_gamma, numpy_beta, numpy_uniform, numpy_exponential

# Exports — AGI strategy ladder (the strategy-ladder design notes)
export STRATEGY_MODES, validate_strategy_config
export strategy_active
export consensus_congestion_forecast, private_edge, complement_shift

# Exports — open-action extension (the strategy-ladder design notes
# §Open-action extension)
export validate_open_action_config
export pivot_haircut, pivot_trigger
export perceived_sector_density, directed_sector_weights

end # module
