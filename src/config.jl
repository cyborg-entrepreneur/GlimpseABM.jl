"""
Configuration module for GlimpseABM.jl

Provides the comprehensive configuration system for the agent-based model,
including all tunable parameters, calibration profiles, and validation utilities.
"""

using Random
using JSON3

const DEFAULT_SECTOR_ORDER = ["tech", "retail", "service", "manufacturing"]

function ordered_sector_keys(sector_profiles::Dict)
    ordered = [sector for sector in DEFAULT_SECTOR_ORDER if haskey(sector_profiles, sector)]
    extras = sort([String(sector) for sector in keys(sector_profiles) if !(String(sector) in ordered)])
    return vcat(ordered, extras)
end

# ============================================================================
# TRAIT DISTRIBUTION SPECIFICATION
# ============================================================================

"""Distribution specification for agent traits."""
struct TraitDistribution
    dist::String
    params::Dict{String,Float64}
end

function TraitDistribution(; dist::String, params::Dict)
    TraitDistribution(dist, Dict{String,Float64}(String(k) => Float64(v) for (k,v) in params))
end

# ============================================================================
# SECTOR PROFILE
# ============================================================================

"""
Sector-specific economic parameters.

Calibration Sources:
- initial_capital_range: NVCA 2024 Yearbook, PitchBook seed/early stage data (scaled for 24-36 round runway)
- operational_cost_range: BLS QCEW quarterly costs by sector (SBA small business benchmarks 2024)
- survival_threshold: BLS Business Employment Dynamics, Fed SBCS 2024 (2-3 quarters operating expenses)
- innovation_probability: NSF BRDIS 2023, USPTO Patent Statistics grant rates
- innovation_return_multiplier: Industry R&D intensity studies
- competition_intensity: Census Bureau Economic Census HHI data, DOJ HHI guidelines
"""
struct SectorProfile
    return_range::Tuple{Float64,Float64}
    return_log_mu::Float64
    return_log_sigma::Float64
    failure_range::Tuple{Float64,Float64}
    failure_volatility_range::Tuple{Float64,Float64}
    capital_range::Tuple{Float64,Float64}
    maturity_range::Tuple{Int,Int}
    # Empirically-calibrated fields
    initial_capital_range::Tuple{Float64,Float64}       # NVCA 2024: sector-specific seed/Series A
    operational_cost_range::Tuple{Float64,Float64}      # BLS QCEW: quarterly operating costs
    survival_threshold::Float64                          # BLS/Fed: ~2 quarters operating expenses
    innovation_probability::Float64                      # NSF BRDIS 2023, USPTO grant rates
    innovation_return_multiplier::Tuple{Float64,Float64} # R&D intensity returns by sector
    competition_intensity::Float64                       # Census HHI-based competition intensity
end

# ============================================================================
# AI LEVEL CONFIGURATION
# ============================================================================

"""AI tier configuration."""
struct AILevelConfig
    cost::Float64
    cost_type::String
    info_quality::Float64
    info_breadth::Float64
    per_use_cost::Float64
end

"""AI domain capability specification."""
struct AIDomainCapability
    accuracy::Float64
    hallucination_rate::Float64
    bias::Float64
end

# ============================================================================
# KNIGHTIAN UNCERTAINTY CALIBRATION
# ============================================================================

"""Producer-side weights for environment-level actor ignorance."""
@kwdef struct ActorIgnoranceProducerWeights
    base_level::Float64 = 0.25
    knowledge_gap_weight::Float64 = 0.28
    gap_pressure_weight::Float64 = 0.18
    gap_coverage_weight::Float64 = 0.12
    ai_disconfirmation_weight::Float64 = 0.15
    exploration_reduction_weight::Float64 = 0.10
    maintain_increase_weight::Float64 = 0.03
end

"""Producer-side weights for environment-level practical indeterminism."""
@kwdef struct PracticalIndeterminismProducerWeights
    crowding_investment_overlap_weight::Float64 = 0.40
    # Opportunity-level competition/saturation exposure (investment-weighted mean
    # over opportunities holding capital). Crowding is modeled as an
    # opportunity-level phenomenon, not a sector-level one, so this single
    # opportunity-level term carries the crowding weight (0.45).
    crowding_opportunity_competition_weight::Float64 = 0.45
    crowding_investment_concentration_weight::Float64 = 0.15
    base_level::Float64 = 0.25
    market_volatility_weight::Float64 = 0.30
    regime_uncertainty_weight::Float64 = 0.25
    timing_pressure_weight::Float64 = 0.25
    path_volatility_weight::Float64 = 0.20
    invest_share_weight::Float64 = 0.15
end

"""Producer-side weights for environment-level agentic novelty."""
@kwdef struct AgenticNoveltyProducerWeights
    disruption_base_bump::Float64 = 0.03
    disruption_novelty_weight::Float64 = 0.04
    disruption_impact_weight::Float64 = 0.02
    disruption_success_weight::Float64 = 0.04
    explore_niche_bump::Float64 = 0.02
    explore_general_bump::Float64 = 0.01
    reuse_combo_hhi_weight::Float64 = 0.60
    reuse_sector_hhi_weight::Float64 = 0.40
    base_level::Float64 = 0.25
    combo_rate_weight::Float64 = 0.20
    new_possibility_rate_weight::Float64 = 0.18
    niche_rate_weight::Float64 = 0.15
    innovation_intensity_weight::Float64 = 0.12
    disruption_weight::Float64 = 0.10
    sector_diversity_weight::Float64 = 0.08
    derivative_adoption_drag::Float64 = 0.15
    reuse_pressure_drag::Float64 = 0.20
end

"""Producer-side weights for environment-level competitive recursion."""
@kwdef struct CompetitiveRecursionProducerWeights
    crowd_weight::Float64 = 0.35
    volatility_weight::Float64 = 0.30
    ai_herd_weight::Float64 = 0.40
    reuse_weight::Float64 = 0.20
    invest_hhi_weight::Float64 = 0.45
    knowledge_overlap_weight::Float64 = 0.12
    ai_action_correlation_excess_weight::Float64 = 0.15
    population_scale_base::Float64 = 0.50
    population_scale_alive_weight::Float64 = 0.50
    no_ai_herding_weight::Float64 = 0.0
end

"""Producer-side calibration for measured Knightian uncertainty dimensions."""
@kwdef struct KnightianProducerWeights
    actor_ignorance::ActorIgnoranceProducerWeights = ActorIgnoranceProducerWeights()
    practical_indeterminism::PracticalIndeterminismProducerWeights = PracticalIndeterminismProducerWeights()
    agentic_novelty::AgenticNoveltyProducerWeights = AgenticNoveltyProducerWeights()
    competitive_recursion::CompetitiveRecursionProducerWeights = CompetitiveRecursionProducerWeights()
end

"""Agent-side calibration for subjective Knightian uncertainty perception."""
@kwdef struct UncertaintyPerceptionWeights
    action_history_blend::Float64 = 0.85
    actor_prior_weight::Float64 = 0.45
    practical_prior_weight::Float64 = 0.45
    novelty_prior_weight::Float64 = 0.50
    recursion_prior_weight::Float64 = 0.50
    actor_short_min::Float64 = 0.03
    actor_short_max::Float64 = 0.45
    practical_short_min::Float64 = 0.04
    practical_short_max::Float64 = 0.50
    novelty_short_min::Float64 = 0.03
    novelty_short_max::Float64 = 0.45
    recursion_short_min::Float64 = 0.04
    recursion_short_max::Float64 = 0.50

    opportunity_legibility_complexity_weight::Float64 = 0.35
    opportunity_legibility_novelty_weight::Float64 = 0.25
    opportunity_legibility_combo_uncertainty_weight::Float64 = 0.20
    opportunity_legibility_discovery_threshold_weight::Float64 = 0.20
    opportunity_market_invested_weight::Float64 = 0.45
    opportunity_market_competition_weight::Float64 = 0.35
    opportunity_market_impact_weight::Float64 = 0.20
    opportunity_signal_intrinsic_weight::Float64 = 0.60
    opportunity_signal_market_weight::Float64 = 0.25
    opportunity_signal_lifecycle_weight::Float64 = 0.15
    truly_unknown_signal_multiplier::Float64 = 0.08
    discovery_threshold_signal_discount::Float64 = 0.70

    observable_quality_sector_familiarity_weight::Float64 = 0.28
    observable_quality_component_familiarity_weight::Float64 = 0.20
    observable_quality_signal_strength_weight::Float64 = 0.22
    observable_quality_visible_coverage_weight::Float64 = 0.18
    observable_quality_data_rich_weight::Float64 = 0.12
    observable_breadth_sector_span_weight::Float64 = 0.45
    observable_breadth_visible_coverage_weight::Float64 = 0.35
    observable_breadth_data_rich_weight::Float64 = 0.20
    evidence_quality_weight::Float64 = 0.55
    evidence_breadth_weight::Float64 = 0.45

    ai_trust_trait_weight::Float64 = 0.50
    ai_trust_learned_weight::Float64 = 0.50
    avg_knowledge_personal_weight::Float64 = 0.75
    avg_knowledge_component_weight::Float64 = 0.25
    discovery_momentum_base::Float64 = 0.10
    discovery_momentum_explore_weight::Float64 = 0.30
    discovery_momentum_coverage_weight::Float64 = 0.10
    discovery_shortfall_explore_weight::Float64 = 0.20
    discovery_shortfall_sector_span_weight::Float64 = 0.10
    actor_info_quality_gap_weight::Float64 = 0.20
    actor_info_breadth_gap_weight::Float64 = 0.15
    actor_knowledge_deficit_weight::Float64 = 0.35
    actor_discovery_shortfall_weight::Float64 = 0.15
    actor_signal_gap_weight::Float64 = 0.15
    actor_info_quality_reduction::Float64 = 0.16
    actor_info_breadth_reduction::Float64 = 0.10
    actor_evidence_reliability_reduction::Float64 = 0.08
    analytical_gap_modifier_weight::Float64 = 0.40
    opportunity_signal_gap_discount::Float64 = 0.35
    actor_explore_reduction::Float64 = 0.08
    actor_volatility_reduction::Float64 = 0.05
    actor_maintain_increase::Float64 = 0.05
    actor_personal_knowledge_reduction::Float64 = 0.18
    actor_knowledge_span_reduction::Float64 = 0.08
    actor_visible_coverage_reduction::Float64 = 0.06
    actor_disconfirmation_increase::Float64 = 0.08
    actor_incompetence_increase::Float64 = 0.15
    actor_short_base::Float64 = 0.26
    actor_short_evidence_reduction::Float64 = 0.16
    actor_short_discovery_shortfall_increase::Float64 = 0.10

    practical_regime_info_reduction::Float64 = 0.20
    practical_regime_disconfirmation_increase::Float64 = 0.20
    known_sector_threshold::Float64 = 0.40
    timing_competition_weight::Float64 = 0.40
    timing_adoption_weight::Float64 = 0.30
    path_risk_base::Float64 = 0.18
    path_risk_gap_pressure_weight::Float64 = 0.22
    path_risk_invest_share_weight::Float64 = 0.12
    path_risk_innovate_over_explore_weight::Float64 = 0.10
    agent_uncertainty_incompetence_weight::Float64 = 0.12
    agent_uncertainty_knowledge_reference::Float64 = 0.60
    agent_uncertainty_knowledge_shortfall_weight::Float64 = 0.08
    agent_uncertainty_disconfirmation_weight::Float64 = 0.05
    agent_uncertainty_discovery_shortfall_weight::Float64 = 0.08
    system_market_volatility_weight::Float64 = 0.08
    system_regime_uncertainty_weight::Float64 = 0.07
    system_timing_pressure_weight::Float64 = 0.06
    system_path_complexity_weight::Float64 = 0.05
    system_volatility_weight::Float64 = 0.05
    crowding_pressure_weight::Float64 = 0.06
    opportunity_overlap_weight::Float64 = 0.04
    opportunity_competition_weight::Float64 = 0.03
    ai_herding_pressure_weight::Float64 = 0.05
    practical_agent_volatility_weight::Float64 = 0.15
    practical_short_base::Float64 = 0.22
    practical_short_market_volatility_weight::Float64 = 0.18
    practical_short_agent_volatility_weight::Float64 = 0.10
    practical_short_evidence_reduction::Float64 = 0.12

    novelty_combo_rate_weight::Float64 = 0.25
    novelty_reuse_pressure_drag::Float64 = 0.20
    novelty_new_possibility_rate_weight::Float64 = 0.15
    novelty_derivative_adoption_drag::Float64 = 0.10
    novelty_innovation_trait_weight::Float64 = 0.22
    novelty_exploration_trait_weight::Float64 = 0.15
    novelty_personal_knowledge_weight::Float64 = 0.10
    novelty_short_base::Float64 = 0.20
    novelty_short_volatility_weight::Float64 = 0.15
    novelty_short_scarcity_weight::Float64 = 0.10
    novelty_short_info_breadth_reduction::Float64 = 0.10

    recursion_action_correlation_baseline::Float64 = 0.30
    recursion_local_base::Float64 = 0.12
    recursion_visible_pressure_weight::Float64 = 0.35
    recursion_confidence_miscalibration_weight::Float64 = 0.18
    recursion_disconfirmation_weight::Float64 = 0.12
    recursion_volatility_weight::Float64 = 0.08
    recursion_invest_share_weight::Float64 = 0.08
    recursion_ai_herding_weight::Float64 = 0.15
    recursion_capital_crowding_weight::Float64 = 0.08
    recursion_opportunity_overlap_weight::Float64 = 0.06
    recursion_combo_reuse_weight::Float64 = 0.06
    recursion_action_correlation_excess_weight::Float64 = 0.08
    recursion_knowledge_span_threshold::Float64 = 0.35
    recursion_knowledge_span_excess_weight::Float64 = 0.10
    recursion_exploration_trait_reduction::Float64 = 0.10
    recursion_low_knowledge_reference::Float64 = 0.55
    recursion_low_knowledge_weight::Float64 = 0.08
    strategic_opacity_decay::Float64 = 0.90
    strategic_opacity_disconfirmation_weight::Float64 = 0.30
    strategic_opacity_pressure_weight::Float64 = 0.10
    strategic_opacity_volatility_weight::Float64 = 0.10
    recursion_short_base::Float64 = 0.20
    recursion_short_volatility_weight::Float64 = 0.15
    recursion_short_herding_weight::Float64 = 0.10
    recursion_short_info_quality_reduction::Float64 = 0.10
    total_uncertainty_actor_weight::Float64 = 0.35
    total_uncertainty_practical_weight::Float64 = 0.25
    total_uncertainty_novelty_resolved_weight::Float64 = 0.20
    total_uncertainty_recursion_weight::Float64 = 0.20
end

# ============================================================================
# MAIN CONFIGURATION STRUCT
# ============================================================================

"""
Enhanced configuration for emergent simulation with all AI learning features.

This is a direct port of the Python EmergentConfig dataclass with all 100+
parameters preserved for exact behavioral compatibility.
"""
@kwdef mutable struct EmergentConfig
    # ========================================================================
    # AGENT CONFIGURATION
    # ========================================================================
    # Default "fixed" because fixed-tier assignment is the main-analysis design
    # and gives a clean average treatment effect across AI tiers. Emergent mode
    # (agents choose their own tier) is the robustness-check design and must be
    # opted into explicitly per run.
    AGENT_AI_MODE::String = "fixed"
    N_AGENTS::Int = 1000
    # Initial-capital precedence: the EmergentAgent constructor honors the
    # `initial_capital` kwarg first, then sector_profile.initial_capital_range,
    # then INITIAL_CAPITAL_RANGE below. This singular INITIAL_CAPITAL scalar is
    # consulted only when USE_UNIFORM_INITIAL_CAPITAL is true; otherwise sector
    # profiles take precedence over it. To force uniform starting capital, set
    # that flag (or pass initial_capital per-agent). The field is retained
    # because production scripts reference it.
    INITIAL_CAPITAL::Float64 = 5_000_000.0
    # Explicit flag to force uniform starting capital across all agents. When
    # true, the EmergentAgent constructor uses the config.INITIAL_CAPITAL scalar
    # instead of sampling from sector_profile.initial_capital_range. An explicit
    # flag (rather than a sentinel value of INITIAL_CAPITAL) lets scripts request
    # uniform capital even when the scalar equals the default.
    USE_UNIFORM_INITIAL_CAPITAL::Bool = false
    INITIAL_CAPITAL_RANGE::Tuple{Float64,Float64} = (2_500_000.0, 10_000_000.0)
    SURVIVAL_THRESHOLD::Float64 = 2_000_000.0
    # Explicit flag. When true, check_survival! uses the scalar SURVIVAL_THRESHOLD
    # instead of sector_profile.survival_threshold.
    USE_UNIFORM_SURVIVAL_THRESHOLD::Bool = false
    SURVIVAL_CAPITAL_RATIO::Float64 = 0.40  # equity_failure floor
    # Survival counts in-flight (not-yet-matured) investment capital as a solvency
    # asset at book value (net worth = cash + deployed), rather than gating purely
    # on cash-on-hand. Default true: deployed-but-maturing capital is an asset, not
    # a loss — a firm should not be declared insolvent merely for the *timing* of
    # capital deployment. A bad investment still triggers failure once it matures
    # and the loss is realized. Set false for the cash-only solvency rule, which
    # gates purely on cash-on-hand (the CASH_ONLY_SURVIVAL robustness condition).
    SURVIVAL_COUNTS_INFLIGHT::Bool = true
    INSOLVENCY_GRACE_ROUNDS::Int = 6  # months of insolvency tolerated before failure
    RANDOM_SEED::Int = 42
    USE_NUMPY_RNG::Bool = false  # Use the deterministic MT19937 (NumpyRNG) stream instead of native MersenneTwister

    # Monthly operational cost — calibrated for ~55% 5-yr survival (BLS benchmark)
    # and set to reflect 2026-era compute / SaaS / cloud infrastructure costs that
    # all entrepreneurial ventures face. AI subscription costs ($3500/month for
    # the frontier tier) stack on top, representing dedicated AI tooling above
    # this baseline compute environment.
    BASE_OPERATIONAL_COST::Float64 = 22500.0
    MAX_AGENT_KNOWLEDGE::Int = 90
    SECTOR_STRENGTH_PRUNE_THRESHOLD::Float64 = 0.1
    MAX_INVESTMENT_FRACTION::Float64 = 0.037  # Monthly cap on capital deployed per investment
    TARGET_INVESTMENT_FRACTION::Float64 = 0.033  # Monthly target deployment fraction
    # High-conviction bet sizing for power-law right-tail outcomes. When
    # confidence × signal_score exceeds HIGH_CONVICTION_THRESHOLD, the bet can
    # scale up to MAX_HIGH_CONVICTION_FRACTION of capital (vs the standard
    # MAX_INVESTMENT_FRACTION). Real founders deploy 10-30% of capital on
    # conviction plays, and the standard 3.7% cap suppresses the right tail that
    # produces venture-scale outcomes. Calibrated to 0.06 / 1.5, which yields a
    # meaningful right tail (~5% of survivors >1.0×, max growth ~3-4×) without
    # materially regressing survival; pushing further requires structural changes
    # such as exit events or longer compounding horizons.
    MAX_HIGH_CONVICTION_FRACTION::Float64 = 0.06
    HIGH_CONVICTION_THRESHOLD::Float64 = 1.5

    # Robustness test parameters for scaling AI failure modes and costs
    HALLUCINATION_INTENSITY::Float64 = 1.0  # Scale AI hallucination rates (0.0=none, 1.0=baseline, 2.0=double)
    OVERCONFIDENCE_INTENSITY::Float64 = 1.0  # Scale AI overconfidence effects (0.0=none, 1.0=baseline, 2.0=double)
    # Scale the systematic bias term applied to AI estimates (information.jl:
    # estimated_return += bias·0.3, estimated_uncertainty -= bias·0.2, where
    # bias comes from AI_DOMAIN_CAPABILITIES). 0.0 = bias-free (the NO_AI_BIAS
    # robustness-addendum cell, completing the pathology-removal family alongside
    # HALLUCINATION_INTENSITY/OVERCONFIDENCE_INTENSITY); 1.0 = baseline.
    AI_BIAS_INTENSITY::Float64 = 1.0
    AI_COST_INTENSITY::Float64 = 1.0        # Scale AI subscription/usage costs (0.0=free, 1.0=baseline, 2.0=double)
    # Cost backend:
    # - "fixed": legacy per-use/subscription costs in AI_LEVELS.
    # - "hybrid": subscription seat fees plus task-specific token-equivalent
    #   usage costs. Token counts are model abstractions for compute/search
    #   effort, not literal API-token telemetry from an external provider.
    # - "token": task-specific usage costs only; subscription schedules are
    #   skipped even for tiers whose AI_LEVELS cost_type is "subscription".
    #
    # DEFAULT = "token" (token-core migration, 2026-06-11; design
    # ruling). Intelligence is priced at the margin: every analysis is billed
    # by task-specific token budgets x per-million prices, scaled by problem
    # complexity, novelty, uncertainty, and candidate count (the multipliers
    # below) - cost scales with the scale and scope of usage, with no seat
    # fees. Rationale: (a) marginal-compute pricing is the architecture the
    # usage-based-pricing literature indicates as the realistic world, and it is now the
    # baseline rather than an ablation; (b) usage-priced costs are endogenous
    # to behavior rather than contractual; (c) per-analysis effective costs
    # are auditable abstractions of published API-billed agentic-task costs.
    # Effective per-market-analysis costs at defaults (complexity 0.5):
    # basic $0.96 / advanced $4.35 / premium $18.89 (ratio ~1:4.5:20); a
    # frontier founder evaluating ~106 visible opportunities/month bills
    # ~$2,000/month - heavy-agentic-usage scale. Calibration check
    # (2026-06-11, from a robustness suite run): under token
    # pricing, population 5-yr survival = 78.0% (none 80.9) - inside the
    # venture-class anchor band (P&Z 65-80%), so the dual-anchor calibration
    # carries over without retuning. The subscription-era architecture is
    # retained as the SUBSCRIPTION_ERA robustness cell (AI_COST_MODEL=
    # "hybrid": $400/$3,500 seat fees + metered usage), and pricing remains
    # the most heavily swept assumption (FREE, 2X, difficulty-scaled,
    # wealth-scaled).
    AI_COST_MODEL::String = "token"
    AI_TOKEN_INPUT_PRICE_PER_MILLION::Dict{String,Float64} = Dict(
        "none" => 0.0,
        "basic" => 120.0,
        "advanced" => 250.0,
        "premium" => 700.0,
    )
    AI_TOKEN_OUTPUT_PRICE_PER_MILLION::Dict{String,Float64} = Dict(
        "none" => 0.0,
        "basic" => 240.0,
        "advanced" => 550.0,
        "premium" => 1500.0,
    )
    AI_TOKEN_REASONING_PRICE_PER_MILLION::Dict{String,Float64} = Dict(
        "none" => 0.0,
        "basic" => 60.0,
        "advanced" => 400.0,
        "premium" => 1200.0,
    )
    AI_TOKEN_DEPTH_MULTIPLIER::Dict{String,Float64} = Dict(
        "none" => 0.0,
        "basic" => 0.70,
        "advanced" => 1.15,
        "premium" => 1.80,
    )
    AI_TOKEN_ACTION_BASE_INPUT::Dict{String,Float64} = Dict(
        "market_analysis" => 5_000.0,
        "innovation_potential" => 8_000.0,
        "exploration" => 4_000.0,
        "tier_review" => 2_500.0,
    )
    AI_TOKEN_ACTION_BASE_OUTPUT::Dict{String,Float64} = Dict(
        "market_analysis" => 1_200.0,
        "innovation_potential" => 1_800.0,
        "exploration" => 1_000.0,
        "tier_review" => 600.0,
    )
    AI_TOKEN_ACTION_REASONING_FACTOR::Dict{String,Float64} = Dict(
        "market_analysis" => 0.60,
        "innovation_potential" => 0.90,
        "exploration" => 0.50,
        "tier_review" => 0.30,
    )
    AI_TOKEN_COMPLEXITY_MULTIPLIER::Float64 = 0.35
    AI_TOKEN_NOVELTY_MULTIPLIER::Float64 = 0.25
    AI_TOKEN_UNCERTAINTY_MULTIPLIER::Float64 = 0.20
    AI_TOKEN_CANDIDATE_MULTIPLIER::Float64 = 0.08
    AI_TOKEN_OUTPUT_COMPLEXITY_MULTIPLIER::Float64 = 0.15
    OPS_COST_INTENSITY::Float64 = 1.0       # Scale agent.operating_cost_estimate (0.0=free, 1.0=baseline, 2.0=double); refutation knob

    # ========================================================================
    # NETWORK CONFIGURATION
    # ========================================================================
    # Gates the per-round neighbor-sampling blocks in simulation.jl. When false,
    # agents receive an empty neighbor list (no peer signals visible).
    USE_NETWORK_EFFECTS::Bool = true
    # Number of randomly sampled peers visible per decision (simulation.jl).
    NETWORK_N_NEIGHBORS::Int = 5

    # ========================================================================
    # AGENT TRAIT DISTRIBUTIONS
    # ========================================================================
    TRAIT_DISTRIBUTIONS::Dict{String,TraitDistribution} = Dict(
        "uncertainty_tolerance" => TraitDistribution(dist="beta", params=Dict("a" => 1.05, "b" => 0.65)),
        # Innovativeness respecified (robustness check of
        # Table 1): the prior clamp(LogNormal(0.5, 0.5), 0, 1) was degenerate —
        # P(X ≥ 1) = Φ(1) ≈ 0.841, so ~84% of agents sat at exactly 1.0 and the
        # "heterogeneous innovativeness" the paper describes was a point mass
        # with a thin left tail. Beta(6, 2) gives a continuous right-leaning
        # distribution (mean 0.75, mode ≈ 0.83, sd ≈ 0.14): entrepreneurs are
        # selected for innovativeness, but the population genuinely varies.
        # Shape parameters are a structural choice pending a psychometric
        # anchor (EO/innovativeness scale data) in the calibration pass — do
        # not retune against outcome levels or tier patterns.
        "innovativeness" => TraitDistribution(dist="beta", params=Dict("a" => 6.0, "b" => 2.0)),
        "competence" => TraitDistribution(dist="uniform", params=Dict("low" => 0.1, "high" => 0.8)),
        "ai_trust" => TraitDistribution(dist="normal_clipped", params=Dict("mean" => 0.5, "std" => 0.38)),
        "trait_momentum" => TraitDistribution(dist="uniform", params=Dict("low" => 0.6, "high" => 0.9)),
        "cognitive_style" => TraitDistribution(dist="uniform", params=Dict("low" => 0.8, "high" => 1.2)),
        "analytical_ability" => TraitDistribution(dist="uniform", params=Dict("low" => 0.1, "high" => 0.9)),
        "exploration_tendency" => TraitDistribution(dist="beta", params=Dict("a" => 0.85, "b" => 0.85)),
        "market_awareness" => TraitDistribution(dist="uniform", params=Dict("low" => 0.1, "high" => 0.9)),
    )

    # ========================================================================
    # MARKET CONFIGURATION
    # ========================================================================
    N_ROUNDS::Int = 120  # 120 months = 10 years (monthly cadence)
    N_RUNS::Int = 50
    BASE_OPPORTUNITIES::Int = 5
    DISCOVERY_PROBABILITY::Float64 = 0.20  # Monthly probability
    INNOVATION_PROBABILITY::Float64 = 0.14  # Monthly probability
    AI_HERDING_DECAY::Float64 = 0.5  # Per-round retention of herding-pattern stocks (~1-month half-life)
    AI_SIGNAL_HISTORY::Int = 420  # 420 months of signal history

    # ========================================================================
    # INNOVATION ECONOMICS
    # ========================================================================
    INNOVATION_BASE_SPEND_RATIO::Float64 = 0.025
    INNOVATION_MAX_SPEND::Float64 = 2667.0  # Monthly max R&D spend
    INNOVATION_FAIL_RECOVERY_RATIO::Float64 = 0.12
    INNOVATION_SUCCESS_BASE_RETURN::Float64 = 0.08  # Monthly return
    INNOVATION_SUCCESS_RETURN_MULTIPLIER::Tuple{Float64,Float64} = (1.8, 3.2)
    INNOVATION_REUSE_PROBABILITY::Float64 = 0.07  # Monthly probability
    INNOVATION_REUSE_LOOKBACK::Int = 300  # 300 months lookback

    # ========================================================================
    # REFUTATION TEST PARAMETERS
    # These allow testing model robustness by modifying AI tier advantages
    # ========================================================================

    # Execution success multipliers by AI tier
    # Default: all 1.0 (no AI advantage in execution)
    AI_EXECUTION_SUCCESS_MULTIPLIERS::Dict{String,Float64} = Dict(
        "none" => 1.0,
        "basic" => 1.0,
        "advanced" => 1.0,
        "premium" => 1.0
    )

    # Innovation quality boost by AI tier (additive, on 0-1 scale).
    # Default: all 0.0; favorable-AI/refutation cells opt in explicitly.
    AI_QUALITY_BOOST::Dict{String,Float64} = Dict(
        "none" => 0.0,
        "basic" => 0.0,
        "advanced" => 0.0,
        "premium" => 0.0
    )

    # Information-quality boost by AI tier (additive, on 0-1 scale).
    # This is separate from AI_QUALITY_BOOST, which affects innovation quality.
    # Robustness scripts can set both when testing a broad "quality" channel.
    AI_INFORMATION_QUALITY_BOOSTS::Dict{String,Float64} = Dict(
        "none" => 0.0,
        "basic" => 0.0,
        "advanced" => 0.0,
        "premium" => 0.0
    )

    # Optional strategic anticipation: AI-assisted agents can discount crowded
    # opportunities when competitive recursion suggests other agents see the
    # same signal. Default off to preserve baseline calibration.
    STRATEGIC_ANTICIPATION_ENABLED::Bool = false
    STRATEGIC_ANTICIPATION_STRENGTH::Float64 = 0.0

    # Optional strategic diversification: AI-assisted agents can deliberately
    # choose among near-top opportunities instead of deterministically taking
    # the highest-scored opportunity when shared-signal/crowding pressure is
    # high. Default off to preserve baseline calibration.
    STRATEGIC_DIVERSIFICATION_ENABLED::Bool = false
    STRATEGIC_DIVERSIFICATION_STRENGTH::Float64 = 0.0
    STRATEGIC_DIVERSIFICATION_TOP_K::Int = 5
    STRATEGIC_DIVERSIFICATION_SCORE_BAND::Float64 = 0.10

    # ── AGI strategy ladder (the strategy-ladder design notes) ────────
    # Decision-strategy rung for the agents the mode applies to:
    #   "reactive"              L0 — current model. The DEFAULT and required to
    #                           be bit-identical to the pre-ladder model: every
    #                           hook short-circuits before any strategy
    #                           computation (pinned by test_strategy_ladder.jl).
    #   "consensus_discounting" S1 — anticipatory congestion shade on expected
    #                           returns of consensus-legible opportunities
    #                           (information cascades).
    #   "comparative_advantage" S2 — re-rank by private edge (sector
    #                           familiarity, knowledge overlap, execution
    #                           traits; Foss & Klein judgment).
    #   "complement_seeking"    S3 — reallocate explore/innovate effort toward
    #                           opportunities where the agent's OWN instrument
    #                           is unsure (Hayek tacit knowledge; effectuation).
    #   "agi_native"            S4 — S1+S2+S3 jointly (the strongest
    #                           ecologically defensible operationalization).
    # Unknown modes ERROR loudly at initialize! and at first strategy_active
    # call — they never fall through to reactive (loud-failure convention,
    # the design notes).
    STRATEGY_MODE::String = "reactive"
    # Tiers the mode applies to; frontier-only suite conditions set
    # ["premium"]. Agents whose decision-time behavioral tier is outside this
    # list behave reactively (bit-identical to default; pinned by test).
    STRATEGY_TIERS::Vector{String} = ["none", "basic", "advanced", "premium"]
    # S1 strength: weight on the anticipatory congestion forecast. The shaded
    # expected EXCESS margin is margin × (1 − discount × forecast), with
    # forecast ∈ [0,1]; 1.0 = full forecast weight (conservative because the
    # forecast itself is a triple product of [0,1] observables).
    STRATEGY_CONSENSUS_DISCOUNT::Float64 = 1.0
    # S2 strength: ranking multiplier 1 + weight × (edge − 0.5), edge ∈ [0,1],
    # so a population-neutral edge of 0.5 leaves scores unchanged.
    STRATEGY_EDGE_WEIGHT::Float64 = 1.0
    # S3 strength: scales (a) the additive explore/innovate utility shift
    # (≤ 0.25 pre-sigmoid at strength 1.0) and (b) the softening of the
    # low-confidence score penalty inside invest ranking.
    STRATEGY_COMPLEMENT_SHIFT::Float64 = 1.0

    # ── Open-action extension (the strategy-ladder design notes, Open-
    #    action extension) ─────────────────────────────────────────────────
    # A1 abandonment option (real options; lean pivots). When true, an agent
    # reviews its in-flight investments once per round (make_decision!, after
    # the invest-utility information pass) and may liquidate a holding at an
    # age-dependent haircut, redeploying the recovered capital. Default false
    # is bit-identical to the closed-action model (pinned by
    # test_open_action_space.jl: the review short-circuits before any
    # computation).
    ENABLE_PIVOT::Bool = false
    # Recovered fraction of committed capital rises LINEARLY with maturity
    # progress (investment age / time_to_maturity) from floor → ceiling:
    # real-options abandonment value rises as resolution approaches, while
    # early kills recover less (sunk setup costs dominate). Simple,
    # inspectable form; validated 0 ≤ floor ≤ ceiling ≤ 1 at initialize! and
    # first use.
    PIVOT_HAIRCUT_FLOOR::Float64 = 0.40
    PIVOT_HAIRCUT_CEILING::Float64 = 0.75
    # Redeployment margin in the pivot rule: liquidation must beat holding by
    # this margin (PIVOT ⇔ expected_residual < recoverable × (1 + margin))
    # because holding has option value the simple residual-value comparison
    # does not capture. Promoted from a module constant to a documented config
    # field (engine invariants F3a, 2026-06-09).
    PIVOT_REDEPLOY_MARGIN::Float64 = 0.05
    # Conviction base in the pivot trigger's deterioration modulation:
    # d_eff = clamp(deterioration × (BASE − decision_confidence) × GAIN, 0, 1).
    # A low-confidence committer (conf→0, factor→BASE) folds quickly; a
    # high-conviction one (conf→1, factor→BASE−1) holds through the same
    # signal. Default 1.5 preserves the original hardcoded factor.
    PIVOT_CONVICTION_BASE::Float64 = 1.5
    # Deterioration gain in the pivot trigger (engine invariants F3a, 2026-06-09).
    # Default 1.0 is bit-identical to the pre-parameter trigger. Rationale for
    # the dial: the verified censoring analysis found the maximum observed
    # perceived-crowding deterioration ≈0.28 at BASELINE, while early/mid-life
    # pivots require d_eff ≈0.45–0.79 — so at gain 1.0 the trigger region is
    # mostly unreachable and the pivot arm risks being dead. A gain of ≈2.5–3
    # maps the observed deterioration range onto the trigger region at
    # mid-progress. The OPEN_ACTION_* suite cells should set the gain per the
    # liveness probe (scripts/probe_pivot_liveness.jl — MANDATORY before any
    # production launch). Probe result 2026-06-09 (64 agents × 40 rounds × 3 seeds):
    #   gain 1.0 ⇒ 0 pivots in BOTH cells (dead arm; censoring confirmed);
    #   gain 2.0 ⇒ 3.4% of invests at BASELINE, 4.7% under dense/low-capacity
    #              (HIGH-crowding-style) markets;
    #   gain 3.0 ⇒ 9.1% BASELINE (overshoots the ≈1–3% target), 7.7% HIGH.
    # ⇒ RECOMMENDED GAIN FOR OPEN_ACTION_* CELLS: 2.0 (set in the suite
    # condition, NOT here — the default stays 1.0 / bit-identical).
    PIVOT_DETERIORATION_GAIN::Float64 = 1.0
    # A2 Hayekian redirection. When true, innovating agents bias the SECTOR of
    # new combinations away from sectors they perceive as crowded and toward
    # sparse ones, using only their own visible-opportunity sector mix
    # (weighted by observable competition traces). The DIRECTION of creation
    # changes; the RATE of innovation attempts does not (the elasticity probe
    # showed volume was never the problem — crowding depletes creators rather
    # than redirecting creation). Default false leaves sector selection
    # exactly on the deterministic knowledge-domain mapping (no extra RNG
    # draws — bit-identical, pinned by test).
    ENABLE_DIRECTED_CREATION::Bool = false
    # Redirection strength dial (G3 mixture form, 2026-06-09): with
    # probability exp(−strength) the deterministic knowledge-mapped sector is
    # kept; otherwise the sector is drawn from density-tilted weights
    # w_s ∝ prior_s × exp(−strength × density_z_s), where density_z is the
    # z-score of the agent's perceived sector density across config.SECTORS
    # and prior anchors the knowledge-mapped sector (see
    # directed_sector_weights, open_action.jl; mixture branch in
    # determine_innovation_sector, innovation.jl). strength = 0 ⇒ the anchor
    # is kept with probability 1 — the dial NESTS the flag-off behavior
    # exactly (distributionally) and interpolates cleanly from there.
    DIRECTED_CREATION_STRENGTH::Float64 = 1.0

    # ── Emergence audit extension (the strategy-ladder design notes,
    #    emergence-audit extension) ──────────────────────────────────────
    # Signal-commonality instrument: cross-agent correlation rho of the CONTINUOUS AI
    # return-estimate error within a (opportunity, round, tier) cell.
    #
    #   eps = sqrt(rho) * eps_common(opp, round, tier) + sqrt(1 - rho) * eps_idio
    #
    # Variance algebra (the entire point of the dial): eps_common and eps_idio
    # are independent standard normals, so
    #   Var(eps) = rho * Var(eps_common) + (1 - rho) * Var(eps_idio)
    #            = rho + (1 - rho) = 1
    # for EVERY rho in [0, 1] — total error variance, and therefore estimate
    # ACCURACY, is preserved by construction. The dial moves signal
    # COMMONALITY at fixed signal QUALITY, decoupling the two channels that
    # are mechanically coupled in the default fully-idiosyncratic draw.
    #
    #   rho = 0.0 (DEFAULT): current fully-idiosyncratic errors. Bit-identical
    #         to the pre-dial model (the blend short-circuits before any
    #         common draw is consumed; agent-rng draw order unchanged). This
    #         is the MOST decorrelated, trap-MINIMIZING error structure —
    #         the conservative reading of all default-rho results.
    #   rho = 1.0: fully shared per-(opportunity, round, tier) errors — the
    #         shared-foundation-model world in which every same-tier agent's
    #         instrument has the same blind spot on the same opportunity.
    #
    # SCOPE (documented, conservative): only the continuous return-estimate
    # error component is correlated in this pass. The hallucination FLAG
    # draws, the lognormal accuracy noise, the uncertainty-estimate noise,
    # and the overconfidence chain stay fully idiosyncratic — and the "none"
    # tier (no AI instrument) is exempt entirely. Correlating those channels
    # too would only deepen convergence, so default-rho results remain a
    # lower bound; noted as a future extension.
    #
    # The common draw eps_common(opp, round, tier) is cached on the
    # InformationSystem and drawn from a DEDICATED RNG stream (derived from
    # the simulation seed at construction; see InformationSystem docs), so
    # enabling rho never desynchronizes market/agent draws. Validated in
    # [0, 1] at initialize! and at first use (loud-failure convention).
    AI_ERROR_CORRELATION::Float64 = 0.0
    # Decision-temperature instrument: decision temperature T, exposed at the single
    # action-selection chokepoint (make_decision! softmax). T DIVIDES the
    # action utilities pre-softmax — equivalently, it multiplies the existing
    # calibrated ACTION_SELECTION_TEMPERATURE:
    #   P(a) ∝ exp(u_a / (ACTION_SELECTION_TEMPERATURE * T))
    # T > 1 flattens choice toward uniform (noisier humans); T < 1 sharpens
    # toward argmax (deterministic optimizers). T = 1.0 (DEFAULT) is
    # bit-identical: the multiply is short-circuited entirely so the default
    # path performs no extra computation and consumes no extra RNG.
    # Operationalizes the mixed-equilibrium conjecture (noise in
    # human decisions approximates the mixed-strategy congestion equilibrium).
    # Validated > 0 at initialize! and at first use.
    DECISION_TEMPERATURE::Float64 = 1.0

    # Placebo switch. When true, non-none AI labels remain available for
    # grouping/ATE placebo checks, but runtime behavior treats them as no-AI.
    # This is stricter than merely copying human-level information parameters
    # because it also suppresses AI-use learning and telemetry.
    SHAM_AI_LABELS_USE_NONE::Bool = false

    # Optional marginal compute surcharge for the DIFFICULTY_SCALED_AI_COST
    # robustness cell. In fixed mode this preserves the legacy per-use
    # complexity surcharge; in hybrid/token modes it adds to the token
    # complexity multiplier. Default 0.0 preserves baseline pricing.
    AI_DIFFICULTY_COST_SCALING::Float64 = 0.0

    # Optional agent-specific complementarity between AI outputs and the
    # entrepreneur's traits/sector knowledge. Default 0.0 preserves baseline.
    AI_COMPLEMENTARITY_STRENGTH::Float64 = 0.0

    # Optional wealth-scaled analysis breadth for the WEALTH_SCALED_COMPUTE
    # robustness cell (the rich-get-richer dynamic: resource-rich founders
    # buy more searches). When > 0, the stochastic visibility budget in
    # get_perceived_opportunities is multiplied by
    # clamp(capital/initial_equity, 0.25, 4.0)^scaling, uniformly across tiers
    # (no tier-keyed advantage; exact-equality discipline preserved). Under the
    # hybrid/token billing backends the extra visibility is billed per analysis,
    # so the compute is genuinely bought at the margin. Default 0.0 skips the
    # computation entirely (bit-identical baseline).
    WEALTH_COMPUTE_SCALING::Float64 = 0.0

    # Excess ROI (above break-even 1.0×) required for a matured investment to
    # count as a success: success = ret_multiple >= 1.0 + max(0, threshold)
    # (agents.jl). Default 0.0 preserves the previously hardcoded >= 1.0 rule.
    INVESTMENT_SUCCESS_ROI_THRESHOLD::Float64 = 0.0
    # Supply/demand return adjustments in market.jl get_demand_adjustments.
    # Defaults match the previously hardcoded 0.70/0.45 multipliers that
    # produced all validated results (pre-cleanup struct defaults 0.35/0.37
    # were never live).
    RETURN_OVERSUPPLY_PENALTY::Float64 = 0.70
    RETURN_UNDERSUPPLY_BONUS::Float64 = 0.45
    # --- Opportunity-value tail controls (added 2026-06-14; defaults = byte-identical) ---
    # The opportunity latent return is lognormal: latent = exp(log_mu + log_sigma·z),
    # truncated to the sector return_range[2] (market.jl:159). TWO things suppress the
    # right tail at baseline: (a) the per-sector ceiling (~4x for tech) and (b) the raw
    # sector log_sigma (~0.32-0.45) — both small. The TAIL-HEAVINESS driver is therefore
    # LOG_SIGMA_MULT (scales sigma); a *cap* alone is inert because raw sigma < 1.0.
    # For a venture-realistic / true-unicorn tail (option B, 2026-06-14): raise
    # LOG_SIGMA_MULT (sigma) AND RETURN_RANGE_MAX_MULT + RETURN_CLAMP_MAX (ceilings, so
    # the heavy draws are not re-clamped). log_mu (the bulk/mean) stays clamped to the
    # original sector range (market.jl:667), so the BULK stays modest while the TAIL grows.
    # Recalibrate survival to the venture-class band (Puri & Zarutskie 65-80%) afterward.
    RETURN_RANGE_MAX_MULT::Float64 = 1.0   # scales per-sector return_range[2] draw ceiling (market.jl:159)
    RETURN_CLAMP_MAX::Float64 = 25.0       # global latent_return ceiling (market.jl:201,1491,1541,1616)
    LOG_SIGMA_MULT::Float64 = 1.0          # scales sector log_sigma (market.jl:109) — THE tail-heaviness driver
    LOG_SIGMA_CAP::Float64 = 1.0           # ceiling on log_sigma after the mult (market.jl:109,155)
    # When true, both an investor's REALIZED return and the AI/agent's perceived
    # ESTIMATE track the opportunity's full latent heavy-tail value (bounded only by
    # RETURN_CLAMP_MAX) instead of the legacy thin-tail caps (~10x base / ~200x realized
    # ceiling in realized_return, ~25x estimate ceiling via OPPORTUNITY_RETURN_RANGE);
    # the convex crowding penalty still dilutes crowded opportunities (the trap). Off
    # preserves the legacy engine and leaves canonical results identical. (2026-06-15 fix.)
    HEAVY_TAIL_RETURNS::Bool = false
    # Idiosyncratic execution/market noise on realized returns (models.jl):
    # scaled_return *= exp(σ·z − σ²/2), applied PER INVESTMENT REALIZATION on
    # top of the opportunity-level Pareto draw. The Pareto draw is common
    # opportunity quality; this term is agent-specific execution risk — the
    # only channel that decorrelates co-investors' outcomes in the same
    # opportunity, and the operationalization of practical indeterminism at
    # the execution stage (uncertainty that no foresight quality resolves).
    #
    # DESIGN PROPERTY (stated, not accidental): the multiplier is
    # mean-preserving in levels but has median exp(−σ²/2) < 1, and success is
    # classified at ret_multiple ≥ 1.0 + threshold — so marginal winners flip
    # to losers more often than marginal losers are rescued (P(X<1) = Φ(σ/2)
    # ≈ 0.575 at σ=0.38). This is the honest behavior of multiplicative noise
    # under a binary success threshold and applies identically to every tier.
    #
    # CALIBRATION STATUS (2026-06-09): σ=0.38 is a WORKING VALUE pending the
    # venture-dispersion benchmark pass — the target is total realized-multiple
    # dispersion (Pareto structural + this idiosyncratic term) matched against
    # observed venture return dispersion (Hall & Woodward 2010; Kerr, Nanda &
    # Rhodes-Kropf 2014-class data), with σ backed out as the residual. Until
    # that pass lands, robustness is carried by the robustness-suite rows
    # NO_RETURN_NOISE (σ=0) and RETURN_NOISE_LOW/HIGH (σ=0.19/0.57), which
    # bracket the default. Do not retune σ against survival levels or tier
    # patterns (calibration principle: observable benchmarks only).
    RETURN_NOISE_SCALE::Float64 = 0.38
    BRANCH_LOG_MEAN_DRIFT::Float64 = 0.11
    BRANCH_LOG_SIGMA_DRIFT::Float64 = 0.07
    BRANCH_FAILURE_DRIFT::Float64 = 0.04
    BRANCH_FEEDBACK_RATE::Float64 = 0.05

    # ========================================================================
    # OPPORTUNITY CHARACTERISTICS
    # ========================================================================
    OPPORTUNITY_RETURN_RANGE::Tuple{Float64,Float64} = (1.1, 25.0)
    # Floor must sit below the calibrated sector failure minima (service 0.033)
    # or uncertainty estimates censor to a constant and carry no cross-
    # opportunity risk information for high-accuracy tiers.
    OPPORTUNITY_UNCERTAINTY_RANGE::Tuple{Float64,Float64} = (0.02, 0.60)

    # ========================================================================
    # AI TOOL CONFIGURATION
    # ========================================================================
    # AI capability tiers. These are abstract capability/cost levels, not
    # predictions about a specific future system or a claim that frontier AI tier is full AGI.
    #
    # Tier definitions:
    # - none: Human decision-making only (no AI augmentation)
    # - basic: consumer-level AI tools / low-cost per-use AI
    # - advanced: high-capability AI systems with subscription pricing
    # - premium: canonical internal key for the frontier AI capability tier,
    #   with high marginal compute cost
    AI_LEVELS::Dict{String,AILevelConfig} = Dict(
        "none" => AILevelConfig(0.0, "none", 0.25, 0.20, 0.0),           # Human baseline
        "basic" => AILevelConfig(30.0, "per_use", 0.43, 0.38, 3.0),      # Consumer capability tier
        "advanced" => AILevelConfig(400.0, "subscription", 0.70, 0.65, 35.0),  # Advanced capability tier
        "premium" => AILevelConfig(3500.0, "subscription", 0.97, 0.92, 150.0), # Frontier AI capability tier
    )

    AI_DOMAIN_CAPABILITIES::Dict{String,Dict{String,AIDomainCapability}} = Dict(
        "none" => Dict(
            "market_analysis" => AIDomainCapability(0.38, 0.28, 0.06),
            "technical_assessment" => AIDomainCapability(0.40, 0.26, -0.04),
            "uncertainty_evaluation" => AIDomainCapability(0.35, 0.30, -0.05),
            "innovation_potential" => AIDomainCapability(0.33, 0.29, 0.07),
        ),
        "basic" => Dict(
            "market_analysis" => AIDomainCapability(0.52, 0.20, 0.04),
            "technical_assessment" => AIDomainCapability(0.54, 0.19, -0.03),
            "uncertainty_evaluation" => AIDomainCapability(0.50, 0.21, -0.04),
            "innovation_potential" => AIDomainCapability(0.48, 0.22, 0.05),
        ),
        "advanced" => Dict(
            "market_analysis" => AIDomainCapability(0.78, 0.10, 0.025),
            "technical_assessment" => AIDomainCapability(0.80, 0.09, -0.015),
            "uncertainty_evaluation" => AIDomainCapability(0.76, 0.11, -0.02),
            "innovation_potential" => AIDomainCapability(0.74, 0.12, 0.02),
        ),
        "premium" => Dict(
            "market_analysis" => AIDomainCapability(0.96, 0.015, 0.003),
            "technical_assessment" => AIDomainCapability(0.97, 0.012, -0.002),
            "uncertainty_evaluation" => AIDomainCapability(0.95, 0.018, -0.003),
            "innovation_potential" => AIDomainCapability(0.94, 0.02, 0.004),
        ),
    )

    # ========================================================================
    # LEARNING PARAMETERS
    # ========================================================================
    INITIAL_RESPONSE_VARIANCE::Float64 = 0.3
    LEARNING_RATE::Float64 = 0.05  # Learning rate for uncertainty response adaptation
    MIN_FUNDING_FRACTION::Float64 = 0.25

    # ========================================================================
    # UNCERTAINTY AND MARKET DYNAMICS
    # ========================================================================
    BLACK_SWAN_PROBABILITY::Float64 = 0.017  # Monthly probability
    MARKET_VOLATILITY::Float64 = 0.15  # Monthly volatility
    COMPETITION_SCALE_FACTOR::Float64 = 1.0  # Multiplier for sector-specific competition_intensity (0.0=no competition, 1.0=baseline, 2.0=double)
    DISABLE_COMPETITION_DYNAMICS::Bool = false  # Set to true to freeze competition at 0.0 (for counterfactual analysis)

    # ========================================================================
    # CAPACITY-CONVEXITY CROWDING MODEL
    # ========================================================================
    # Unified crowding penalty using a capacity + convexity approach:
    #   penalty = λ · max(0, C/K - 1)^γ
    #   net_return = base_return · exp(-penalty)
    #
    # A theoretically grounded model where:
    #   - No penalty until crowding exceeds capacity (K)
    #   - Penalty increases convexly beyond capacity (γ controls sharpness)
    #   - Exponential decay keeps returns positive
    # ========================================================================
    USE_CAPACITY_CONVEXITY_CROWDING::Bool = true  # Enable the capacity-convexity crowding model

    # γ = Convexity exponent: how sharply penalties increase beyond capacity
    # γ = 1: linear (gentle)
    # γ = 1.5: mildly convex (gradual ramp — realistic for competitive markets)
    # γ = 2: quadratic (convex - "a little crowded is OK, very crowded is brutal")
    # γ = 3: cubic (very sharp)
    # Calibrated: γ=1.5 produces gradual penalty from C=5 to C=15, avoiding
    # the cliff-like profile of γ=2 where C<K is fine but C>K is catastrophic.
    CROWDING_CONVEXITY_GAMMA::Float64 = 1.5

    # λ = Strength: maps crowding into payoff reduction
    # At 2× capacity (C/K = 2), penalty = λ · 1^γ = λ
    # exp(-λ) gives the return multiplier at 2× capacity:
    #   λ = 0.50 → ~39% drop at 2× capacity (exp(-0.50) ≈ 0.61)
    #   λ = 1.00 → ~63% drop at 2× capacity (exp(-1.00) ≈ 0.37)
    #   λ = 1.50 → ~78% drop at 2× capacity (exp(-1.50) ≈ 0.22)
    # The configured λ = 1.5 (with K_sat = 1.5 and γ = 1.5 below) is calibrated
    # to the aggregate survival target at observed competition levels.
    CROWDING_STRENGTH_LAMBDA::Float64 = 1.5

    # Capital-saturation convexity threshold. K_sat is the saturation ratio
    # (total_invested / capacity) above which the convexity penalty starts. It
    # is a capital ratio and does NOT scale with N: capacity and invested capital
    # scale together, so the ratio is population-invariant. This is the threshold
    # read by the active convex-crowding model.
    #   K_sat = 1.0 → penalty starts at full capacity
    #   K_sat = 1.5 → penalty starts 50% above capacity (meaningfully oversubscribed)
    # Calibrated to the ~50% aggregate survival target while preserving the
    # advanced > basic tier ordering.
    CROWDING_CAPACITY_RATIO_K::Float64 = 1.5

    # Parameters for the alternative crowding model (used when
    # USE_CAPACITY_CONVEXITY_CROWDING = false)
    OPPORTUNITY_COMPETITION_PENALTY::Float64 = 0.5  # Return penalty per unit of opp.competition (0.5 = 50% max penalty at competition=1.0)
    OPPORTUNITY_COMPETITION_THRESHOLD::Float64 = 0.2  # Competition level above which penalty starts applying
    OPPORTUNITY_COMPETITION_FLOOR::Float64 = 0.1  # Minimum return multiplier after competition penalty (0.1 = 90% max reduction)

    # ── Per-niche crowding aversion in the DECISION score (canonical change,
    #    2026-06-22) ──────────────────────────────────────────────────────────
    # Weight on the agent's ANTICIPATION of the realized convex capacity penalty
    # when it ranks opportunities. At decision time the agent observes a niche's
    # saturation S = total_invested / capacity (a visible field) and shades that
    # opportunity's score by exp(-w · λ · max(0, S/K − 1)^γ), reusing the SAME
    # λ/γ/K as the realized penalty (CROWDING_STRENGTH_LAMBDA,
    # CROWDING_CONVEXITY_GAMMA, CROWDING_CAPACITY_RATIO_K) so the agent prices
    # exactly the cost it will incur. This is PER-NICHE (steers the ranking
    # toward emptier niches), unlike the market-level 0.25 competitive-recursion
    # ACTION penalty in calculate_investment_utility, which only throttles
    # investing overall. Observability-clean: reads only the visible capacity and
    # committed-capital tally — no latent_return_potential or other hidden field.
    #   w = 0.0 → off; bit-identical to the pre-change model (pinned by test).
    #   w = 1.0 → the agent fully prices the realized penalty it would incur.
    # Residual convergence under w = 1.0 is the realistic "anticipation relocates"
    # dynamic: the agent prices CURRENT saturation, but rivals pile in after
    # commitment, so realized crowding still exceeds what was anticipated.
    DECISION_CROWDING_AVERSION_WEIGHT::Float64 = 0.0

    MACRO_REGIME_STATES::Tuple{String,String,String,String,String} =
        ("crisis", "recession", "normal", "growth", "boom")

    # NBER Business Cycle-calibrated transition matrix (monthly rounds)
    # Source: NBER Business Cycle Dating Committee data 1945-2024
    # Average expansion: 64 months, average recession: 11 months
    # Crisis frequency: ~1 per decade, boom frequency: ~2-3 per decade
    # Adjusted for monthly cadence - higher self-transition to maintain regime duration
    # Formula: p_monthly = 1 - (1 - p_quarterly)/3
    MACRO_REGIME_TRANSITIONS::Dict{String,Dict{String,Float64}} = Dict(
        "crisis" => Dict("crisis" => 0.75, "recession" => 0.12, "normal" => 0.12, "growth" => 0.01, "boom" => 0.0),
        "recession" => Dict("crisis" => 0.01, "recession" => 0.77, "normal" => 0.17, "growth" => 0.04, "boom" => 0.01),
        "normal" => Dict("crisis" => 0.003, "recession" => 0.027, "normal" => 0.85, "growth" => 0.09, "boom" => 0.03),
        "growth" => Dict("crisis" => 0.003, "recession" => 0.01, "normal" => 0.07, "growth" => 0.84, "boom" => 0.077),
        "boom" => Dict("crisis" => 0.003, "recession" => 0.013, "normal" => 0.067, "growth" => 0.13, "boom" => 0.787),
    )

    # Regime modifiers - balanced for realistic business cycle dynamics
    MACRO_REGIME_RETURN_MODIFIERS::Dict{String,Float64} = Dict(
        "crisis" => 0.85, "recession" => 0.98, "normal" => 1.10, "growth" => 1.28, "boom" => 1.45
    )

    MACRO_REGIME_FAILURE_MODIFIERS::Dict{String,Float64} = Dict(
        "crisis" => 1.18, "recession" => 1.05, "normal" => 1.0, "growth" => 0.90, "boom" => 0.78
    )

    MACRO_REGIME_TREND::Dict{String,Float64} = Dict(
        "crisis" => -0.8, "recession" => -0.35, "normal" => 0.0, "growth" => 0.35, "boom" => 0.6
    )

    MACRO_REGIME_VOLATILITY::Dict{String,Float64} = Dict(
        "crisis" => 0.55, "recession" => 0.35, "normal" => 0.2, "growth" => 0.25, "boom" => 0.3
    )

    RETURN_DEMAND_CROWDING_THRESHOLD::Float64 = 0.42
    RETURN_DEMAND_CROWDING_PENALTY::Float64 = 0.48
    FAILURE_DEMAND_PRESSURE::Float64 = 0.20

    # ========================================================================
    # RECURSION WEIGHTS
    # ========================================================================
    RECURSION_WEIGHTS::Dict{String,Float64} = Dict(
        "crowd_weight" => 0.35,
        "volatility_weight" => 0.30,
        "ai_herd_weight" => 0.40,
        "reuse_weight" => 0.20,
    )

    # Optional direct AI novelty channel for robustness/refutation cells.
    # Baseline novelty should emerge from observed combinations, niches,
    # innovation births, diversity, and reuse rather than AI usage share.
    AI_NOVELTY_UPLIFT::Float64 = 0.0
    DOWNSIDE_OVERSUPPLY_WEIGHT::Float64 = 0.30  # Balanced penalty for oversupply conditions
    RETURN_LOWER_BOUND::Float64 = 0.0  # Minimum return multiple (0.0 = total loss, 0.5 = 50% back)

    # ========================================================================
    # COMPETITION / NOVELTY-STATE SCALES
    # (The novelty-disruption mechanism that once lived here was excised
    # as unreachable. The two fields below
    # survive because live code still reads them.)
    # ========================================================================
    DISRUPTION_COMPETITION_THRESHOLD::Float64 = 10.0  # Competition normalization scale for strategic anticipation/diversification (agents.jl)
    DISRUPTION_STATE_DECAY::Float64 = 0.92            # Per-round retention of the disruption state signal (uncertainty.jl)
    # Explicit robustness stress for AI generalization on novel/data-poor
    # opportunities. Baseline is off; novelty difficulty should normally emerge
    # from opportunity features, discovery, concentration, and outcome feedback.
    NOVELTY_NOISE_INVERSION_FACTOR::Float64 = 0.0

    # ========================================================================
    # CAPACITY CONSTRAINT PARAMETERS
    # Capacity represents max concurrent capital an opportunity can absorb.
    # With 1000 agents at $5M, ~400 investing $185K/round across 30 opportunities,
    # outstanding capital per opportunity is ~$10-30M at steady state.
    # The capital-saturation convexity (models.jl) uses total_invested/capacity
    # as its crowding signal.
    # ========================================================================
    OPPORTUNITY_BASE_CAPACITY::Float64 = 15_000_000.0  # Base capacity in dollars
    OPPORTUNITY_CAPACITY_VARIANCE::Float64 = 0.3       # Random variance in capacity
    # Heavy-tailed niche/market size: when > 0, opportunity capacity is drawn from a
    # mean-preserving lognormal (median below base, heavy right tail) instead of the
    # uniform +/-VARIANCE band — most niches small, a few enormous. Relocates the
    # venture-realistic heavy tail from opportunity RETURNS (a category error) to market
    # SIZE, where it belongs. 0 = legacy uniform (canonical). (2026-06-15.)
    NICHE_SIZE_LOG_SIGMA::Float64 = 0.0

    # ========================================================================
    # SEQUENTIAL DECISION PARAMETERS
    # Information cascades from observing early investors
    # ========================================================================
    SEQUENTIAL_DECISIONS_ENABLED::Bool = true
    EARLY_DECISION_FRACTION::Float64 = 0.3            # Fraction deciding in first wave
    SIGNAL_VISIBILITY_WEIGHT::Float64 = 0.15          # How much early signals influence later agents

    # ========================================================================
    # SCALING PARAMETERS
    # ========================================================================
    OPPORTUNITIES_PER_CAPITA::Float64 = 0.04
    MIN_OPPORTUNITIES::Int = 5
    POWER_LAW_SHAPE_A::Float64 = 2.2  # Heavier tails for VC-like power-law returns (finite mean since α>2)

    # ========================================================================
    # POPULATION SCALING PARAMETERS
    # ========================================================================
    # Reference population for which all absolute parameters were calibrated.
    # When N_AGENTS differs, initialize!() applies scaling adjustments to keep
    # model dynamics comparable across population sizes.
    SCALE_REFERENCE_N::Int = 1000
    ENABLE_POPULATION_SCALING::Bool = true   # Set false to use raw (unscaled) parameters
    _POPULATION_SCALING_APPLIED::Bool = false  # Idempotency guard for initialize!
    # Per-round decay of opportunity competition (market.jl), applied as
    # competition *= (1 - rate). Default 0.10 matches the previously hardcoded
    # 0.9 retention multiplier.
    COMPETITION_DECAY_RATE::Float64 = 0.10
    MAX_KNOWLEDGE_PIECES::Int = 5000         # Cap on knowledge registry size (age-based eviction)
    # Quality gate above which a SUCCESSFUL innovation mints a new derived
    # knowledge component (create_derived_knowledge → learn_from_success!).
    # Default 0.7 reproduces the prior hardcoded gate exactly (byte-identical).
    # DIAGNOSED 2026-06-22: base components are capped at level 0.3, and
    # innovation quality ≈ mean(component_level)·0.7 + 0.2 + ~0.1, so base-only
    # quality maxes near 0.61 — BELOW 0.7 — which DEADLOCKS the recombinant cycle:
    # no derived (higher-level) knowledge can be minted to raise future quality,
    # so the gate is never reached and the pool grows only via failure residue.
    # Lowering this gate (~0.5) lets above-average successful recombinations mint
    # derived knowledge, whose higher levels then raise future innovation quality,
    # igniting the recombinant-growth loop (Weitzman / Schumpeterian new
    # combinations) the model is meant to exhibit.
    DERIVED_KNOWLEDGE_QUALITY_GATE::Float64 = 0.7

    # ── S2 per-opportunity knowledge edge (comparative_advantage; gated) ───────
    # When true, opportunities carry a knowledge_components vector and S2's
    # private-edge re-ranking uses the agent's COMPONENT-knowledge overlap with
    # the opportunity (genuine founder-market fit) instead of the coarse
    # sector-familiarity fallback. Relies on the recombinant-knowledge engine
    # (DERIVED_KNOWLEDGE_QUALITY_GATE ~0.55) for agents to hold heterogeneous
    # components. Default false = empty components everywhere = byte-identical
    # (S2 keeps using the sector-familiarity fallback).
    ENABLE_OPPORTUNITY_COMPONENTS::Bool = false
    # Number of components an ENDOWED opportunity draws from its sector's domain
    # pool (innovation-spawned opportunities inherit the recombination's
    # components directly). Only consulted when ENABLE_OPPORTUNITY_COMPONENTS.
    OPPORTUNITY_N_COMPONENTS::Int = 3

    AI_TRUST_ADJUSTMENT_RATE::Float64 = 0.033  # Monthly adjustment
    # Months between emergent-mode tier re-evaluations. A quarterly cadence
    # matches observed behavior: users rarely reconsider AI subscriptions every
    # month, and re-deciding every round would produce more switching than reality.
    AI_TIER_REVIEW_INTERVAL::Int = 3
    # Initial freeze period: defer the FIRST tier review until this many rounds
    # have elapsed, so investments mature and tier_roi_history populates with
    # real performance data before agents make their first informed switch.
    # Without it, the early review fires while ROI is 0 for all tiers and
    # decisions devolve into cost-vs-peer-signal cascades. Default 12 is one
    # investment cycle.
    AI_TIER_INITIAL_FREEZE_ROUNDS::Int = 12

    # ========================================================================
    # ACTION SELECTION CONTROLS
    # ========================================================================
    ACTION_SELECTION_TEMPERATURE::Float64 = 0.45
    ACTION_SELECTION_NOISE::Float64 = 0.10
    ACTION_BIAS_SIGMA::Float64 = 0.05
    RECENT_OUTCOME_MEMORY::Int = 24

    # ========================================================================
    # UNCERTAINTY VOLATILITY CONTROLS
    # ========================================================================
    UNCERTAINTY_SHORT_WINDOW::Int = 18  # 18 months
    UNCERTAINTY_SHORT_DECAY::Float64 = 0.0
    NOVELTY_HISTORY_WINDOW::Int = 15  # Months of novelty-event history used by uncertainty.jl novelty measures
    UNCERTAINTY_VOLATILITY_DECAY::Float64 = 0.87  # Monthly decay
    UNCERTAINTY_VOLATILITY_SCALING::Float64 = 0.45
    UNCERTAINTY_AI_SWITCH_WEIGHT::Float64 = 0.09
    UNCERTAINTY_MARKET_RETURN_WEIGHT::Float64 = 0.14
    UNCERTAINTY_ACTION_VARIANCE_WEIGHT::Float64 = 0.14
    UNCERTAINTY_CROWDING_WEIGHT::Float64 = 0.18
    UNCERTAINTY_COMPETITIVE_WEIGHT::Float64 = 0.12
    UNCERTAINTY_AI_HERDING_WEIGHT::Float64 = 0.10
    # Producer weights govern the environment's measured Knightian state.
    # Perception weights govern how agents locally read that measured state.
    # These structs expose calibration assumptions without making them AI-tier
    # bonuses; later sensitivity sweeps should vary families of these fields.
    KNIGHTIAN_PRODUCER_WEIGHTS::KnightianProducerWeights = KnightianProducerWeights()
    UNCERTAINTY_PERCEPTION_WEIGHTS::UncertaintyPerceptionWeights = UncertaintyPerceptionWeights()

    # ========================================================================
    # KNOWLEDGE & INNOVATION
    # ========================================================================
    SECTORS::Vector{String} = String[]

    # NVCA 2024-calibrated sector weights for agent initial sector assignment
    # Reflects dominant VC deal flow by sector
    SECTOR_WEIGHTS::Dict{String,Float64} = Dict(
        "tech" => 0.60,           # 60% - dominant VC deal flow
        "service" => 0.15,        # 15% - B2B services
        "manufacturing" => 0.15,  # 15% - hardware/industrial
        "retail" => 0.10,         # 10% - consumer/retail
    )

    # Sector profiles with empirically-calibrated parameters
    # Sources: NVCA 2024, BLS BED/QCEW, Fed SBCS, NSF BRDIS, USPTO, Census HHI
    # Capital ranges calibrated for 24-36 month runway (monthly cadence)
    SECTOR_PROFILES::Dict{String,SectorProfile} = Dict(
        "tech" => SectorProfile(
            (1.60, 4.00), log(2.40), 0.45,                         # return params
            (0.1, 0.17), (0.013, 0.04), (300000.0, 1200000.0),     # failure (monthly), capital
            (12, 36),                                              # maturity: 12-36 months
            # Empirically-calibrated fields. Reflects Series A/B rounds with
            # 24-36 month runway before profitability.
            (3_000_000.0, 6_000_000.0),  # initial_capital_range
            (20_000.0, 30_000.0),        # operational_cost_range: NET BURN RATE for tech.
            # The per-round op cost represents NET burn (gross opex minus revenue
            # from existing operations), not gross opex. Gross opex from BLS QCEW NAICS 5415
            # wages ($130k/yr × 3-7 engineers / 12) × 1.4 overhead = $45-105k/mo, but going
            # concerns typically have revenue covering 60-80% of opex, leaving ~30% net burn:
            # 0.3 × ($45-105k) ≈ $13-32k. Center on $20-30k.
            1_950_000.0,                  # survival_threshold: ~65% of min sector capital
            0.16,                         # innovation_probability: monthly
            (2.0, 4.0),                   # innovation_return_multiplier: high tech upside
            1.2                           # competition_intensity: HHI 1500-2500
        ),
        "retail" => SectorProfile(
            (1.40, 2.80), log(1.85), 0.32,                         # return params
            (0.07, 0.13), (0.013, 0.033), (50000.0, 400000.0),     # failure (monthly), capital
            (6, 24),                                               # maturity: 6-24 months
            # Empirically-calibrated fields:
            (2_200_000.0, 4_000_000.0),  # initial_capital_range
            (13_000.0, 23_000.0),        # operational_cost_range: net burn for retail; 0.3 × BLS QCEW NAICS 44-45 gross opex
            1_430_000.0,                  # survival_threshold: ~65% of min sector capital
            0.11,                         # innovation_probability: monthly
            (1.6, 2.5),                   # innovation_return_multiplier: moderate returns
            0.7                           # competition_intensity: HHI 500-1000
        ),
        "service" => SectorProfile(
            (1.50, 3.00), log(1.95), 0.36,                         # return params
            (0.033, 0.093), (0.01, 0.027), (15000.0, 200000.0),    # failure (monthly), capital
            (6, 18),                                               # maturity: 6-18 months
            # Empirically-calibrated fields:
            (1_400_000.0, 2_500_000.0),  # initial_capital_range
            (8_300.0, 15_000.0),         # operational_cost_range: net burn for services; 0.3 × BLS QCEW NAICS 56/81 gross opex
            910_000.0,                    # survival_threshold: ~65% of min sector capital
            0.13,                         # innovation_probability: monthly
            (1.6, 2.5),                   # innovation_return_multiplier: moderate returns
            0.9                           # competition_intensity: HHI 800-1500
        ),
        "manufacturing" => SectorProfile(
            (1.60, 3.50), log(2.20), 0.4,                          # return params
            (0.083, 0.14), (0.013, 0.033), (250000.0, 1500000.0),  # failure (monthly), capital
            (18, 48),                                              # maturity: 18-48 months
            # Empirically-calibrated fields:
            (4_000_000.0, 7_500_000.0),  # initial_capital_range
            (26_700.0, 40_000.0),        # operational_cost_range: net burn for manufacturing; 0.3 × BLS QCEW NAICS 31-33 gross opex
            2_600_000.0,                  # survival_threshold: ~65% of min sector capital
            0.17,                         # innovation_probability: monthly
            (1.5, 2.8),                   # innovation_return_multiplier: incremental improvements
            1.4                           # competition_intensity: HHI 1800-3000
        ),
    )

    # ========================================================================
    # CALIBRATION
    # ========================================================================
    calibration_targets::Dict{String,Any} = Dict{String,Any}()
    active_calibration::Union{String,Nothing} = nothing

    # ========================================================================
    # PERFORMANCE / IO
    # ========================================================================
    round_log_interval::Int = 12  # Log every year (12 months)
    enable_round_logging::Bool = true
    use_float32::Bool = true
end

# ── AGI strategy ladder mode registry + validation ──────────────────────────
# (the strategy-ladder design notes). Lives in config.jl because the
# valid-mode set is a config contract; the behavioral functions live in
# src/strategy.jl.
const STRATEGY_MODES = (
    "reactive",
    "consensus_discounting",
    "comparative_advantage",
    "complement_seeking",
    "agi_native",
)

"""
Validate the AGI-strategy-ladder fields. Unknown STRATEGY_MODE or
STRATEGY_TIERS entries ERROR loudly — they must never fall through to reactive
behavior (loud-failure convention, the design notes). Called from
`initialize!` (config build) and from `strategy_active` (first use), so a
config that bypasses `initialize!` still fails fast.
"""
function validate_strategy_config(config::EmergentConfig)::Nothing
    mode = config.STRATEGY_MODE
    if !(mode in STRATEGY_MODES)
        error("Unknown STRATEGY_MODE '$(mode)'. Valid modes: " *
              "$(join(STRATEGY_MODES, ", ")). Unknown modes never fall " *
              "through to reactive (the strategy-ladder design notes).")
    end
    for tier in config.STRATEGY_TIERS
        if !haskey(config.AI_LEVELS, tier)
            error("Unknown STRATEGY_TIERS entry '$(tier)'. Valid tiers: " *
                  "$(join(sort!(collect(keys(config.AI_LEVELS))), ", ")).")
        end
    end
    # G2 (2026-06-09): S1 is the manuscript's canonical operationalization of
    # anticipatory congestion discounting; the legacy
    # STRATEGIC_ANTICIPATION_ENABLED multiplier is retained ONLY as an
    # independent-implementation robustness check, run in its own cell.
    # Co-activating both on the same score path is forbidden loudly.
    if config.STRATEGIC_ANTICIPATION_ENABLED &&
       (mode == "consensus_discounting" || mode == "agi_native")
        error("STRATEGIC_ANTICIPATION_ENABLED=true with STRATEGY_MODE=" *
              "'$(mode)': two operationalizations of anticipatory congestion " *
              "discounting would compose multiplicatively; S1 is canonical " *
              "(the strategy-ladder design notes, Canonicality); run " *
              "the legacy cell separately.")
    end
    # STRATEGIC_DIVERSIFICATION_ENABLED is deliberately NOT forbidden
    # alongside the ladder: it is a post-ranking stochastic RE-DRAW among
    # near-top evaluations (select_portfolio_evaluation, src/agents.jl), not a
    # second discount on the same score path. Co-activation composes selection
    # with shading — a meaningful robustness cell, not double-counting.
    return nothing
end

"""
Validate the open-action extension fields (the strategy-ladder design notes
§Open-action extension). Malformed haircut bounds or a negative redirection
strength ERROR loudly — never silently clamp (loud-failure convention,
the design notes). Called from `initialize!` and at first use of each
enabled channel (pivot review / directed sector selection), so a config that
bypasses `initialize!` still fails fast.
"""
function validate_open_action_config(config::EmergentConfig)::Nothing
    floor_v = config.PIVOT_HAIRCUT_FLOOR
    ceiling_v = config.PIVOT_HAIRCUT_CEILING
    if !(0.0 <= floor_v <= ceiling_v <= 1.0)
        error("Invalid pivot haircut bounds: PIVOT_HAIRCUT_FLOOR=$(floor_v), " *
              "PIVOT_HAIRCUT_CEILING=$(ceiling_v). Require " *
              "0 ≤ floor ≤ ceiling ≤ 1 (recovered FRACTION of committed " *
              "capital; the strategy-ladder design notes).")
    end
    if config.PIVOT_REDEPLOY_MARGIN < 0.0
        error("Invalid PIVOT_REDEPLOY_MARGIN=$(config.PIVOT_REDEPLOY_MARGIN). " *
              "Require ≥ 0 (liquidation must beat holding by a non-negative " *
              "margin).")
    end
    if config.PIVOT_CONVICTION_BASE < 1.0
        error("Invalid PIVOT_CONVICTION_BASE=$(config.PIVOT_CONVICTION_BASE). " *
              "Require ≥ 1 so the conviction factor (BASE − confidence) stays " *
              "non-negative over confidence ∈ [0, 1].")
    end
    if config.PIVOT_DETERIORATION_GAIN < 0.0
        error("Invalid PIVOT_DETERIORATION_GAIN=" *
              "$(config.PIVOT_DETERIORATION_GAIN). Require ≥ 0 (1.0 = the " *
              "bit-identical pre-parameter trigger).")
    end
    if config.DIRECTED_CREATION_STRENGTH < 0.0
        error("Invalid DIRECTED_CREATION_STRENGTH=" *
              "$(config.DIRECTED_CREATION_STRENGTH). Require ≥ 0. The dial " *
              "is a mixture: P(keep knowledge-mapped sector) = exp(−strength)" *
              ", else a density-tilted draw — strength = 0 reproduces the " *
              "flag-off sector distribution exactly (ENABLE_DIRECTED_" *
              "CREATION=false additionally consumes zero RNG).")
    end
    return nothing
end

"""
Validate the emergence-audit dials (the strategy-ladder design notes,
emergence-audit extension). Out-of-range values ERROR loudly — never
silently clamp (loud-failure convention). Called from `initialize!` and at
first use of each enabled dial (get_information when rho != 0;
make_decision! when T != 1), so a config that bypasses `initialize!` still
fails fast.
"""
function validate_emergence_audit_config(config::EmergentConfig)::Nothing
    rho = config.AI_ERROR_CORRELATION
    if !(0.0 <= rho <= 1.0)
        error("Invalid AI_ERROR_CORRELATION=$(rho). Require 0 ≤ rho ≤ 1 " *
              "(variance-preserving blend sqrt(rho)*eps_common + " *
              "sqrt(1-rho)*eps_idio; the strategy-ladder design notes " *
              "emergence-audit extension).")
    end
    if !(config.DECISION_TEMPERATURE > 0.0)
        error("Invalid DECISION_TEMPERATURE=$(config.DECISION_TEMPERATURE). " *
              "Require > 0 (softmax temperature multiplier; 1.0 = " *
              "bit-identical default; the strategy-ladder design notes " *
              "emergence-audit extension).")
    end
    if config.DECISION_CROWDING_AVERSION_WEIGHT < 0.0
        error("Invalid DECISION_CROWDING_AVERSION_WEIGHT=" *
              "$(config.DECISION_CROWDING_AVERSION_WEIGHT). Require ≥ 0 " *
              "(0.0 = off / bit-identical; weight on the anticipated convex " *
              "capacity penalty in the per-niche decision score).")
    end
    return nothing
end

# Post-initialization: sync SECTORS from SECTOR_PROFILES keys and apply population scaling
function initialize!(config::EmergentConfig)
    config.SECTORS = ordered_sector_keys(config.SECTOR_PROFILES)
    # Loud validation of the AGI strategy ladder fields at config build time
    # (the strategy-ladder design notes).
    validate_strategy_config(config)
    validate_open_action_config(config)
    validate_emergence_audit_config(config)

    # ── Population scaling ──────────────────────────────────────────
    # All absolute parameters were calibrated at SCALE_REFERENCE_N (default 1000).
    # When N_AGENTS differs, we adjust capacity/crowding thresholds so that
    # per-agent dynamics remain comparable.
    #
    # Scaling uses √(N/N_ref) for thresholds that gate on aggregate quantities.
    # Competition delta (market.jl) uses √N scaling with reference_population=100
    # to maintain calibrated per-capita pressure. These two scaling regimes
    # work together: √N thresholds absorb √N-scaled aggregate competition.
    if config.ENABLE_POPULATION_SCALING && config.N_AGENTS != config.SCALE_REFERENCE_N && !config._POPULATION_SCALING_APPLIED
        scale = config.N_AGENTS / config.SCALE_REFERENCE_N  # ratio (e.g., 10 for 10K, 100 for 100K)
        sqrt_scale = sqrt(scale)

        # The crowding threshold CROWDING_CAPACITY_RATIO_K is an economic per-opp
        # property and intentionally does NOT scale with N: saturation =
        # total_invested/capacity stays bounded as N grows because
        # OPPORTUNITY_BASE_CAPACITY also scales by √N (below).

        # Opportunity base capacity: max capital an opportunity absorbs.
        # Total capital scales linearly with N, opportunities sub-linearly.
        # Scale by √N (not linearly) to maintain utilization pressure.
        config.OPPORTUNITY_BASE_CAPACITY *= sqrt_scale

        # Disruption competition threshold: absolute level, scale with K
        config.DISRUPTION_COMPETITION_THRESHOLD *= sqrt_scale

        config._POPULATION_SCALING_APPLIED = true
        @info "Population scaling applied" N=config.N_AGENTS ref=config.SCALE_REFERENCE_N scale sqrt_scale K_ratio=config.CROWDING_CAPACITY_RATIO_K capacity=config.OPPORTUNITY_BASE_CAPACITY
    end

    return config
end

"""
Compute integer opportunity targets even if overrides supply floats.

Scaling policy: above the reference population, opportunities grow
sub-linearly to model realistic market saturation (not every additional
entrepreneur creates a proportional number of new markets); below the
reference population, opportunities scale linearly per capita so smaller
runs keep the calibrated opportunity density.

  n_opps = max(MIN, BASE, per_capita * N)                      for N <  N_ref
  n_opps = max(MIN, BASE, N_ref * per_capita * (1 + ln(N/N_ref))) for N >= N_ref

This gives 20 at 500, 40 at 1K, ~133 at 10K, ~225 at 100K (vs. 4000 with
linear scaling).
"""
function get_scaled_opportunities(config::EmergentConfig, n_agents::Int)::Int
    min_ops = ceil(Int, Float64(config.MIN_OPPORTUNITIES))
    base_ops = ceil(Int, Float64(config.BASE_OPPORTUNITIES))
    per_capita = max(config.OPPORTUNITIES_PER_CAPITA, 0.0)

    if config.ENABLE_POPULATION_SCALING
        ref_n = config.SCALE_REFERENCE_N
        if n_agents < ref_n
            # Below reference: linear per-capita scaling preserves the
            # calibrated opportunity density for smaller populations.
            scaled = ceil(Int, n_agents * per_capita)
        else
            # At/above reference: reference-level linear + log surplus
            ref_opps = ref_n * per_capita
            log_scale = 1.0 + max(0.0, log(n_agents / ref_n))
            scaled = ceil(Int, ref_opps * log_scale)
        end
    else
        scaled = ceil(Int, n_agents * per_capita)
    end

    return max(min_ops, base_ops, scaled)
end

"""
Get AI domain capability for a given tier and domain.
"""
function get_ai_domain_capability(config::EmergentConfig, ai_level::String, domain::String)::AIDomainCapability
    if haskey(config.AI_DOMAIN_CAPABILITIES, ai_level)
        tier_caps = config.AI_DOMAIN_CAPABILITIES[ai_level]
        if haskey(tier_caps, domain)
            return tier_caps[domain]
        end
    end
    return AIDomainCapability(0.5, 0.1, 0.0)
end

"""
Return a deep-copied, JSON-safe representation of the configuration.
"""
function snapshot(config::EmergentConfig)::Dict{String,Any}
    result = Dict{String,Any}()
    for field_name in fieldnames(EmergentConfig)
        result[String(field_name)] = _to_plain_value(getfield(config, field_name))
    end
    return result
end

"""
Return a new config with the provided overrides merged in.
"""
function copy_with_overrides(config::EmergentConfig, overrides::Dict{String,Any})::EmergentConfig
    new_cfg = deepcopy(config)
    apply_overrides!(new_cfg, overrides)
    new_cfg.SECTORS = ordered_sector_keys(new_cfg.SECTOR_PROFILES)
    return new_cfg
end

"""
Recursively merge overrides into config.
"""
function apply_overrides!(config::EmergentConfig, overrides::Dict{String,Any})
    for (key, value) in overrides
        key_sym = Symbol(key)
        if !hasproperty(config, key_sym)
            error("Unknown configuration attribute '$key' in calibration override.")
        end
        current = getfield(config, key_sym)
        field_type = fieldtype(typeof(config), key_sym)
        plain_value = _plain_json_value(value)
        if _supports_partial_override(current) && isa(plain_value, Dict)
            # Expand the existing typed value to a plain dict tree first so that
            # partial overrides of Dict-of-struct fields or top-level calibration
            # structs deep-merge with the rest of the fields before
            # _coerce_override_value rebuilds the typed structure.
            base_plain = _to_plain_value(current)
            merged = deep_merge_dict(base_plain, plain_value)
            setfield!(config, key_sym, _coerce_override_value(field_type, merged))
        else
            setfield!(config, key_sym, _coerce_override_value(field_type, plain_value))
        end
    end
end

function _supports_partial_override(value)::Bool
    if value isa AbstractDict
        return true
    end
    T = typeof(value)
    return isstructtype(T) &&
        fieldcount(T) > 0 &&
        !(value isa Union{Number,AbstractString,Symbol,Nothing})
end

"""
Convert JSON3 containers to plain Julia containers before applying config
overrides. JSON snapshots otherwise leave nested values as JSON3.Object /
JSON3.Array, which cannot be assigned back into typed config fields.
"""
function _plain_json_value(value)
    if value isa JSON3.Object
        return Dict{String,Any}(String(k) => _plain_json_value(v) for (k, v) in pairs(value))
    elseif value isa JSON3.Array
        return Any[_plain_json_value(v) for v in value]
    else
        return value
    end
end

"""
Recursively expand a typed config value into a plain Julia container tree:
structs become Dict{String,Any}, Tuples and AbstractArrays become Any[], and
nested Dicts are re-keyed to String. Used by apply_overrides! so that the
deep-merge step operates on a uniform dict-of-dicts tree regardless of whether
the existing field holds structs, tuples, or primitives.
"""
function _to_plain_value(v)
    if v isa AbstractDict
        return Dict{String,Any}(String(k) => _to_plain_value(vv) for (k, vv) in v)
    elseif v isa Tuple
        return Any[_to_plain_value(x) for x in v]
    elseif v isa AbstractArray
        return Any[_to_plain_value(x) for x in v]
    elseif v isa Union{Number,AbstractString,Symbol,Nothing}
        return v
    elseif isstructtype(typeof(v)) && fieldcount(typeof(v)) > 0
        return Dict{String,Any}(
            String(f) => _to_plain_value(getfield(v, f)) for f in fieldnames(typeof(v))
        )
    else
        return v
    end
end

function _coerce_override_value(::Type{Any}, value)
    return _plain_json_value(value)
end

function _coerce_override_value(::Type{T}, value) where T
    value = _plain_json_value(value)

    if value isa T
        return value
    end
    if value === nothing
        if Nothing <: T
            return nothing
        end
        error("Cannot assign nothing to config field of type $T")
    end

    if T isa Union
        last_err = nothing
        for subtype in Base.uniontypes(T)
            subtype === Nothing && continue
            try
                return _coerce_override_value(subtype, value)
            catch err
                last_err = err
            end
        end
        rethrow(last_err)
    elseif T === Float64
        return Float64(value)
    elseif T === Int
        return Int(value)
    elseif T === Bool
        return Bool(value)
    elseif T === String
        return String(value)
    elseif T <: Tuple
        vals = collect(value)
        params = T.parameters
        length(vals) == length(params) || error("Tuple override length mismatch for $T")
        return tuple((_coerce_override_value(params[i], vals[i]) for i in eachindex(vals))...)
    elseif T <: AbstractVector
        element_type = eltype(T)
        return element_type[_coerce_override_value(element_type, v) for v in value]
    elseif T <: Dict
        key_type, value_type = T.parameters
        result = Dict{key_type,value_type}()
        for (k, v) in value
            result[_coerce_override_value(key_type, k)] = _coerce_override_value(value_type, v)
        end
        return result
    elseif T <: Number
        return T(value)
    elseif value isa Dict
        names = fieldnames(T)
        types = fieldtypes(T)
        args = Any[]
        for (i, name) in enumerate(names)
            key = String(name)
            haskey(value, key) || error("Missing field '$key' while reconstructing $T")
            push!(args, _coerce_override_value(types[i], value[key]))
        end
        return T(args...)
    else
        return value
    end
end

"""
Deep merge two dictionaries without mutating the originals.
"""
function deep_merge_dict(base::Dict, updates::Dict)::Dict
    merged = Dict{Any,Any}(k => deepcopy(v) for (k, v) in base)
    for (key, value) in updates
        if isa(value, Dict) && haskey(merged, key) && isa(merged[key], Dict)
            merged[key] = deep_merge_dict(merged[key], value)
        else
            merged[key] = deepcopy(value)
        end
    end
    return merged
end

# ============================================================================
# CALIBRATION PROFILES
# ============================================================================

"""
Reusable parameter bundle with empirical targets for reproducibility.
"""
struct CalibrationProfile
    name::String
    description::String
    overrides::Dict{String,Any}
    target_metrics::Dict{String,Dict{String,Any}}
    source::String
end

function CalibrationProfile(;
    name::String,
    description::String,
    overrides::Dict{String,Any} = Dict{String,Any}(),
    target_metrics::Dict{String,Dict{String,Any}} = Dict{String,Dict{String,Any}}(),
    source::String = "built-in"
)
    CalibrationProfile(name, description, overrides, target_metrics, source)
end

"""
Return a config with the calibration overrides applied.
"""
function apply_calibration_profile(config::EmergentConfig, profile::Union{CalibrationProfile,Nothing})::EmergentConfig
    if isnothing(profile)
        return config
    end
    updated = copy_with_overrides(config, profile.overrides)
    updated.active_calibration = profile.name
    updated.calibration_targets = deepcopy(profile.target_metrics)
    return updated
end

"""
Fetch a built-in calibration profile by name (case-insensitive).
"""
function get_calibration_profile(name::String)::CalibrationProfile
    normalized = lowercase(strip(name))
    for profile in values(CALIBRATION_LIBRARY)
        if lowercase(profile.name) == normalized
            return profile
        end
    end
    available = join(keys(CALIBRATION_LIBRARY), ", ")
    error("Unknown calibration profile '$name'. Available: $available")
end

"""
Load a calibration profile definition from disk.
"""
function load_calibration_profile(path::String)::CalibrationProfile
    file_path = abspath(expanduser(path))
    if !isfile(file_path)
        error("Calibration file not found: $file_path")
    end
    payload = JSON3.read(read(file_path, String))
    # Materialize JSON3 containers to plain Dict{String,Any} trees (via
    # _plain_json_value, the same helper apply_overrides! uses): JSON3.Object
    # cannot be assigned to the Dict-typed CalibrationProfile fields.
    overrides_raw = _plain_json_value(get(payload, :overrides, get(payload, :parameters, Dict{String,Any}())))
    overrides = Dict{String,Any}(String(k) => v for (k, v) in overrides_raw)
    target_raw = _plain_json_value(get(payload, :target_metrics, Dict{String,Any}()))
    target_metrics = Dict{String,Dict{String,Any}}(
        String(k) => Dict{String,Any}(String(kk) => vv for (kk, vv) in v) for (k, v) in target_raw
    )
    name = String(get(payload, :name, splitext(basename(file_path))[1]))
    description = String(get(payload, :description, "Custom calibration loaded from $(basename(file_path))"))
    source = String(get(payload, :source, file_path))
    return CalibrationProfile(
        name=name,
        description=description,
        overrides=overrides,
        target_metrics=target_metrics,
        source=source
    )
end

# ============================================================================
# BUILT-IN CALIBRATION LIBRARY
# ============================================================================

const CALIBRATION_LIBRARY = Dict{String,CalibrationProfile}(
    "minimal_causal" => CalibrationProfile(
        name="minimal_causal",
        description="Minimal model for causal identification. Strips simulation to ~25 essential parameters. Monthly cadence.",
        overrides=Dict{String,Any}(
            "N_AGENTS" => 500,
            "N_ROUNDS" => 600,  # 600 months = 50 years
            "N_RUNS" => 30,
            # Forcing uniform initial capital for causal identification requires
            # BOTH keys: the INITIAL_CAPITAL scalar is a no-op unless
            # USE_UNIFORM_INITIAL_CAPITAL is true (otherwise the EmergentAgent
            # constructor samples from sector_profile.initial_capital_range).
            "USE_UNIFORM_INITIAL_CAPITAL" => true,
            "INITIAL_CAPITAL" => 5_000_000.0,
            "SURVIVAL_CAPITAL_RATIO" => 0.40,
            "USE_NETWORK_EFFECTS" => false,
            "RETURN_NOISE_SCALE" => 0.20,
            "BLACK_SWAN_PROBABILITY" => 0.0,
        ),
        target_metrics=Dict{String,Dict{String,Any}}(
            "causal_identification" => Dict{String,Any}(
                "description" => "Minimal model for isolating AI tier causal effects",
                "fixed_tier_required" => true,
            ),
        ),
    ),
    "venture_baseline_2024" => CalibrationProfile(
        name="venture_baseline_2024",
        description="Dual-anchor calibration (approved 2026-06-09, the calibration design notes): PRIMARY anchor = venture-class 5-yr survival band 0.65-0.80 (the modeled population is \$2.5-10M-capitalized tech ventures; Puri & Zarutskie 2012 JF: VC-financed firms fail materially less in years 1-5 than matched non-VC). CONTEXT anchor = BLS BED economy-wide ~0.50-0.55. Anchor choice forced by population validity, not convenience: the frontier-tier pattern is anchor-independent. Monthly cadence.",
        overrides=Dict{String,Any}(
            # This profile does not override BASE_OPERATIONAL_COST: the production
            # charge path reads agent.operating_cost_estimate (sector-derived from
            # sector_profile.operational_cost_range), so the global scalar has no
            # effect on actual burn. Scale operating costs via OPS_COST_INTENSITY.
            "SURVIVAL_CAPITAL_RATIO" => 0.56,
            "INSOLVENCY_GRACE_ROUNDS" => 6,  # 6 months grace period
            "DISCOVERY_PROBABILITY" => 0.073,  # Monthly
            "INNOVATION_PROBABILITY" => 0.123,  # Monthly
            "OPPORTUNITY_RETURN_RANGE" => (0.92, 6.0),  # Better returns for 10-yr survival
            "INVESTMENT_SUCCESS_ROI_THRESHOLD" => 0.04,  # Monthly
            "MAX_INVESTMENT_FRACTION" => 0.04,  # Monthly
        ),
        target_metrics=Dict{String,Dict{String,Any}}(
            # PRIMARY anchor — focal population (venture-class). Post-fix smoke
            # (2026-06-09): overall 0.67-0.68 sits inside this band; no
            # cost/threshold tightening indicated.
            "survival_rate_month60" => Dict{String,Any}(
                "target" => 0.725,
                "tolerance" => 0.075,  # band 0.65-0.80
                "role" => "primary",
                "source" => "Venture-class 5-yr survival (Puri & Zarutskie 2012 JF, " *
                            "VC-financed vs matched non-VC failure dynamics; " *
                            "Crunchbase-class graduation data). Focal population: " *
                            "\$2.5-10M-capitalized tech ventures.",
            ),
            # CONTEXT anchor — economy-wide reference, reported alongside the
            # primary, not targeted by any knob.
            "survival_rate_month60_economy_context" => Dict{String,Any}(
                "target" => 0.525,
                "tolerance" => 0.025,  # band 0.50-0.55
                "role" => "context",
                "source" => "BLS Business Employment Dynamics, 5-year establishment " *
                            "survival (economy-wide, all sectors/sizes). Context only: " *
                            "wrong population for the modeled firms.",
            ),
            "survival_rate_month120" => Dict{String,Any}(
                "target" => 0.25,
                "tolerance" => 0.10,
                "role" => "context",
                "source" => "BLS 10-year survival estimates (economy-wide context).",
            ),
        ),
    ),
)
