"""
AGI strategy ladder — S1/S2/S3 decision strategies.

Implements the strategy rungs specified in
the strategy-ladder design notes:

  S1 `consensus_discounting`  anticipatory congestion shade on the expected
                              return of consensus-legible opportunities
                              (information cascades, Bikhchandani et al. 1992)
  S2 `comparative_advantage`  private-edge re-ranking from self-observable
                              founder–market fit (Foss & Klein 2012 judgment)
  S3 `complement_seeking`     reallocate effort toward opportunities where the
                              agent's OWN instrument is unsure (Hayek 1945;
                              Sarasvathy 2001 means-based logic)
  S4 `agi_native`             S1+S2+S3 jointly

OBSERVABILITY CONTRACT (design doc "Observability audit", enforced by
test/test_strategy_ladder.jl): functions in this module may read ONLY
decision-time observables — the agent's own Information fields
(estimated_return, estimated_uncertainty, confidence), the agent's own
traits / sector knowledge / knowledge set, Perception fields, and observable
opportunity attributes already used by production scoring (sector and the
sector_familiarity opacity inputs). They must NEVER read
opp.latent_return_potential, opp.latent_failure_potential,
info.hidden_factors, info.contains_hallucination, info.actual_accuracy,
other agents' current-round choices, or any RNG stream (all functions are
deterministic given their inputs, preserving seed pairing).

The reactive default is bit-identical to the pre-ladder model: every hook in
agents.jl gates on `strategy_active`, which returns `false` for
STRATEGY_MODE="reactive" before any strategy computation runs (pinned by the
default-neutrality test and the debug counter below).
"""

# Debug counter: incremented by every strategy component computation. The
# default-neutrality test asserts it stays at 0 for an entire reactive
# simulation — structural proof that no strategy code path is reachable when
# inactive. Atomic because suite runs execute simulations on multiple threads.
#
# G4 hot-path hygiene (2026-06-09): increments are gated behind a debug flag
# (Ref{Bool}, default false) so production ladder cells pay one non-atomic
# boolean load per computation instead of a contended atomic add.
# `reset_strategy_eval_count!()` ENABLES counting as a side effect — resetting
# only ever happens in tests about to measure the counter, so the ==0
# neutrality pins and the >0 liveness pins keep their meaning without every
# test file having to know about the flag. Production never resets, so the
# flag stays false for production runs.
const STRATEGY_EVAL_COUNT = Threads.Atomic{Int}(0)
const STRATEGY_EVAL_COUNT_ENABLED = Ref{Bool}(false)

set_strategy_eval_count_enabled!(on::Bool) = (STRATEGY_EVAL_COUNT_ENABLED[] = on; nothing)
strategy_eval_count_enabled()::Bool = STRATEGY_EVAL_COUNT_ENABLED[]
function reset_strategy_eval_count!(; enable::Bool = true)
    STRATEGY_EVAL_COUNT_ENABLED[] = enable
    Threads.atomic_xchg!(STRATEGY_EVAL_COUNT, 0)
    return nothing
end
strategy_eval_count()::Int = STRATEGY_EVAL_COUNT[]
@inline function _strategy_count!()
    if STRATEGY_EVAL_COUNT_ENABLED[]
        Threads.atomic_add!(STRATEGY_EVAL_COUNT, 1)
    end
    return nothing
end

# ============================================================================
# DISPATCHER
# ============================================================================

"""
    strategy_active(config, tier)::Bool

Whether the AGI strategy ladder applies to a decision taken at behavioral
`tier`. Reactive mode short-circuits BEFORE validation or any other work so
the default path costs one string comparison (bit-identical requirement).
Unknown modes ERROR loudly at first use via `validate_strategy_config`.

G4 hygiene note: this gate is evaluated once per decision CALL, never per
candidate — the hook sites cache the result in `ladder_active` before their
scoring loops, so an inactive config does no per-candidate string work.
"""
function strategy_active(config::EmergentConfig, tier::AbstractString)::Bool
    config.STRATEGY_MODE == "reactive" && return false
    validate_strategy_config(config)
    return String(tier) in config.STRATEGY_TIERS
end

"""
    strategy_active(agent, config)::Bool

Agent-level convenience form: gates on the agent's current effective
(behavioral) tier. Hook sites inside the decision path use the
(config, tier) form directly because they already hold the effective
decision-time AI level.
"""
function strategy_active(agent, config::EmergentConfig)::Bool
    return strategy_active(config, behavior_ai_level(config, String(get_ai_level(agent))))
end

# Per-component gates. `agi_native` composes all three rungs (design doc S4).
strategy_uses_s1(mode::AbstractString)::Bool =
    mode == "consensus_discounting" || mode == "agi_native"
strategy_uses_s2(mode::AbstractString)::Bool =
    mode == "comparative_advantage" || mode == "agi_native"
strategy_uses_s3(mode::AbstractString)::Bool =
    mode == "complement_seeking" || mode == "agi_native"

# ============================================================================
# OBSERVABLE EXTRACTORS
# ============================================================================
# Single funnel through which strategy code touches Information objects: only
# the visible fields, with the same clamps evaluate_opportunity_basic applies,
# so the observability audit has exactly one surface to check per input.

_observable_confidence(info::Information)::Float64 = clamp(Float64(info.confidence), 0.05, 0.99)
_observable_confidence(::Nothing)::Float64 = 0.5  # neutral, matches evaluate_opportunity_basic fallback

_observable_uncertainty(info::Information)::Float64 = clamp(Float64(info.estimated_uncertainty), 0.0, 1.0)
_observable_uncertainty(::Nothing)::Float64 = 0.5

"""
Market-level perceived signal-commonality in [0,1]: how much of the market is
acting on shared (AI-legible) signals. Combines the visible AI-adoption share
with the perceived competitive-recursion level (the agent's existing
"everyone sees what I see" signal) — both already exposed through production
perception. Used by S3 (and reuses S1's adoption input at market level, per
the design doc).
"""
function market_signal_commonality(perception::Perception)::Float64
    adoption = clamp(Float64(perception.competition_signal.ai_usage_share), 0.0, 1.0)
    recursion = clamp(Float64(perception.competitive_recursion.level), 0.0, 1.0)
    return clamp(0.5 * adoption + 0.5 * recursion, 0.0, 1.0)
end

"""
Market-level perceived crowding in [0,1] (S1 input c): the
practical-indeterminism crowding pressure already computed by production
perception. Deliberately market-level, not per-opportunity — the design doc
specifies S1's per-opportunity differentiation comes through own-instrument
confidence (consensus legibility), not through opportunity attributes.
"""
perceived_crowding(perception::Perception)::Float64 =
    clamp(Float64(perception.practical_indeterminism.crowding_pressure), 0.0, 1.0)

# ============================================================================
# S1 — CONSENSUS DISCOUNTING
# ============================================================================

"""
    consensus_congestion_forecast(own_confidence, ai_adoption_share, crowding)

Anticipatory congestion shade in [0,1] (design doc S1): a simple product of
three [0,1] observables —

  (a) the agent's OWN instrument confidence on this opportunity ("my model is
      confident here ⇒ this opportunity is data-rich ⇒ rivals' instruments
      are confident too" — consensus legibility),
  (b) the visible AI-adoption share (how many rivals hold such instruments),
  (c) current perceived market crowding (how aggressively capital is already
      converging).

A product (not a weighted sum) because all three are necessary for the
anticipated pile-in: a confident signal nobody else can buy, or universal AI
with no observed convergence, forecasts no congestion.
"""
function consensus_congestion_forecast(
    own_confidence::Float64,
    ai_adoption_share::Float64,
    crowding::Float64,
)::Float64
    return clamp(own_confidence, 0.0, 1.0) *
           clamp(ai_adoption_share, 0.0, 1.0) *
           clamp(crowding, 0.0, 1.0)
end

"""
S1 hook: shade the agent's decision-time expected return BEFORE scoring and
commitment. Only the EXCESS margin over break-even (est_return − 1) is
shaded — shading a negative margin by (1 − d·f) would shrink the expected
loss and make already-unattractive opportunities MORE attractive, inverting
the mechanism. The reactive 0.25 competitive-recursion utility penalty
(calculate_investment_utility) stays untouched: S1 ADDS the anticipatory,
per-opportunity component; it does not retune the reactive coefficient
(no-double-count requirement in the design brief).

Reads only: info.confidence (visible), perception market aggregates.
"""
function strategy_shaded_return(
    config::EmergentConfig,
    est_return::Float64,
    info::Union{Information,Nothing},
    perception::Perception,
)::Float64
    strategy_uses_s1(config.STRATEGY_MODE) || return est_return
    _strategy_count!()
    margin = est_return - 1.0
    margin <= 0.0 && return est_return
    forecast = consensus_congestion_forecast(
        _observable_confidence(info),
        Float64(perception.competition_signal.ai_usage_share),
        perceived_crowding(perception),
    )
    discount = clamp(config.STRATEGY_CONSENSUS_DISCOUNT, 0.0, 1.0)
    return 1.0 + margin * (1.0 - discount * forecast)
end

# ============================================================================
# S2 — COMPARATIVE ADVANTAGE (PRIVATE EDGE)
# ============================================================================
# G1 redesign (2026-06-09, approved): the edge is RELATIVE within the agent's
# current choice set. The original form 1 + w·(edge − 0.5) anchored on an
# absolute 0.5 "neutral" center that the edge formula could not reach (max
# achievable 0.465, typical 0.21–0.35), so EVERY opportunity took a sub-1
# multiplier — a flat invest-propensity suppression confounding the re-ranking
# treatment in every S2/composite cell. The fix centers the multiplier on the
# choice-set mean edge: mean multiplier ≡ 1 per choice set BY CONSTRUCTION
# (whenever the [0.25, 2.0] inspectability clamp does not bind, which at the
# default weight requires deviations beyond ±0.75 — impossible for edges in
# [0,1] unless the weight is raised above 1.5), so only the RANKING signal
# survives. The agent-constant trait term (competence/analytical_ability) is
# dropped: per-agent constants cannot re-rank a choice set; they only shifted
# the level of every multiplier (and partially double-counted channels the
# traits already act through — the base score's uncertainty_tolerance scaling
# and confidence sizing).

# Knowledge-component overlap mirrors the production perception pattern
# (uncertainty.jl ~1916): when the opportunity type carries a
# knowledge_components vector, overlap = |agent_knowledge ∩ components| /
# |components|. The base Opportunity struct has NO components field, so the
# EXPLICIT production fallback is sector familiarity (same fallback production
# perception uses). The agent-knowledge Set is built lazily INSIDE the
# hasfield branch (G1 latent-trap fix): production never pays the allocation,
# and the caller passes the knowledge dict itself rather than pre-building a
# Set of sector names that would silently mis-intersect component IDs. An
# extended opportunity type using this branch must share its component-ID
# ontology with the agent knowledge keys.
function _component_overlap(
    opp,
    agent_components::Set{String},
    fallback::Float64,
)::Float64
    # S2 per-opportunity edge: overlap between the agent's COMPONENT knowledge
    # (the kb.agent_knowledge mirror synced in make_decision!) and the
    # opportunity's knowledge_components. Empty agent_components (flag off, or an
    # agent with no component knowledge) → sector-familiarity fallback, the
    # pre-edge behavior, so the model is byte-identical when
    # ENABLE_OPPORTUNITY_COMPONENTS is off.
    isempty(agent_components) && return fallback
    if hasfield(typeof(opp), :knowledge_components)
        comps = getfield(opp, :knowledge_components)
        if !isnothing(comps) && !isempty(comps)
            component_ids = Set(String.(comps))
            return length(intersect(agent_components, component_ids)) /
                   length(component_ids)
        end
    end
    return fallback
end

"""
    private_edge(opp, sector_knowledge, sector_familiarity_value)

Per-opportunity self-observable founder–market-fit edge in [0,1] (design doc
S2, G1 form): the knowledge-component overlap when the opportunity type
carries the field, else sector familiarity (the base Opportunity path).

Agent-constant traits are deliberately EXCLUDED (G1 fix, 2026-06-09): they
cannot re-rank a within-agent choice set and only added level noise plus a
partial double-count of the channels the traits already act through (base
score, confidence). Documented in the strategy-ladder design notes.
"""
function private_edge(
    opp,
    sector_knowledge,
    sector_familiarity_value::Float64,
    agent_components::Set{String}=Set{String}(),
)::Float64
    # agent_components empty (default / flag off) → _component_overlap returns the
    # sector-familiarity fallback, preserving the pre-edge behavior. sector_knowledge
    # is retained for signature compatibility with existing direct callers but no
    # longer drives the overlap (the agent's COMPONENT set does).
    return clamp(
        _component_overlap(opp, agent_components,
                           clamp(sector_familiarity_value, 0.0, 1.0)),
        0.0, 1.0,
    )
end

"""
    StrategyEdgeContext

Choice-set context for the S2 multiplier: per-opportunity edges (keyed by
opportunity id) and their mean over the opportunities scored in this call.
Built ONCE per choice-set pass by `strategy_edge_context`;
`strategy_score_multiplier` then does dictionary lookups only (G4 hygiene —
no per-candidate `sector_familiarity` calls inside the scoring loop).
"""
struct StrategyEdgeContext
    edges::Dict{String,Float64}
    edge_mean::Float64
end

"""
    strategy_edge_context(config, sector_knowledge, opportunities)
        -> Union{StrategyEdgeContext,Nothing}

First pass of the two-pass S2 hook (G1 fix): compute EVERY candidate's private
edge, then the multiplier centers each candidate on the choice-set mean —
multiplier_i = 1 + STRATEGY_EDGE_WEIGHT × (edge_i − mean_j edge_j) — so the
mean multiplier over the choice set is ≡ 1 by construction (no level
confound), while the founder-market-fit RANKING signal is preserved.

Returns `nothing` when S2 is inactive for `config.STRATEGY_MODE` or the
candidate set is empty (the multiplier then applies no S2 term). Deterministic
and observable-only: reads the agent's own sector knowledge and the same
opportunity opacity inputs production scoring already uses; consumes no RNG.

NOTE (documented double-dip resolution): `evaluate_opportunity_basic` already
applies ×(1 + 0.5·fam) to the same score. With within-set centering the S2
term adds a GRADIENT on top of that base gradient — an intentional, dialable
steepening of the founder-market-fit slope (dial: STRATEGY_EDGE_WEIGHT), not
a hidden level artifact. See the strategy-ladder design notes.
"""
function strategy_edge_context(
    config::EmergentConfig,
    sector_knowledge,
    opportunities::AbstractVector,
    agent_components::Set{String}=Set{String}(),
)::Union{StrategyEdgeContext,Nothing}
    strategy_uses_s2(config.STRATEGY_MODE) || return nothing
    isempty(opportunities) && return nothing
    _strategy_count!()
    edges = Dict{String,Float64}()
    total = 0.0
    for opp in opportunities
        # Single fam computation per (agent, opportunity) per choice-set pass
        # (G4b): the multiplier reuses the cached edge instead of re-calling
        # sector_familiarity per candidate.
        fam = sector_familiarity(
            sector_knowledge,
            opp.sector;
            opportunity=opp,
            config=config,
            default=0.1,
        )
        # agent_components empty (default / flag off) → private_edge returns the
        # sector-familiarity fallback (fam), so this is byte-identical when off.
        e = private_edge(opp, sector_knowledge, fam, agent_components)
        edges[String(opp.id)] = e
        total += e
    end
    return StrategyEdgeContext(edges, total / length(opportunities))
end

# ============================================================================
# COMBINED PER-OPPORTUNITY SCORE MULTIPLIER (S2 + S3b)
# ============================================================================

"""
Per-opportunity score multiplier applied at both invest-scoring hook sites
(calculate_investment_utility and evaluate_portfolio_opportunities ranking).
Second pass of the two-pass S2 hook: the caller builds a
`StrategyEdgeContext` over the FULL candidate set once per choice-set pass
(`strategy_edge_context`), then this multiplier centers each candidate on the
choice-set mean edge.

S2 (G1 form): 1 + STRATEGY_EDGE_WEIGHT × (edge_i − mean_j edge_j) — re-ranks
    toward the agent's private edge with mean multiplier ≡ 1 per choice set
    (no flat invest-propensity suppression; level confound removed by
    construction). The gradient stacks on top of the base score's (1+0.5·fam)
    — an intentional, documented steepening with STRATEGY_EDGE_WEIGHT as the
    dial (see strategy_edge_context docstring).
S3b: when the agent's own instrument reports LOW confidence (< 0.5), partially
    cancel evaluate_opportunity_basic's confidence penalty (score × (0.75 +
    0.5·conf)) by the ratio with conf moved toward neutral 0.5, scaled by
    STRATEGY_COMPLEMENT_SHIFT × market signal-commonality. Oracle-blind
    opportunities become RELATIVELY more attractive exactly when shared
    signals dominate the market — never by boosting beyond what a neutral
    signal would have scored. High-confidence boosts are not dampened.

Reads only: the precomputed edge context (agent's own knowledge + the
sector_familiarity opacity inputs production scoring already uses),
info.confidence, perception aggregates. Clamped to [0.25, 2.0] to keep the
adjustment inspectable (at the default weight the clamp never binds for
edges in [0,1], preserving the exact mean-1 invariant).
"""
function strategy_score_multiplier(
    config::EmergentConfig,
    edge_context::Union{StrategyEdgeContext,Nothing},
    opp,
    info::Union{Information,Nothing},
    perception::Perception,
)::Float64
    mode = config.STRATEGY_MODE
    mult = 1.0

    if strategy_uses_s2(mode) && !isnothing(edge_context)
        _strategy_count!()
        # Candidates outside the context (cannot occur at the production hook
        # sites, which build the context over the same pool they score) take
        # the mean edge — a neutral multiplier of exactly 1.
        edge = get(edge_context.edges, String(opp.id), edge_context.edge_mean)
        edge_weight = clamp(Float64(config.STRATEGY_EDGE_WEIGHT), 0.0, 2.0)
        mult *= 1.0 + edge_weight * (edge - edge_context.edge_mean)
    end

    if strategy_uses_s3(mode)
        _strategy_count!()
        conf = _observable_confidence(info)
        if conf < 0.5
            strength = clamp(Float64(config.STRATEGY_COMPLEMENT_SHIFT), 0.0, 1.0)
            commonality = market_signal_commonality(perception)
            conf_softened = conf + strength * commonality * (0.5 - conf)
            # Ratio of the confidence factor evaluate_opportunity_basic applies
            # (0.75 + 0.5·conf) at the softened vs raw confidence — cancels
            # part of the penalty without touching the reactive function.
            mult *= (0.75 + 0.5 * conf_softened) / (0.75 + 0.5 * conf)
        end
    end

    return clamp(mult, 0.25, 2.0)
end

# ============================================================================
# S3a — COMPLEMENT-SEEKING ACTION SHIFT
# ============================================================================

"""
    complement_shift(signal_commonality, low_confidence_availability, strength)

Additive pre-sigmoid explore/innovate utility shift (design doc S3): rises
with market signal-commonality (everyone is acting on the same legible
signals) AND with the availability of "my AI is unsure here" targets in the
agent's visible set. The 0.25 cap at strength 1.0 keeps the shift on the same
scale as the existing raw-score terms (which range ~0.1–0.4).
"""
function complement_shift(
    signal_commonality::Float64,
    low_confidence_availability::Float64,
    strength::Float64,
)::Float64
    return clamp(strength, 0.0, 2.0) * 0.25 *
           clamp(signal_commonality, 0.0, 1.0) *
           clamp(low_confidence_availability, 0.0, 1.0)
end

"""
S3a hook (called from make_decision! AFTER calculate_investment_utility has
populated the information cache): compute the explore/innovate utility shift
from the agent's OWN already-purchased Information objects.

Reads info_system.information_cache directly (keyed (opp.id, tier, agent.id))
rather than calling get_information: re-reading paid-for information is not a
new analytical task (no charge) and consumes no RNG (determinism/pairing
requirement). Opportunities the agent holds no information on are not counted
as complement targets — the agent cannot see that its instrument is unsure
about a signal it never bought.
"""
function strategy_complement_action_shift(
    config::EmergentConfig,
    opportunities::AbstractVector,
    info_system,
    ai_level::AbstractString,
    agent_id::Int,
    perception::Perception,
)::Float64
    strategy_uses_s3(config.STRATEGY_MODE) || return 0.0
    _strategy_count!()
    isempty(opportunities) && return 0.0

    n_targets = 0
    for opp in opportunities
        info = nothing
        if !isnothing(info_system)
            info = get(info_system.information_cache,
                       (opp.id, String(ai_level), agent_id), nothing)
        end
        isnothing(info) && continue
        # A complement target: the agent's own instrument reports low
        # confidence OR high estimated uncertainty (both visible fields).
        if _observable_confidence(info) < 0.5 || _observable_uncertainty(info) > 0.5
            n_targets += 1
        end
    end

    availability = n_targets / length(opportunities)
    return complement_shift(
        market_signal_commonality(perception),
        availability,
        Float64(config.STRATEGY_COMPLEMENT_SHIFT),
    )
end
