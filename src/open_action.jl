"""
Open-action extension — A1 pivot (abandonment option) + A2 directed creation
(Hayekian redirection).

Implements the open-action extension specified in
the design notes §"Open-action extension":

  A1 `ENABLE_PIVOT`             per-round portfolio review; an agent may
                                liquidate an in-flight investment at an
                                age-dependent haircut and redeploy. Theory:
                                real options (abandonment value); lean pivots.
  A2 `ENABLE_DIRECTED_CREATION` innovating agents bias new-combination SECTOR
                                selection away from perceived-crowded sectors
                                toward sparse ones (Hayek 1945 local
                                knowledge). Direction changes; the RATE of
                                innovation attempts does not.

OBSERVABILITY CONTRACT (same audit as strategy.jl, enforced by
test/test_open_action_space.jl): functions in this file may read ONLY
decision-time observables —

  A1 pivot trigger: values the agent itself stored at commitment time
  (committed amount, round invested / maturity round, decision-time expected
  return, decision confidence, perceived crowding at commitment), CURRENT
  market-level perceived crowding (Perception aggregates), and the agent's
  OWN instrument's current-round estimate IF already cached (free cache
  read — the review never forces a charged get_information call).
  A2 density estimate: the sector mix of the agent's own visible-opportunity
  set, weighted by each opportunity's observable competition trace (the same
  field production perception already reads).

They must NEVER read opp.latent_return_potential, opp.latent_failure_potential,
info.hidden_factors, info.contains_hallucination, info.actual_accuracy, other
agents' current-round choices, or any RNG stream (the directed-creation DRAW
uses the agent's rng, but only when the channel is enabled — the disabled
path consumes zero extra RNG, preserving bit-identical defaults and seed
pairing).

The disabled default is bit-identical to the closed-action model: the pivot
review and the density computation gate on the config flags BEFORE any
computation runs (pinned by the default-neutrality test and the debug counter
below).
"""

# Debug counter, mirror of strategy.jl's STRATEGY_EVAL_COUNT: incremented by
# every open-action computation (pivot-trigger evaluation, sector-density
# estimate). The default-neutrality test asserts it stays at 0 for an entire
# flags-off simulation. Atomic because suite runs execute on multiple threads.
#
# Hot-path hygiene (design note): atomic increments on every
# trigger evaluation are pure debug overhead in production runs, so they are
# gated behind OPEN_ACTION_COUNT_ENABLED (default false). The neutrality tests
# flip the flag on around their assertions; the structural ==0 guarantee is
# unchanged because the counted call sites are identical.
const OPEN_ACTION_EVAL_COUNT = Threads.Atomic{Int}(0)
const OPEN_ACTION_COUNT_ENABLED = Ref{Bool}(false)

# Debug-only deterioration samples collected by pivot_trigger when the debug
# flag is on (consumed by a liveness probe to print the
# observed deterioration distribution). Lock-guarded for the threaded suite;
# zero cost in production (single Bool check, flag default false).
const PIVOT_DETERIORATION_SAMPLES = Float64[]
const _OPEN_ACTION_DEBUG_LOCK = ReentrantLock()

set_open_action_count_enabled!(flag::Bool) = (OPEN_ACTION_COUNT_ENABLED[] = flag; nothing)
reset_open_action_eval_count!() = (Threads.atomic_xchg!(OPEN_ACTION_EVAL_COUNT, 0); nothing)
open_action_eval_count()::Int = OPEN_ACTION_EVAL_COUNT[]
_open_action_count!() =
    (OPEN_ACTION_COUNT_ENABLED[] && Threads.atomic_add!(OPEN_ACTION_EVAL_COUNT, 1); nothing)

function reset_pivot_deterioration_samples!()
    lock(_OPEN_ACTION_DEBUG_LOCK) do
        empty!(PIVOT_DETERIORATION_SAMPLES)
    end
    return nothing
end

_record_pivot_deterioration_sample!(d::Float64) =
    (OPEN_ACTION_COUNT_ENABLED[] && lock(() -> push!(PIVOT_DETERIORATION_SAMPLES, d),
                                          _OPEN_ACTION_DEBUG_LOCK); nothing)

# ============================================================================
# A1 — PIVOT (ABANDONMENT OPTION)
# ============================================================================

# The redeployment margin, conviction base, and deterioration gain were
# promoted to documented config fields (PIVOT_REDEPLOY_MARGIN,
# PIVOT_CONVICTION_BASE, PIVOT_DETERIORATION_GAIN — (design note)
# 2026-06-09; defaults bit-identical to the original hardcoded trigger). See
# config.jl for the field documentation, including the censoring analysis
# motivating the gain dial.

"""
    pivot_haircut(config, progress)::Float64

Recovered FRACTION of committed capital at maturity progress
`progress = investment_age / time_to_maturity ∈ [0,1]`: linear from
PIVOT_HAIRCUT_FLOOR at commitment to PIVOT_HAIRCUT_CEILING just before
resolution. Real-options logic: abandonment value rises as resolution
approaches (more of the venture is verifiable/salvageable), while early kills
recover less because sunk setup costs dominate. Deliberately the simplest
inspectable form.
"""
function pivot_haircut(config::EmergentConfig, progress::Float64)::Float64
    p = clamp(progress, 0.0, 1.0)
    return config.PIVOT_HAIRCUT_FLOOR +
           (config.PIVOT_HAIRCUT_CEILING - config.PIVOT_HAIRCUT_FLOOR) * p
end

"""
    pivot_trigger(config; progress, crowding_now, crowding_at_commit,
                  decision_confidence, est_multiple)
        -> (pivot::Bool, expected_residual_per_dollar, recoverable_per_dollar)

The exact pivot rule (per-dollar form; both sides scale by the committed
amount, so the comparison is amount-free). All inputs are decision-time
observables (see file docstring):

  deterioration = clamp(crowding_now − crowding_at_commit, 0, 1)
      perceived market crowding has worsened since the agent committed
      (current Perception aggregate vs the agent's own stored perception).
  d_eff = clamp(deterioration × (PIVOT_CONVICTION_BASE − decision_confidence)
                × PIVOT_DETERIORATION_GAIN, 0, 1)
      the agent's ORIGINAL conviction modulates how hard the deterioration
      signal bites: a low-confidence committer (conf→0, factor→BASE) folds
      quickly; a high-conviction one (conf→1, factor→BASE−1) holds through
      the same signal. The GAIN dial (default 1.0 — bit-identical) exists
      because the verified censoring analysis showed the observed
      deterioration range (≈0–0.28 at BASELINE) sits mostly below the
      trigger region (d_eff ≈0.45–0.79 for early/mid-life pivots); gain
      ≈2.5–3 maps the observed range onto the trigger region at mid-progress
      (see the config-field docs).
  expected_residual = max(est_multiple, 0) × (1 − d_eff)
      decision-relevant expected payoff per committed dollar at maturity:
      the agent's stored commitment-time INSTRUMENT estimate (the raw
      Information.estimated_return persisted at decision time — F1),
      discounted by the conviction-weighted perceived deterioration.
  recoverable = pivot_haircut(config, progress)

  PIVOT  ⇔  expected_residual < recoverable × (1 + PIVOT_REDEPLOY_MARGIN)

i.e. liquidate when what the agent now expects the position to pay is worth
less than what it can recover today plus a small redeployment margin.
"""
function pivot_trigger(
    config::EmergentConfig;
    progress::Float64,
    crowding_now::Float64,
    crowding_at_commit::Float64,
    decision_confidence::Float64,
    est_multiple::Float64,
)::Tuple{Bool,Float64,Float64}
    _open_action_count!()
    deterioration = clamp(
        clamp(crowding_now, 0.0, 1.0) - clamp(crowding_at_commit, 0.0, 1.0),
        0.0, 1.0,
    )
    _record_pivot_deterioration_sample!(deterioration)
    conviction_factor = config.PIVOT_CONVICTION_BASE - clamp(decision_confidence, 0.0, 1.0)
    d_eff = clamp(deterioration * conviction_factor * config.PIVOT_DETERIORATION_GAIN,
                  0.0, 1.0)
    expected_residual = max(est_multiple, 0.0) * (1.0 - d_eff)
    recoverable = pivot_haircut(config, progress)
    return (expected_residual < recoverable * (1.0 + config.PIVOT_REDEPLOY_MARGIN),
            expected_residual, recoverable)
end

# Observable-extractor surface (mirrors strategy.jl's extractors): only the
# VISIBLE estimated_return field of an Information object — never
# hidden_factors, contains_hallucination, or actual_accuracy. Since
# design note the pivot trigger reads the decision-time-
# STORED instrument estimate rather than a same-round cache re-query, so this
# extractor is no longer on the trigger path; it is retained as the
# documented observable surface for any future open-action reader of a live
# Information object.
_observable_estimated_return(info::Information)::Float64 = Float64(info.estimated_return)
_observable_estimated_return(::Nothing)::Union{Float64,Nothing} = nothing

# ============================================================================
# A2 — DIRECTED CREATION (HAYEKIAN REDIRECTION)
# ============================================================================

"""
    perceived_sector_density(opportunities)::Dict{String,Float64}

The agent's seat-level perceived sector density: the sector mix of its OWN
visible-opportunity set, with each opportunity weighted by (1 + observable
competition trace), normalized to sum to 1. Sectors overrepresented in the
visible set with high competition traces are "crowded" from this agent's
seat; sectors absent from the visible set have density 0 (maximally sparse).

This is the density estimate the design doc specifies for the case actually
present in the model: only market-LEVEL crowding is exposed through
perception, so per-sector density must come from what the agent can see —
its visible opportunities' sectors (observable) and their competition traces
(the same opp.competition field production perception already aggregates).
No latent field is read.
"""
function perceived_sector_density(opportunities::AbstractVector)::Dict{String,Float64}
    _open_action_count!()
    density = Dict{String,Float64}()
    for opp in opportunities
        sector = String(opp.sector)
        trace = hasfield(typeof(opp), :competition) ?
            clamp(Float64(opp.competition), 0.0, 1.0) : 0.0
        density[sector] = get(density, sector, 0.0) + (1.0 + trace)
    end
    total = sum(values(density); init=0.0)
    if total > 0.0
        for k in keys(density)
            density[k] /= total
        end
    end
    return density
end

# Knowledge-anchor prior: the sector the deterministic knowledge-domain
# mapping would have chosen keeps a 2:1 prior over every other sector, so
# directed creation REDIRECTS the knowledge linkage rather than severing it
# (an agent still innovates from what it knows; density reweights where it
# points that knowledge).
const DIRECTED_CREATION_ANCHOR_PRIOR = 2.0

"""
    directed_sector_weights(sectors, density, anchor_sector, strength)
        -> Vector{Float64}

Pure weights function for the TILTED component of the directed-creation
mixture (G3, 2026-06-09): `determine_innovation_sector` (innovation.jl) keeps
the knowledge-mapped anchor with probability exp(−strength) and only draws
from these weights otherwise, so the strength dial nests the flag-off
baseline. Sector-selection weights:

  w_s ∝ prior_s × exp(−strength × clamp(z_s, −3, 3))

where z_s is the z-score of the agent's perceived density share of sector s
across `sectors` (all-equal densities ⇒ z ≡ 0 ⇒ weights reduce to the prior),
and prior_s = DIRECTED_CREATION_ANCHOR_PRIOR for the knowledge-mapped
`anchor_sector`, 1 otherwise. The exponential form makes the redirection
strength a clean log-linear dial; the ±3 z-clamp keeps weights inspectable
(max density tilt e^{6·strength} between the most- and least-crowded sector).
"""
function directed_sector_weights(
    sectors::AbstractVector{String},
    density::AbstractDict{String,Float64},
    anchor_sector::String,
    strength::Float64,
)::Vector{Float64}
    isempty(sectors) && return Float64[]
    shares = [get(density, s, 0.0) for s in sectors]
    mu = mean(shares)
    sd = length(shares) > 1 ? std(shares) : 0.0
    weights = Vector{Float64}(undef, length(sectors))
    for (i, s) in enumerate(sectors)
        z = sd > 1e-12 ? (shares[i] - mu) / sd : 0.0
        prior = s == anchor_sector ? DIRECTED_CREATION_ANCHOR_PRIOR : 1.0
        weights[i] = prior * exp(-max(strength, 0.0) * clamp(z, -3.0, 3.0))
    end
    return weights
end
