"""
Typed market-state payload consumed by agent decisions, `realized_return`,
and the uncertainty layer. A typed struct (rather than a `Dict{String,Any}`)
guarantees every field is present and typed, so a misspelled key fails at
field-access time rather than silently returning a default. Producer is
`market.jl:get_market_conditions`.

The `extras::Dict{String,Any}` escape hatch holds one-off experimental
fields (e.g., `combo_hhi`) with no settled producer/consumer contract.
Prefer adding a typed field above to reaching for `extras`.
"""
struct MarketConditions
    # Regime + macro
    regime::String
    volatility::Float64
    regime_return_multiplier::Float64
    regime_failure_multiplier::Float64

    # Market scale
    round::Int

    # Crowding / clearing
    tier_invest_share::Dict{String,Float64}
    sector_clearing_index::Dict{String,Float64}
    aggregate_clearing_ratio::Float64
    crowding_metrics::Dict{String,Float64}
    sector_demand_adjustments::Dict{String,Dict{String,Float64}}
    avg_competition::Float64

    # Uncertainty hook — genuinely variant shape, stays flexible
    uncertainty_state::Dict{String,Any}

    # Escape hatch for one-off experimental fields
    extras::Dict{String,Any}
end

# ────────────────────────────────────────────────────────────────────────
# Dict-like read access. Keeps `get(mc, "regime", …)`-style call sites
# working; core consumers use field access directly, so this shim is
# defensive for diagnostic and script code.
#
# Silent-zero prevention: a read against a truly-unknown key returns the
# provided `default` (matching Dict semantics), but a read against a field
# that EXISTS always returns the field value (no default applied). So a
# present-but-zero field can never be masked by a default.
# ────────────────────────────────────────────────────────────────────────

function Base.get(mc::MarketConditions, key::AbstractString, default)
    sym = Symbol(key)
    if hasfield(MarketConditions, sym)
        return getfield(mc, sym)
    end
    return get(mc.extras, String(key), default)
end

function Base.haskey(mc::MarketConditions, key::AbstractString)
    sym = Symbol(key)
    return hasfield(MarketConditions, sym) || haskey(mc.extras, String(key))
end

function Base.getindex(mc::MarketConditions, key::AbstractString)
    sym = Symbol(key)
    if hasfield(MarketConditions, sym)
        return getfield(mc, sym)
    end
    return mc.extras[String(key)]
end
