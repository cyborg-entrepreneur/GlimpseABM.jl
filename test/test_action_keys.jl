# Producer/consumer consistency check for action-dict keys.
#
# v2.3 auditing caught three "silent zero" bugs where a consumer read key X but
# producers emitted key Y (e.g. "investment_amount" vs "amount"). Both names
# look reasonable in isolation; the bug is invisible to a code reader.
#
# This test scans the source for every consumer key (`get(<dict>, "X", ...)`
# and `<dict>["X"]` reads) and asserts that at least one producer somewhere
# in the codebase writes that key. Catches typos automatically.

using Test

const SRC_DIR = joinpath(@__DIR__, "..", "src")

# Files to scan (skip action_keys.jl itself — it's the constants module)
const SCAN_FILES = filter(f -> endswith(f, ".jl") && f != "action_keys.jl",
                          readdir(SRC_DIR))

# Keys explicitly allowed to be read without a matching write in src/.
# These either come from external sources (config, perception payloads,
# market_conditions populated by market.jl which uses different patterns)
# or are read with documented fallback defaults that don't depend on a
# producer existing.
const ALLOWED_CONSUMER_ONLY = Set([
    # Market conditions / config — populated by market.jl get_market_conditions
    # under different patterns (Symbol keys, struct fields)
    "regime", "uncertainty_state", "volatility", "regime_return_multiplier",
    "regime_failure_multiplier", "avg_competition", "sector_clearing_index",
    "tier_invest_share",
    # Perception payload — populated by uncertainty.jl perceive_uncertainty
    "actor_ignorance", "practical_indeterminism", "agentic_novelty",
    "competitive_recursion", "decision_confidence", "knowledge_signal",
    "execution_risk", "innovation_signal", "competition_signal",
    "neighbor_signals", "level", "ignorance_level", "confidence",
    "gap_pressure", "indeterminism_level", "novelty_potential",
    "recursion_level", "opportunity_interest", "opportunity_sentiment",
    "sector_sentiment", "peer_roi_gap", "ai_adoption_pressure",
    "ai_distribution",
    # Optional config-driven fields with documented fallback defaults
    "info_quality", "info_breadth", "per_use_cost",
    # TraitDistribution params consumed by utils.jl sample_trait with a
    # documented fallback (1.0). No default trait uses the lognormal spec
    # since the 2026-06-09 innovativeness respecification, so no Dict-literal
    # producer remains in src/, but the lognormal branch is a supported
    # config option.
    "sigma",
    # Innovation outcome details (subdict of action)
    "chosen_opportunity_details", "innovation_details", "innovation_obj",
    # Agent fields read via getattr-style get (not action dict)
    "operating_cost_estimate", "primary_sector", "risk_tolerance",
    # Recovery / fallback only consulted when matured-outcome record is missing
    "expected_return",
    # Legacy alias for "amount" — kept as fallback in market.jl consumers
    "capital_deployed",
    # market_conditions fields populated by market.jl get_market_conditions
    "ai_tier_shares", "sector_demand_adjustments", "aggregate_clearing_ratio",
    # v3.0: MarketConditions is now a typed struct; field access replaces
    # string-keyed reads. The consistency scan still sees these as orphan
    # "consumer-only" keys because writes are now field assignments, not
    # string-keyed dict entries. All allowlisted here — see
    # market_conditions.jl for the authoritative schema.
    "trend", "momentum", "n_opportunities", "exploration_activity",
    "crowding_metrics", "combo_hhi", "crowding_index", "extras",
    # Phase 1 of dict→struct migration: OpportunityEvaluation is now a typed
    # struct (models.jl). These keys were previously consumed via dict access
    # on the eval_record returned from evaluate_portfolio_opportunities;
    # consumers now use field access (best_eval.final_score etc.) so the
    # scanner no longer sees matching producer entries.
    "final_score", "competition_at_evaluation",
])

function read_source(path::String)::String
    read(path, String)
end

function extract_consumer_keys(src::String)::Set{String}
    keys = Set{String}()
    # Pattern A: get(<anything>, "key", ...)
    for m in eachmatch(r"\bget\s*\(\s*[A-Za-z_][\w\.\[\]]*\s*,\s*\"([a-z_][a-z_0-9]*)\"", src)
        push!(keys, m.captures[1])
    end
    # Pattern B: get!(<anything>, "key", ...)
    for m in eachmatch(r"\bget!\s*\(\s*[A-Za-z_][\w\.\[\]]*\s*,\s*\"([a-z_][a-z_0-9]*)\"", src)
        push!(keys, m.captures[1])
    end
    # Pattern C: <dict>["key"] (read in any context, including assign — we'll
    # trim down by also collecting writes, then any key with a write is fine).
    # Match anything ending in ["key"] regardless of bracket count quirks
    for m in eachmatch(r"\[\s*\"([a-z_][a-z_0-9]*)\"\s*\]", src)
        push!(keys, m.captures[1])
    end
    return keys
end

function extract_producer_keys(src::String)::Set{String}
    keys = Set{String}()
    # Pattern A: <dict>["key"] = value
    for m in eachmatch(r"\[\s*\"([a-z_][a-z_0-9]*)\"\s*\]\s*=", src)
        push!(keys, m.captures[1])
    end
    # Pattern B: "key" => value  (Dict literal entry)
    for m in eachmatch(r"\"([a-z_][a-z_0-9]*)\"\s*=>\s*", src)
        push!(keys, m.captures[1])
    end
    return keys
end

@testset "Action-dict producer/consumer consistency" begin
    all_consumers = Set{String}()
    all_producers = Set{String}()

    for f in SCAN_FILES
        path = joinpath(SRC_DIR, f)
        src = read_source(path)
        union!(all_consumers, extract_consumer_keys(src))
        union!(all_producers, extract_producer_keys(src))
    end

    # Every key consumed must either be produced somewhere or be on the allowlist.
    orphan_keys = setdiff(all_consumers, all_producers, ALLOWED_CONSUMER_ONLY)

    if !isempty(orphan_keys)
        println("\nORPHAN consumer keys (read but never written, not allowlisted):")
        for k in sort(collect(orphan_keys))
            println("  - \"$k\"")
        end
        println()
    end
    @test isempty(orphan_keys)
end

println("Action-key consistency check passed.")
