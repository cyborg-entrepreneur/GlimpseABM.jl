"""
Action-dict field name constants.

The agent action / outcome dicts (`Dict{String,Any}`) flow from agent code
(producer) to simulation/market code (consumer). Centralizing shared key names
keeps producers and consumers aligned when a field has multiple plausible
labels.

This module centralizes the names most prone to that mismatch so producer/
consumer pairs reference the same constant. The accompanying
`test/test_action_keys.jl` enforces consistency by scanning the source for
every consumer key and asserting at least one producer writes it.

Not every action-dict key lives here — only the ones enforced by that test.
Renaming an existing one requires updating producers, consumers, and the
constant together.
"""
module ActionKeys

# Identification
const ACTION              = "action"               # invest / innovate / explore / maintain
const ACTION_TYPE         = "action_type"          # compatibility alias
const AGENT_ID            = "agent_id"
const ROUND               = "round"

# AI tier metadata
const AI_LEVEL            = "ai_level"
const AI_LEVEL_USED       = "ai_level_used"
const AI_TIER_USED        = "ai_tier_used"

# Investment action — capital + identifiers
# amount = capital deployed in THIS round's invest action.
# investment_amount = lifetime / matured-outcome variant (DIFFERENT semantic).
const AMOUNT              = "amount"               # primary key for current-round invest
const INVESTMENT_AMOUNT   = "investment_amount"    # only emitted by matured-outcome records
const OPPORTUNITY_ID      = "opportunity_id"
const CHOSEN_OPP_OBJ      = "chosen_opportunity_obj"
const MATURITY_ROUND      = "maturity_round"

# Investment action — perception
const ESTIMATED_RETURN    = "estimated_return"     # AI-tier-aware estimate (canonical)
const EXPECTED_RETURN     = "expected_return"      # compatibility alias
const INFO_QUALITY_USED   = "info_quality_used"
const INFO_BREADTH_USED   = "info_breadth_used"
const COMPETITION_AT_INV  = "competition_at_investment"

# Information / hallucination propagation (record_ai_signals!)
const AI_CONTAINS_HALLUC  = "ai_contains_hallucination"
const AI_CONFIDENCE       = "ai_confidence"
const AI_ACTUAL_ACCURACY  = "ai_actual_accuracy"

# Innovation action — full Information propagation
const INNOVATION_ID       = "innovation_id"
const INNOVATION_TYPE     = "innovation_type"
const INNOVATION_QUALITY  = "innovation_quality"
const INNOVATION_NOVELTY  = "innovation_novelty"
const INNOVATION_SCARCITY = "innovation_scarcity"
const INNOVATION_SECTOR   = "innovation_sector"
const INNOVATION_RETURN   = "innovation_return"
const COMBINATION_SIG     = "combination_signature"
const KNOWLEDGE_COMPS     = "knowledge_components"
const IS_NEW_COMBINATION  = "is_new_combination"
const AI_ASSISTED         = "ai_assisted"
const AI_DOMAINS_USED     = "ai_domains_used"
const CASH_MULTIPLE       = "cash_multiple"
const MARKET_IMPACT       = "market_impact"
const SUCCESS             = "success"

# Explore action
# explore_cost = capital spent on exploration in this round (distinct from the
# generic "cost" key; consumers must read this exact name).
const EXPLORE_COST        = "explore_cost"
const DISCOVERED_SECTOR   = "discovered_sector"
const EXPLORATION_TYPE    = "exploration_type"      # "niche_discovery" triggers create_niche_opportunity
const CREATED_NICHE       = "created_niche"
const DISCOVERED_NICHE    = "discovered_niche"
const KNOWLEDGE_GAIN      = "knowledge_gain"
const NICHE_MODIFIER      = "niche_modifier"
const NICHE_SECTOR        = "niche_sector"

# Generic / cost accounting
const CAPITAL_BEFORE      = "capital_before"
const CAPITAL_AFTER       = "capital_after"
const CAPITAL_RETURNED    = "capital_returned"
const RECOVERY            = "recovery"
const REASON              = "reason"
const FAILURE_REASON      = "failure_reason"

end  # module
